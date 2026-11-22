#!/usr/bin/env python3
"""Translation worker for the yk-auto-translate WordPress plugin.

Pulls pending jobs over HTTPS, translates Arabic <-> English with an open
model on CPU (CTranslate2), posts the results back. No vendor API, no cap.
Env: YK_SITE_URL, YK_WORKER_TOKEN, YK_ENGINE (opus|m2m100), YK_MODEL_DIR,
YK_BATCH, YK_LIMIT, YK_MAX_MINUTES.  Flags: --selftest, --engine, --limit.
"""
import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path

import requests

AR_RE = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
LAT_RE = re.compile(r"[A-Za-z]")
LETTER_RE = re.compile(r"[^\W\d_]")
# Items kept verbatim: HTML tags/comments, URLs, e-mails, phone numbers, numbers.
PROTECT_RE = re.compile(
    r"<!--.*?-->|</?[A-Za-z][^<>]*>|(?:https?://|www\.)[^\s<>\"']+"
    r"|[\w.+-]+@[\w-]+(?:\.[\w-]+)+|\+?\d[\d\s().\-]{6,}\d|\d+(?:[.,٫٬]\d+)*",
    re.S,
)
# Placeholder: plain Latin letters only (digits become A-J), so the SentencePiece
# vocabularies know the characters and the model tends to copy it as is.
LETTERS = "ABCDEFGHIJ"
PH_RE = re.compile(r"QX\s*([A-J]+)\s*XQ", re.I)
SENT_RE = re.compile(r"(\s*)([^.!?؟؛]*[.!?؟؛]+|[^.!?؟؛]+)(\s*)")
SENT_CAP = 400

STATS = {"done": 0, "failed": 0, "chars": 0, "model_s": 0.0, "kept": 0}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- text handling
def ph(i):
    return "QX" + "".join(LETTERS[int(c)] for c in str(i)) + "XQ"


def protect(text):
    items = []

    def sub(m):
        s, tail = m.group(0), ""
        if s.lower().startswith(("http", "www")):
            core = s.rstrip(".,;:!?)،؛؟")
            tail, s = s[len(core):], core
        items.append(s)
        return ph(len(items) - 1) + tail

    return PROTECT_RE.sub(sub, text), items


def restore(text, items):
    def sub(m):
        i = int("".join(str(LETTERS.index(c.upper())) for c in m.group(1)))
        return items[i] if i < len(items) else m.group(0)

    return PH_RE.sub(sub, text)


def ph_ids(s):
    return sorted(m.group(1).upper() for m in PH_RE.finditer(s))


def needs_model(piece):
    return bool(LETTER_RE.search(PH_RE.sub("", piece)))


def script_matches(prot, target):
    t = PH_RE.sub("", prot)
    ar, lat = len(AR_RE.findall(t)), len(LAT_RE.findall(t))
    if ar + lat == 0:
        return True  # nothing translatable
    return ar > lat if target == "ar" else lat > ar


def hard_split(s, cap):
    out, cur = [], ""
    for w in s.split(" "):
        if cur and len(cur) + 1 + len(w) > cap:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    out.append(cur)
    return out


def split_long(core, cap=SENT_CAP):
    if len(core) <= cap:
        return [core.strip()]
    pieces, cur = [], ""
    for part in re.split(r"(?<=[,،])", core):
        if cur and len(cur) + len(part) > cap:
            pieces.append(cur)
            cur = part
        else:
            cur += part
    pieces.append(cur)
    final = []
    for p in pieces:
        final.extend(hard_split(p.strip(), cap) if len(p) > cap else [p])
    return [p.strip() for p in final if p.strip()]


def parse(text):
    """-> list of str (literal) or (lead, [pieces], trail)."""
    items = []
    for part in re.split(r"(\n+)", text):
        if not part:
            continue
        if part.startswith("\n"):
            items.append(part)
            continue
        for m in SENT_RE.finditer(part):
            lead, core, trail = m.groups()
            if not core.strip():
                items.append(m.group(0))
            else:
                items.append((lead, split_long(core), trail))
    return items


# ---------------------------------------------------------------- engines
class Engine:
    """Interface: translate(texts, src, tgt) -> list of translated strings."""
    name = "base"

    def translate(self, texts, src, tgt):
        raise NotImplementedError


class CT2Engine(Engine):
    """Shared logic: convert a HF model to CTranslate2 int8 once, then load it."""

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.loaded = {}

    def _load(self, model_id, tok_cls):
        if model_id in self.loaded:
            return self.loaded[model_id]
        import ctranslate2
        path = self.model_dir / model_id.replace("/", "__")
        if not (path / "model.bin").exists():
            from ctranslate2.converters import TransformersConverter
            from transformers import AutoTokenizer
            log(f"converting {model_id} to CTranslate2 int8 (first use only)")
            tmp = Path(str(path) + ".tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            self.model_dir.mkdir(parents=True, exist_ok=True)
            TransformersConverter(model_id).convert(str(tmp), quantization="int8", force=True)
            AutoTokenizer.from_pretrained(model_id).save_pretrained(str(tmp))
            shutil.rmtree(path, ignore_errors=True)
            tmp.rename(path)
        tok = tok_cls.from_pretrained(str(path))
        trans = ctranslate2.Translator(str(path), device="cpu", compute_type="int8",
                                       inter_threads=1, intra_threads=os.cpu_count() or 2)
        self.loaded[model_id] = (trans, tok)
        return self.loaded[model_id]

    @staticmethod
    def _decode(tok, tokens):
        return tok.decode(tok.convert_tokens_to_ids(tokens), skip_special_tokens=True).strip()


class OpusEngine(CT2Engine):
    """Helsinki-NLP opus-mt-ar-en / opus-mt-en-ar (Apache-2.0)."""
    name = "opus"
    MODELS = {("ar", "en"): "Helsinki-NLP/opus-mt-ar-en", ("en", "ar"): "Helsinki-NLP/opus-mt-en-ar"}

    def translate(self, texts, src, tgt):
        from transformers import MarianTokenizer
        trans, tok = self._load(self.MODELS[(src, tgt)], MarianTokenizer)
        prefix = ">>ara<< " if tgt == "ar" else ""  # en-ar needs a target-language token
        batch = [tok.convert_ids_to_tokens(tok.encode(prefix + t, truncation=True, max_length=512))
                 for t in texts]
        res = trans.translate_batch(batch, beam_size=2, max_decoding_length=300)
        return [self._decode(tok, r.hypotheses[0]) for r in res]


class M2MEngine(CT2Engine):
    """facebook/m2m100_418M (MIT)."""
    name = "m2m100"
    MODEL = "facebook/m2m100_418M"

    def translate(self, texts, src, tgt):
        from transformers import M2M100Tokenizer
        trans, tok = self._load(self.MODEL, M2M100Tokenizer)
        tok.src_lang = src
        batch = [tok.convert_ids_to_tokens(tok.encode(t, truncation=True, max_length=512)) for t in texts]
        res = trans.translate_batch(batch, target_prefix=[[f"__{tgt}__"]] * len(batch),
                                    beam_size=2, max_decoding_length=300)
        return [self._decode(tok, r.hypotheses[0][1:]) for r in res]


ENGINES = {"opus": OpusEngine, "m2m100": M2MEngine}


class Translator:
    def __init__(self, engine, batch):
        self.engine, self.batch = engine, max(1, batch)

    def _run(self, pieces, src, tgt):
        done = {}
        for i in range(0, len(pieces), self.batch):
            chunk = pieces[i:i + self.batch]
            t0 = time.time()
            outs = self.engine.translate(chunk, src, tgt)
            STATS["model_s"] += time.time() - t0
            STATS["chars"] += sum(len(c) for c in chunk)
            for s, o in zip(chunk, outs):
                if not o or ph_ids(o) != ph_ids(s):
                    STATS["kept"] += 1  # model lost a placeholder: keep source piece
                    log(f"placeholder lost or empty output, piece kept untranslated: {s[:60]!r}")
                    o = s
                done[s] = o
        return done

    def field(self, text, src, tgt):
        if not isinstance(text, str) or not text.strip():
            return text
        prot, items = protect(text)
        if script_matches(prot, tgt):
            return text
        parsed = parse(prot)
        todo = sorted({p for it in parsed if isinstance(it, tuple) for p in it[1] if needs_model(p)},
                      key=len)
        done = self._run(todo, src, tgt)
        out = []
        for it in parsed:
            if isinstance(it, str):
                out.append(it)
            else:
                lead, pieces, trail = it
                out.append(lead + " ".join(done.get(p, p) for p in pieces) + trail)
        return restore("".join(out), items)


# ---------------------------------------------------------------- site client
class Fatal(Exception):
    pass


class Site:
    def __init__(self, url, token):
        self.base = url.rstrip("/") + "/wp-json/yk-translate/v1"
        self.http = requests.Session()
        self.http.headers.update({"X-YK-Token": token, "User-Agent": "yk-translate-worker/1.0"})

    def call(self, method, path, **kw):
        last = ""
        for attempt in range(4):  # first try + 3 retries
            try:
                r = self.http.request(method, self.base + path, timeout=30, **kw)
            except (requests.ConnectionError, requests.Timeout) as e:
                last = f"network error ({type(e).__name__})"
            else:
                if r.status_code == 401:
                    raise Fatal("HTTP 401: the site rejected the worker token (check YK_WORKER_TOKEN).")
                if r.status_code < 500:
                    r.raise_for_status()
                    return r.json()
                last = f"HTTP {r.status_code}"
            if attempt < 3:
                time.sleep(2 * 2 ** attempt)
        raise RuntimeError(f"{method} {path} failed after retries: {last}")

    def pending(self, limit):
        data = self.call("GET", "/pending", params={"limit": limit})
        return data.get("jobs", []) if isinstance(data, dict) else data

    def result(self, job, engine, fields):
        return self.call("POST", "/result", json={
            "job_id": job["job_id"], "post_id": job["post_id"], "target": job["target"],
            "hash": job.get("hash", ""), "engine": engine, "fields": fields})

    def fail(self, job_id, reason):
        return self.call("POST", "/fail", json={"job_id": job_id, "reason": reason})


# ---------------------------------------------------------------- main
def selftest(tr):
    samples = [("Furniture store in Salmiya open daily", "en", "ar"),
               ("محل أثاث في السالمية يفتح يوميا", "ar", "en"),
               ("Call +965 5555 1234 or visit https://example.com/shop today.", "en", "ar")]
    for text, s, t in samples:
        out = tr.field(text, s, t)
        print(f"[{s}->{t}] {text}\n      => {out}", flush=True)


def summary(wall):
    cps = STATS["chars"] / STATS["model_s"] if STATS["model_s"] else 0.0
    log("---- summary ----")
    log(f"jobs done: {STATS['done']}  jobs failed: {STATS['failed']}  pieces kept untranslated: {STATS['kept']}")
    log(f"characters translated: {STATS['chars']}  seconds in model: {STATS['model_s']:.1f}  "
        f"chars/sec: {cps:.1f}  wall seconds: {wall:.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true", help="translate two built-in samples, no site needed")
    ap.add_argument("--engine", default=os.environ.get("YK_ENGINE", "opus"), choices=sorted(ENGINES))
    ap.add_argument("--limit", type=int, default=int(os.environ.get("YK_LIMIT", "25") or 25))
    args = ap.parse_args()

    engine = ENGINES[args.engine](os.environ.get("YK_MODEL_DIR", "./models"))
    tr = Translator(engine, int(os.environ.get("YK_BATCH", "16") or 16))
    start = time.time()
    log(f"engine={engine.name} batch={tr.batch} cpus={os.cpu_count()}")

    if args.selftest:
        selftest(tr)
        summary(time.time() - start)
        return 0

    url, token = os.environ.get("YK_SITE_URL", ""), os.environ.get("YK_WORKER_TOKEN", "")
    if not url or not token:
        log("YK_SITE_URL and YK_WORKER_TOKEN must be set.")
        return 2
    site = Site(url, token)
    deadline = start + float(os.environ.get("YK_MAX_MINUTES", "20") or 20) * 60
    seen = set()
    try:
        while time.time() < deadline:
            jobs = [j for j in site.pending(args.limit) if str(j["job_id"]) not in seen]
            if not jobs:
                log("no more pending jobs")
                break
            for job in jobs:
                if time.time() >= deadline:
                    log("time limit reached, stopping")
                    break
                seen.add(str(job["job_id"]))
                try:
                    src, tgt = job["source"], job["target"]
                    fields = {k: tr.field(v, src, tgt) for k, v in job["fields"].items()}
                    site.result(job, engine.name, fields)
                    STATS["done"] += 1
                    log(f"job {job['job_id']} (post {job['post_id']}, {src}->{tgt}) done")
                except Fatal:
                    raise
                except Exception as e:  # one bad job must not stop the run
                    STATS["failed"] += 1
                    reason = f"{type(e).__name__}: {e}"[:300]
                    log(f"job {job.get('job_id')} failed: {reason}")
                    try:
                        site.fail(job["job_id"], reason)
                    except Fatal:
                        raise
                    except Exception as e2:
                        log(f"could not report failure: {type(e2).__name__}")
    except Fatal as e:
        log(f"FATAL: {e}")
        summary(time.time() - start)
        return 1
    summary(time.time() - start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
