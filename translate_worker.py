#!/usr/bin/env python3
"""Translation worker for the yk-auto-translate WordPress plugin.

Pulls pending jobs over HTTPS, translates Arabic <-> English with an open
model on CPU (CTranslate2), posts the results back. No vendor API, no cap.
Env: YK_SITE_URL, YK_WORKER_TOKEN, YK_ENGINE (opus|m2m100|m2m100-big|madlad|tgemma), YK_OLLAMA_URL, YK_MODEL_DIR,
YK_BATCH, YK_LIMIT, YK_MAX_MINUTES, YK_GLOSSARY (path of glossary.json).
Flags: --selftest, --engine, --limit.
"""
import argparse
import json
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
# Items kept verbatim: HTML tags/comments, shortcodes, %variables%, URLs, e-mails and
# phone-like numbers (7+ digits). Plain numbers (prices, years, times) go to the model
# and are verified afterwards (see Translator._checked).
PROTECT_RE = re.compile(
    r"<!--.*?-->|</?[A-Za-z][^<>]*>|\[/?[A-Za-z][^\[\]]*\]|%[A-Za-z0-9_]+%"
    r"|(?:https?://|www\.)[^\s<>\"']+"
    r"|[\w.+-]+@[\w-]+(?:\.[\w-]+)+|(?P<phone>\+?\d[\d\s().\-]{6,}\d)",
    re.S,
)
NUM_RE = re.compile(r"\d+(?:[.,٫٬]\d+)*")
DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
# Formatting tags that may be dropped from one sentence. Block tags, <yk-keep> and
# anything else are never in this list. Links (<a>) are excluded again in _droppable.
INLINE_RE = re.compile(r"</?(?:strong|b|em|i|u|span|a)(?=[\s>/])[^<>]*>", re.I)
TIME_RE = re.compile(r"(?<![\w:.])(\d{1,2})(?::(\d{2}))?\s*([ap])(?:\.m\.?|m)(?![A-Za-z])", re.I)
# Placeholder: plain Latin letters only (digits become A-J), so the SentencePiece
# vocabularies know the characters and the model tends to copy it as is.
LETTERS = "ABCDEFGHIJ"
PH_RE = re.compile(r"QX\s*([A-J]+)\s*XQ", re.I)
SENT_RE = re.compile(r"(\s*)([^.!?؟؛]*[.!?؟؛]+|[^.!?؟؛]+)(\s*)")
SENT_CAP = 400

STATS = {"done": 0, "failed": 0, "chars": 0, "model_s": 0.0, "load_s": 0.0, "kept": 0, "split": 0,
         "num_retry": 0, "fmt_dropped": 0, "script_retry": 0, "len_reject": 0}
GLOSS = {}  # source language -> (compiled regex, lookup); filled by load_glossary()


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- text handling
def ph(i):
    return "QX" + "".join(LETTERS[int(c)] for c in str(i)) + "XQ"


def protect(text):
    items = []

    def sub(m):
        s, tail = m.group(0), ""
        if m.group("phone") and sum(c.isdigit() for c in s) < 7:
            return s  # not phone-like: leave it to the model
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


def digit_runs(s):
    return sorted(re.findall(r"\d+", s.translate(DIGIT_MAP)))


def protect_numbers(piece, items):
    """Old way: every plain number in the piece becomes a placeholder."""
    def sub(m):
        items.append(m.group(0))
        return ph(len(items) - 1)

    return NUM_RE.sub(sub, piece)


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


def pack_sentences(core, cap):
    """Greedy: whole sentences joined into pieces of at most cap characters."""
    pieces, cur = [], ""
    for m in SENT_RE.finditer(core):
        s = m.group(2).strip()
        if not s:
            continue
        if len(s) > cap:
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.extend(split_long(s, cap))
        elif cur and len(cur) + 1 + len(s) > cap:
            pieces.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}" if cur else s
    if cur:
        pieces.append(cur)
    return pieces


def parse(text, cap=SENT_CAP, sentences=True):
    """-> list of str (literal) or (lead, [pieces], trail).
    sentences=True: one item per sentence. False: one item per paragraph (line), cut on
    sentence ends only when it is longer than cap."""
    items = []
    for part in re.split(r"(\n+)", text):
        if not part:
            continue
        if part.startswith("\n"):
            items.append(part)
            continue
        if not sentences:
            core = part.strip()
            if not core:
                items.append(part)
            else:
                lead = part[:len(part) - len(part.lstrip())]
                trail = part[len(part.rstrip()):]
                items.append((lead, [core] if len(core) <= cap else pack_sentences(core, cap), trail))
            continue
        for m in SENT_RE.finditer(part):
            lead, core, trail = m.groups()
            if not core.strip():
                items.append(m.group(0))
            else:
                items.append((lead, split_long(core, cap), trail))
    return items


def split_at_ph(piece):
    """-> [("t", text) | ("p", placeholder)] in source order."""
    parts, pos = [], 0
    for m in PH_RE.finditer(piece):
        parts.append(("t", piece[pos:m.start()]))
        parts.append(("p", m.group(0)))
        pos = m.end()
    parts.append(("t", piece[pos:]))
    return parts


def _time_value(m):
    """'11:00 am' -> '11:00 صباحاً'; 12 pm -> '12 ظهراً'; 12 am -> '12 منتصف الليل'; None if not a time."""
    h, mi = int(m.group(1)), m.group(2)
    if not 1 <= h <= 12 or (mi is not None and int(mi) > 59):
        return None
    digits = m.group(1) + (":" + mi if mi else "")
    pm = m.group(3).lower() == "p"
    word = ("ظهراً" if h == 12 else "مساءً") if pm else ("منتصف الليل" if h == 12 else "صباحاً")
    return f"{digits} {word}"


# ---------------------------------------------------------------- glossary
AR_FOLD = str.maketrans("أإآ", "ااا")
AR_PRE = {"ب": "في ", "ل": "إلى ", "و": "و ", "ف": "ف ", "ك": "مثل "}  # attached prefix -> own word


def _norm_en(s):
    return re.sub(r"[\s\-]+", " ", s.strip().lower())


def _norm_ar(s):
    return " ".join(s.split())


def _ar_variants(f):
    s = {_norm_ar(f)}
    s |= {v.translate(AR_FOLD) for v in s}
    s |= {re.sub(r"ة(?=\s|$)", "ه", v) for v in s}
    return s


def load_glossary(path):
    """Compile glossary.json. A missing or invalid file logs one line and leaves no glossary."""
    GLOSS.clear()
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError("top level must be a list")
    except (OSError, ValueError) as e:
        log(f"glossary not loaded ({type(e).__name__}); continuing without one")
        return 0
    en_map, ar_map, ll_map, n = {}, {}, {}, 0
    for e in data:
        try:
            en = [str(x) for x in e["en"] if str(x).strip()]
            ar = [str(x) for x in e["ar"] if str(x).strip()]
            amb = bool(e.get("ar_ambiguous", False))
        except (TypeError, KeyError):
            continue
        if not en or not ar:
            continue
        n += 1
        for v in en:
            en_map.setdefault(_norm_en(v), _norm_ar(ar[0]))
        if not amb:  # an ordinary Arabic word is only replaced English -> Arabic
            for v in ar:
                for w in _ar_variants(v):
                    ar_map.setdefault(w, en[0])
                    if w.startswith("ال") and len(w) > 3:
                        ll_map.setdefault(w[2:], en[0])  # للسالمية = ل + ال + سالمية
    if en_map:
        alt = "|".join(sorted((r"[\s\-]+".join(re.escape(w) for w in k.split(" ")) for k in en_map),
                              key=len, reverse=True))
        GLOSS["en"] = (re.compile(r"(?<![A-Za-z0-9])(?:" + alt + r")(?![A-Za-z0-9])", re.I), en_map)
    if ar_map:
        def alt_ar(keys):
            return "|".join("[  ]+".join(re.escape(w) for w in k.split(" "))
                            for k in sorted(keys, key=len, reverse=True))
        core = r"(?P<pre>[بلوفك])?(?P<form>" + alt_ar(ar_map) + r")"
        if ll_map:
            core += r"|(?P<ll>لل)(?P<rest>" + alt_ar(ll_map) + r")"
        GLOSS["ar"] = (re.compile(r"(?<!\w)(?:" + core + r")(?!\w)"), (ar_map, ll_map))
    return n


def apply_glossary(prot, items, src, tgt):
    """Replace times (en->ar) and glossary terms by protected items holding the other language's form."""
    if (src, tgt) == ("en", "ar"):
        def tsub(m):
            val = _time_value(m)
            if val is None:
                return m.group(0)
            items.append(val)
            return ph(len(items) - 1)

        prot = TIME_RE.sub(tsub, prot)
    g = GLOSS.get(src)
    if not g or {src, tgt} != {"en", "ar"}:
        return prot
    rx, look = g

    def sub(m):
        pre = ""
        if src == "en":
            val = look[_norm_en(m.group(0))]
        else:
            ar_map, ll_map = look
            if m.groupdict().get("ll"):
                val, pre = ll_map[_norm_ar(m.group("rest"))], AR_PRE["ل"]
            else:
                val, pre = ar_map[_norm_ar(m.group("form"))], AR_PRE.get(m.group("pre") or "", "")
        items.append(val)
        return pre + ph(len(items) - 1)

    return rx.sub(sub, prot)


# ---------------------------------------------------------------- engines
class Engine:
    """Interface: translate(texts, src, tgt) -> list of translated strings."""
    name = "base"
    piece_cap = SENT_CAP        # longest piece sent in one call
    split_sentences = True      # False: pieces are whole paragraphs (LLM-style engines)
    script_check = True         # output must be mostly in the target script
    len_check = False           # reject outputs far longer or shorter than the source
    sentence_retry = False      # on a failed check, retry the piece sentence by sentence

    def start(self):
        """Called once before the first job; raise RuntimeError when the engine is unusable."""

    def translate(self, texts, src, tgt):
        raise NotImplementedError


def sh_report(label, path):
    """Log memory and disk (Linux `free -h`, `df -h`); silently nothing where they do not exist."""
    import subprocess
    for cmd in (["free", "-h"], ["df", "-h", str(path)]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.rstrip()
        except Exception:
            continue
        log(f"{label}: {' '.join(cmd)}\n{out}")


class CT2Engine(Engine):
    """Shared logic: convert a HF model to CTranslate2 int8 once, then load it."""

    SAVE_SLOW_TOKENIZER = False   # True: save the slow tokenizer class (needs no protobuf)
    REPORT_MEMORY = False         # True: log `free -h` and `df -h` around the conversion

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.loaded = {}

    def _load(self, model_id, tok_name):
        if model_id in self.loaded:
            return self.loaded[model_id]
        t0 = time.time()
        import ctranslate2
        import transformers
        tok_cls = getattr(transformers, tok_name)
        path = self.model_dir / model_id.replace("/", "__")
        if not (path / "model.bin").exists():
            from ctranslate2.converters import TransformersConverter
            from transformers import AutoTokenizer
            log(f"converting {model_id} to CTranslate2 int8 (first use only)")
            tmp = Path(str(path) + ".tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            self.model_dir.mkdir(parents=True, exist_ok=True)
            if self.REPORT_MEMORY:
                sh_report("before conversion", self.model_dir)
            # low_cpu_mem_usage: constructor argument in ctranslate2 4.6.3 (read in its source)
            TransformersConverter(model_id, low_cpu_mem_usage=True).convert(str(tmp), quantization="int8", force=True)
            saver = tok_cls if self.SAVE_SLOW_TOKENIZER else AutoTokenizer
            saver.from_pretrained(model_id).save_pretrained(str(tmp))
            shutil.rmtree(path, ignore_errors=True)
            tmp.rename(path)
            if os.environ.get("YK_PURGE_HF_CACHE") == "1":  # set by the workflow: keep the job cache small
                shutil.rmtree(Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface"),
                              ignore_errors=True)
            if self.REPORT_MEMORY:
                sh_report("after conversion", self.model_dir)
        tok = tok_cls.from_pretrained(str(path))
        trans = ctranslate2.Translator(str(path), device="cpu", compute_type="int8",
                                       inter_threads=1, intra_threads=os.cpu_count() or 2)
        self.loaded[model_id] = (trans, tok)
        STATS["load_s"] += time.time() - t0
        return self.loaded[model_id]

    @staticmethod
    def _timed(trans, batch, **kw):
        t0 = time.time()
        res = trans.translate_batch(batch, **kw)
        STATS["model_s"] += time.time() - t0
        return res

    @staticmethod
    def _decode(tok, tokens):
        return tok.decode(tok.convert_tokens_to_ids(tokens), skip_special_tokens=True).strip()


class OpusEngine(CT2Engine):
    """Helsinki-NLP opus-mt-ar-en / opus-mt-en-ar (Apache-2.0)."""
    name = "opus"
    MODELS = {("ar", "en"): "Helsinki-NLP/opus-mt-ar-en", ("en", "ar"): "Helsinki-NLP/opus-mt-en-ar"}

    def translate(self, texts, src, tgt):
        trans, tok = self._load(self.MODELS[(src, tgt)], "MarianTokenizer")
        prefix = ">>ara<< " if tgt == "ar" else ""  # en-ar needs a target-language token
        batch = [tok.convert_ids_to_tokens(tok.encode(prefix + t, truncation=True, max_length=512))
                 for t in texts]
        res = self._timed(trans, batch, beam_size=2, max_decoding_length=300)
        return [self._decode(tok, r.hypotheses[0]) for r in res]


class M2MEngine(CT2Engine):
    """facebook/m2m100_418M (MIT)."""
    name = "m2m100"
    MODEL = "facebook/m2m100_418M"

    def translate(self, texts, src, tgt):
        trans, tok = self._load(self.MODEL, "M2M100Tokenizer")
        tok.src_lang = src
        batch = [tok.convert_ids_to_tokens(tok.encode(t, truncation=True, max_length=512)) for t in texts]
        res = self._timed(trans, batch, target_prefix=[[f"__{tgt}__"]] * len(batch),
                          beam_size=2, max_decoding_length=300)
        return [self._decode(tok, r.hypotheses[0][1:]) for r in res]


class M2MBigEngine(M2MEngine):
    """facebook/m2m100_1.2B (MIT): same code path as m2m100, larger model."""
    name = "m2m100-big"
    MODEL = "facebook/m2m100_1.2B"


class MadladEngine(CT2Engine):
    """google/madlad400-3b-mt (Apache-2.0, T5): target language chosen by a `<2xx> ` prefix."""
    name = "madlad"
    MODEL = "google/madlad400-3b-mt"
    SAVE_SLOW_TOKENIZER = True
    REPORT_MEMORY = True

    def translate(self, texts, src, tgt):
        trans, tok = self._load(self.MODEL, "T5Tokenizer")
        batch = [tok.convert_ids_to_tokens(tok.encode(f"<2{tgt}> {t}", truncation=True, max_length=512))
                 for t in texts]
        res = self._timed(trans, batch, beam_size=2, max_decoding_length=300)
        return [self._decode(tok, r.hypotheses[0]) for r in res]


_PREAMBLE_RE = re.compile(r"^\s*(?:here(?: is|'s) the (?:arabic |english )?translation|"
                          r"(?:arabic |english )?translation)\s*:\s*", re.I)


def clean_llm(out, src):
    """Strip whitespace, a code fence, wrapping quotes and a one-line preamble from a model answer."""
    s = out.strip()
    m = re.fullmatch(r"```[A-Za-z]*\n?(.*?)\n?```", s, re.S)
    if m:
        s = m.group(1).strip()
    s = _PREAMBLE_RE.sub("", s, count=1).strip()
    for a, b in (('"', '"'), ("“", "”"), ("«", "»"), ("'", "'")):
        if len(s) > 1 and s[0] == a and s[-1] == b and a not in src and b not in src:
            s = s[1:-1].strip()
            break
    return s


class TGemmaEngine(Engine):
    """TranslateGemma 4B (Gemma Terms of Use) served by a local Ollama; one request per piece."""
    name = "tgemma"
    MODEL = "translategemma:4b"
    piece_cap = 1200
    split_sentences = False
    len_check = True
    sentence_retry = True
    LANGS = {"en": ("English", "en"), "ar": ("Arabic", "ar")}
    # Wording and layout from https://ollama.com/library/translategemma (two blank lines before the text).
    PROMPT = ("You are a professional {S} ({sc}) to {T} ({tc}) translator. Your goal is to accurately "
              "convey the meaning and nuances of the original {S} text while adhering to {T} grammar, "
              "vocabulary, and cultural sensitivities.\n"
              "Produce only the {T} translation, without any additional explanations or commentary. "
              "Please translate the following {S} text into {T}:\n\n\n{text}")

    def __init__(self, model_dir=None):
        self.url = (os.environ.get("YK_OLLAMA_URL") or "http://127.0.0.1:11434").rstrip("/")

    def start(self):
        try:
            requests.get(self.url + "/api/tags", timeout=10).raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Ollama is not running at {self.url} ({type(e).__name__})") from e

    def prompt(self, text, src, tgt):
        (S, sc), (T, tc) = self.LANGS[src], self.LANGS[tgt]
        return self.PROMPT.format(S=S, sc=sc, T=T, tc=tc, text=text)

    def translate(self, texts, src, tgt):
        out = []
        for t in texts:
            body = {"model": self.MODEL, "stream": False, "keep_alive": "3h",
                    "messages": [{"role": "user", "content": self.prompt(t, src, tgt)}],
                    "options": {"temperature": 0, "seed": 42, "num_ctx": 2048,
                                "num_predict": min(1024, max(64, 3 * (len(t) // 3 + 1)))}}
            t0 = time.time()
            r = requests.post(self.url + "/api/chat", json=body, timeout=180)
            STATS["model_s"] += time.time() - t0
            r.raise_for_status()
            out.append(clean_llm(r.json().get("message", {}).get("content", ""), t))
        return out


ENGINES = {"opus": OpusEngine, "m2m100": M2MEngine, "m2m100-big": M2MBigEngine,
           "madlad": MadladEngine, "tgemma": TGemmaEngine}


class Translator:
    def __init__(self, engine, batch):
        self.engine, self.batch = engine, max(1, batch)

    def _batches(self, pieces, src, tgt):
        raw = {}
        for i in range(0, len(pieces), self.batch):
            chunk = pieces[i:i + self.batch]
            outs = self.engine.translate(chunk, src, tgt)
            STATS["chars"] += sum(len(c) for c in chunk)
            raw.update(zip(chunk, outs))
        return raw

    def _checked(self, texts, src, tgt, items):
        """-> {text: output or None}. An output is accepted only if every placeholder came back
        and the digit runs equal the source's (digits then written 0-9). A digit mismatch is
        redone once with the numbers protected as placeholders (STATS["num_retry"])."""
        res, retry, cand, bad = {}, {}, {}, []
        for t, o in self._batches(texts, src, tgt).items():
            if not o or ph_ids(o) != ph_ids(t):
                res[t] = None
            elif self._insane(t, o, tgt):
                bad.append(t)
                cand[t] = o
            else:
                cand[t] = o
        eng = self.engine
        redo = [t for t in bad if eng.sentence_retry and len(self._sentences(t)) > 1]
        if redo:  # same instruction, one sentence per call
            outs = self._batches(sorted({s for t in redo for s in self._sentences(t)}, key=len), src, tgt)
            for t in redo:
                o = " ".join(outs.get(s) or "" for s in self._sentences(t)).strip()
                if o and ph_ids(o) == ph_ids(t) and not self._insane(t, o, tgt, count=False):
                    cand[t] = o
                    bad.remove(t)
        for t in bad:  # still failing: keep the source piece
            del cand[t]
            res[t] = t
            STATS["kept"] += 1
            log(f"output failed the script/length check, kept untranslated: {t[:60]!r}")
        for t, o in cand.items():
            if not digit_runs(t) or digit_runs(o) == digit_runs(t):
                res[t] = o.translate(DIGIT_MAP)
            else:
                retry[t] = protect_numbers(t, items)
        if retry:
            STATS["num_retry"] += len(retry)
            outs = self._batches(sorted(set(retry.values()), key=len), src, tgt)
            for t, t2 in retry.items():
                o = outs[t2]
                res[t] = o if o and ph_ids(o) == ph_ids(t2) else None
        return res

    @staticmethod
    def _sentences(t):
        return [m.group(2).strip() for m in SENT_RE.finditer(t) if m.group(2).strip()]

    def _insane(self, t, o, tgt, count=True):
        """Why an output is rejected ("script" / "length") or None. Source-verbatim tokens (brands,
        units) and placeholders do not count against the target script."""
        eng = self.engine
        if eng.script_check:
            known = {w.lower() for w in re.findall(r"[^\W\d_]+", PH_RE.sub(" ", t))}
            rest = " ".join(w for w in re.findall(r"[^\W\d_]+", PH_RE.sub(" ", o)) if w.lower() not in known)
            ar, lat = len(AR_RE.findall(rest)), len(LAT_RE.findall(rest))
            bad = bool(ar + lat) and not (ar > lat if tgt == "ar" else lat > ar)
            if not ar + lat:  # nothing new: an echo of a real sentence is a failure, a brand name is not
                norm = lambda x: " ".join(PH_RE.sub(" ", x).lower().split())  # noqa: E731
                bad = len(norm(o).split()) >= 4 and norm(o) == norm(t)
            if bad:
                if count:
                    STATS["script_retry"] += 1
                return "script"
        if eng.len_check and (len(o) > 3 * len(t) + 40 or (len(t) > 60 and len(o) < len(t) / 5)):
            if count:
                STATS["len_reject"] += 1
            return "length"
        return None

    @staticmethod
    def _droppable(s, items):
        """True when every placeholder of the piece is an inline formatting tag and none is a link."""
        ids = [int("".join(str(LETTERS.index(c.upper())) for c in m.group(1))) for m in PH_RE.finditer(s)]
        return bool(ids) and all(
            i < len(items) and INLINE_RE.fullmatch(items[i]) and not re.match(r"</?a[\s>/]", items[i], re.I)
            and not items[i].lower().startswith(("<yk-keep", "</yk-keep"))
            for i in ids)

    def _run(self, pieces, src, tgt, items):
        res = self._checked(pieces, src, tgt, items)
        done = {s: o for s, o in res.items() if o is not None}
        fail = [s for s in pieces if s not in done]
        if fail:  # formatting-only placeholders: retry once without those tags
            drop = {s: re.sub(r" {2,}", " ", PH_RE.sub("", s)).strip()
                    for s in fail if self._droppable(s, items)}
            if drop:
                r3 = self._checked(sorted(set(drop.values()), key=len), src, tgt, items)
                for s, t in drop.items():
                    if r3[t] is not None:
                        done[s] = r3[t]
                        STATS["fmt_dropped"] += 1
                fail = [s for s in fail if s not in done]
        plan = {}
        for s in fail:  # split at placeholders (numbers too, when only they were the problem)
            if PH_RE.search(s):
                plan[s] = split_at_ph(s)
            elif digit_runs(s):
                plan[s] = split_at_ph(protect_numbers(s, items))
            else:
                STATS["kept"] += 1
                log(f"empty output, piece kept untranslated: {s[:60]!r}")
                done[s] = s
        if plan:
            segs = sorted({t.strip() for parts in plan.values() for k, t in parts
                           if k == "t" and LETTER_RE.search(t)}, key=len)
            seg_out = self._checked(segs, src, tgt, items)
            for s, parts in plan.items():
                STATS["split"] += 1
                out = ""
                for k, t in parts:
                    seg = t.strip()
                    if k == "t" and seg in seg_out:
                        o = seg_out[seg]
                        if o is None:
                            STATS["kept"] += 1
                            log(f"segment output unusable, kept untranslated: {seg[:60]!r}")
                            o = seg
                        out += (" " if t[:1].isspace() else "") + o + (" " if t[-1:].isspace() else "")
                    else:
                        out += t
                done[s] = out.strip()
        return done

    def field(self, text, src, tgt):
        if not isinstance(text, str) or not text.strip():
            return text
        prot, items = protect(text)
        if script_matches(prot, tgt):
            return text
        prot = apply_glossary(prot, items, src, tgt)
        parsed = parse(prot, self.engine.piece_cap, self.engine.split_sentences)
        todo = sorted({p for it in parsed if isinstance(it, tuple) for p in it[1] if needs_model(p)},
                      key=len)
        done = self._run(todo, src, tgt, items)
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
SAMPLES = [
    # English -> Arabic
    ("Family-run furniture shop in Salmiya offering sofas, bedroom sets and custom-made wardrobes. "
     "Free delivery inside Kuwait on orders above 200 KWD.", "en", "ar"),
    ("2018 Toyota Land Cruiser GXR for sale. 85,000 km, full service history, single owner. "
     "Price: 9,500 KWD (negotiable). Call 5555 1234.", "en", "ar"),
    ("Spacious 3-bedroom apartment for rent in Hawally, 180 sqm, 2 bathrooms, central A/C, parking. "
     "Rent 450 KWD per month, available from next month.", "en", "ar"),
    ("We are hiring a sales executive for our showroom in Farwaniya. Minimum two years of experience, "
     "fluent English and Arabic, valid Kuwait driving licence. Send your CV today.", "en", "ar"),
    ("Traditional Kuwaiti restaurant in Jabriya serving machboos, murabyan and fresh seafood. "
     "Open daily from 11:00 am to 11:00 pm, Fridays from 1:00 pm. Family seating available.", "en", "ar"),
    ("Used iPhone 13, 128 GB, blue, excellent condition with original box and charger. "
     "120 KWD, serious buyers only.", "en", "ar"),
    ("Welcome to our dental clinic in Kuwait City.\n\nBook your appointment by calling +965 2222 3333 "
     "or online at https://example.com/book. We accept KNET and all major cards.", "en", "ar"),
    ('<p>Our <strong>bakery</strong> in Mangaf bakes fresh bread every morning.<br>Visit our '
     '<a href="https://example.com/menu">full menu</a> for prices.</p>', "en", "ar"),
    ('[gallery ids="1,2,3"]\nThe best car wash in Ahmadi. %title% %sep% %sitename%', "en", "ar"),
    ("<!-- wp:paragraph -->\n<p>Ladies-only beauty salon in Mahboula.</p>\n<!-- /wp:paragraph -->", "en", "ar"),
    ("Looking for a reliable cleaning company? We serve Jleeb Al-Shuyoukh, Khaitan and Mishref "
     "with full-time and part-time cleaners.", "en", "ar"),
    ("Wedding hall in Sharq with capacity for 400 guests. Contact us for packages and availability.",
     "en", "ar"),
    ("Pay securely with KNET at checkout. Orders ship within 2 business days.", "en", "ar"),
    ("Open daily from 9 am to 5 pm, and Fridays from 2:30 p.m. to 10 pm.", "en", "ar"),
    ("Monthly rent is 350 KD, plus a service charge of 25 KWD.", "en", "ar"),
    ("Our <strong>handmade</strong> sweets are baked fresh every day.", "en", "ar"),
    # Arabic -> English
    ("محل لبيع الأجهزة الكهربائية في حولي، أسعار منافسة وضمان سنة على جميع المنتجات.", "ar", "en"),
    ("للبيع سيارة نيسان باترول موديل 2016، ماشية 120 ألف كم، نظيفة وصبغ وكالة، السعر 6,200 دينار "
     "قابل للتفاوض. للتواصل 5555 1234", "ar", "en"),
    ("شقة للإيجار في السالمية، ثلاث غرف وصالة ومطبخ، المساحة 150 متر، الإيجار 400 دينار شهرياً "
     "مع تكييف مركزي.", "ar", "en"),
    ("مطلوب محاسب للعمل في شركة تجارية في الشويخ. خبرة لا تقل عن ثلاث سنوات، والراتب حسب الخبرة. "
     "يرجى إرسال السيرة الذاتية.", "ar", "en"),
    ("مطعم مأكولات بحرية في الفحيحيل، نفتح كل يوم من الساعة 12 ظهراً حتى 12 منتصف الليل، "
     "وعندنا جلسات عوائل وخدمة توصيل.", "ar", "en"),
    ("للبيع غسالة سامسونج مستعملة بحالة ممتازة، استعمال سنتين فقط، السعر 60 دينار.", "ar", "en"),
    ("مرحباً بكم في صالوننا في مدينة الكويت.\n\nللحجز اتصلوا على +965 2222 3333 أو زوروا موقعنا "
     "https://example.com/book وراح نرد عليكم بأسرع وقت.", "ar", "en"),
    ('<p>نقدم <strong>أفضل</strong> خدمات تصليح الأجهزة في الجهراء.<br>تفضلوا بزيارة '
     '<a href="https://example.com/services">صفحة الخدمات</a> لمعرفة الأسعار.</p>', "ar", "en"),
    ('[contact-form-7 id="5"] تواصلوا معنا للاستفسار | %title% %sep% %sitename%', "ar", "en"),
    ("شلونكم؟ عندنا خصم 20% على كل الملابس الرجالية هالأسبوع، وتقدرون تدفعون بالكي نت أو كاش. "
     "لا يفوتكم العرض!", "ar", "en"),
    ("تأجير شاليهات للعوائل في الخيران، مع مسبح خاص وإطلالة مباشرة على البحر. الحجز مسبقاً.", "ar", "en"),
    ("مدرسة خاصة في الفروانية تقبل الطلبات للعام الدراسي الجديد، من الروضة حتى الصف الثاني عشر.",
     "ar", "en"),
    ("بيت للبيع في مبارك الكبير، مساحة 400 متر، خمس غرف نوم وملحق، قريب من الخدمات والمدارس.",
     "ar", "en"),
    ("سيارة تويوتا كامري موديل 2019، ماشية 65,000 كم، السعر 4,800 دينار كويتي.", "ar", "en"),
]


def selftest(tr):
    for text, s, t in (("Hello", "en", "ar"), ("مرحبا", "ar", "en")):  # warm-up, not counted
        tr.field(text, s, t)
    for k in ("chars", "model_s", "kept", "split", "num_retry", "fmt_dropped", "script_retry", "len_reject"):
        STATS[k] = type(STATS[k])()
    for text, s, t in SAMPLES:
        out = tr.field(text, s, t)
        print(f"[{s}->{t}] {text}\n      => {out}", flush=True)


def summary(wall):
    cps = STATS["chars"] / STATS["model_s"] if STATS["model_s"] else 0.0
    log("---- summary ----")
    log(f"jobs done: {STATS['done']}  jobs failed: {STATS['failed']}  "
        f"pieces split at placeholders: {STATS['split']}  segments kept untranslated: {STATS['kept']}  "
        f"num_retry: {STATS['num_retry']}  fmt_dropped: {STATS['fmt_dropped']}  "
        f"script_retry: {STATS['script_retry']}  len_reject: {STATS['len_reject']}")
    log(f"characters translated: {STATS['chars']}  seconds loading model: {STATS['load_s']:.1f}  "
        f"seconds in translate_batch: {STATS['model_s']:.1f}  chars/sec: {cps:.1f}  wall seconds: {wall:.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true", help="translate the built-in samples, no site needed")
    ap.add_argument("--engine", default=os.environ.get("YK_ENGINE", "opus"), choices=sorted(ENGINES))
    ap.add_argument("--limit", type=int, default=int(os.environ.get("YK_LIMIT", "25") or 25))
    args = ap.parse_args()

    gpath = os.environ.get("YK_GLOSSARY") or str(Path(__file__).resolve().parent / "glossary.json")
    n = load_glossary(gpath)
    if n:
        log(f"glossary: {n} entries")
    engine = ENGINES[args.engine](os.environ.get("YK_MODEL_DIR", "./models"))
    tr = Translator(engine, int(os.environ.get("YK_BATCH", "16") or 16))
    start = time.time()
    log(f"engine={engine.name} batch={tr.batch} cpus={os.cpu_count()}")
    try:
        engine.start()
    except RuntimeError as e:
        log(f"FATAL: {e}")
        return 2

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
