"""Plain-assert tests for the text handling in translate_worker.py (no model, no network).

Run:  python -I test_text.py
"""
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    import requests  # noqa: F401
except ImportError:  # keep the test runnable without installing anything
    sys.modules["requests"] = types.ModuleType("requests")

import translate_worker as tw  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


class KeepEngine(tw.Engine):
    """Tags every piece and keeps placeholders, like a well-behaved model."""
    name = "keep"

    def __init__(self):
        self.calls = []

    def translate(self, texts, src, tgt):
        self.calls.append(list(texts))
        return ["T{" + t + "}" for t in texts]


class DropEngine(KeepEngine):
    """Like the real model on the failing sample: loses every placeholder."""
    name = "drop"

    def translate(self, texts, src, tgt):
        self.calls.append(list(texts))
        return ["T{" + tw.PH_RE.sub("", t).strip() + "}" for t in texts]


class EmptyEngine(KeepEngine):
    def translate(self, texts, src, tgt):
        self.calls.append(list(texts))
        return ["" for _ in texts]


def fresh(engine_cls=KeepEngine):
    for k in ("chars", "model_s", "load_s", "kept", "split"):
        tw.STATS[k] = type(tw.STATS[k])()
    eng = engine_cls()
    return tw.Translator(eng, 16), eng


def protected(s):
    return [m.group(0) for m in tw.PROTECT_RE.finditer(s)]


n_glossary = tw.load_glossary(os.path.join(HERE, "glossary.json"))
with open(os.path.join(HERE, "glossary.json"), encoding="utf-8") as fh:
    entries = json.load(fh)
n_amb = sum(1 for e in entries if e.get("ar_ambiguous"))
assert n_glossary == len(entries) and n_glossary > 60
print(f"glossary entries: {n_glossary}, ambiguous: {n_amb}")

# ---- 1. PROTECT_RE: new classes, byte-identical round trip
cases = [
    '[gallery ids="1,2,3"]', "[/gallery]", "[contact-form-7 id=\"5\" title=\"x\"]",
    "%title%", "%sep%", "%sitename%", "%a_b2%",
    "<!-- wp:paragraph -->", "<!-- /wp:paragraph -->", '<!-- wp:image {"id":5} -->',
    "https://example.com/shop?a=1", "+965 5555 1234", "<a href=\"https://x.com\">", "</p>", "<br>",
]
for c in cases:
    assert protected(c) == [c], (c, protected(c))
    prot, items = tw.protect(c)
    assert tw.restore(prot, items) == c
assert protected("50% off and 20% more") == ["50", "20"]  # a lone % is not a variable
print("ok 1 protect classes")

# ---- 2. fallback split keeps every protected item, in order, batched
src = "Call +965 5555 1234 or visit https://example.com/shop today."
tr, eng = fresh(DropEngine)
out = tr.field(src, "en", "ar")
assert out == "T{Call} +965 5555 1234 T{or visit} https://example.com/shop T{today.}", out
assert tw.STATS["split"] == 1 and tw.STATS["kept"] == 0, tw.STATS
assert len(eng.calls) == 2 and len(eng.calls[1]) == 3, eng.calls  # one batch for all segments
print("ok 2 split with plain items")

mixed = ('<p>Visit <strong>our shop</strong> at www.example.com or [shortcode a="b"]click here[/shortcode] '
         'for %title% %sep% %sitename% now.<br>Open 9:00 daily</p>')
tr, eng = fresh(DropEngine)
out = tr.field(mixed, "en", "ar")
assert protected(out) == protected(mixed), (protected(out), protected(mixed))
assert tw.STATS["split"] >= 1
assert "QX" not in out
print("ok 3 split keeps html/url/shortcode/percent in order")

# ---- 4. healthy engine: no split, items intact
tr, eng = fresh(KeepEngine)
out = tr.field(mixed, "en", "ar")
assert protected(out) == protected(mixed) and tw.STATS["split"] == 0
print("ok 4 no fallback when placeholders survive")

# ---- 5. empty output keeps source segments and counts them
tr, eng = fresh(EmptyEngine)
out = tr.field("Call 5555 1234 today", "en", "ar")
assert out == "Call 5555 1234 today", out
assert tw.STATS["kept"] >= 1
print("ok 5 empty output kept untranslated")

# ---- 6. Gutenberg block comment survives
gb = "<!-- wp:paragraph -->\n<p>Ladies beauty salon.</p>\n<!-- /wp:paragraph -->"
for cls in (KeepEngine, DropEngine):
    tr, eng = fresh(cls)
    out = tr.field(gb, "en", "ar")
    assert protected(out) == protected(gb) == ["<!-- wp:paragraph -->", "<p>", "</p>", "<!-- /wp:paragraph -->"]
    assert out.startswith("<!-- wp:paragraph -->\n") and out.endswith("\n<!-- /wp:paragraph -->")
print("ok 6 gutenberg comments")

# ---- 7. glossary en -> ar
tr, eng = fresh(KeepEngine)
assert tr.field("Salmiya", "en", "ar") == "السالمية"
out = tr.field("Shop in Salmiya, Hawalli and Al-Salmiyah.", "en", "ar")
assert "السالمية" in out and "حولي" in out and "Salmiy" not in out and "Hawal" not in out, out
assert out.count("السالمية") == 2 and "الال" not in out
out = tr.field("Offices in KUWAIT CITY and Kuwait.", "en", "ar")
assert "مدينة الكويت" in out and out.count("الكويت") == 2, out
assert "Salmiyas" in tr.field("Salmiyas shop", "en", "ar").replace("T{", "")  # whole words only
assert tr.field("Pay with KNET", "en", "ar").endswith("كي نت}")
print("ok 7 glossary en->ar")

# ---- 8. glossary ar -> en, prefixes
tr, eng = fresh(KeepEngine)
assert tr.field("السالمية", "ar", "en") == "Salmiya"
o = tr.field("محل في السالمية", "ar", "en")
assert o == "T{محل في Salmiya}", o
o = tr.field("اشتري من محل بالسالمية", "ar", "en")
assert "في Salmiya" in o, o
o = tr.field("اتجه للسالمية", "ar", "en")
assert "إلى Salmiya" in o, o
o = tr.field("الفحيحيل والجهراء", "ar", "en")
assert "Fahaheel" in o and "Jahra" in o, o
o = tr.field("نقبل بالكي نت", "ar", "en")
assert "KNET" in o, o
o = tr.field("مدينة الكويت", "ar", "en")
assert o == "Kuwait City", o
o = tr.field("حول السالميةX وبالسالميه", "ar", "en")  # glued Latin letter: no match; double prefix: not matched
assert o.count("Salmiya") == 0, o
o = tr.field("حول السالميةX بالسالميه", "ar", "en")  # one prefix, ه spelling: match
assert "السالميةX" in o and o.count("Salmiya") == 1, o
print("ok 8 glossary ar->en and prefixes")

# ---- 9. ambiguous entries are English -> Arabic only
tr, eng = fresh(KeepEngine)
o = tr.field("شقة في شرق قرب بيان", "ar", "en")
assert "Sharq" not in o and "Bayan" not in o and "شرق" in o, o
o = tr.field("Flat in Sharq near Bayan", "en", "ar")
assert "شرق" in o and "بيان" in o, o
print("ok 9 ambiguous entries")

# ---- 10. script_matches still skips text already in the target language
tr, eng = fresh(KeepEngine)
assert tr.field("مرحبا بكم في محلنا", "en", "ar") == "مرحبا بكم في محلنا" and not eng.calls
assert tr.field("Welcome to our shop", "ar", "en") == "Welcome to our shop" and not eng.calls
assert tr.field("https://example.com", "en", "ar") == "https://example.com" and not eng.calls
print("ok 10 script_matches skip")

# ---- 11. missing / invalid glossary: one log line, no glossary, no crash
assert tw.load_glossary(os.path.join(HERE, "no-such-file.json")) == 0 and not tw.GLOSS
bad = os.path.join(HERE, "_bad_glossary_test.json")
with open(bad, "w", encoding="utf-8") as fh:
    fh.write("{not json")
try:
    assert tw.load_glossary(bad) == 0 and not tw.GLOSS
finally:
    os.remove(bad)
tr, eng = fresh(KeepEngine)
assert tr.field("Shop in Salmiya", "en", "ar") == "T{Shop in Salmiya}"
print("ok 11 no glossary")

print("ALL TESTS PASSED")
