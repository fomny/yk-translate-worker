"""Plain-assert tests for the text handling in translate_worker.py (no model, no network).

Run:  python -I test_text.py
"""
import json
import os
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    import requests  # noqa: F401
except ImportError:  # keep the test runnable without installing anything
    _stub = types.ModuleType("requests")
    _stub.ConnectionError = type("ConnectionError", (Exception,), {})
    _stub.Timeout = type("Timeout", (Exception,), {})
    _stub.Session = type("Session", (), {"__init__": lambda self: setattr(self, "headers", {}) or None})
    _stub.Session.headers = type("H", (dict,), {})()
    sys.modules["requests"] = _stub

import translate_worker as tw  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


class KeepEngine(tw.Engine):
    """Tags every piece and keeps placeholders, like a well-behaved model."""
    name = "keep"
    script_check = False  # these fakes tag text with Latin letters; the script check has its own tests

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
    for k in ("chars", "model_s", "load_s", "kept", "split", "num_retry", "fmt_dropped",
              "script_retry", "len_reject", "inline_ok", "protected_retry", "glossary_miss"):
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
assert protected("50% off and 20% more") == []  # plain numbers go to the model; a lone % is no variable
assert protected("3 - 5 rooms, 2018, 11:00, 85,000 km, 9,500 KWD") == []
assert protected("Call 5555 1234 now") == ["5555 1234"]  # 8 digits: phone-like
assert protected("ring 22 33 44 55") == ["22 33 44 55"]
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

tw.load_glossary(os.path.join(HERE, "glossary.json"))


class ArabicDigitsEngine(KeepEngine):
    def translate(self, texts, src, tgt):
        self.calls.append(list(texts))
        return [t.translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")) for t in texts]


class NumMangleEngine(KeepEngine):
    """Strips digits from plain text; copies placeholders when they are present."""
    drop_ph = False

    def translate(self, texts, src, tgt):
        self.calls.append(list(texts))
        out = []
        for t in texts:
            if "QX" in t:
                out.append(tw.PH_RE.sub("", t) if self.drop_ph else t)
            else:
                out.append(re.sub(r"\d", "", t) if re.search(r"\d", t) else "T{" + t + "}")
        return out


class NumMangleDropEngine(NumMangleEngine):
    drop_ph = True


# ---- 12. plain numbers reach the model unprotected; verified; written 0-9
tr, eng = fresh(KeepEngine)
out = tr.field("Price 9,500 for the 2018 model", "en", "ar")
assert eng.calls == [["Price 9,500 for the 2018 model"]] and out == "T{Price 9,500 for the 2018 model}", (eng.calls, out)
assert tw.STATS["num_retry"] == 0
tr, eng = fresh(ArabicDigitsEngine)
out = tr.field("Price 9,500 for the 2018 model", "en", "ar")
assert out == "Price 9,500 for the 2018 model" and tw.STATS["num_retry"] == 0, out
print("ok 12 plain numbers verified, western digits")

# ---- 13. digit mismatch -> numbers protected and redone
tr, eng = fresh(NumMangleEngine)
out = tr.field("Price 9,500 for 2018", "en", "ar")
assert out == "Price 9,500 for 2018" and tw.STATS["num_retry"] == 1 and tw.STATS["split"] == 0, (out, tw.STATS)
tr, eng = fresh(NumMangleDropEngine)  # the redo also loses placeholders: split around the numbers
out = tr.field("Price 9,500 for 2018", "en", "ar")
assert out == "T{Price} 9,500 T{for} 2018" and tw.STATS["num_retry"] == 1 and tw.STATS["split"] == 1, (out, tw.STATS)
print("ok 13 number retry and split")

# ---- 14. times, English -> Arabic only
tr, eng = fresh(KeepEngine)
out = tr.field("Open 11:00 am to 11:00 pm, 12 pm, 12 am, 7:30PM and 5 p.m. daily.", "en", "ar")
for want in ("11:00 صباحاً", "11:00 مساءً", "12 ظهراً", "12 منتصف الليل", "7:30 مساءً", "5 مساءً"):
    assert want in out, (want, out)
assert " am" not in out and "pm" not in out.lower()
assert "13 pm" in tr.field("Closes at 13 pm sharp", "en", "ar")  # not a valid 12-hour time: untouched
tr, eng = fresh(KeepEngine)
o = tr.field("نفتح 11:00 am", "ar", "en")
assert "11:00 am" in o and "صباحاً" not in o
print("ok 14 times")

# ---- 15. inline formatting tags: dropped for one sentence, links and block tags never
tr, eng = fresh(DropEngine)
out = tr.field("Our <strong>bakery</strong> bakes bread.", "en", "ar")
assert out == "T{Our bakery bakes bread.}" and tw.STATS["fmt_dropped"] == 1 and tw.STATS["split"] == 0, (out, tw.STATS)
tr, eng = fresh(DropEngine)
src = 'Visit our <a href="https://x.com/m">full menu</a> now.'
out = tr.field(src, "en", "ar")
assert protected(out) == protected(src) and tw.STATS["fmt_dropped"] == 0 and tw.STATS["split"] == 1, (out, tw.STATS)
tr, eng = fresh(DropEngine)
src = "<p>Some <em>text</em> here</p>"
out = tr.field(src, "en", "ar")
assert out.startswith("<p>") and out.endswith("</p>") and tw.STATS["fmt_dropped"] == 0
print("ok 15 inline tags")

# ---- 16. adjacent <yk-keep> tags survive, in order, byte-identical
src = 'Hello <yk-keep n="1"></yk-keep><yk-keep n="2"></yk-keep> world'
for cls in (KeepEngine, DropEngine):
    tr, eng = fresh(cls)
    out = tr.field(src, "en", "ar")
    assert '<yk-keep n="1"></yk-keep><yk-keep n="2"></yk-keep>' in out, (cls.__name__, out)
    assert protected(out) == protected(src) and tw.STATS["fmt_dropped"] == 0
print("ok 16 yk-keep")

# ---- 17. glossary additions
tr, eng = fresh(KeepEngine)
out = tr.field("Rent 350 KD, 120 sqm, A/C, near Samsung and iPhone shops; Toyota Land Cruiser, Nissan Patrol.", "en", "ar")
for want in ("350", "دينار كويتي", "متر مربع", "تكييف", "سامسونج", "آيفون", "تويوتا لاند كروزر", "نيسان باترول"):
    assert want in out, (want, out)
o = tr.field("للبيع نيسان باترول وغسالة سامسونج و تويوتا لاند كروزر", "ar", "en")
assert "Nissan Patrol" in o and "Samsung" in o and "Toyota Land Cruiser" in o, o
o = tr.field("دينار كويتي وتكييف", "ar", "en")  # abbreviations are English -> Arabic only
assert "KD" not in o and "KWD" not in o and "A/C" not in o, o
print("ok 17 glossary additions")

assert "m2m100-big" in tw.ENGINES and tw.ENGINES["m2m100-big"].MODEL == "facebook/m2m100_1.2B"
assert tw.ENGINES["m2m100"].MODEL == "facebook/m2m100_418M"
print("ok 18 engines")
# ---- 19. paragraph pieces for LLM-style engines
items = tw.parse("First sentence. Second sentence.\nThird.", 1200, False)
assert items == [("", ["First sentence. Second sentence."], ""), "\n", ("", ["Third."], "")], items
long_par = " ".join(f"Sentence number {i} is here." for i in range(12))
items = tw.parse(long_par, 80, False)
assert len(items) == 1 and len(items[0][1]) > 1 and all(len(p) <= 80 for p in items[0][1]), items
assert " ".join(items[0][1]) == long_par
assert len(tw.parse("One. Two. Three.", 400, True)) == 3  # default: one item per sentence
print("ok 19 parse without sentence split")


class ScriptEngine(tw.Engine):
    """Returns whatever `answers` maps the input to (default: an untranslated copy)."""
    name = "script"

    def __init__(self, answers=None):
        self.answers, self.calls = answers or {}, []

    def translate(self, texts, src, tgt):
        self.calls.append(list(texts))
        return [self.answers.get(t, t) for t in texts]


def with_engine(eng):
    for k in ("chars", "model_s", "load_s", "kept", "split", "num_retry", "fmt_dropped",
              "script_retry", "len_reject", "inline_ok", "protected_retry", "glossary_miss"):
        tw.STATS[k] = type(tw.STATS[k])()
    return tw.Translator(eng, 16)


# ---- 20. target-script check, all engines
tw.load_glossary(os.path.join(HERE, "no-such-file.json"))
tr = with_engine(ScriptEngine({"Hello there friend": "Hello there buddy"}))
assert tr.field("Hello there friend", "en", "ar") == "Hello there friend"  # still English: kept
assert tw.STATS["script_retry"] == 1 and tw.STATS["kept"] == 1, tw.STATS
tr = with_engine(ScriptEngine({"Visit the Nokia store": "زيارة متجر Nokia"}))  # brand from the source is fine
assert tr.field("Visit the Nokia store", "en", "ar") == "زيارة متجر Nokia" and tw.STATS["script_retry"] == 0
tr = with_engine(ScriptEngine({"محل في المدينة القديمة": "محل في المدينة القديمة"}))  # echo, ar->en
assert tr.field("محل في المدينة القديمة", "ar", "en") == "محل في المدينة القديمة" and tw.STATS["kept"] == 1
tr = with_engine(ScriptEngine({"محل في المدينة القديمة": "shop in the old city"}))  # real translation passes
assert tr.field("محل في المدينة القديمة", "ar", "en") == "shop in the old city" and tw.STATS["kept"] == 0
tr = with_engine(ScriptEngine({"Call me": "Call me"}))  # a short echo (could be a name) is not rejected
assert tr.field("Call me", "en", "ar") == "Call me" and tw.STATS["script_retry"] == 0
print("ok 20 script check")


class LLMFake(ScriptEngine):
    piece_cap, split_sentences, len_check, sentence_retry = 1200, False, True, True

    def translate(self, texts, src, tgt):
        self.calls.append(list(texts))
        return [self.answers[t] if t in self.answers else
                (t if len(tw.SENT_RE.findall(t)) > 1 else "ترجمة") for t in texts]


# ---- 21. LLM engine: failed paragraph retried sentence by sentence, then kept
two = "Hello there my friend. How are you today?"
tr = with_engine(LLMFake())
out = tr.field(two, "en", "ar")
assert out == "ترجمة ترجمة" and tw.STATS["script_retry"] == 1 and tw.STATS["kept"] == 0, (out, tw.STATS)
assert tr.engine.calls[0] == [two] and sorted(tr.engine.calls[1]) == ["Hello there my friend.", "How are you today?"]
one = "A single short sentence that is long enough to matter here."
tr = with_engine(LLMFake({one: "ن" * 500}))  # absurdly long answer
assert tr.field(one, "en", "ar") == one and tw.STATS["len_reject"] == 1 and tw.STATS["kept"] == 1
longer = "This sentence is over sixty characters long so that a tiny answer is suspicious."
tr = with_engine(LLMFake({longer: "ن"}))
assert tr.field(longer, "en", "ar") == longer and tw.STATS["len_reject"] == 1
print("ok 21 llm retry and length check")

# ---- 22. cleaning an LLM answer
assert tw.clean_llm('  "Hello"  ', "x") == "Hello"
assert tw.clean_llm('```\nمرحبا\n```', "x") == "مرحبا"
assert tw.clean_llm("Here is the translation:\nمرحبا بك", "x") == "مرحبا بك"
assert tw.clean_llm("Translation: مرحبا", "x") == "مرحبا"
assert tw.clean_llm("The translation of the poem is long", "x") == "The translation of the poem is long"
assert tw.clean_llm('"Quoted" source', '"Quoted" source') == '"Quoted" source'
print("ok 22 clean_llm")

# ---- 23. engine definitions: prompt, caps, fatal start, converter settings
g = tw.TGemmaEngine()
assert g.prompt("Hello, how are you?", "en", "ar") == (
    "You are a professional English (en) to Arabic (ar) translator. Your goal is to accurately convey the "
    "meaning and nuances of the original English text while adhering to Arabic grammar, vocabulary, and "
    "cultural sensitivities.\nProduce only the Arabic translation, without any additional explanations or "
    "commentary. Please translate the following English text into Arabic:\n\n\nHello, how are you?")
assert (g.piece_cap, g.split_sentences, g.MODEL) == (1200, False, "translategemma:4b")
assert (tw.MadladEngine.piece_cap, tw.MadladEngine.split_sentences) == (400, True)
assert tw.MadladEngine.MODEL == "google/madlad400-3b-mt" and tw.MadladEngine.SAVE_SLOW_TOKENIZER
os.environ["YK_OLLAMA_URL"] = "http://127.0.0.1:9"
try:
    tw.TGemmaEngine().start()
    raise AssertionError("start() should fail when nothing listens")
except RuntimeError as e:
    assert "Ollama is not running at http://127.0.0.1:9" in str(e)
assert set(tw.ENGINES) == {"opus", "m2m100", "m2m100-big", "madlad", "tgemma"}
print("ok 23 engines")

# ---- 24. spacing when restoring word-like items; structural items never get spaces
assert tw.restore("120 QXAXQفقط", ["دينار كويتي"]) == "120 دينار كويتي فقط"
assert tw.restore("QXAXQو أيام", ["5 مساءً"]) == "5 مساءً و أيام"
assert tw.restore("inQXAXQ.", ["Salmiya"]) == "in Salmiya."
assert tw.restore("atQXAXQ", ["11:00 صباحاً"]) == "at 11:00 صباحاً"
assert tw.restore("wordQXAXQnext", ["<br>"]) == "word<br>next"
assert tw.restore("wordQXAXQnext", ["https://x.com/a"]) == "wordhttps://x.com/anext"
assert tw.restore("aQXAXQQXBXQb", ["السالمية", '<yk-keep n="1">']) == 'a السالمية<yk-keep n="1">b'
assert tw.restore("QXAXQ QXBXQ", ["السالمية", "<b>"]) == "السالمية <b>"
print("ok 24 restore spacing")


class NatEngine(tw.Engine):
    """Natural-first fake: answers by exact input text (placeholders as the code builds them)."""
    name = "nat"
    natural_first = True
    piece_cap, split_sentences = 1200, False

    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def translate(self, texts, src, tgt):
        self.calls.append(list(texts))
        if callable(self.answers):
            return [self.answers(t) for t in texts]
        return [self.answers.get(t, "؟؟") for t in texts]


tw.load_glossary(os.path.join(HERE, "glossary.json"))
S1 = "Rent 450 دينار كويتي per month."  # what attempt 1 sends for "Rent 450 KWD per month."
P1 = "Rent 450 QXAXQ per month."        # what the protected method sends

# ---- 25. attempt 1: glossary written inline in the target language, no placeholders
tr = with_engine(NatEngine({S1: "الإيجار 450 دينار كويتي شهريا.",
                            "Open 9 am to 5 pm.": "مفتوح من 9 صباحاً إلى 5 مساءً.",
                            "Wedding hall in شرق.": "قاعة زفاف في شرق."}))
assert tr.field("Rent 450 KWD per month.", "en", "ar") == "الإيجار 450 دينار كويتي شهريا."
assert tr.field("Open 9 am to 5 pm.", "en", "ar") == "مفتوح من 9 صباحاً إلى 5 مساءً."
assert tr.field("Wedding hall in Sharq.", "en", "ar") == "قاعة زفاف في شرق."
assert tw.STATS["inline_ok"] == 3 and tw.STATS["protected_retry"] == 0, tw.STATS
assert all("QX" not in t for c in tr.engine.calls for t in c) and tr.engine.calls[0] == [S1]
tr = with_engine(NatEngine({"Family shop in السالمية.": "متجر عائلي في السالمية.", "شقة في Salmiya": "Apartment in Salmiya"}))
assert tr.field("Family shop in Salmiya.", "en", "ar") == "متجر عائلي في السالمية."
assert tr.field("شقة في السالمية", "ar", "en") == "Apartment in Salmiya" and tw.STATS["inline_ok"] == 2
tr = with_engine(NatEngine({}))
assert tr.field("Salmiya", "en", "ar") == "السالمية" and not tr.engine.calls  # only a glossary term: no call
print("ok 25 inline glossary")

# ---- 26. glossary form missing -> redone with the protected method
tr = with_engine(NatEngine({S1: "الإيجار 450 دولار شهريا.", P1: "الإيجار 450 QXAXQ شهريا."}))
assert tr.field("Rent 450 KWD per month.", "en", "ar") == "الإيجار 450 دينار كويتي شهريا."
assert tw.STATS["inline_ok"] == 0 and tw.STATS["protected_retry"] == 1, tw.STATS
tr = with_engine(NatEngine({"Wedding hall in شرق.": "قاعة زفاف في الشمال.",
                            "Wedding hall in QXAXQ.": "قاعة زفاف في QXAXQ."}))
assert tr.field("Wedding hall in Sharq.", "en", "ar") == "قاعة زفاف في شرق."
tr = with_engine(NatEngine({"شقة في Salmiya": "Apartment in Salmia", "شقة في QXAXQ": "Apartment in QXAXQ"}))
assert tr.field("شقة في السالمية", "ar", "en") == "Apartment in Salmiya"
assert tw.STATS["protected_retry"] == 1
print("ok 26 protected retry on glossary miss")

# ---- 27. digit mismatch -> protected method, numbers as placeholders when needed
tr = with_engine(NatEngine({"Rent 450 per month.": "الإيجار 45 شهريا.",
                            "Rent QXAXQ per month.": "الإيجار QXAXQ شهريا."}))
out = tr.field("Rent 450 per month.", "en", "ar")
assert out == "الإيجار 450 شهريا." and tw.STATS["protected_retry"] == 1 and tw.STATS["num_retry"] == 1, (out, tw.STATS)
print("ok 27 digit mismatch")

# ---- 28. structure: edges peeled, the rest sent raw and verified
assert tw.peel("QXAXQ  hello QXBXQ world QXCXQ \n") == ("QXAXQ  ", "hello QXBXQ world", " QXCXQ \n")
assert tw.peel("QXAXQQXBXQ") == ("QXAXQQXBXQ", "", "") and tw.peel("plain text") == ("", "plain text", "")
tr = with_engine(NatEngine({"Visit https://example.com now.": "زوروا https://example.com الآن."}))
out = tr.field("Visit https://example.com now.", "en", "ar")
assert out == "زوروا https://example.com الآن." and tw.STATS["inline_ok"] == 1, (out, tw.STATS)
html = "<p>Our <strong>bakery</strong> sells bread.</p>"
tr = with_engine(NatEngine({"Our <strong>bakery</strong> sells bread.": "مخبزنا <strong>الخبز</strong> يبيع."}))
out = tr.field(html, "en", "ar")
assert out == "<p>مخبزنا <strong>الخبز</strong> يبيع.</p>" and tw.STATS["inline_ok"] == 1, (out, tw.STATS)
assert tr.engine.calls == [["Our <strong>bakery</strong> sells bread."]]  # <p> and </p> never sent
# a dropped inline tag: attempt 1 refused, the protected method takes over
tr = with_engine(NatEngine(lambda t: "مخبز" if "<" in t else "مخبزنا " + " ".join(tw.PH_RE.findall(t)) + " يبيع."))
out = tr.field(html, "en", "ar")
assert tw.STATS["protected_retry"] == 1 and out.startswith("<p>") and out.endswith("</p>"), (out, tw.STATS)
print("ok 28 structure raw, edges peeled")

# phones: digits identical once spaces are removed; a changed digit is refused
ph_src = "Call +965 2222 3333 now."
tr = with_engine(NatEngine({ph_src: "اتصل +965 2222 3333 الآن."}))
assert tr.field(ph_src, "en", "ar") == "اتصل +965 2222 3333 الآن." and tw.STATS["inline_ok"] == 1
tr = with_engine(NatEngine({ph_src: "اتصل +9652222 3333 الآن."}))
assert tr.field(ph_src, "en", "ar") == "اتصل +9652222 3333 الآن." and tw.STATS["inline_ok"] == 1
tr = with_engine(NatEngine(lambda t: "اتصل +965 2222 3334 الآن." if "+965" in t else "اتصل QXAXQ الآن."))
assert tr.field(ph_src, "en", "ar") == "اتصل +965 2222 3333 الآن." and tw.STATS["inline_ok"] == 0
# adjacent <yk-keep> tags: byte-identical, in order, once each
yk = 'Hello <yk-keep n="1"></yk-keep><yk-keep n="2"></yk-keep> world'
tr = with_engine(NatEngine({yk: 'مرحبا <yk-keep n="1"></yk-keep><yk-keep n="2"></yk-keep> عالم'}))
assert tr.field(yk, "en", "ar") == 'مرحبا <yk-keep n="1"></yk-keep><yk-keep n="2"></yk-keep> عالم'
tr = with_engine(NatEngine(lambda t: 'مرحبا <yk-keep n="2"></yk-keep><yk-keep n="1"></yk-keep> عالم' if "<" in t
                           else "مرحبا " + "".join(tw.PH_RE.findall(t)) + " عالم"))
out = tr.field(yk, "en", "ar")  # swapped order refused, the protected method restores the right order
assert 'n="1"></yk-keep><yk-keep n="2"' in out and tw.STATS["protected_retry"] == 1, (out, tw.STATS)
print("ok 28b phones and yk-keep")


class InventEngine(KeepEngine):
    """Drops placeholders in whole pieces; makes up digits for short segments."""

    def translate(self, texts, src, tgt):
        self.calls.append(list(texts))
        return [tw.PH_RE.sub("", t).strip() if "QX" in t else "x 25" for t in texts]


# ---- 29. split fallback: a segment with invented digits keeps its source
tr, eng = fresh(InventEngine)
out = tr.field("Call 5555 1234 today", "en", "ar")
assert out == "Call 5555 1234 today" and tw.STATS["kept"] == 1 and tw.STATS["split"] == 1, (out, tw.STATS)
print("ok 29 split segments keep digits strict")

assert tw.TGemmaEngine.natural_first and not any(
    getattr(tw.ENGINES[n], "natural_first", False) for n in ("opus", "m2m100", "m2m100-big", "madlad"))
print("ok 30 natural_first only for tgemma")

# ---- 31. numbers are compared by value
ok, no = tw.numbers_ok, lambda *a, **k: not tw.numbers_ok(*a, **k)
assert ok("ماشية 120 ألف كم", "mileage 120,000 km") and ok("ماشية 120 ألف كم", "120 thousand km")
assert ok("6,200 دينار", "6200 dinars") and ok("Open 11:00 am", "مفتوح 11 صباحاً") and ok("3-bedroom flat", "شقة 3 غرف")
assert ok("استعمال سنتين فقط، السعر 60", "used for only 2 years, priced at 60")  # number word -> digit
assert ok("Minimum two years, salary 500", "الحد الأدنى 2 سنوات، الراتب 500")
assert ok("12 ظهراً حتى 12 منتصف الليل", "12 noon until midnight")  # a repeated value may be written once
assert ok("الصف الثاني عشر", "12th grade") and ok("Open 24/7", "مفتوح 24/7")
assert ok("السعر 2,5 دينار", "price 2.5 dinars")
assert no("السعر 60", "price 50") and no("السعر 60 دينار", "priced in dinars")  # a price must not change or vanish
assert no("Rent 450", "450 and 20") and no("Rent 450 per month", "الإيجار 45 شهريا")
assert no("Call us", "5 minutes") is True
assert ok("a", "b", strict=True) and no("Rent 450", "Rent 450 and 2", strict=True)
assert tw.number_values("1,234.5 and ٣٫٥ and 2 million") == {1234.5, 3.5, 2e6}
print("ok 31 numbers by value")

# ---- 32. natural-first final choice: translation, natural output, or the whole source; never a mix
src_t = "Rent 450 KWD per month."
prot_t = P1
tr = with_engine(NatEngine({S1: "الإيجار 450 دولار شهريا.", prot_t: "الإيجار 450 QXAXQ شهريا."}))
assert tr.field(src_t, "en", "ar") == "الإيجار 450 دينار كويتي شهريا." and tw.STATS["glossary_miss"] == 0  # (ii)
tr = with_engine(NatEngine({S1: "الإيجار 450 دولار شهريا.", prot_t: "Rent QXAXQ monthly please"}))
out = tr.field(src_t, "en", "ar")  # protected answer is not Arabic: (iii) attempt 1, glossary missed
assert out == "الإيجار 450 دولار شهريا." and tw.STATS["glossary_miss"] == 1 and tw.STATS["kept"] == 0, (out, tw.STATS)
tr = with_engine(NatEngine({S1: "الإيجار 40 دولار شهريا.", prot_t: "Rent QXAXQ monthly please"}))
out = tr.field(src_t, "en", "ar")  # numbers wrong AND protected unusable: (iv) the whole source piece
assert out == src_t and tw.STATS["kept"] == 1 and "دينار" not in out, (out, tw.STATS)
# a piece the protected path could only half translate is kept whole, glossary terms in their source form
tr, eng = fresh(InventEngine)
out = tr.field("Rent in Salmiya call 5555 1234 today", "en", "ar")
assert out == "Rent in Salmiya call 5555 1234 today" and tw.STATS["kept"] == 1, (out, tw.STATS)
print("ok 32 no mixed pieces")

# ---- 33. natural answers that only rewrite numbers harmlessly are accepted
tr = with_engine(NatEngine({"للبيع سيارة ماشية 120 ألف كم، السعر 6,200 دينار": "Car for sale, 120,000 km, price 6200 dinars",
                            "استعمال سنتين فقط، السعر 60 دينار": "Used for only 2 years, priced at 60 dinars"}))
assert tr.field("للبيع سيارة ماشية 120 ألف كم، السعر 6,200 دينار", "ar", "en").startswith("Car for sale")
assert tr.field("استعمال سنتين فقط، السعر 60 دينار", "ar", "en").startswith("Used for only 2")
assert tw.STATS["inline_ok"] == 2 and tw.STATS["protected_retry"] == 0, tw.STATS
print("ok 33 harmless number rewrites")

# ---- 34. YK_DEBUG lines (only with YK_DEBUG=1)
import contextlib  # noqa: E402
import io  # noqa: E402

for flag, want in (("1", True), ("0", False)):
    os.environ["YK_DEBUG"] = flag
    tr = with_engine(NatEngine({S1: "الإيجار 450 دولار شهريا.", prot_t: "الإيجار 450 QXAXQ شهريا."}))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tr.field(src_t, "en", "ar")
    text = buf.getvalue()
    assert ("DEBUG attempt 1 (inline) rejected [glossary:دينار كويتي]" in text) == want, text
    assert ("DEBUG" in text) == want
os.environ.pop("YK_DEBUG")
print("ok 34 debug lines")


# ---- 35. REST handling with a fake session
class FakeResp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


class FakeSession:
    """Answers scripted per route; an entry can be an exception to raise. Records every call."""

    def __init__(self, script):
        self.script, self.calls = script, []

    def request(self, method, url, timeout=None, **kw):
        route = url.split("/yk-translate/v1")[1]
        self.calls.append((method, route, kw))
        item = self.script[route].pop(0) if len(self.script[route]) > 1 else self.script[route][0]
        if isinstance(item, Exception):
            raise item
        return FakeResp(*item)


def site_with(script):
    site = tw.Site("https://example.test", "t")
    site.http = FakeSession(script)
    return site


def reset_stats():
    for k in tw.STATS:
        tw.STATS[k] = type(tw.STATS[k])()


JOB = {"job_id": 7, "post_id": 3, "target": "ar", "source": "en", "hash": "h1", "fields": {"title": "Hi"}}
real_sleep = tw.time.sleep
tw.time.sleep = lambda s: None
try:
    def run(script, fields=None):
        reset_stats()
        site = site_with(script)
        return tw.deliver(site, JOB, "tgemma", fields or {"title": "مرحبا"}), site

    res, site = run({"/result": [(200, {"ok": True, "already": True})]})
    assert res == "done" and tw.STATS["done"] == 1
    for code in ("yk_stale_hash", "yk_not_claimed", "yk_not_needed"):
        res, site = run({"/result": [(409, {"code": code, "message": "m"})], "/fail": [(200, {})]})
        assert res == "skipped" and tw.STATS["skipped"] == 1 and tw.STATS["failed"] == 0, code
        assert [c[1] for c in site.http.calls] == ["/result"], site.http.calls  # no /fail
    res, site = run({"/result": [(404, {"code": "yk_unknown_job"})], "/fail": [(200, {})]})
    assert res == "skipped" and [c[1] for c in site.http.calls] == ["/result"]
    # yk_busy: 3 posts at most, then dropped without /fail
    res, site = run({"/result": [(409, {"code": "yk_busy"})], "/fail": [(200, {})]})
    assert res == "skipped" and [c[1] for c in site.http.calls] == ["/result"] * 3 and tw.STATS["skipped"] == 1
    res, site = run({"/result": [(409, {"code": "yk_busy"}), (200, {"ok": True})]})
    assert res == "done" and len(site.http.calls) == 2
    # the same body each time, hash and target as issued
    bodies = [c[2]["json"] for c in site.http.calls]
    assert bodies[0] == bodies[1] and bodies[0]["hash"] == "h1" and bodies[0]["target"] == "ar"
    # codes that call /fail once
    for status, code in ((422, "yk_protected_lost"), (422, "yk_missing_fields"), (413, "yk_too_large")):
        res, site = run({"/result": [(status, {"code": code, "message": "bad"})], "/fail": [(200, {"ignored": True})]})
        assert res == "failed" and tw.STATS["failed"] == 1, code
        assert [c[1] for c in site.http.calls] == ["/result", "/fail"], site.http.calls
        assert code in site.http.calls[1][2]["json"]["reason"]
    # 5xx on /result: no retry, no /fail
    res, site = run({"/result": [(500, {"code": "yk_write_failed"})], "/fail": [(200, {})]})
    assert res == "failed" and tw.STATS["failed"] == 1 and [c[1] for c in site.http.calls] == ["/result"]
    res, site = run({"/result": [(502, None)], "/fail": [(200, {})]})  # not JSON
    assert res == "failed" and len(site.http.calls) == 1
    # 401 / 403 are fatal on any route
    for status, code in ((401, "rest_forbidden"), (403, "yk_no_secret"), (403, "yk_https_required")):
        try:
            run({"/result": [(status, {"code": code})]})
            raise AssertionError("expected Fatal")
        except tw.Fatal as e:
            assert str(status) in str(e) and code in str(e)
    try:
        site_with({"/pending": [(403, {"code": "yk_no_secret"})]}).pending(5)
        raise AssertionError("expected Fatal")
    except tw.Fatal:
        pass
    # network error on /result: one re-POST, only after GET /pending?count_only=1 answered
    res, site = run({"/result": [tw.requests.ConnectionError("x"), (200, {"already": True})],
                     "/pending": [(200, {"pending": 2})]})
    assert res == "done" and [c[1] for c in site.http.calls] == ["/result", "/pending", "/result"]
    assert site.http.calls[1][2]["params"] == {"count_only": 1}
    reset_stats()
    site = site_with({"/result": [tw.requests.Timeout("x")], "/pending": [tw.requests.ConnectionError("x")]})
    try:
        tw.deliver(site, JOB, "tgemma", {"title": "x"})
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass
    assert [c[1] for c in site.http.calls].count("/result") == 1  # site unreachable: no second POST
    # 5xx on /pending is retried as before
    site = site_with({"/pending": [(503, {}), (200, {"jobs": [JOB]})]})
    assert site.pending(5) == [JOB]
    # a lost <yk-keep> tag: /fail "protected lost locally", nothing posted
    job = dict(JOB, fields={"body": 'A <yk-keep n="1"></yk-keep> B <yk-keep n="2"></yk-keep>'})
    reset_stats()
    site = site_with({"/result": [(200, {})], "/fail": [(200, {})]})
    assert tw.deliver(site, job, "tgemma", {"body": 'أ <yk-keep n="1"></yk-keep> ب'}) == "failed"
    assert [c[1] for c in site.http.calls] == ["/fail"] and site.http.calls[0][2]["json"]["reason"] == "protected lost locally"
    reset_stats()
    site = site_with({"/result": [(200, {})]})
    ok_out = {"body": 'أ <yk-keep n="1"></yk-keep> ب <yk-keep n="2"></yk-keep>'}
    assert tw.deliver(site, job, "tgemma", ok_out) == "done"
finally:
    tw.time.sleep = real_sleep

# fields whose value is not a string pass through untouched
tr = with_engine(KeepEngine())
assert tr.field(5, "en", "ar") == 5 and tr.field(None, "en", "ar") is None and tr.field(["a"], "en", "ar") == ["a"]
assert tr.field("", "en", "ar") == ""
print("ok 35 REST handling")
print("ALL TESTS PASSED")
