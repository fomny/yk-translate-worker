# yk-translate worker

Translation worker for the `yk-auto-translate` WordPress plugin (spec 003). It pulls pending jobs from the site over HTTPS, translates Arabic and English in both directions with an open model on CPU (CTranslate2, int8), and posts the results back. No usage cap, no paid service, no translation-vendor key.

## Secrets and variables

Set these in the GitHub repository (Settings > Secrets and variables > Actions) or as environment variables on a PC:

| Name | Meaning |
|---|---|
| `YK_SITE_URL` | Site address, for example `https://example.com` (secret) |
| `YK_WORKER_TOKEN` | The shared secret sent as `X-YK-Token` (secret) |
| `YK_ENGINE` | Optional: `tgemma` (default), `opus`, `m2m100`, `m2m100-big` or `madlad`. In Actions it comes from the run input |

Optional tuning (PC or workflow env): `YK_MODEL_DIR` (default `./models`), `YK_BATCH` (16), `YK_LIMIT` (25), `YK_MAX_MINUTES` (20; in Actions the run input `max_minutes`, for one long backfill run, with a 330 minute job limit), `YK_GLOSSARY` (path of the glossary file, default `glossary.json` next to the script).

## Protected items and the glossary

URLs, e-mails, phone-like numbers (7 or more digits), HTML tags and comments (including Gutenberg `<!-- wp:... -->` blocks), WordPress shortcodes (`[name ...]`, `[/name]`) and `%variables%` such as `%title%` are replaced by placeholders before translation and put back byte-identical. If the model drops a placeholder, the worker tries in this order:

1. Pieces whose only placeholders are inline formatting tags (strong, b, em, i, u, span): translate once more without those tags (`fmt_dropped`). Links (`<a>`), block tags and `<yk-keep>` are never dropped.
2. Otherwise split the piece at each placeholder, translate the text segments between them in one batch, and join them back in the original order (`split`).

Plain numbers (prices, years, areas, times like 11:00, 20%) are not protected up front: the model sees them, and the output is accepted only if its digit runs equal the source's (Arabic-Indic digits are written as 0-9). If not, that piece is redone with its numbers as placeholders (`num_retry`). English to Arabic, a time with am/pm becomes a protected item: `11:00 am` becomes `11:00 صباحاً`, `1 pm` becomes `1 مساءً`, `12 pm` becomes `12 ظهراً`, `12 am` becomes `12 منتصف الليل`.

Natural first: an engine with `natural_first` (only `tgemma`) first translates the natural sentence, with only structure protected (tags, shortcodes, `%variables%`, URLs, e-mails, phone-like numbers). Times, currency, units and glossary terms are left to the model. The answer is accepted (`natural_ok`) only if all structural placeholders came back, the digit runs match, the script and length checks pass, and every glossary entry found in the source has one of its accepted target forms in the output (the entry's own forms plus the optional `ar_ok` / `en_ok` lists, which are accepted in an output but never written by us). Otherwise the piece is redone by the protected method above (`protected_retry`). Restored glossary and time phrases get a space where they would touch a letter or digit; tags, URLs and shortcodes never do. The segments of the split fallback must keep their digits exactly, or the segment is kept in the source language.

| Engine | Needs | Speed on the runner | Quality, one line |
|---|---|---|---|
| `tgemma` (default) | Ollama, 3.3 GB model, `requests` only | about 20 chars/s | Best so far: fluent, understands colloquial Arabic; slow |
| `opus` | torch stack, small models | about 637 chars/s | Fast, but poor in places (dropped currency, mangled times) |
| `m2m100` (418M) | torch stack | 55 to 85 chars/s | Better than opus, about a third of sentences have meaning errors |
| `m2m100-big` (1.2B) | torch stack, more memory | about 40 chars/s | Not better than the 418M model |
| `madlad` (3B) | torch stack, swap for the first conversion | not measured | Not yet compared |

- `madlad` is `google/madlad400-3b-mt` (Apache-2.0), converted to CTranslate2 int8 on the runner from Google's own repository (about 12 GB download, a 16 GB swap file is added when the cache is cold, and the download cache is deleted afterwards; the cached converted model is about 3 GB). The target language is chosen by a `<2ar> ` or `<2en> ` prefix.
- `tgemma` is TranslateGemma 4B (`translategemma:4b`, about 3.3 GB) run by a local Ollama server (`YK_OLLAMA_URL`, default `http://127.0.0.1:11434`), one request per piece with the prompt format from the [Ollama model page](https://ollama.com/library/translategemma). It translates whole paragraphs (pieces up to 1200 characters) and retries a failed paragraph sentence by sentence. The model is provided under Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
- For every engine an output that is not mostly in the target script (ignoring placeholders, digits and words copied from the source) is not accepted: the source piece is kept and counted (`script_retry`; `tgemma` first retries by sentence). For `tgemma` an answer much longer or shorter than its source is rejected the same way (`len_reject`).

`glossary.json` fixes place names and fixed terms (Salmiya = السالمية, KNET = كي نت). It is a list of `{"en": [...], "ar": [...], "ar_ambiguous": false}`. The first item of each list is the form written out; the others are accepted spellings. A match is replaced by a protected item holding the other language's first form, so it follows the same mechanism as a URL. English matches are case-insensitive whole words, longest first; Arabic matches are whole words and accept one attached prefix (ب ل و ف ك, and لل for ل + ال). `"ar_ambiguous": true` marks an Arabic form that is also an ordinary word or name (شرق, بيان); it is applied English to Arabic only. A missing or invalid file logs one line and the worker continues without a glossary.

## Selftest from the Actions tab

Actions > translate > Run workflow > tick `selftest`, pick the engine, run. It needs no site and no secrets. It runs one uncounted warm-up per direction, then 30 realistic samples (16 English to Arabic, 14 Arabic to English) and prints each as `[en->ar] source` and `=> output`. The summary shows: jobs done and failed, pieces split at placeholders, segments kept untranslated, `num_retry`, `fmt_dropped`, `script_retry`, `len_reject`, `natural_ok`, `protected_retry`, characters translated, seconds loading the model (download, conversion, load), seconds inside `translate_batch`, and chars/sec (characters divided by the `translate_batch` time only, so it is the real translation speed). The first run downloads and converts the model (a few minutes); later runs reuse the cache.

## Tests without a model

`python -I test_text.py` runs plain asserts with fake engines: placeholder fallback, glossary in both directions, prefixes, ambiguous entries, shortcodes, `%variables%`, Gutenberg comments. No model, network or extra install needed.

## Run on a PC

```
pip install -r requirements.txt
python translate_worker.py --selftest
set YK_SITE_URL=https://example.com
set YK_WORKER_TOKEN=...
python translate_worker.py
```

(On Linux or macOS use `export` instead of `set`.) The worker stops when no job is pending or after `YK_MAX_MINUTES`.

## How it runs on GitHub

The workflow starts on `repository_dispatch` (type `translate`), by hand, and every 30 minutes. The first step asks the site how many jobs are pending and stops with success when there are none, before any Python is set up. Runs never overlap (`concurrency: translate`).

## Model licences

- `Helsinki-NLP/opus-mt-ar-en` and `opus-mt-en-ar`: Apache-2.0, commercial use allowed.
- `facebook/m2m100_418M`: MIT.
- NLLB-200 is CC-BY-NC (non-commercial) and is deliberately not used.

## Important: 60-day inactivity

GitHub disables scheduled workflows in a repository with no activity for 60 days. If the schedule stops, open the Actions tab and re-enable the workflow (a new commit also resets the timer). Jobs queued on the site wait until then.
