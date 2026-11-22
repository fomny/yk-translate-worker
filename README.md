# yk-translate worker

Translation worker for the `yk-auto-translate` WordPress plugin (spec 003). It pulls pending jobs from the site over HTTPS, translates Arabic and English in both directions with an open model on CPU (CTranslate2, int8), and posts the results back. No usage cap, no paid service, no translation-vendor key.

## Secrets and variables

Set these in the GitHub repository (Settings > Secrets and variables > Actions) or as environment variables on a PC:

| Name | Meaning |
|---|---|
| `YK_SITE_URL` | Site address, for example `https://example.com` (secret) |
| `YK_WORKER_TOKEN` | The shared secret sent as `X-YK-Token` (secret) |
| `YK_ENGINE` | Optional: `opus` (default) or `m2m100`. In Actions it comes from the run input |

Optional tuning (PC or workflow env): `YK_MODEL_DIR` (default `./models`), `YK_BATCH` (16), `YK_LIMIT` (25), `YK_MAX_MINUTES` (20), `YK_GLOSSARY` (path of the glossary file, default `glossary.json` next to the script).

## Protected items and the glossary

URLs, e-mails, phone numbers, numbers, HTML tags and comments (including Gutenberg `<!-- wp:... -->` blocks), WordPress shortcodes (`[name ...]`, `[/name]`) and `%variables%` such as `%title%` are replaced by placeholders before translation and put back byte-identical. If the model drops a placeholder, that piece is split at each placeholder, the text segments between them are translated separately in one batch, and everything is joined back in the original order.

`glossary.json` fixes place names and fixed terms (Salmiya = السالمية, KNET = كي نت). It is a list of `{"en": [...], "ar": [...], "ar_ambiguous": false}`. The first item of each list is the form written out; the others are accepted spellings. A match is replaced by a protected item holding the other language's first form, so it follows the same mechanism as a URL. English matches are case-insensitive whole words, longest first; Arabic matches are whole words and accept one attached prefix (ب ل و ف ك, and لل for ل + ال). `"ar_ambiguous": true` marks an Arabic form that is also an ordinary word or name (شرق, بيان); it is applied English to Arabic only. A missing or invalid file logs one line and the worker continues without a glossary.

## Selftest from the Actions tab

Actions > translate > Run workflow > tick `selftest`, pick the engine, run. It needs no site and no secrets. It runs one uncounted warm-up per direction, then 26 realistic samples (13 English to Arabic, 13 Arabic to English) and prints each as `[en->ar] source` and `=> output`. The summary shows: jobs done and failed, pieces split at placeholders, segments kept untranslated, characters translated, seconds loading the model (download, conversion, load), seconds inside `translate_batch`, and chars/sec (characters divided by the `translate_batch` time only, so it is the real translation speed). The first run downloads and converts the model (a few minutes); later runs reuse the cache.

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
