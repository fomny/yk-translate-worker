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

Natural first (only `tgemma`): attempt 1 peels the structural items (tags, shortcodes, `%variables%`, URLs, e-mails, phone-like numbers) off both ends of the piece and never sends them. Inside the core the glossary terms are written in the target language in place (`Family shop in السالمية`, `شقة في Salmiya`) and the remaining structure is sent raw. The answer is accepted (`inline_ok`) only if every structural item is back byte-identical, the same number of times and in the same order (a phone: the same digits once spaces are removed), the numbers agree by value, the script and length checks pass, and every glossary entry found in the source has one of its accepted target forms in the output (the entry's own forms plus the optional `ar_ok` / `en_ok` lists, accepted in an output but never written by us). Otherwise attempt 2 is the protected method above (`protected_retry`). Numbers are compared by value (`6,200` = `6200`, `120 ألف` = `120,000`, `11:00` = `11`); a number may also be written as a word (two = 2 = سنتين). A piece is never returned half translated: attempt 1, else attempt 2 if it has no kept segment, else the attempt-1 answer when only the glossary check failed (`glossary_miss`), else the whole source piece (`kept`). Restored glossary and time phrases get a space where they would touch a letter or digit; tags, URLs and shortcodes never do. Set `YK_DEBUG=1` (the workflow does it for the selftest) to log every rejected attempt with the check that failed (`placeholders`, `structure`, `digits`, `script`, `length`, `glossary:<entry>`) and every segment the protected path kept.

Every answer of every engine is also refused when it holds a bracketed span (`[..]`, `{..}`, `<word>`), template filler ("your contact information", "insert ", "أدخل ", "اكتب هنا" ...) or Markdown (`**`, `__`, `#`, `- `, `* ` list markers, code fences) that the source did not have (`invention` / `markdown` in the debug lines). The Ollama request asks for `num_thread` = the CPU count (`YK_OLLAMA_THREADS` overrides) and `num_ctx` 2048; the summary shows the model's own numbers: model calls, prompt and generation tokens per second.

Site answers: a 200 (also `already`) is done. 409 (`yk_stale_hash`, `yk_not_claimed`, `yk_not_needed`), 404 `yk_unknown_job` and, after three posts, 409 `yk_busy` drop the job (`skipped`, no `/fail`). 413 and 422 and other 4xx call `/fail` once with the code. A 5xx on `/result` is not retried and not reported (the site records it). After a network error the identical body is posted once more, only after `GET /pending?count_only=1` answered. 401 and 403 stop the run. A text field whose `<yk-keep>` tags did not all come back is not posted: the job gets `/fail` `protected lost locally`. Fields that are not strings are passed through untouched.

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

Actions > translate > Run workflow > tick `selftest`, pick the engine, run. It needs no site and no secrets. It runs one uncounted warm-up per direction, then 30 realistic samples (16 English to Arabic, 14 Arabic to English) and prints each as `[en->ar] source` and `=> output`. The summary shows: jobs done and failed, pieces split at placeholders, segments kept untranslated, `num_retry`, `fmt_dropped`, `script_retry`, `len_reject`, `inline_ok`, `protected_retry`, `glossary_miss`, `skipped`, characters translated, seconds loading the model (download, conversion, load), seconds inside `translate_batch`, and chars/sec (characters divided by the `translate_batch` time only, so it is the real translation speed). The first run downloads and converts the model (a few minutes); later runs reuse the cache.

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
