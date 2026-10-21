# Retrieval benchmark (Qasper)

`evals/qasper_benchmark.py` measures how well the app finds the right part of a PDF, on real
papers and on questions written by other people. It tests **retrieval only**: whether the
chunks handed to the language model come from the pages that hold the answer. It does not
grade the generated answers.

The 21-question check in `evals/retrieval_eval.py` stays as a quick regression test. It is too
small, and its questions reuse the document's wording, to say which retrieval mode is better.

## Data

[Qasper](https://allenai.org/data/qasper) (Dasigi et al., 2021, CC BY 4.0) has questions about
NLP papers from arXiv. The question writers saw only the title and abstract, so their
questions are not echoes of the paper's wording. Annotators then marked the paragraphs of the
full text that answer each question.

- The benchmark downloads the JSON release and each paper's PDF from `export.arxiv.org`
  (one request about every 3 seconds). Nothing is committed except `evals/qasper_manifest.json`,
  which pins the SHA-256 of every PDF used, so a changed upstream PDF is reported.
- PDFs are the **latest** arXiv version. Qasper does not record the version; on 20 validation
  papers, 94.5% of evidence paragraphs were found in the latest version against 76.4% in v1.
- Downloads live in `.eval_cache/` (git-ignored), including the Chroma indexes.

## How a question gets its gold pages

The PDF is read with the app's own extraction. Each evidence paragraph is located by word
4-gram overlap (lower-cased, hyphenation and ligatures undone, Qasper's `BIBREF`/`INLINEFORM`
placeholders removed): a page holds the paragraph if it contains at least 20% of its 4-grams,
and the paragraph counts as found if the pages together contain at least 60%. A question's gold
pages are all pages holding any answerable annotator's evidence.

Questions are **dropped, and counted in the output**, when they are unanswerable, have only
table/figure ("FLOAT SELECTED") or heading evidence, or have evidence that cannot be found in the
PDF (for example because the PDF text differs from the text Qasper used). Papers that fail to
download or read are dropped too. Use `inspect` to print random matches and check them by eye.

## Metrics

All use the top *k* chunks, with 95% bootstrap intervals that resample **papers** (questions
about the same paper are not independent).

- **page hit@k**: a top-k chunk comes from a gold page. This is what the page viewer shows.
- **chunk hit@k**: a top-k chunk is mostly evidence text (at least half of its 4-grams occur in
  the evidence). Stricter, because a lucky page does not count.
- **MRR@10** of the first gold page.

Methods: `vector`, `bm25`, `hybrid` (as in the app). Chunk sizes 500, 1000 (the default) and
1500, with overlap 20% of the size. Differences between methods, and between chunk sizes at an
equal context budget (k × chunk size = 3000 characters), are **paired** per question.

## Running it

```bash
pip install -r requirements.txt -r requirements-local.txt
python -m evals.qasper_benchmark fetch   --split validation --limit 40
python -m evals.qasper_benchmark inspect --split validation --limit 40
python -m evals.qasper_benchmark run     --split validation --limit 40 --out evals/results/qasper_validation.json
python -m evals.qasper_benchmark fetch   --split test
python -m evals.qasper_benchmark run     --split test --out evals/results/qasper_test.json
```

Embeddings are local (`all-MiniLM-L6-v2`), so no API key is needed. On 4 CPU cores the test
split took about 45 minutes to download and about 2 hours to index and score (three chunk
sizes); the 40-paper validation run took about 25 minutes. It is not run in CI because it
downloads data. The result JSON (kept in `evals/results/`) records the commit, embedding model,
counts kept and dropped, and every number in the tables.

## Protocol

To avoid tuning on the answers, the matcher thresholds, PDF version and report format were
chosen on the **validation** split only. The decision rule was written down before the test
split was scored, and the test split is scored once:

1. Hybrid stays the default retrieval mode unless its paired page hit@3 difference from vector
   has a 95% interval entirely below zero on test. BM25 alone is never the default.
2. The default chunk size (1000, k=3 at the time) was never changed by the benchmark itself.
   Another size is only *recommended* if, at an equal context budget, it beats the default in hybrid page hit
   with a paired interval excluding zero on **both** validation and test. Small chunks make both
   page hit and chunk hit easier, so the recommendation comes with its costs: a new size
   re-indexes every stored PDF and changes how much text each prompt holds.

## Results

Scored once, on the Qasper **test** split with the rules above. 1270 of 1451 questions from 400
papers were kept (79 unanswerable, 62 with only table/figure or heading evidence, 34 whose
evidence could not be located in the PDF, 6 from 2 papers that failed to download). 92.2% of
evidence paragraphs were located. Embeddings: `all-MiniLM-L6-v2`. Intervals are 95%, resampling
papers. Validation (100 questions, 38 papers) is in `evals/results/qasper_validation.json`.

**Retrieval mode, default chunk size (1000), page hit:**

| | vector | BM25 | hybrid |
|---|---|---|---|
| @1 | 46.2% | 38.8% | 47.3% |
| @3 | 72.2% [69.4–74.9] | 67.1% [64.1–70.1] | 73.6% [71.1–76.2] |
| @5 | 82.8% | 79.7% | 85.0% |
| @10 | 93.9% | 92.7% | 95.5% |
| MRR@10 | 61.7% | 55.6% | 62.8% |

- **Hybrid vs vector, page hit@3 (paired):** +1.4% [-0.9, +3.6] at size 1000, +2.2% [-0.1, +4.7]
  at 500 and +3.3% [+0.9, +5.7] at 1500. Hybrid is never significantly worse, so by rule 1 it
  stays the default. Its edge at the default settings is small: at @3 it is within noise, and
  it is clearer at @5 (+2.3% [+0.1, +4.4]) and @10 (+1.6% [+0.2, +2.9]).
- **BM25 alone** is 4.6–5.3 points below vector at @3 (intervals exclude zero), as expected for
  questions that were not written from the paper's wording.

**Chunk size at an equal context budget (k × size = 3000 characters), hybrid, page hit:**

| chunk size, k | page hit | vs 1000, k=3 (paired) |
|---|---|---|
| 500, k=6 | 86.6% [84.6–88.6] | **+13.0%** [+10.9, +15.1] |
| 1000, k=3 (default) | 73.6% [71.1–76.2] | - |
| 1500, k=2 | 64.3% [61.4–67.1] | -9.4% [-11.6, -7.1] |

Validation agrees (500/k=6 was +10.0% [+2.0, +17.9]; 1500/k=2 was -8.0% [-15.5, -1.8]), so
rule 2 is met: **smaller chunks with more of them retrieve the evidence page more often at the
same amount of text.** The same holds for vector and BM25. As an exploratory check (not part of
the rule), the strict chunk-level hit rate also favours 500/k=6 (55.8%) over 1000/k=3 (31.3%).

What this does **not** show: that answers improve. More, smaller chunks make a page hit easier
by covering more places, and each prompt then contains six excerpts instead of three. That
question is tested in the next section, and the default was then changed (see **Decision**
there).

## Answer quality (local model)

`evals/qasper_answers.py` asks whether the retrieval gain carries over to answers. It generates
answers with the app's own prompt and local-model code (MiniCPM5-1B) but fixed generation
settings (greedy decoding, thinking off, at most 200 new tokens) instead of the UI sliders,
which the user sets and which default to a higher temperature and a longer limit for the same questions under two settings, both hybrid:
**default** (1000-character chunks, k=3) and **small** (500-character chunks, k=6). A seeded
random sample of 150 test papers gave 457 questions from 144 papers. Scoring needs no judge
model: yes/no questions use the answer's leading "yes"/"no", the others use Qasper's token-level
Answer-F1 (maximum over annotators); page citations like "(p. 3)" and reasoning blocks are
removed first. The decision rule was written before any answer was generated: *small* replaces
*default* only if the paired difference's 95% lower bound is above -0.02 and its answers are not
more than 25% longer.

| | default | small | small - default (paired) |
|---|---|---|---|
| Answer score | 0.297 | 0.309 | **+0.012** [-0.013, +0.036] |
| Gold-token recall | 0.391 | 0.408 | +0.017 [-0.015, +0.047] |
| Mean answer length | 19.0 words | 21.3 words | x1.12 |
| Seconds per answer (4 CPU cores) | 7.6 | 8.5 | |
| extractive (253 questions) | 0.256 | 0.259 | +0.002 [-0.021, +0.026] |
| abstractive (127) | 0.204 | 0.208 | +0.004 [-0.020, +0.027] |
| yes/no accuracy (77) | 0.584 | 0.639 | +0.055 [-0.050, +0.167] |

By the rule, small is **not meaningfully worse**, so switching is allowed. It is not shown to be
better: every interval includes zero. The +13-point page-hit gain does not show up as a
measurable answer gain with this 1B model. The cost of small chunks is about 12% more time per
answer (more excerpts to read) and a re-index of every stored PDF.

Caveats: only the local 1B model was tested (no OpenAI key was available), and a larger model
may use better context differently; token F1 is a crude measure of free-form answers; and the
oracle arm (annotators' evidence as context), which would separate retrieval from model error,
was not run.

**Decision.** The answer-quality rule was met, and the maintainer chose to adopt the candidate, so
the app defaults changed from 1000-character chunks with overlap 200 and `k=3` to 500-character
chunks with overlap 100 and `k=6` (the overlap stays at 20% of the chunk size, as benchmarked).
This was decided with the local 1B model only; the OpenAI backend was not tested. The 21-question
check in `evals/retrieval_eval.py` was re-run under the new defaults and no longer shows hybrid
ahead (86% vector, 86% BM25, 81% hybrid at `k=6`; 21 questions, wording-sharing questions), so it
is a regression check only. Hybrid stays the default on the strength of the 1270-question test.

## Scanned pages (OCR)

A separate, descriptive experiment (`evals/qasper_scanned.py`; no default depends on it). It asks
how much retrieval is lost when a paper has to be read with OCR. Each paper is rasterised to an
image-only PDF at 200 dpi (the run stops if a copy still has a text layer), indexed through the
app's own Tesseract path, and scored on the **same questions and gold pages** as the original
text PDF. Settings are the current defaults (500-character chunks, overlap 100, hybrid and vector
retrieval, MiniLM embeddings). 96 papers drawn at random (seed 2) from the test split gave 320
questions; intervals are 95%, resampling papers, and the differences are paired.

| Page hit | Text layer | Tesseract | Tesseract - text layer | Unlimited-OCR (remote) | Remote - text layer |
| --- | --- | --- | --- | --- | --- |
| hybrid, top 3 | 74.7% | 72.8% | -1.9% [-6.8, +3.3] | 71.9% | -2.8% [-8.1, +2.2] |
| hybrid, top 6 | 87.5% | 86.6% | -0.9% [-4.6, +2.7] | 88.8% | +1.2% [-2.8, +5.3] |
| vector, top 3 | 73.4% | 72.5% | -0.9% [-6.3, +4.2] | 75.0% | +1.6% [-3.2, +6.5] |
| vector, top 6 | 87.8% | 84.1% | -3.8% [-8.1, +0.3] | 87.8% | +0.0% [-3.7, +3.8] |

On clean, born-digital arXiv pages rendered at 200 dpi, Tesseract costs between about 1 and 4
points of page hit, and Unlimited-OCR between about -3 and +2 points. Every interval, for both
engines, includes zero, so this sample cannot separate a small loss from none. The two engines
were run on the same papers and questions, but I did not compute a paired Tesseract-versus-remote
interval, and the point estimates differ by less than the width of the intervals above; this
experiment does not show that one engine retrieves better than the other. Hybrid lost less than
vector for Tesseract at top 6 (-0.9% against -3.8%), but that comparison was not tested and should
not be read as a finding.

The remote run used `baidu/Unlimited-OCR` (Hub revision `07dea83`) served by vLLM on one NVIDIA L4
(`evals/modal_unlimited_ocr.py`), one request per page, with the app's prompt and parameters.
Of the 889 pages, 4 (the first four pages of one paper) timed out at the 120 s client limit and
were indexed as empty; they are counted in the scores above. Reading the 889 pages took about
1 hour 45 minutes, roughly 7 seconds per page with four requests in flight.

Three measures need care:

- **Chunk hit is not comparable here.** It falls from 57% to 17% under OCR, because it matches
  word 4-grams of the chunk against evidence taken from the text PDF, and OCR breaks those
  sequences (line breaks, hyphenation, ligatures, column order). That is a measurement artefact,
  not a retrieval result. Page hit is the figure to use.
- **OCR fidelity** is 64.9% for Tesseract and 83.1% for Unlimited-OCR: the average share of each
  text page's 4-grams found in the indexed OCR of that page. It counts exact word sequences, so
  it is a strict measure.
- **Chunk hit under the remote engine** was 62.8% (vector) and 64.7% (hybrid) against 57.2% and
  56.6% for the text layer (differences +5.6% and +8.1%, intervals excluding zero). I did not
  investigate why it exceeds the text layer; it likely reflects how the text is segmented, not
  better retrieval, so page hit is still the figure to use.

### Degraded scans

Clean renders are the easy case, so the same 96 papers and 320 questions were also run on
synthetically degraded copies (`--degrade`). The levels and the reading rule were fixed before the
runs: a difference counts only if its paired 95% interval excludes zero. *Severe* is 80 dpi, a
random page rotation of 1-2 degrees, Gaussian noise (sigma 16) and JPEG quality 30; the noise
and rotation are seeded per paper and page, so both engines read identical images. This is a
simulation, not real scans.

| Page hit, severe scans | Text layer | Tesseract | Unlimited-OCR | Remote - Tesseract (paired) |
| --- | --- | --- | --- | --- |
| hybrid, top 3 | 74.7% | 71.9% | 74.7% | +2.8% [-3.3, +8.9] |
| hybrid, top 6 | 87.5% | 86.9% | 90.0% | +3.1% [-0.6, +6.8] |
| vector, top 3 | 73.4% | 72.8% | 76.2% | +3.4% [-1.7, +8.8] |
| vector, top 6 | 87.8% | 86.9% | 89.1% | +2.2% [-1.5, +5.9] |

Neither engine differs from the text layer (every interval includes zero), and the remote engine
is ahead of Tesseract by 2 to 3 points in all four settings, but **no interval excludes zero**,
so retrieval does not separate them. Retrieval looks too forgiving to show the gap: Tesseract's
page hit on severe scans is about the same as on the clean ones (vector top 6: 86.9% against
84.1%) although its OCR fidelity fell from 64.9% to 13.8%. A plausible reason, which I did not
test, is that a paper has roughly nine pages, so the top six chunks cover much of it even when
the text is poor. OCR fidelity is **78.9% for Unlimited-OCR against 13.8% for Tesseract** on
these scans; one remote page was indexed as empty after a connection error to the server.
Chunk hit is again not comparable.

A *moderate* level (120 dpi, rotation 0.5-1 degrees, noise sigma 10, JPEG quality 45) gave the same
picture with a smaller text gap: OCR fidelity 83.3% for Unlimited-OCR against 61.9% for Tesseract,
and remote - Tesseract page hit between -1.2% and +3.8% across the same four settings, every
interval including zero (vector top 6: +3.8% [+0.0, +7.3], which the harness counts as including
zero). Results: `evals/results/qasper_scanned_moderate.json`.

Because retrieval hides that gap, the answer-quality benchmark was run on the severe scans only.

### Answer quality on severe scans

`evals/qasper_answers.py generate --degrade severe` answers the same 320 questions with the same
1B local model and the app's defaults, from three indexes of each paper: the text-layer PDF
(*small*), the Tesseract read of the severe scan and the Unlimited-OCR read of it. Scoring is as
above (token F1, yes/no accuracy, no judge model). The primary comparison, written down before
the run, was Unlimited-OCR minus Tesseract on severe scans.

| | Text layer | Tesseract | Unlimited-OCR |
|---|---|---|---|
| Answer score | 0.315 | 0.244 | 0.317 |
| Gold-token recall | 0.417 | 0.311 | 0.444 |
| extractive (185 questions) | 0.245 | 0.156 | 0.256 |
| abstractive (80) | 0.207 | 0.178 | 0.235 |
| yes/no accuracy (55) | 0.709 | 0.636 | 0.643 |

| Paired difference (answer score) | |
|---|---|
| Unlimited-OCR - Tesseract | **+0.073** [+0.044, +0.104] |
| Tesseract - text layer | -0.071 [-0.101, -0.045] |
| Unlimited-OCR - text layer | +0.002 [-0.022, +0.025] (includes zero) |

On heavily degraded scans, answers built from Tesseract's text are measurably worse than answers
from the text layer, and answers built from Unlimited-OCR's text are not distinguishably
different from the text layer. The gain over Tesseract is largest on extractive questions
(0.156 to 0.256); abstractive questions also gained (0.178 to 0.235), and I did not compute
intervals per question type. The yes/no subset is small
(55 questions) and both OCR engines are below the text layer there; its gap was not tested on
its own.

Caveats: one 1B model, one degradation level for answers, a simulated rather than real scan, a
single seed, and a crude metric. The GPU server ran for roughly 9 hours in total (an estimate,
not a billed figure); the Tesseract arm needs none. The OCR itself ran once per page; answers were generated afterwards from
the cached indexes.

Not covered: real scans (photographs, creases, bleed-through), handwriting, other languages, the
OpenAI backend. Results: `evals/results/qasper_scanned.json` (Tesseract, clean),
`qasper_scanned_remote.json` (remote, clean), `qasper_scanned_severe.json` (both engines, severe)
and `answers_severe.jsonl` (the 960 generated answers). `--ocr-engine ... --ocr-url ...
--ocr-api-key ...` reruns the retrieval runs, and `python -m evals.qasper_answers score --answers
evals/results/answers_severe.jsonl` rescored the answers.

## Limitations of this benchmark

- Gold pages are found by text matching, so some evidence is missed or mislocated. The drop
  counts show how many questions that affects.
- Questions are about NLP papers with a clear structure; scanned PDFs, other languages and
  layout-heavy documents are not covered (OCR is switched off for the main benchmark; see
  [Scanned pages](#scanned-pages-ocr) for a small OCR experiment on rendered pages and simulated degraded scans).
- Page hit is easier to reach with more, smaller chunks, so read it together with chunk hit.
- Retrieval only: a good retrieval score does not guarantee a good answer.
