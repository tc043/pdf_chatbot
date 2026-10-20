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
python -m evals.qasper_benchmark run     --split validation --limit 40 --out qasper_results_validation.json
python -m evals.qasper_benchmark fetch   --split test
python -m evals.qasper_benchmark run     --split test --out qasper_results_test.json
```

Embeddings are local (`all-MiniLM-L6-v2`), so no API key is needed. Expect the test split
(416 papers) to take about 30 minutes to download and a few more to index on CPU. It is not run
in CI because it downloads data. The result JSON records the commit, embedding model, counts
kept and dropped, and every number in the tables.

## Protocol

To avoid tuning on the answers, the matcher thresholds, PDF version and report format were
chosen on the **validation** split only. The decision rule was written down before the test
split was scored, and the test split is scored once:

1. Hybrid stays the default retrieval mode unless its paired page hit@3 difference from vector
   has a 95% interval entirely below zero on test. BM25 alone is never the default.
2. The default chunk size (1000, k=3) is never changed by the benchmark itself. Another size is
   only *recommended* if, at an equal context budget, it beats the default in hybrid page hit
   with a paired interval excluding zero on **both** validation and test. Small chunks make both
   page hit and chunk hit easier, so the recommendation comes with its costs: a new size
   re-indexes every stored PDF and changes how much text each prompt holds.

## Results

Test results are added here after the test split is scored.

## Limitations of this benchmark

- Gold pages are found by text matching, so some evidence is missed or mislocated. The drop
  counts show how many questions that affects.
- Questions are about NLP papers with a clear structure; scanned PDFs, other languages and
  layout-heavy documents are not covered (OCR is switched off for this benchmark).
- Page hit is easier to reach with more, smaller chunks, so read it together with chunk hit.
- Retrieval only: a good retrieval score does not guarantee a good answer.
