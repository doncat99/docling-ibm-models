# TableFormer verified draft decoding

Investigation: 2026-10-22. This is source-fork validation, not an installed
OntoScience release or a general speed guarantee.

## Algorithm and remaining limits

The decoder proposes up to 15 tags from the longest previously accepted tag
suffix. One causal decoder call verifies the current token and the proposals.
Only the greedy-consistent prefix is accepted. The first mismatch supplies the
next accepted token; later speculative cache entries are overwritten on the next
call. Existing structure corrections and EOS rules remain in force. Model
weights, attention context, numerical precision, and table accuracy mode do not
change. This groups work within one table; it does not batch different tables.

The original prototype rebuilt its suffix index from the entire history on
every verification call, even at verification width 1. With a fixed 16-token
context, cumulative index-building work could grow quadratically with output
length. `_TagDraft` now indexes each accepted transition once and uses only a
bounded suffix for proposals. Rejected proposals never enter the index. The
draft-matching test compares every incremental update with an independent
full-history search over repeated and irregular tags.

KV projection caching and BBox vectorization are shared by both sides of the
comparison below. The incremental draft index is also shared: the experiment
isolates verification width 1 versus 16, not the incremental index's standalone
latency benefit. A causal block can still waste computation on rejected tokens.
It can change floating-point rounding, so equivalence must be measured on each
target backend and workload; it is not a bitwise-equivalence guarantee.

## Matched full-report experiment

Apple M2, 24 GB RAM, macOS 15.6.1, PyTorch 2.14.1, four CPU threads,
float32, accurate TableFormer, no OCR, no CPU fallback. Layout and TableFormer
both use MPS, with Docling commit `2d511ba3`'s shared inference submission guard.
Both runs use the same working model fork sources, weights, input, options,
instrumentation, and measurement boundary. Each starts in a fresh process.
The only decoder option changed is verification width.

Input: 77-page Vermilion report, SHA-256
`12655a44a303dbd7cb24e16ccd204afbda6624ddf45bd3234d40f0358a5c3410`.

| Verification width | Conversion wall time | Model prediction time, summed | Outcome |
| --- | ---: | ---: | --- |
| 1, serial KV decoder | 76.219 s | 46.924 s | SUCCESS, 77 pages, 37 tables, 5877 cells, zero errors |
| 16, verified drafts | 44.163 s | 14.585 s | SUCCESS, 77 pages, 37 tables, 5877 cells, zero errors |

A second fresh-process width-16 conversion completed in 41.079 seconds with
the same counts, zero errors, and the same canonical exported document hash.
Only the first pair is used for the percentage comparison below; serial was
not repeated in this experiment, so this is not a latency distribution.

Conversion wall time includes converter/model initialization and excludes JSON
export. Per-table timings include instrumentation and must not be interpreted
as isolated GPU kernel times. This pair observed a 42.1% reduction in conversion
wall time and a 68.9% reduction in summed model prediction time. It is one report
on one machine, not a broad benchmark or evidence about CUDA performance.

The complete exported documents match after removing only `bbox` fields:
canonical JSON SHA-256
`7cd05bf6a69f358b9cce467c6f7b693239d864f0b8f7a7aef90fef460947a4f6`.
This includes cell text, structure, spans, and other exported metadata. Raw tag
sequence multisets also match; pipeline call order differs and raw array index
is not a stable table identity. Matching a reference does not establish human
ground truth. The existing warning about one unmatched PDF cell remains.
The maximum bounding-box coordinate difference in the first pair was 0.017214
PDF points over 52000 numeric bounding-box fields.

The serial full conversion completed without OOM. Maximum driver-memory samples
taken after table calls were approximately 2.62 GB for serial and 2.61 GB for
verified drafts; these are snapshots, not measured process peaks. This does not
explain the earlier OOM or establish that it cannot recur. The earlier failed
run must not replace a fresh control run.

Local diagnostic evidence (not distributed with this fork):

- `tableformer-full-compare.py`: same runner for both verification widths.
- `vermilion-algorithm-matched-serial/{metadata,summary,document,raw-tables}.json`.
- `vermilion-algorithm-matched-spec/{metadata,summary,document,raw-tables}.json`.
- `vermilion-algorithm-matched-spec-repeat/{metadata,summary,document,raw-tables}.json`.
- Separate captured-memory replay: `tableformer-fork-decode-*.jsonl`.

All 20 focused tests passed and `make validate` passed. Focused tests cover
decoder/KV equality, causal block equality, EOS bounds,
incremental positions, BBox equivalence, draft indexing, and rejected-proposal
rollback. They do not replace multi-document quality or installed-app acceptance.
No model weight, baseline, product acceptance threshold, or installed runtime
was changed by this experiment.
