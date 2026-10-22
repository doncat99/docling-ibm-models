# TableFormer verified draft decoding

Investigation: 2026-10-22. This is source-fork validation, not an installed
OntoScience release or a general speed guarantee.

## Algorithm and remaining limits

The decoder proposes up to 31 tags on MPS (15 on other backends) from the longest previously accepted tag
suffix. One causal decoder call verifies the current token and the proposals.
Only the greedy-consistent prefix is accepted. The first mismatch supplies the
next accepted token; later speculative cache entries are overwritten on the next
call. Existing structure corrections and EOS rules remain in force. Model
weights, attention context, numerical precision, and table accuracy mode do not
change. This groups work within one table; it does not batch different tables.

The MPS width increased from 16 to 32 after captured-memory comparisons at
widths 8, 16, 32, and 64. All 37 sequences matched; width 32 reduced decoder
calls from 1299 to 1058. Width 64 made fewer calls but was slower in the
follow-up sweep, so it was not selected. CPU/CUDA retain width 16 because this
measurement does not establish a benefit there. Verification and EOS semantics
are unchanged. The rejected-proposal test exercises widths 16, 32, and 64.

Combined with eval-only ResNet convolution/normalization folding in the Docling
fork, three alternating full-report pairs reduced the 77-page median from
39.129 to 33.056 seconds (15.5%). All exported content except bbox matched;
maximum bbox difference was 0.000145 PDF points. Three additional PDF fixtures
(9, 18, and 4 pages) preserved content and tags too. This combined result cannot
be attributed to the width change alone. See the Docling fork's
`docs/usage/mps_performance.md` for all runs and memory/validation limits.

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

## Follow-up: bounded-memory cell matching

The next measured improvement is in CPU post-processing, not decoder batching.
The original PDF-cell matcher checks every predicted-cell/PDF-token pair.
The new matcher sorts PDF tokens by vertical start and builds a maximum-end
interval index. Each predicted cell queries only vertically overlapping tokens.
The exact intersection and area arithmetic remain unchanged. Candidates return
to original PDF-token order before matching. This preserves duplicate-ID,
tie, and insertion-order behavior, including reversed and zero-area boxes.

The index uses O(P) memory for P PDF tokens, plus at most O(P) temporary
candidates for one predicted cell. It never stores the full T-by-P pair set.
Dense overlapping boxes can still require all pairwise checks; this is an
optimization for spatially separated rows, not a new matching rule.
Response assembly also indexes predicted cells by ID once instead of scanning
all predicted cells for each match. Duplicate IDs retain first-match behavior.

Captured inputs from the same 77-page report were replayed against the exact
committed implementation. All returned values and match-key order agreed.
Four repeated totals, in seconds:

| Operation | Old implementation | Interval/index implementation |
| --- | --- | --- |
| 74 PDF-cell matching calls | 1.3303, 1.2755, 1.3091, 1.1497 | 0.2611, 0.2531, 0.4209, 0.2555 |
| 37 response assembly calls | 0.2781, 0.2851, 0.2887, 0.6201 | 0.0167, 0.0163, 0.0164, 0.0163 |

These are local CPU operation timings, not full-conversion speedup ratios.
The disjoint-row regression makes 100 exact intersection calls for 100 rows;
the old implementation fails the same specification with 10000 calls.
Five hundred seeded comparisons also cover duplicates, reversed boxes,
degenerate boxes, and output order. No expected-output baseline was changed.

Fresh-process full conversions of the final interval-index implementation:

| Input | Pages / tables / cells | Control | Candidate | Maximum bbox delta |
| --- | --- | ---: | ---: | ---: |
| Vermilion | 77 / 37 / 5877 | 40.734 s | 36.273 s | 0 |
| normal_4pages | 4 / 1 / 15 | 4.197 s | 3.140 s | 0.0000611 PDF points |
| 2206.01062 | 9 / 5 / 420 | 6.616 s | 5.783 s | 0 |
| redp5110_sampled | 18 / 6 / 198 | 6.392 s | 5.959 s | 0 |

Every pair preserved all exported content except bbox and the raw table-tag
sequence multiset, with zero conversion errors. Each fixture has only one
control/candidate pair. Background load and cache effects were not controlled;
these times confirm completion and are not a stable end-to-end gain claim.
The report still has the existing unmatched-token warning described above.
The final code passed 49 affected tests and `make validate` (including Ruff
and ty). Evidence: `interval-final-oct22-{report,normal,paper,redbook}-*`,
`compare-interval-final-oct22.py`, and `replay-cpu-source-oct22.py` in the local
OntoScience `.tmp/` workspace.

Acceptance scope: external-library internal optimization; no OntoScience
product contract, Case pass criterion, or baseline changed. App-level
`change-verification` and installed-runtime acceptance were not run because
this fork revision has not been pinned or packaged into OntoScience.

## Rejected follow-up experiments

- Moving CPU preparation and table matching outside the shared MPS guard did
  not improve conversion time. Three full-report controls took 31.7975,
  37.5452, and 35.3659 seconds; the narrowed guard took 38.5764, 37.1698,
  and 37.4659 seconds. The median regressed by 5.9%. All outputs matched
  apart from bbox rounding up to 0.0000611 PDF points. The guard change was
  reverted. Both models still use MPS with the existing crash protection.
- Position-aware drafts reduced decoder calls from 1058 to 887 on captured
  encoder memories. This alone did not establish a stable latency benefit.
  With one warmed model and all 37 table memories, three alternating rounds
  took 9.052/11.048/12.274 seconds for the committed draft and
  9.186/10.865/9.513 seconds for the position-aware draft. All tags agreed.
  This prototype was not selected for production.
- Confidence-based draft widths raised accepted-output/verified-position
  utilization from 32.3% to 50.5%, but increased calls from 887 to 1029.
  The same warmed-model experiment took 11.404/10.232/12.625 seconds:
  slower than the position-aware draft in all three rounds. Improving the
  utilization ratio alone is not a valid performance acceptance criterion.

These decoder replays exclude image encoding and full conversion. They ran
on a machine with concurrent background activity; three rounds are not a
latency distribution. A preliminary all-candidates sweep also had noisy full
conversion timings, so it was replaced with the bounded-memory index above.
The replay scripts, captured inputs, and full conversion evidence remain in
the local OntoScience `.tmp/` workspace. No adaptive decoding or narrower MPS
guard is shipped by this follow-up.
