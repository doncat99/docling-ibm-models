# TableFormer verified draft decoding

Investigation: 2026-10-22. This is source-fork validation, not an installed
OntoScience release or a general speed guarantee.

## Algorithm and remaining limits

### Shared additive causal mask

The block decoder previously supplied the same boolean causal mask to each
of its six attention layers. PyTorch SDPA converted that mask to additive
zero/negative-infinity form once per layer. The decoder now constructs the
additive mask once per verified block and reuses it across layers. The dtype
matches the decoder input. Prefix offsets and causal visibility are unchanged.
This removes repeated mask conversion without changing verification width,
proposal generation, weights, precision, concurrency, or table batching.

Two alternating fresh-process full-report pairs, using the configuration and
77-page input identified below, measured conversion wall time including model
initialization:

| Pair | Boolean mask | Shared additive mask |
| --- | ---: | ---: |
| 1 | 30.434 s | 28.621 s |
| 2, candidate first | 31.760 s | 28.807 s |
| Median | 31.097 s | 28.714 s |

The observed median reduction is 7.7%. All four complete exported documents,
including coordinates, were exactly equal: 77 pages, 37 tables, 5877 cells,
zero conversion errors. This is one machine and one report, not a general
speed guarantee. An isolated four-round replay preserved all accepted hidden
states and tags bitwise; one replay round was slower with the candidate.
CPU profiler spans that include tensor materialization also include device
completion waits and must not be reported as pure transfer overhead.

The regression specification failed before the source change because SDPA
received a boolean mask. It checks serial-versus-block hidden states with a
nonempty cached prefix, exact causal visibility, dtype, and reuse across layers.
Local evidence: `decoder-mask-full-{baseline,candidate}-{1,2}/`,
`decoder-mask-replay-oct22.json`, and `decoder-operators-oct22-trace.json` in
OntoScience's ignored `.tmp` directory. This source change does not update the
installed application or establish the 7–14 second full-report goal.

### Draft algorithm

The decoder proposes up to 31 tags on MPS (15 on other backends). It first
searches accepted suffixes at the same tag offset within a row, then searches
suffixes without a position constraint if that position has no match. One
causal decoder call verifies the current token and the proposals.
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
bounded suffix for proposals. The position-aware implementation uses an
eight-token context, as in the evaluated prototype. Rejected proposals never
enter either index. The
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

## Follow-up experiments and decisions

- An initial three-pair lock-scope experiment observed a 5.9% median
  regression and was temporarily reverted. Background load was not isolated;
  this did not establish that a narrower guard was ineffective. Six later
  alternating full-report pairs are recorded below. The narrower boundary is
  retained to remove unnecessary serialization, without a stable speedup claim.
- Position-aware drafts reduced decoder calls from 1058 to 887 on captured
  encoder memories. This alone did not establish a stable latency benefit.
  With one warmed model and all 37 table memories, three alternating rounds
  took 9.052/11.048/12.274 seconds for the committed draft and
  9.186/10.865/9.513 seconds for the position-aware draft. All tags agreed.
  This latency evidence alone did not select the prototype. It was later
  retained for its measured reduction in verification calls; see below.
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

## Position-aware candidates

Suffix-only drafting conflates identical tag suffixes at different positions
within a row. `_TagDraft` now keeps accepted continuations keyed by both the
suffix and its row offset, before consulting position-independent suffixes.
The offset counts accepted OTSL tags since the latest `nl`, not visual PDF
columns. The `nl` token ID comes from the model vocabulary. Each proposal
advances a local offset; only accepted tags change the persistent offset and
indexes. Rejected speculative tags cannot train the next proposal.

This improves the proposal heuristic, not the model's correctness rules:
the greedy verifier, EOS/budget limits, table corrections, and KV-cache rollback
remain unchanged. A poor proposal was wasted work rather than evidence of a
wrong final table. No weight or acceptance threshold changed. The additional
position index grows linearly with accepted output for a fixed context bound;
it does add CPU bookkeeping and memory, which can offset fewer decoder calls.

With the final fork implementation, all 37 captured MPS table memories again
produced the same 9180 accepted tags, including EOS. Verification calls fell
from 1058 to 887 (16.2%). All corresponding accepted decoder hidden states
were exactly equal in this replay (maximum absolute delta 0). This is observed
equality on these memories, not a general bitwise guarantee. The earlier
full-report 32.18-to-29.91-second median also included spatial matching and
cannot be attributed to position-aware candidates alone.

Tests compare each incremental update against an independent full-history
position-aware search, distinguish the same suffix at different row offsets,
exercise different vocabulary IDs for `nl`, and check that repeated proposals
do not alter accepted state. Existing random-model tests continue to compare
greedy tags and hidden states after rejected proposals at widths 16/32/64.

The final implementation passed 51 affected tests and `make validate`.
The same four full PDF fixtures also completed with zero conversion errors:

| Input | Conversion time | Maximum bbox delta from the pre-change export |
| --- | ---: | ---: |
| Vermilion, 77 pages / 37 tables / 5877 cells | 32.976 s | 0 |
| normal_4pages, 4 pages / 1 table / 15 cells | 4.323 s | 0.0000229 PDF points |
| 2206.01062, 9 pages / 5 tables / 420 cells | 5.185 s | 0 |
| redp5110_sampled, 18 pages / 6 tables / 198 cells | 6.443 s | 0.0001221 PDF points |

All exported content except bbox and all raw tag-sequence multisets agreed
with the interval-index-only reference exports. Those reference conversions
ran earlier, so these times must not be used as a fresh paired speed test.
Local evidence: `position-fork-replay-oct22.json`, `position-fork-oct22-*`,
`position-fork-validation-oct22.py`, and `compare-position-fork-oct22.py`.
The scope remains external-library performance: no App Case delta or baseline
promotion, and no installed-runtime acceptance before an immutable artifact
is pinned and packaged by OntoScience.

## Narrower submission guard: retained, no stable speedup claim

Six alternating pairs used the same 77-page report, both models on MPS,
PyTorch 2.14.1, float32, and four CPU threads. GPU benchmarks did not overlap.
Other system activity remained; these are not isolated performance baselines.

| Pair | Coarse guard | Narrow guard |
| --- | ---: | ---: |
| 1 | 29.88 s | 28.63 s |
| 2 | 29.71 s | 29.05 s |
| 3 | 29.62 s | 29.27 s |
| 4 | 29.77 s | 30.19 s |
| 5, complete lock timing | 32.34 s | 34.10 s |
| 6, complete lock timing | 29.29 s | 30.28 s |

The first four pairs reduced median wall time by 1.9%; the last two were
slower. Across all six pairs the medians were effectively equal. Summed
layout/table host lock waits in pairs 5 and 6 fell from 12.90 to 9.34 seconds
and 11.68 to 9.47 seconds. CPU preparation and matching overlapped layout's
held-guard intervals. These host intervals are not GPU execution timings and
must not be added to infer critical-path savings.

All twelve conversions produced 77 pages, 37 tables, 5877 cells, and zero
conversion errors or crashes. Complete exports matched after removing only
bbox fields; raw tag-sequence multisets matched (9180 tags including EOS).
Maximum bbox deviation was 0.0000611 PDF points. The existing unmatched-cell
warning remains. Reference equality is not independently verified ground truth.

Retain the narrower guard to eliminate unnecessary CPU serialization. Keep all
device transfers, inference, tensor postprocessing, and materialization under
the shared guard. This does not remove MPS crash protection, increase worker
counts, or establish a stable end-to-end speedup. Regression tests check both
predictor entry points and the layout boundary. Local evidence:
`mps-lock-retest-oct22-results.json` in OntoScience's ignored `.tmp/` directory.

The integrated source revision completed a fresh full-report run in 36.457
seconds: 77 pages, 37 tables, 5877 cells, zero errors. Content and raw tags
matched the coarse-guard reference; maximum bbox delta was 0.0000611 PDF
points. This is an integration observation, not a paired speedup measurement.
The model revision passed 53 focused tests and changed-file `make validate`.
Version `4.0.4+ontoscience.2` includes this guard interface and the earlier
verified-decoding, position-aware-draft, and matching-index changes. No
checkpoint weights or installed application are changed by this release.
