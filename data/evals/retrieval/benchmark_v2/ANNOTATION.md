# SEC retrieval benchmark v2: annotation contract

The 120 scored queries and six exclusions were authored from six preselected,
tracked annual filings, before any retrieval result was observed. The nine
primary strata are mutually exclusive; counts are in `dataset_manifest.json`.
Each company contributes 40 scored queries (20 per filing). The corpus includes
all 948 extracted chunks, not only the labeled material. Each mode receives the
same query, ticker/year/form filter and both text/table document types; stratum
labels never select a narrower retrieval filter.

## Relevance rules

- **2 — relevant:** directly supports the requested fact/explanation or supplies
  one independently necessary facet of a multi-evidence question. A table must
  contain the requested measure, not a similar tax-credit row. A narrative must
  contain the explanation, not merely mention the metric. Exact anchored
  duplicates in multiple chunks are all valid alternatives.
- **1 — partially relevant:** supports related context or only part of the
  requested lookup without the requested explanation/fact. These explicitly
  annotated judgments receive graded nDCG credit, but do not count as hits for
  binary Recall/MRR. For example, a numeric sales table is partial support for
  a question asking management's growth explanation.
- **0 — irrelevant:** an explicitly inspected similar-but-wrong passage, such
  as borrowing instead of cash-equivalent classification or facility area
  instead of segment sales. These are existing corpus chunks, not injected
  artificial documents. Wrong companies/filings do not become relevant merely
  through shared wording.
- **Unjudged:** all remaining chunks. They receive zero gain for metric
  computation, but must be counted separately from explicitly irrelevant hits.
  This is a known-gold benchmark, not exhaustive human judgment of every pair.

Every judgment includes evidence ID, content/source SHA-256, original extracted
section path, and exact content spans with quoted text. The source-first
annotation specification contains literal supporting anchors, which are matched
against *all* chunks of the specified filing without a ranking function. Each
anchor is independently verified against normalized source HTML text (NFKC,
whitespace removed), with a corresponding offset. This catches incompatible
source versions; it does not turn string matching into semantic adjudication.
An exact R&D table-cell rule excludes the similarly named tax-credit measure.

The annotator is the coding assistant, with source inspection; there is no
claim of independent human, blinded, dual-annotator agreement. Query/anchor
generation and span mapping are independent of the retriever being tested.
Additional human review would strengthen the labels but must produce a new
version, never silently amend this frozen dataset after seeing scores.

## Multiple valid chunks and multi-evidence questions

All known grade-2 chunks contribute to ordinary chunk Recall@K. Returning one
of several duplicate supporting chunks does not count as retrieving every
alternative: this denominator can penalize chunking overlap. Therefore also
report required-evidence-group coverage@10: each required group is satisfied by
any of its grade-2 alternatives. Multi-evidence cases require multiple groups;
their ordinary Recall/MRR/nDCG alone do not prove complete evidence coverage.
Do not collapse IDs or grades differently for different retrieval modes.

## Exclusions and missing labels

Six future-audited-result questions are explicitly unanswerable from their
source-year filings. They remain in the dataset and exclusion counts, but are
not retrieval ranking or latency denominators. A retriever has no abstention
contract here; no unanswerable-detection score is claimed.

An answerable case with no grade-2 judgment, a missing corpus ID, incompatible
source/content hash or invalid span is a **validation failure**, never an
exclusion chosen after seeing results. Duplicate query IDs, duplicate corpus
IDs and repeated judgment IDs are errors. Duplicate returned IDs are counted
and later occurrences occupy ranks with zero gain, rather than earning repeated
credit or silently improving the ranks of later hits. Retrieval errors remain
in overall denominators with zero quality scores; latency/error coverage is
reported explicitly.

## Historical 75-case seed

The original file and SHA remain unchanged. `seed_compatibility.json` records
original IDs/UIDs and all exact normalized-anchor mappings. Truncated anchors
are unresolved, not guessed using nearest neighbors, numbering or rankings.
Exact-anchor compatibility alone is not a semantic label rewrite. Only
`AAPL10K24_TXT_001` is adopted into the balanced benchmark, as
`KBV2_AAPL_2024_09`, with identical query semantics and a separately inspected
source span. Other cases remain immutable reference material and compatibility
audit entries, not silently scored or relabeled. PR20's text exclusion remains
historically correct for its different collection and unchanged.

## Claims and limitations

This benchmark supports paired comparisons of the four frozen KB retrieval
modes on this corpus. It is not an estimate of production traffic quality,
end-to-end answer correctness, SEC/XBRL correctness or broader-market coverage.
Three very large technology/commerce companies are not sector-diverse; only
annual filings are present. Topic templates repeated across years are
correlated: 120 queries are not 120 independent information needs. The nine
small strata and latency tails need cautious interpretation; no significance
or universal superiority claim follows from one run.

Existing chunker heading errors remain, including misleading Item 6/forward-
looking headings around other sections. Evidence spans and source filings,
not heading correctness, anchor judgments. Full raw tables are used without
generated summaries or row documents; this representation differs from PR20.
It is not valid to subtract these scores from historical PR20 scores as though
only the retrieval algorithm changed. No production code or defaults changed.
