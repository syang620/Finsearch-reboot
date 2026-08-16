from __future__ import annotations

import asyncio
import json
import time
import re
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict

from agents.contracts import (
    AnalysisTask,
    FilingMetadata,
    FormType,
    OpenIssue,
    PlannerIntent,
    PlannerRuntimeOutput,
    Severity,
)
from pydantic import BaseModel, Field
from llm_client import build_chat_model


_TICKER_STOPWORDS = {
    "SEC",
    "GAAP",
    "IFRS",
    "FY",
    "FQ",
    "USD",
    "EPS",
    "FCF",
    "EBITDA",
    "EBIT",
    "COGS",
    "YOY",
    "Q1",
    "Q2",
    "Q3",
    "Q4",
    "ITEM",
    "NOTE",
    "MDA",
    "CAPEX",
    "IPO",
    "AI",
    "ML",
    "API",
    "PDF",
    "AWS",
}

_TICKER_RE = re.compile(
    r"""
    (?:
        \$([A-Za-z]{1,5})
        |
        \b([A-Z]{1,5}(?:\.[A-Z])?)\b
    )
""",
    re.VERBOSE,
)

_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_FY_RE = re.compile(r"\bFY\s*(19\d{2}|20\d{2})\b", re.IGNORECASE)
_FY_SHORT_RE = re.compile(r"\bFY\s*'?(\d{2})\b", re.IGNORECASE)
_10K_RE = re.compile(r"\b10[-\s]?K\b", re.IGNORECASE)
_10Q_RE = re.compile(r"\b10[-\s]?Q\b", re.IGNORECASE)
_QUARTER_RE = re.compile(r"\b(Q[1-4])\b", re.IGNORECASE)
_QUARTER_WORD_RE = re.compile(r"\b(first|1st|second|2nd|third|3rd|fourth|4th)[-\s]?quarter\b", re.IGNORECASE)
_ANNUAL_REPORT_RE = re.compile(r"\bannual report\b", re.IGNORECASE)
_QUARTERLY_REPORT_RE = re.compile(r"\bquarterly report\b", re.IGNORECASE)
_MULTI_YEAR_RE = re.compile(
    r"\b(?:FY\s*)?(19\d{2}|20\d{2}|\d{2})\s*(?:/|-|to|or|and)\s*(?:FY\s*)?(\d{2}|19\d{2}|20\d{2})\b",
    re.IGNORECASE,
)
_MULTI_COMPANY_CUE_RE = re.compile(r"\b(compare|versus|vs\.?|and)\b", re.IGNORECASE)

_POSSESSIVE_NAME_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9&\.\-]*(?:\s+[A-Z][A-Za-z0-9&\.\-]*){0,4})\s*[']s\b"
)
_OF_FOR_NAME_RE = re.compile(
    r"\b(?:of|for)\s+([A-Z][A-Za-z0-9&\.\-]*(?:\s+[A-Z][A-Za-z0-9&\.\-]*){0,4})\b"
)
_LEADING_NAME_RE = re.compile(
    r"\b(?:how does|how did|what did|what does|using|from|for|per|in)\s+"
    r"([A-Z][A-Za-z0-9&\.\-]*(?:\s+(?:of|[A-Z][A-Za-z0-9&\.\-]*)){0,4})\b"
)


def _normalize_company_key(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"[^\w\s&\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\b(incorporated|inc|corp|corporation|co|company|ltd|limited|plc)\b", "", s).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s


_KNOWN_COMPANY_ALIASES: Dict[str, Optional[str]] = {
    "3M": "MMM",
    "AT&T": "T",
    "Alphabet": None,
    "Alphabet Class A": "GOOGL",
    "Alphabet Class C": "GOOG",
    "AmEx": "AXP",
    "Amazon": "AMZN",
    "American Express": "AXP",
    "Apple": "AAPL",
    "BNY Mellon": "BK",
    "Bank of America": "BAC",
    "Bank of New York Mellon": "BK",
    "Berkshire Hathaway (Class B)": "BRK.B",
    "Berkshire Hathaway B": "BRK.B",
    "Big Blue": "IBM",
    "BlackRock": "BLK",
    "Chevron": "CVX",
    "Cisco": "CSCO",
    "Coca-Cola": "KO",
    "Coke": "KO",
    "Comcast": "CMCSA",
    "Costco": "COST",
    "Disney": "DIS",
    "ExxonMobil": "XOM",
    "FedEx": "FDX",
    "GE Aerospace": "GE",
    "Google": None,
    "Home Depot": "HD",
    "IBM": "IBM",
    "Intuit": "INTU",
    "J&J": "JNJ",
    "JPMorgan": "JPM",
    "JPMorgan Chase": "JPM",
    "Lowe's": "LOW",
    "Mastercard": "MA",
    "McDonalds": "MCD",
    "Meta": "META",
    "Microsoft": "MSFT",
    "Mondelez": "MDLZ",
    "Netflix": "NFLX",
    "Nvidia": "NVDA",
    "PepsiCo": "PEP",
    "Qualcomm": "QCOM",
    "Salesforce": "CRM",
    "T-Mobile": "TMUS",
    "Texas Instruments": "TXN",
    "U.S. Bancorp": "USB",
    "US Bancorp": "USB",
    "UnitedHealth Group": "UNH",
    "Visa": "V",
    "Walmart": "WMT",
    "Walt Disney": "DIS",
}

_KNOWN_COMPANY_KEYS = {
    _normalize_company_key("Alphabet"),
    _normalize_company_key("Google"),
}

_DEFAULT_COMPANY_TICKER_MAP: Dict[str, str] = {
    **{
        "apple": "AAPL",
        "apple inc": "AAPL",
        "microsoft": "MSFT",
        "amazon": "AMZN",
        "meta": "META",
        "meta platforms": "META",
        "nvidia": "NVDA",
        "tesla": "TSLA",
        "netflix": "NFLX",
        "berkshire hathaway": "BRK.B",
        "walmart": "WMT",
        "jpmorgan chase": "JPM",
    },
    **{
        _normalize_company_key(name): ticker
        for name, ticker in _KNOWN_COMPANY_ALIASES.items()
        if ticker is not None and _normalize_company_key(name) not in _KNOWN_COMPANY_KEYS
    },
}


def _strip_company_prefix(candidate: Optional[str]) -> Optional[str]:
    if not candidate:
        return None
    out = str(candidate).strip()
    parts = out.split()
    while len(parts) > 1 and parts[0].lower() in {"in", "using", "from", "for", "per"}:
        parts = parts[1:]
    cleaned = " ".join(parts).strip()
    return cleaned or None


def _looks_like_ticker_token(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.fullmatch(r"[A-Z]{1,5}(?:\.[A-Z])?", str(text).strip()))


def _looks_like_non_company_candidate(text: Optional[str]) -> bool:
    if not text:
        return False
    candidate = str(text).strip()
    if _looks_like_ticker_token(candidate):
        return True
    if re.fullmatch(r"FY\d{2,4}", candidate, re.IGNORECASE):
        return True
    if re.fullmatch(r"Q[1-4]", candidate, re.IGNORECASE):
        return True
    if _10K_RE.fullmatch(candidate) or _10Q_RE.fullmatch(candidate):
        return True
    return False


def _find_company_alias_spans(query: str) -> List[Tuple[int, int]]:
    q = (query or "").replace("’", "'")
    spans: List[Tuple[int, int]] = []
    for alias in sorted(_KNOWN_COMPANY_ALIASES, key=len, reverse=True):
        for match in re.finditer(re.escape(alias), q, flags=re.IGNORECASE):
            spans.append((match.start(), match.end()))
    spans.sort()
    return spans


def _find_company_mentions(query: str) -> List[str]:
    normalized_query = _normalize_company_key((query or "").replace("’", "'"))
    if not normalized_query:
        return []

    matches: List[Tuple[int, int, str]] = []
    for alias in _KNOWN_COMPANY_ALIASES:
        alias_key = _normalize_company_key(alias)
        if not alias_key:
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(alias_key)}(?!\w)")
        for match in pattern.finditer(normalized_query):
            matches.append((match.start(), match.end(), alias))

    matches.sort(key=lambda item: (-(item[1] - item[0]), item[0], item[2]))
    kept: List[Tuple[int, int, str]] = []
    for start, end, alias in matches:
        overlaps = any(not (end <= kept_start or start >= kept_end) for kept_start, kept_end, _ in kept)
        if overlaps:
            continue
        kept.append((start, end, alias))

    kept.sort(key=lambda item: item[0])
    mentions: List[str] = []
    seen = set()
    for _, _, alias in kept:
        key = _normalize_company_key(alias)
        if key in seen:
            continue
        seen.add(key)
        mentions.append(alias)
    return mentions


def _extract_company_name(query: str) -> Optional[str]:
    q = (query or "").replace("’", "'").strip()
    if not q:
        return None
    mentions = _find_company_mentions(q)
    if len(mentions) == 1:
        return mentions[0]
    if len(mentions) > 1:
        return None
    m = _POSSESSIVE_NAME_RE.search(q)
    if m:
        candidate = _strip_company_prefix(m.group(1))
        return None if _looks_like_non_company_candidate(candidate) else candidate
    m2 = _OF_FOR_NAME_RE.search(q)
    if m2:
        candidate = _strip_company_prefix(m2.group(1))
        return None if _looks_like_non_company_candidate(candidate) else candidate
    m3 = _LEADING_NAME_RE.search(q)
    if m3:
        candidate = _strip_company_prefix(m3.group(1))
        return None if _looks_like_non_company_candidate(candidate) else candidate
    return None


def _resolve_ticker_from_company_name(company_name: Optional[str], mapping: Dict[str, str]) -> Optional[str]:
    if not company_name:
        return None
    key = _normalize_company_key(company_name)
    if not key:
        return None
    if key in _KNOWN_COMPANY_KEYS:
        return None
    if key in mapping:
        return mapping[key]
    for k, v in mapping.items():
        if key == k or key.startswith(k) or k.startswith(key):
            return v
    return None


def _extract_ticker_candidates(query: str) -> List[str]:
    candidates: List[str] = []
    alias_spans = _find_company_alias_spans(query)
    for m in _TICKER_RE.finditer(query):
        tok = (m.group(1) or m.group(2) or "").strip()
        if not tok:
            continue
        span = (m.start(), m.end())
        if any(
            not (span[1] <= alias_start or span[0] >= alias_end)
            for alias_start, alias_end in alias_spans
        ):
            continue
        if m.group(2):
            group_start = m.start(2)
            group_end = m.end(2)
            window = query[max(0, group_start - 3) : min(len(query), group_end + 2)]
            if re.search(r"10[-\s]?[KQ]s?", window, re.IGNORECASE):
                continue
            if tok.upper() in {"A", "Q"}:
                prefix = query[max(0, group_start - 2) : group_start]
                suffix = query[group_end : min(len(query), group_end + 2)]
                if prefix in {"\n", "\r", "\r\n"} and suffix.startswith(":"):
                    continue
        t = tok.upper()
        if t in _TICKER_STOPWORDS:
            continue
        if _YEAR_RE.fullmatch(t):
            continue
        if t not in candidates:
            candidates.append(t)
    return candidates


def _pick_fiscal_quarter(query: str) -> Optional[str]:
    m = _QUARTER_RE.search(query)
    if m:
        return m.group(1).upper()
    m_word = _QUARTER_WORD_RE.search(query)
    if not m_word:
        return None
    token = m_word.group(1).lower()
    mapping = {
        "first": "Q1",
        "1st": "Q1",
        "second": "Q2",
        "2nd": "Q2",
        "third": "Q3",
        "3rd": "Q3",
        "fourth": "Q4",
        "4th": "Q4",
    }
    return mapping.get(token)


def _expand_short_year(year_text: str) -> int:
    year = int(year_text)
    if year >= 100:
        return year
    return 2000 + year


def _has_ambiguous_fiscal_year(query: str) -> bool:
    q = (query or "").replace("–", "-").replace("—", "-")
    if _MULTI_YEAR_RE.search(q):
        return True
    explicit_years = {int(y) for y in _YEAR_RE.findall(q)}
    if len(explicit_years) > 1:
        return True
    short_fy_years = {_expand_short_year(y) for y in _FY_SHORT_RE.findall(q)}
    if len(short_fy_years) > 1:
        return True
    return False


def _pick_fiscal_year(query: str) -> Optional[int]:
    if _has_ambiguous_fiscal_year(query):
        return None
    m = _FY_RE.search(query)
    if m:
        return int(m.group(1))
    m_short = _FY_SHORT_RE.search(query)
    if m_short:
        return _expand_short_year(m_short.group(1))
    years = [int(y) for y in _YEAR_RE.findall(query)]
    return years[0] if years else None


def _pick_form_type(query: str) -> Optional[FormType]:
    if _10K_RE.search(query) or _ANNUAL_REPORT_RE.search(query):
        return FormType.TEN_K
    if _10Q_RE.search(query) or _QUARTERLY_REPORT_RE.search(query):
        return FormType.TEN_Q
    quarter = _pick_fiscal_quarter(query)
    if quarter and quarter in {"Q1", "Q2", "Q3"}:
        return FormType.TEN_Q
    if re.search(r"\byear[-\s]?end\b|\byear[-\s]?ended\b|\bat year[-\s]?end\b", query, re.IGNORECASE):
        return FormType.TEN_K
    return None


def _guess_metric(query: str) -> str:
    q = (query or "").lower()
    if "net debt" in q:
        return "net debt"
    if "total debt" in q or ("debt" in q and "total" in q):
        return "total debt"
    if "debt" in q or "borrow" in q or "credit facility" in q or "notes payable" in q:
        return "debt"
    if "free cash flow" in q or "fcf" in q:
        return "free cash flow"
    if "capex" in q or "capital expenditure" in q:
        return "capital expenditures"
    if "operating cash" in q or "cash flow from operations" in q:
        return "net cash provided by operating activities"
    if "revenue" in q or "net sales" in q:
        return "revenue"
    if "eps" in q:
        return "earnings per share"
    return "filing facts"


def _intent_hint_from_query(user_query: str, metric_hint: str) -> Tuple[PlannerIntent, str, bool, List[str]]:
    q = (user_query or "").lower()
    metric = (metric_hint or "").lower()

    if re.search(r"\bwhat is\b|\bdefine\b|\bmeaning\b|\bexplain\b", q):
        return PlannerIntent.DEFINITION, "extract", False, ["definition_pattern"]

    calc_cues: List[str] = []
    if re.search(r"\bplus\b|\bsum\b|\badd(ed)?\b|\bcombined\b|\baggregate(d)?\b", q):
        calc_cues.append("additive_word")
    if re.search(r"\b(short[-\s]?term).{0,40}(long[-\s]?term)\b|\b(long[-\s]?term).{0,40}(short[-\s]?term)\b", q):
        calc_cues.append("short_long_term_combo")
    if re.search(r"\bminus\b|\bless\b|\bsubtract\b|\bexcluding\b|\bnet of\b", q):
        calc_cues.append("subtractive_word")
    if re.search(
        r"\b(net debt|ratio|margin|growth|change|delta|vs\.?|versus|compare|difference|yoy|qoq|cagr)\b",
        q,
    ):
        calc_cues.append("comparison_or_derived_metric")
    if metric in {"net debt"}:
        calc_cues.append("metric_requires_compute")

    if calc_cues:
        return PlannerIntent.FILING_CALC, "compute", True, calc_cues
    return PlannerIntent.FILING_FACT, "extract", True, []


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    return None
    return None


def _extract_metadata_hints_and_issues(
    query: str,
    *,
    company_ticker_map: Dict[str, str],
    default_doc_types: Optional[List[str]] = None,
) -> Tuple[FilingMetadata, List[OpenIssue]]:
    q = (query or "").replace("’", "'").strip()
    explicit_tickers = _extract_ticker_candidates(q)
    company_mentions = _find_company_mentions(q)
    company_name = company_mentions[0] if len(company_mentions) == 1 else None
    if company_name is None and not company_mentions:
        company_name = _extract_company_name(q)

    fy = _pick_fiscal_year(q)
    form = _pick_form_type(q)
    fq = _pick_fiscal_quarter(q)

    has_multi_company_cue = bool(_MULTI_COMPANY_CUE_RE.search(q))
    multi_company = (
        len(company_mentions) > 1
        or len(set(explicit_tickers)) > 1
        or (has_multi_company_cue and (len(company_mentions) + len(explicit_tickers) > 1))
    )

    normalized_company = _normalize_company_key(company_name or "")
    resolved_company_ticker = _resolve_ticker_from_company_name(company_name, company_ticker_map)
    company_ticker_conflict = (
        not multi_company
        and company_name is not None
        and resolved_company_ticker is not None
        and any(ticker != resolved_company_ticker for ticker in explicit_tickers)
    )

    final_company_name = company_name
    ticker: Optional[str] = None
    if multi_company or company_ticker_conflict:
        final_company_name = None
    else:
        if explicit_tickers:
            ticker = explicit_tickers[0]
        elif resolved_company_ticker:
            ticker = resolved_company_ticker

    share_class_ambiguous = (
        final_company_name is not None
        and normalized_company in _KNOWN_COMPANY_KEYS
        and ticker is None
    )

    issues: List[OpenIssue] = []
    if multi_company:
        issues.append(
            OpenIssue(
                code="MULTI_COMPANY_QUERY",
                message="Multiple company entities detected; current crawl mode expects one primary company.",
                severity=Severity.WARNING,
            )
        )
    elif company_ticker_conflict:
        issues.append(
            OpenIssue(
                code="COMPANY_TICKER_CONFLICT",
                message="Company name and explicit ticker disagree, so metadata extraction abstained.",
                severity=Severity.WARNING,
            )
        )
    elif share_class_ambiguous:
        issues.append(
            OpenIssue(
                code="SHARE_CLASS_AMBIGUOUS",
                message="Detected Alphabet/Google without a disambiguating share class or ticker.",
                severity=Severity.WARNING,
            )
        )

    if ticker is None:
        issues.append(
            OpenIssue(
                code="TICKER_MISSING",
                message="No unambiguous ticker could be determined from the query.",
                severity=Severity.WARNING,
            )
        )

    if fy is None:
        if _has_ambiguous_fiscal_year(q):
            issues.append(
                OpenIssue(
                    code="MULTI_YEAR_QUERY",
                    message="Multiple fiscal years were referenced in the query.",
                    severity=Severity.WARNING,
                )
            )
            issues.append(
                OpenIssue(
                    code="FISCAL_YEAR_AMBIGUOUS",
                    message="Fiscal year extraction abstained because the query references multiple years.",
                    severity=Severity.WARNING,
                )
            )
        else:
            issues.append(
                OpenIssue(
                    code="FISCAL_YEAR_MISSING",
                    message="No fiscal year detected in the query.",
                    severity=Severity.WARNING,
                )
            )

    if form == FormType.TEN_Q:
        issues.append(
            OpenIssue(
                code="FORM_NOT_10K_DATASET",
                message="Detected a 10-Q/quarterly-report query while the current hard-mode eval set is 10-K scoped.",
                severity=Severity.WARNING,
            )
        )

    hints = FilingMetadata(
        ticker=ticker,
        company_name=final_company_name,
        fiscal_year=fy,
        form_type=form,
        doc_types=list(default_doc_types) if default_doc_types is not None else None,
        fiscal_quarter=fq,
    )
    return hints, issues


DEFAULT_TARGET_RESOLUTION_PROMPT_TEMPLATE = """
You are the Target Resolver Planner for an SEC-filings Agentic RAG system.

Very important system design:
Before you are called, a deterministic resolver has already attempted to extract:
- ticker
- fiscal_year
- form_type

Those deterministic results are provided to you as input.

Your role is:
- trust deterministic values when they are resolved
- never overwrite a resolved deterministic ticker, fiscal_year, or form_type
- only use language reasoning to fill gaps when a blocking field could not be resolved deterministically
- if the meaning is still too ambiguous to resolve correctly, ask for clarification through the JSON fields below
- produce a compact retrieval handoff for the retriever

Because this is a programmatic setting:
- NEVER ask the user questions outside the JSON
- If clarification is needed, set `needs_clarification=true` and populate `clarification_questions`
- Always still return one valid JSON object



Rules:
1. Never invent a company, ticker, fiscal year, or form type.
2. If deterministic extraction resolved ticker, fiscal_year, or form_type, copy those values exactly.
3. Do not "improve", reinterpret, or replace resolved deterministic values.
4. Only infer ticker when deterministic_ticker is null and the query strongly implies a single canonical company.
5. Only infer fiscal_year when deterministic_fiscal_year is null and the query explicitly or strongly implies one year.
6. If the user references a supported group alias, expand it to the underlying canonical entities.
7. If form_type is unresolved and not clearly implied, return form_type=null rather than guessing.
8. When multiple fiscal years apply to the same resolved company set, emit one target per (company, fiscal_year) pair.
9. If clarification is needed, DO NOT return anything to the retrieval plan.
10. `goal` must be a short, atomic, retrieval-only evidence request.
11. The `goal` field is a retrieval-level instruction dispatched to individual filings.
    It must NOT reference specific company names, tickers, or fiscal years.
    It should describe WHAT to extract from a single filing (e.g., "extract annual revenue").
    Comparison, ranking, and aggregation happen downstream — not in the goal.
12. When all targets require the same extraction, emit exactly ONE job whose
    `applies_to_target_ids` lists every target_id. Do NOT create separate jobs
    per company or per company-year pair unless the extraction goal differs.
13. `goal` should be generic and reusable without company name or fiscal year.
14. Do not mention company name(s) or year-to-year comparison language inside `goal`.
15. If `needs_clarification=true`, return `targets=[]`.
16. Do not emit partial, provisional, or guessed targets before clarification is resolved.
17. `target_id` must be an integer. Use sequential integers starting at 1.
18. `target_key` must be a stable readable string key:
    - if ticker and fiscal_year are known: "{TICKER}_FY{YEAR}"
    - if ticker is known but fiscal_year is unknown: "{TICKER}_UNKNOWN_YEAR"
    - if ticker is unknown: "TARGET_{N}"

How to use the deterministic input:
- deterministic_targets: authoritative target candidates from rules/regex/maps
- deterministic_ticker: authoritative if non-null
- deterministic_fiscal_year: authoritative if non-null
- deterministic_form_type: authoritative if non-null
- unresolved_blockers: tells you which blocking fields remain unresolved after deterministic extraction
- clarification_history contains prior clarification questions and the user's answers; treat the answers as authoritative user input
- If unresolved_blockers is empty, do not perform gap filling for ticker or fiscal_year.
- If clarification_history resolves a blocker, do not ask the same question again.
- If unresolved_blockers contains "ticker" or "fiscal_year", you may use the user query plus alias maps to try to fill only those missing fields.
- Choose `route="kb"` for narrative, descriptive, explanatory, qualitative, or filing-evidence questions.
- Choose `route="structured_fact"` for direct supported numeric metric questions that can be answered by a structured SEC fact.
- Choose `route="hybrid"` when the user wants both a direct numeric metric answer and filing-based explanation or context.
- Supported structured-fact metrics are direct reported scalar facts, such as revenue, gross profit, operating income, net income, cash and cash equivalents, total assets, total liabilities, stockholders equity, operating cash flow, capex, and total debt.
- Do not emit final SEC `metric_id`; that mapping happens downstream.
- When `route` is `structured_fact` or `hybrid`, emit one or more `structured_fact_requests` using only:
  - `subquestion`
  - `metric_hint`
  - `entity_hint`
  - `fiscal_year`
  - `fiscal_period`
- Keep `metric_hint` human-readable, such as "revenue" or "cash and cash equivalents". Do not emit snake_case or registry-style IDs such as "cash_and_cash_equivalents", "total_debt", or "stockholders_equity".
- Keep routing conservative. If the question is unsupported, comparative, ratio-based, margin-based, per-share, or otherwise likely to need filing interpretation, prefer `kb` over `structured_fact`.
- Do NOT route derived financial ratios or calculated metrics to `structured_fact`.
- Examples that should remain `kb`: return on equity (ROE), return on assets (ROA), debt-to-equity ratio, gross margin, operating margin, EBITDA margin, free cash flow yield, EV/EBITDA, EPS, diluted EPS, and balance-sheet summary questions.
- Do not decompose unsupported ratios, margins, per-share metrics, or calculated metrics into multiple structured fact requests. Keep those questions on `kb`.
- Do not route multi-company comparison questions such as "Compare Apple and Microsoft revenue in FY2024" to `structured_fact`. Keep them on `kb`.
- Use `hybrid` only when the user clearly asks for both a supported scalar fact and narrative explanation or filing context.

Deterministic extraction results:
{{PLANNER_PAYLOAD_JSON}}

Return exactly one JSON object matching this schema:
{
  "retrieval_needed": bool,
  "route": "kb" | "structured_fact" | "hybrid",
  "structured_fact_requests": [
    {
      "subquestion": string,
      "metric_hint": string | null,
      "entity_hint": string | null,
      "fiscal_year": integer | null,
      "fiscal_period": string | null
    }
  ],
  "task_class": "single_target_fact | multi_target_compare | multi_target_screen | other",
  "targets": [
    {
      "target_id": integer,
      "target_key": string | null,
      "company_name": string | null,
      "ticker": string | null,
      "fiscal_year": integer | null,
      "form_type": "10-K" | "10-Q" | null
    }
  ],
  "retrieval_plan": {
    "fanout_mode": "single_target" | "per_target",
    "jobs": [
      {
        "applies_to_target_ids": [integer],
        "goal": string,
        "job_type": "metric_extract" | "narrative_extract"
      }
    ]
  } | null,
  "needs_clarification": bool,
  "clarification_reason": string | null,
  "clarification_questions": [string],
  "open_issues": [
    {
      "code": string,
      "message": string,
      "severity": "info" | "warning" | "error"
    }
  ]
}
Do not use markdown, code fences, or commentary.
"""


_TRUE_STRINGS = {"1", "true", "yes", "y", "on"}
_ALLOWED_FORM_TYPES = {"10-K", "10-Q"}
_ALLOWED_TASK_CLASSES = {
    "single_target_fact",
    "multi_target_compare",
    "multi_target_screen",
    "other",
}
_ALLOWED_ROUTES = {"kb", "structured_fact", "hybrid"}
_ALLOWED_SEVERITIES = {"info", "warning", "error"}
_MULTI_TARGET_TASK_CLASSES = {"multi_target_compare", "multi_target_screen"}
_ALLOWED_JOB_TYPES = {
    "metric_extract",
    "narrative_extract",
}
_UNSUPPORTED_STRUCTURED_FACT_HINT_PATTERNS = (
    "gross margin",
    "operating margin",
    "ebitda margin",
    "return on equity",
    "roe",
    "return on assets",
    "roa",
    "debt-to-equity",
    "debt to equity",
    "earnings per share",
    "eps",
)

class _StructuredTargetResolutionTarget(BaseModel):
    target_id: int = Field(default=1)
    target_key: Optional[str] = None
    company_name: Optional[str] = None
    ticker: Optional[str] = None
    fiscal_year: Optional[int] = None
    form_type: Optional[Literal["10-K", "10-Q"]] = None


class _StructuredTargetResolutionJob(BaseModel):
    applies_to_target_ids: List[int] = Field(default_factory=list)
    goal: str
    job_type: Literal["metric_extract", "narrative_extract"]


class _StructuredTargetResolutionPlan(BaseModel):
    fanout_mode: Literal["single_target", "per_target"] = "single_target"
    jobs: List[_StructuredTargetResolutionJob] = Field(default_factory=list)


class _StructuredFactRequest(BaseModel):
    subquestion: str
    metric_hint: Optional[str] = None
    entity_hint: Optional[str] = None
    fiscal_year: Optional[int] = None
    fiscal_period: Optional[str] = None


class _StructuredPlannerIssue(BaseModel):
    code: str
    message: str
    severity: Literal["info", "warning", "error"] = "warning"


class _StructuredTargetResolutionOutput(BaseModel):
    retrieval_needed: bool = True
    route: Literal["kb", "structured_fact", "hybrid"] = "kb"
    structured_fact_requests: List[_StructuredFactRequest] = Field(default_factory=list)
    task_class: Literal[
        "single_target_fact",
        "multi_target_compare",
        "multi_target_screen",
        "other",
    ] = "other"
    targets: List[_StructuredTargetResolutionTarget] = Field(default_factory=list)
    retrieval_plan: Optional[_StructuredTargetResolutionPlan] = None
    needs_clarification: bool = False
    clarification_reason: Optional[str] = None
    clarification_questions: List[str] = Field(default_factory=list)
    open_issues: List[_StructuredPlannerIssue] = Field(default_factory=list)


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_metric_hint_text(value: Any) -> Optional[str]:
    text = _normalize_text(value)
    if text is None:
        return None
    return " ".join(text.replace("_", " ").split())


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_STRINGS
    return bool(value)


def _normalize_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _normalize_form_type(value: Any) -> Optional[str]:
    text = _normalize_text(value)
    if text is None:
        return None
    upper = text.upper()
    return upper if upper in _ALLOWED_FORM_TYPES else None


def _build_target_key(
    *,
    ticker: Optional[str],
    fiscal_year: Optional[int],
    index: int,
) -> str:
    if ticker and fiscal_year is not None:
        return f"{ticker}_FY{fiscal_year}"
    if ticker:
        return f"{ticker}_UNKNOWN_YEAR"
    return f"TARGET_{index}"


def _dedupe_ints(values: Sequence[Any], *, limit: Optional[int] = None) -> List[int]:
    out: List[int] = []
    seen = set()
    for value in values:
        number = _normalize_int(value)
        if number is None:
            continue
        if number in seen:
            continue
        seen.add(number)
        out.append(number)
        if limit is not None and len(out) >= limit:
            break
    return out


def _normalize_open_issue(issue: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(issue, dict):
        return None
    code = _normalize_text(issue.get("code")) or "UNSPECIFIED"
    message = _normalize_text(issue.get("message")) or ""
    severity = (_normalize_text(issue.get("severity")) or "warning").lower()
    if severity not in _ALLOWED_SEVERITIES:
        severity = "warning"
    return {
        "code": code,
        "message": message,
        "severity": severity,
    }


def _dedupe_strings(values: Sequence[Any], *, limit: Optional[int] = None) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = _normalize_text(value)
        if text is None:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if limit is not None and len(out) >= limit:
            break
    return out


def _normalize_target(target: Any, *, index: int) -> Optional[Dict[str, Any]]:
    if not isinstance(target, dict):
        return None

    ticker = _normalize_text(target.get("ticker"))
    company_name = _normalize_text(target.get("company_name"))
    fiscal_year = _normalize_int(target.get("fiscal_year"))
    form_type = _normalize_form_type(target.get("form_type"))
    target_id = _normalize_int(target.get("target_id"))
    if target_id is None:
        target_id = index
    target_key = _normalize_text(target.get("target_key")) or _build_target_key(
        ticker=ticker,
        fiscal_year=fiscal_year,
        index=index,
    )

    return {
        "target_id": target_id,
        "target_key": target_key,
        "company_name": company_name,
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "form_type": form_type,
    }


def _normalize_retrieval_job(
    job: Any,
    *,
    target_ids: Sequence[int],
) -> Optional[Dict[str, Any]]:
    if not isinstance(job, dict):
        return None

    goal = _normalize_text(job.get("goal"))
    if goal is None:
        return None

    applies_to_target_ids = _dedupe_ints(job.get("applies_to_target_ids") or [])
    valid_target_ids = set(target_ids)
    applies_to_target_ids = [target_id for target_id in applies_to_target_ids if target_id in valid_target_ids]
    if not applies_to_target_ids:
        applies_to_target_ids = list(target_ids)

    job_type = (_normalize_text(job.get("job_type")) or "metric_extract").strip().lower()
    if job_type == "fact_lookup":
        job_type = "metric_extract"
    elif job_type == "component_extract":
        job_type = "narrative_extract"
    if job_type not in _ALLOWED_JOB_TYPES:
        job_type = "metric_extract"

    return {
        "applies_to_target_ids": applies_to_target_ids,
        "goal": goal,
        "job_type": job_type,
    }


def _normalize_retrieval_plan(
    retrieval_plan: Any,
    *,
    targets: Sequence[Dict[str, Any]],
    needs_clarification: bool,
) -> Optional[Dict[str, Any]]:
    if needs_clarification or not isinstance(retrieval_plan, dict):
        return None

    target_ids = [int(target["target_id"]) for target in targets if _normalize_int(target.get("target_id")) is not None]
    if not target_ids:
        return None

    fanout_mode = (_normalize_text(retrieval_plan.get("fanout_mode")) or "").strip().lower()
    if fanout_mode not in {"single_target", "per_target"}:
        fanout_mode = "single_target" if len(target_ids) == 1 else "per_target"

    jobs = [
        normalized
        for normalized in (
            _normalize_retrieval_job(job, target_ids=target_ids)
            for job in (retrieval_plan.get("jobs") or [])
        )
        if normalized is not None
    ]
    if not jobs:
        return None

    return {
        "fanout_mode": fanout_mode,
        "jobs": jobs,
    }


def _normalize_structured_fact_request(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    subquestion = _normalize_text(value.get("subquestion"))
    if subquestion is None:
        return None
    return {
        "subquestion": subquestion,
        "metric_hint": _normalize_metric_hint_text(value.get("metric_hint")),
        "entity_hint": _normalize_text(value.get("entity_hint")),
        "fiscal_year": _normalize_int(value.get("fiscal_year")),
        "fiscal_period": _normalize_text(value.get("fiscal_period")),
    }


def _normalize_structured_fact_requests(values: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for value in values or []:
        item = _normalize_structured_fact_request(value)
        if item is not None:
            normalized.append(item)
    return normalized


def _should_force_kb_route(
    *,
    route: str,
    structured_fact_requests: Sequence[Dict[str, Any]],
    open_issues: Sequence[Dict[str, Any]],
) -> bool:
    if route not in {"structured_fact", "hybrid"}:
        return False

    issue_codes = {
        (_normalize_text(issue.get("code")) or "")
        for issue in open_issues
        if isinstance(issue, dict)
    }
    if "MULTI_COMPANY_QUERY" in issue_codes:
        return True

    for request in structured_fact_requests:
        if not isinstance(request, dict):
            continue
        combined_text = " ".join(
            part
            for part in (
                _normalize_metric_hint_text(request.get("metric_hint")),
                _normalize_text(request.get("subquestion")),
            )
            if part
        ).lower()
        if any(pattern in combined_text for pattern in _UNSUPPORTED_STRUCTURED_FACT_HINT_PATTERNS):
            return True
    return False


def _normalize_clarification_turns(
    clarification_turns: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for turn in clarification_turns or []:
        if not isinstance(turn, dict):
            continue
        question = _normalize_text(turn.get("question"))
        answer = _normalize_text(turn.get("answer"))
        if not question:
            continue
        out.append(
            {
                "question": question,
                "answer": answer or "",
            }
        )
    return out


def _format_clarification_context(clarification_turns: List[Dict[str, str]]) -> str:
    if not clarification_turns:
        return ""
    lines = ["Clarification answers:"]
    for turn in clarification_turns:
        answer = _normalize_text(turn.get("answer"))
        if answer:
            lines.append(f"Answer: {answer}")
    return "\n".join(lines)


def _build_deterministic_targets(
    hints: Any,
    *,
    candidate_tickers: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    ticker = _normalize_text(getattr(hints, "ticker", None))
    company_name = _normalize_text(getattr(hints, "company_name", None))
    fiscal_year = _normalize_int(getattr(hints, "fiscal_year", None))
    form_type = _normalize_form_type(getattr(hints, "form_type", None))

    candidate_tickers = _dedupe_strings(candidate_tickers or [])
    if ticker and ticker not in candidate_tickers:
        candidate_tickers.append(ticker)
    if not candidate_tickers or fiscal_year is None:
        return []

    resolved_targets = []
    for index, target_ticker in enumerate(candidate_tickers, start=1):
        resolved_targets.append(
            {
                "target_id": index,
                "target_key": _build_target_key(
                    ticker=target_ticker,
                    fiscal_year=fiscal_year,
                    index=index,
                ),
                "company_name": company_name,
                "ticker": target_ticker,
                "fiscal_year": fiscal_year,
                "form_type": form_type,
            }
        )
    return resolved_targets


def _build_planner_state(
    *,
    user_query: str,
    effective_user_query: str,
    clarification_history: List[Dict[str, str]],
    deterministic_targets: List[Dict[str, Any]],
    deterministic_ticker: Optional[str],
    deterministic_fiscal_year: Optional[int],
    deterministic_form_type: Optional[str],
    unresolved_blockers: List[str],
    deterministic_tickers: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    return {
        "original_user_query": user_query,
        "clarification_history": clarification_history,
        "deterministic_tickers": list(_dedupe_strings(deterministic_tickers or [])),
        "deterministic_targets": deterministic_targets,
        "deterministic_ticker": deterministic_ticker,
        "deterministic_fiscal_year": deterministic_fiscal_year,
        "deterministic_form_type": deterministic_form_type,
        "unresolved_blockers": unresolved_blockers,
        "effective_user_query": effective_user_query,
    }


def build_target_resolution_payload(
    *,
    planner: Any,
    user_query: str,
    clarification_turns: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    clarification_history = _normalize_clarification_turns(clarification_turns)
    clarification_context = _format_clarification_context(clarification_history)

    effective_user_query = user_query.strip()
    if clarification_context:
        effective_user_query = f"{effective_user_query}\n\n{clarification_context}"

    hints, issues = _extract_metadata_hints_and_issues(
        effective_user_query,
        company_ticker_map=planner.company_ticker_map,
        default_doc_types=getattr(planner, "default_doc_types", None),
    )
    metric_guess = _guess_metric(effective_user_query)
    intent_hint, task_type_hint, retrieval_needed_hint, calc_cues = _intent_hint_from_query(
        effective_user_query,
        metric_guess,
    )

    candidate_tickers = _dedupe_strings(_extract_ticker_candidates(effective_user_query))
    unresolved_blockers: List[str] = []
    if not candidate_tickers and getattr(hints, "ticker", None) is None:
        unresolved_blockers.append("ticker")
    if getattr(hints, "fiscal_year", None) is None:
        unresolved_blockers.append("fiscal_year")
    if not candidate_tickers and getattr(hints, "ticker", None) is not None:
        candidate_tickers = [_normalize_text(hints.ticker)]

    deterministic_targets = _build_deterministic_targets(
        hints,
        candidate_tickers=candidate_tickers,
    )
    deterministic_ticker = getattr(hints, "ticker", None)
    deterministic_fiscal_year = getattr(hints, "fiscal_year", None)
    deterministic_form_type = _normalize_form_type(getattr(hints, "form_type", None))
    hints_payload = hints.model_dump(mode="json")
    hints_payload["candidate_tickers"] = list(candidate_tickers)

    payload = {
        "user_query": user_query,
        "effective_user_query": effective_user_query,
        "clarification_history": clarification_history,
        "deterministic_targets": deterministic_targets,
        "deterministic_ticker": deterministic_ticker,
        "deterministic_fiscal_year": deterministic_fiscal_year,
        "deterministic_form_type": deterministic_form_type,
        "deterministic_hints": hints_payload,
        "deterministic_open_issues": [issue.model_dump(mode="json") for issue in issues],
        "unresolved_blockers": unresolved_blockers,
        "metric_guess": metric_guess,
        "deterministic_intent_hint": intent_hint.value,
        "deterministic_task_type_hint": task_type_hint,
        "deterministic_retrieval_needed_hint": retrieval_needed_hint,
        "deterministic_calc_cues": calc_cues,
    }

    planner_state = _build_planner_state(
        user_query=user_query,
        effective_user_query=effective_user_query,
        clarification_history=clarification_history,
        deterministic_targets=deterministic_targets,
        deterministic_ticker=deterministic_ticker,
        deterministic_fiscal_year=deterministic_fiscal_year,
        deterministic_form_type=deterministic_form_type,
        unresolved_blockers=unresolved_blockers,
        deterministic_tickers=candidate_tickers,
    )

    return {
        "hints": hints,
        "issues": issues,
        "metric_guess": metric_guess,
        "payload": payload,
        "planner_state": planner_state,
        "effective_user_query": effective_user_query,
        "clarification_history": clarification_history,
    }


def render_target_resolution_prompt(
    prompt_template: str,
    *,
    user_query: str,
    payload: Dict[str, Any],
) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False)
    prompt = str(prompt_template or "")
    prompt = prompt.replace("{{USER_QUERY}}", user_query)
    if "{{PLANNER_PAYLOAD_JSON}}" in prompt:
        prompt = prompt.replace("{{PLANNER_PAYLOAD_JSON}}", payload_json)
    else:
        prompt = prompt + "\n\n" + payload_json
    return prompt


def _coerce_structured_resolution_output(raw_output: Any) -> Optional[Dict[str, Any]]:
    if raw_output is None:
        return None
    if isinstance(raw_output, dict):
        return dict(raw_output)
    if isinstance(raw_output, str):
        return _extract_first_json_object(raw_output)
    for attr in ("model_dump", "dict"):
        if hasattr(raw_output, attr):
            maybe_dict = getattr(raw_output, attr)()
            if isinstance(maybe_dict, dict):
                return dict(maybe_dict)
    return None


def _is_blank_or_none_output(raw_output: Any) -> bool:
    if raw_output is None:
        return True
    if isinstance(raw_output, str):
        normalized = raw_output.strip().lower()
        return not normalized or normalized == "none"
    return False


async def _ainvoke_llm(model: Any, prompt: str) -> Any:
    invoke_async = getattr(model, "ainvoke", None)
    if callable(invoke_async):
        return await invoke_async(prompt)
    invoke_sync = getattr(model, "invoke", None)
    if not callable(invoke_sync):
        raise AttributeError("LLM does not expose invoke() or ainvoke().")
    return await asyncio.to_thread(invoke_sync, prompt)


def _normalize_resolution_output(parsed_output: Any) -> Dict[str, Any]:
    if not isinstance(parsed_output, dict):
        raise ValueError("Parsed output must be a JSON object.")

    route = (_normalize_text(parsed_output.get("route")) or "kb").lower()
    if route not in _ALLOWED_ROUTES:
        route = "kb"

    task_class = _normalize_text(parsed_output.get("task_class")) or "other"
    if task_class not in _ALLOWED_TASK_CLASSES:
        task_class = "other"

    structured_fact_requests = _normalize_structured_fact_requests(parsed_output.get("structured_fact_requests"))

    targets: List[Dict[str, Any]] = []
    for index, target in enumerate(parsed_output.get("targets") or [], start=1):
        normalized = _normalize_target(target, index=index)
        if normalized is not None:
            targets.append(normalized)

    clarification_questions = [
        question
        for question in (
            _normalize_text(question) for question in (parsed_output.get("clarification_questions") or [])
        )
        if question
    ]

    needs_clarification = _normalize_bool(parsed_output.get("needs_clarification"))
    if not needs_clarification:
        clarification_questions = []
    else:
        targets = []

    retrieval_plan = _normalize_retrieval_plan(
        parsed_output.get("retrieval_plan"),
        targets=targets,
        needs_clarification=needs_clarification,
    )

    open_issues = [
        issue
        for issue in (_normalize_open_issue(issue) for issue in (parsed_output.get("open_issues") or []))
        if issue is not None
    ]

    if _should_force_kb_route(
        route=route,
        structured_fact_requests=structured_fact_requests,
        open_issues=open_issues,
    ):
        route = "kb"
        structured_fact_requests = []

    return {
        "retrieval_needed": _normalize_bool(parsed_output.get("retrieval_needed")),
        "route": route,
        "structured_fact_requests": structured_fact_requests,
        "task_class": task_class,
        "targets": targets,
        "retrieval_plan": retrieval_plan,
        "needs_clarification": needs_clarification,
        "clarification_reason": _normalize_text(parsed_output.get("clarification_reason")),
        "clarification_questions": clarification_questions,
        "open_issues": open_issues,
    }


def _build_default_retrieval_plan(
    *,
    targets: Sequence[Dict[str, Any]],
    metric_guess: str,
    user_query: str,
) -> Optional[Dict[str, Any]]:
    target_ids = [int(target["target_id"]) for target in targets if _normalize_int(target.get("target_id")) is not None]
    if not target_ids:
        return None

    goal = _normalize_text(metric_guess) or _normalize_text(user_query) or "extract relevant filing evidence"
    return {
        "fanout_mode": "single_target" if len(target_ids) == 1 else "per_target",
        "jobs": [
            {
                "applies_to_target_ids": target_ids,
                "goal": goal,
                "job_type": "metric_extract",
            }
        ],
    }


def _build_fallback_target_resolution(
    *,
    target_run: Dict[str, Any],
    company_ticker_map: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    planner_state = dict(target_run.get("planner_state") or {})
    unresolved_blockers = list(planner_state.get("unresolved_blockers") or [])
    if unresolved_blockers:
        clarification_questions = []
        if "ticker" in unresolved_blockers:
            clarification_questions.append("Which ticker should be analyzed?")
        if "fiscal_year" in unresolved_blockers:
            clarification_questions.append("Which fiscal year should be used?")
        return {
            "retrieval_needed": False,
            "route": "kb",
            "structured_fact_requests": [],
            "task_class": "other",
            "targets": [],
            "retrieval_plan": None,
            "needs_clarification": True,
            "clarification_reason": (
                "Planner still requires clarification for required extraction fields: "
                + ", ".join(unresolved_blockers)
            ),
            "clarification_questions": clarification_questions,
            "open_issues": [
                {
                    "code": "PLANNER_LLM_FALLBACK_CLARIFICATION_REQUIRED",
                    "message": (
                        "Fallback route detected unresolved required fields and cannot construct "
                        "a deterministic retrieval target."
                    ),
                    "severity": "warning",
                }
            ],
        }

    deterministic_targets = [
        dict(target) for target in (planner_state.get("deterministic_targets") or [])
        if isinstance(target, dict)
    ]
    if not deterministic_targets:
        return None

    for target in deterministic_targets:
        if not _normalize_text(target.get("ticker")) and target.get("company_name"):
            resolved = _resolve_ticker_from_company_name(
                target.get("company_name"),
                company_ticker_map or _DEFAULT_COMPANY_TICKER_MAP,
            )
            if resolved is not None:
                target["ticker"] = resolved

    intent_hint = _normalize_text(target_run.get("deterministic_intent_hint"))
    retrieval_needed_hint = target_run.get("deterministic_retrieval_needed_hint")
    metric_guess = _normalize_text(target_run.get("metric_guess")) or "filing evidence"
    query_text = str(
        planner_state.get("original_user_query")
        or planner_state.get("effective_user_query")
        or ""
    ).lower()
    filing_signal = any(
        token in query_text
        for token in [
            "10-k",
            "10-q",
            "10 k",
            "annual report",
            "fiscal",
            "filing",
            "revenue",
            "profit",
            "segment",
            "debt",
            "loan",
            "asset",
            "liquid",
            "risk",
            "credit",
        ]
    ) and bool(planner_state.get("deterministic_targets") or [])
    needs_filing_retrieval = intent_hint in {"filing_fact", "filing_calc"} or filing_signal
    retrieval_needed = (
        bool(retrieval_needed_hint)
        if retrieval_needed_hint is not None
        else bool(needs_filing_retrieval)
    )
    if not retrieval_needed and needs_filing_retrieval:
        retrieval_needed = True

    retrieval_plan = None
    if retrieval_needed:
        retrieval_plan = _build_default_retrieval_plan(
            targets=deterministic_targets,
            metric_guess=metric_guess,
            user_query=str(
                planner_state.get("original_user_query")
                or planner_state.get("effective_user_query")
                or ""
            ).strip(),
        )

    return {
        "retrieval_needed": bool(retrieval_needed),
        "route": "kb",
        "structured_fact_requests": [],
        "task_class": "single_target_fact" if len(deterministic_targets) == 1 else "other",
        "targets": deterministic_targets,
        "retrieval_plan": retrieval_plan,
        "needs_clarification": False,
        "clarification_reason": None,
        "clarification_questions": [],
        "open_issues": [
            {
                "code": "PLANNER_LLM_FALLBACK",
                "message": (
                    "Planner LLM output could not be produced or validated; "
                    "using deterministic target metadata fallback."
                ),
                "severity": "warning",
            }
        ],
    }


def _build_metadata(
    *,
    targets: Sequence[Dict[str, Any]],
    deterministic_hints: Dict[str, Any],
) -> Dict[str, Any]:
    first_target = dict(targets[0]) if targets else {}
    fiscal_years = sorted(
        {
            int(year)
            for year in (
                _normalize_int(target.get("fiscal_year"))
                for target in targets
            )
            if year is not None
        }
    )
    form_types = _dedupe_strings(
        _normalize_form_type(target.get("form_type"))
        for target in targets
    )

    metadata = FilingMetadata(
        ticker=(
            first_target.get("ticker")
            if len(targets) == 1
            else _normalize_text(deterministic_hints.get("ticker"))
        ),
        company_name=(
            first_target.get("company_name")
            if len(targets) == 1
            else _normalize_text(deterministic_hints.get("company_name"))
        ),
        fiscal_year=fiscal_years[0] if len(fiscal_years) == 1 else _normalize_int(deterministic_hints.get("fiscal_year")),
        form_type=_normalize_form_type(
            first_target.get("form_type")
            if len(targets) == 1
            else (form_types[0] if len(form_types) == 1 else deterministic_hints.get("form_type"))
        ),
        doc_types=deterministic_hints.get("doc_types"),
        fiscal_quarter=deterministic_hints.get("fiscal_quarter"),
    )
    return metadata.model_dump(mode="json", exclude_none=False)


def _build_analysis_task(
    *,
    task_class: str,
    metric_guess: str,
    retrieval_plan: Optional[Dict[str, Any]],
    task_type_hint: str,
) -> Dict[str, Any]:
    metric = _normalize_text(metric_guess)
    if metric is None:
        metric = _normalize_text((((retrieval_plan or {}).get("jobs") or [{}])[0]).get("goal")) or "filing evidence"

    task_type = (_normalize_text(task_type_hint) or "extract").lower()
    if task_class in _MULTI_TARGET_TASK_CLASSES:
        task_type = "compare"
    elif task_type not in {"extract", "compute", "compare", "trend"}:
        task_type = "extract"

    job_types = {
        _normalize_text((job or {}).get("job_type")) or ""
        for job in ((retrieval_plan or {}).get("jobs") or [])
        if isinstance(job, dict)
    }
    requires_calculation = task_type == "compute"

    expected_artifacts = ["table", "row", "text"]
    if job_types and job_types == {"narrative_extract"}:
        expected_artifacts = ["text"]

    analysis_task = AnalysisTask(
        task_type=task_type,
        metric=metric,
        definition_notes=[],
        requires_calculation=requires_calculation,
        expected_artifacts=expected_artifacts,
        output_format="table" if task_class in _MULTI_TARGET_TASK_CLASSES else "short_answer",
    )
    return analysis_task.model_dump(mode="json")


def _merge_open_issues(*issue_lists: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()
    for issue_list in issue_lists:
        for issue in issue_list or []:
            normalized = _normalize_open_issue(issue)
            if normalized is None:
                continue
            key = (
                normalized["code"],
                normalized["message"],
                normalized["severity"],
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(normalized)
    return merged


def _build_planner_output(
    *,
    status: str,
    target_run: Dict[str, Any],
    target_resolution: Optional[Dict[str, Any]],
    clarification_request: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    planner_state = dict(target_run.get("planner_state") or {})
    deterministic_hints = dict(target_run.get("deterministic_hints") or {})
    metric_guess = _normalize_text(target_run.get("metric_guess")) or "filing evidence"
    task_class = _normalize_text((target_resolution or {}).get("task_class")) or "other"
    targets = list((target_resolution or {}).get("targets") or [])
    route = (_normalize_text((target_resolution or {}).get("route")) or "kb").lower()
    if route not in _ALLOWED_ROUTES:
        route = "kb"
    structured_fact_requests = _normalize_structured_fact_requests(
        (target_resolution or {}).get("structured_fact_requests")
    )

    retrieval_needed = bool((target_resolution or {}).get("retrieval_needed")) and status == "completed"
    if status == "completed" and route == "structured_fact":
        retrieval_needed = False
    elif status == "completed" and route == "hybrid":
        retrieval_needed = True
    elif status == "completed" and route == "kb":
        structured_fact_requests = []

    retrieval_plan = (target_resolution or {}).get("retrieval_plan")
    if status == "completed" and route == "structured_fact":
        retrieval_plan = None
    if retrieval_needed and retrieval_plan is None:
        retrieval_plan = _build_default_retrieval_plan(
            targets=targets,
            metric_guess=metric_guess,
            user_query=str(planner_state.get("original_user_query") or target_run.get("user_query") or "").strip(),
        )
    if status != "completed":
        retrieval_plan = None
        structured_fact_requests = []

    intent = _normalize_text(target_run.get("deterministic_intent_hint")) or "filing_fact"
    if intent not in {"filing_fact", "filing_calc", "definition", "other"}:
        intent = "other"

    analysis_task = _build_analysis_task(
        task_class=task_class,
        metric_guess=metric_guess,
        retrieval_plan=retrieval_plan,
        task_type_hint=_normalize_text(target_run.get("deterministic_task_type_hint")) or "extract",
    )
    metadata = _build_metadata(targets=targets, deterministic_hints=deterministic_hints)
    open_issues = _merge_open_issues(
        target_run.get("deterministic_open_issues") or [],
        (target_resolution or {}).get("open_issues") or [],
    )

    original_user_query = str(
        planner_state.get("original_user_query")
        or target_run.get("user_query")
        or ""
    ).strip()
    effective_user_query = str(
        planner_state.get("effective_user_query") or original_user_query
    ).strip()
    payload = {
        "status": status,
        "retrieval_needed": retrieval_needed,
        "intent": intent,
        "route": route,
        "structured_fact_requests": structured_fact_requests,
        "metadata": metadata,
        "analysis_task": analysis_task,
        "open_issues": open_issues,
        "task_class": task_class,
        "targets": targets,
        "retrieval_plan": retrieval_plan,
        "original_user_query": original_user_query,
        "clarification_history": list(planner_state.get("clarification_history") or []),
        "clarification_request": clarification_request,
        "effective_user_query": effective_user_query,
    }
    return PlannerRuntimeOutput.model_validate(payload).model_dump(mode="json")


def run_target_resolution_prompt(
    *,
    planner: Any,
    user_query: str,
    prompt_template: str,
    clarification_turns: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            run_target_resolution_prompt_async(
                planner=planner,
                user_query=user_query,
                prompt_template=prompt_template,
                clarification_turns=clarification_turns,
            )
        )

    raise RuntimeError(
        "run_target_resolution_prompt cannot be used inside a running event loop. "
        "Use run_target_resolution_prompt_async instead."
    )


async def run_target_resolution_prompt_async(
    *,
    planner: Any,
    user_query: str,
    prompt_template: str,
    clarification_turns: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    pre_llm = build_target_resolution_payload(
        planner=planner,
        user_query=user_query,
        clarification_turns=clarification_turns,
    )
    prompt = render_target_resolution_prompt(
        prompt_template,
        user_query=user_query,
        payload=pre_llm["payload"],
    )

    raw_output = ""
    parsed_output: Optional[Dict[str, Any]] = None
    llm_error: Optional[str] = None
    validation_error: Optional[str] = None
    final_resolution: Optional[Dict[str, Any]] = None
    structured_builder = getattr(planner.llm, "with_structured_output", None)
    structured_enabled = (
        str(getattr(planner.llm, "_llm_type", "")).strip().lower() != "google-genai"
        and callable(structured_builder)
    )
    structured_failed = False

    if structured_enabled:
        try:
            structured_llm = structured_builder(_StructuredTargetResolutionOutput)
            response = await _ainvoke_llm(structured_llm, prompt)
            structured_output = _coerce_structured_resolution_output(response)
            if structured_output is None:
                raw_output = str(response)
                structured_failed = True
            else:
                raw_output = json.dumps(structured_output, ensure_ascii=False)
                parsed_output = structured_output
        except Exception:
            structured_failed = True
            raw_output = ""

    if parsed_output is None and llm_error is None:
        try:
            response = await _ainvoke_llm(planner.llm, prompt)
            response_text = response.content if hasattr(response, "content") else str(response)
            raw_output = response_text.strip() if isinstance(response_text, str) else str(response_text)
            parsed_output = _coerce_structured_resolution_output(raw_output)
        except Exception as exc:
            llm_error = f"LLM_CALL_FAILED: {exc}"

    if _is_blank_or_none_output(raw_output):
        raw_output = ""

    if llm_error is None and parsed_output is None:
        raw_output = raw_output or ""
        validation_error = "PARSE_FAILED"

    if llm_error is None and parsed_output is not None:
        try:
            final_resolution = _normalize_resolution_output(parsed_output)
        except Exception as exc:
            validation_error = f"VALIDATION_FAILED: {exc}"

    return {
        "user_query": user_query,
        "planner_state": pre_llm["planner_state"],
        "effective_user_query": pre_llm["effective_user_query"],
        "clarification_history": pre_llm["clarification_history"],
        "deterministic_hints": pre_llm["hints"].model_dump(mode="json", exclude_none=True),
        "deterministic_open_issues": [issue.model_dump(mode="json") for issue in pre_llm["issues"]],
        "metric_guess": pre_llm["metric_guess"],
        "deterministic_intent_hint": pre_llm["payload"].get("deterministic_intent_hint"),
        "deterministic_task_type_hint": pre_llm["payload"].get("deterministic_task_type_hint"),
        "deterministic_retrieval_needed_hint": pre_llm["payload"].get("deterministic_retrieval_needed_hint"),
        "deterministic_calc_cues": list(pre_llm["payload"].get("deterministic_calc_cues") or []),
        "expanded_queries": [],
        "expansion_error": None,
        "prompt": prompt,
        "raw_output": raw_output,
        "parsed_output": parsed_output,
        "llm_error": llm_error,
        "validation_error": validation_error,
        "model_output_present": not _is_blank_or_none_output(raw_output),
        "used_fallback": bool(structured_failed),
        "needs_clarification": bool(final_resolution and final_resolution.get("needs_clarification")),
        "clarification_questions": list((final_resolution or {}).get("clarification_questions") or []),
        "final_resolution": final_resolution,
        "final_plan": final_resolution,
    }


def _coerce_answers(
    answers: Any,
    *,
    questions: Sequence[str],
) -> List[str]:
    if isinstance(answers, str):
        if len(questions) != 1:
            raise ValueError("Expected one answer per clarification question.")
        return [answers.strip()]

    if isinstance(answers, (list, tuple)):
        answer_list = [str(answer).strip() for answer in answers]
        if len(answer_list) != len(questions):
            raise ValueError("Number of answers must match number of clarification questions.")
        return answer_list

    raise TypeError("answers must be a string or a list/tuple of strings.")


class InteractivePlannerAgent:
    """
    Primary planner implementation with a structured clarification loop.
    """

    def __init__(
        self,
        planner: Optional[Any] = None,
        *,
        llm: Optional[Any] = None,
        model: str = "ollama/qwen2.5:14b-instruct",
        temperature: float = 0.0,
        target_resolution_prompt_template: str = DEFAULT_TARGET_RESOLUTION_PROMPT_TEMPLATE,
        default_doc_types: Optional[List[str]] = None,
        company_ticker_map: Optional[Dict[str, str]] = None,
        enable_query_expansion: bool = True,
        auto_run_full_planner: bool = False,
        full_planner_include_trace: bool = False,
        log_timing: bool = True,
        **_: Any,
    ) -> None:
        inherited_doc_types = getattr(planner, "default_doc_types", None)
        inherited_company_ticker_map = getattr(planner, "company_ticker_map", None) or {}
        self.llm = llm or getattr(planner, "llm", None) or build_chat_model(
            model=model,
            temperature=temperature,
        )
        self.default_doc_types = (
            list(default_doc_types)
            if default_doc_types is not None
            else (list(inherited_doc_types) if inherited_doc_types is not None else None)
        )
        self.company_ticker_map = {
            **_DEFAULT_COMPANY_TICKER_MAP,
            **dict(inherited_company_ticker_map),
            **(company_ticker_map or {}),
        }
        self.model = model
        self.enable_query_expansion = bool(enable_query_expansion)
        self.target_resolution_prompt_template = str(target_resolution_prompt_template or "")
        self.auto_run_full_planner = bool(auto_run_full_planner)
        self.full_planner_include_trace = bool(full_planner_include_trace)
        self.log_timing = bool(log_timing)
        self.last_timing_ms: Dict[str, int] = {}

    async def aplan_turn(
        self,
        user_query: str,
        clarification_turns: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        query = str(user_query or "").strip()
        target_run = await run_target_resolution_prompt_async(
            planner=self,
            user_query=query,
            prompt_template=self.target_resolution_prompt_template,
            clarification_turns=list(clarification_turns or []),
        )
        return self._package_turn(target_run)

    def _run_planner_turn(
        self,
        user_query: str,
        clarification_turns: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.aplan_turn(
                    user_query=user_query,
                    clarification_turns=clarification_turns,
                )
            )
        raise RuntimeError(
            "Planner sync methods cannot be used from a running event loop. "
            "Use the async planner API (aplan_turn)."
        )

    def start(self, user_query: str) -> Dict[str, Any]:
        return self._run_planner_turn(user_query=user_query, clarification_turns=[])

    def resume(self, prior_turn: Dict[str, Any], answers: Any) -> Dict[str, Any]:
        state = dict(prior_turn.get("planner_state") or {})
        questions = list(((prior_turn.get("clarification_request") or {}).get("questions")) or [])
        if not questions:
            raise ValueError("prior_turn does not contain pending clarification questions.")

        answer_list = _coerce_answers(answers, questions=questions)
        clarification_history = _normalize_clarification_turns(state.get("clarification_history"))
        for question, answer in zip(questions, answer_list):
            clarification_history.append({"question": str(question), "answer": answer})

        return self._run_planner_turn(
            user_query=str(state.get("original_user_query") or prior_turn.get("user_query") or "").strip(),
            clarification_turns=clarification_history,
        )

    def plan(self, user_query: str, *, include_trace: bool = True) -> Any:
        t0 = time.perf_counter()
        turn = self.start(user_query)
        timing_ms = {"plan_total_ms": int((time.perf_counter() - t0) * 1000)}
        self.last_timing_ms = timing_ms
        if self.log_timing:
            ordered = " ".join(f"{k}={timing_ms[k]}" for k in sorted(timing_ms))
            print(f"[planner_timing_ms] {ordered}")

        plan_payload = dict(turn.get("planner_output") or {})
        if not include_trace:
            return plan_payload

        trace: Dict[str, Any] = {
            "timing_ms": timing_ms,
            "status": turn.get("status"),
            "clarification_request": turn.get("clarification_request"),
        }
        if turn.get("llm_error"):
            trace["error"] = str(turn["llm_error"])
        if turn.get("validation_error"):
            trace["validation_error"] = str(turn["validation_error"])
        return {"plan": plan_payload, "trace": trace}

    def chat(
        self,
        user_query: str,
        *,
        max_rounds: int = 3,
        input_fn: Callable[[str], str] = input,
        print_fn: Callable[[str], None] = print,
    ) -> Dict[str, Any]:
        turn = self.start(user_query)
        remaining_rounds = max(1, int(max_rounds))

        while turn.get("status") == "needs_clarification" and remaining_rounds > 0:
            clarification_request = turn.get("clarification_request") or {}
            reason = _normalize_text(clarification_request.get("reason"))
            if reason:
                print_fn(f"Clarification needed: {reason}")

            questions = list(clarification_request.get("questions") or [])
            answers: List[str] = []
            for question in questions:
                answers.append(str(input_fn(f"{question}\n> ") or "").strip())

            turn = self.resume(turn, answers)
            remaining_rounds -= 1

        return turn

    def _package_turn(self, target_run: Dict[str, Any]) -> Dict[str, Any]:
        planner_state = dict(target_run.get("planner_state") or {})
        target_resolution = target_run.get("final_resolution")
        clarification_request = None
        fallback_used = False
        downstream_skipped_reason = None

        if target_run.get("llm_error") or target_run.get("validation_error"):
            target_resolution = _build_fallback_target_resolution(
                target_run=target_run,
                company_ticker_map=self.company_ticker_map,
            )
            if target_resolution is None:
                status = "error"
            elif target_resolution.get("needs_clarification"):
                status = "needs_clarification"
                clarification_request = {
                    "reason": (target_resolution or {}).get("clarification_reason"),
                    "questions": list((target_resolution or {}).get("clarification_questions") or []),
                }
            else:
                status = "completed"
            fallback_used = True
        elif target_run.get("needs_clarification"):
            status = "needs_clarification"
            clarification_request = {
                "reason": (target_resolution or {}).get("clarification_reason"),
                "questions": list((target_resolution or {}).get("clarification_questions") or []),
            }
        else:
            status = "completed"

        planner_output = _build_planner_output(
            status=status,
            target_run=target_run,
            target_resolution=target_resolution,
            clarification_request=clarification_request,
        )

        if not self.auto_run_full_planner:
            downstream_skipped_reason = "FULL_PLANNER_DISABLED_BY_CONFIG"

        return {
            "status": status,
            "user_query": target_run.get("user_query"),
            "planner_state": planner_state,
            "target_resolution": target_resolution,
            "planner_output": planner_output,
            "clarification_request": clarification_request,
            "prompt": target_run.get("prompt"),
            "raw_output": target_run.get("raw_output"),
            "parsed_output": target_run.get("parsed_output"),
            "llm_error": target_run.get("llm_error"),
            "validation_error": target_run.get("validation_error"),
            "model_output_present": not _is_blank_or_none_output(target_run.get("raw_output")),
            "fallback_used": fallback_used,
            "structured_fallback_used": bool(target_run.get("used_fallback")),
            "full_plan": planner_output if status == "completed" else None,
            "full_plan_trace": None,
            "downstream_skipped_reason": downstream_skipped_reason,
        }


def run_interactive_target_resolution(
    *,
    planner: Any,
    user_query: str,
    prompt_template: str,
    max_rounds: int = 3,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> Dict[str, Any]:
    agent = InteractivePlannerAgent(
        planner=planner,
        target_resolution_prompt_template=prompt_template,
        auto_run_full_planner=False,
    )
    turn = agent.start(user_query)
    rounds: List[Dict[str, Any]] = [turn]
    remaining_rounds = max(1, int(max_rounds))

    while turn.get("status") == "needs_clarification" and remaining_rounds > 0:
        clarification_request = turn.get("clarification_request") or {}
        reason = _normalize_text(clarification_request.get("reason"))
        if reason:
            print_fn(f"Clarification needed: {reason}")

        questions = list(clarification_request.get("questions") or [])
        answers: List[str] = []
        for question in questions:
            answers.append(str(input_fn(f"{question}\n> ") or "").strip())

        turn = agent.resume(turn, answers)
        rounds.append(turn)
        remaining_rounds -= 1

    planner_state = dict(turn.get("planner_state") or {})
    return {
        "user_query": user_query,
        "clarification_turns": list(planner_state.get("clarification_history") or []),
        "rounds": rounds,
        "round_count": len(rounds),
        "stopped_due_to_max_rounds": bool(turn.get("status") == "needs_clarification" and remaining_rounds <= 0),
        "final_resolution": turn.get("target_resolution"),
        "last_run": turn,
        "status": turn.get("status"),
    }


__all__ = [
    "DEFAULT_TARGET_RESOLUTION_PROMPT_TEMPLATE",
    "InteractivePlannerAgent",
    "build_target_resolution_payload",
    "render_target_resolution_prompt",
    "run_target_resolution_prompt",
    "run_interactive_target_resolution",
]
