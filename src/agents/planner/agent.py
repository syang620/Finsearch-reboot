"""
Crawl-mode Planner Agent (minimal LangGraph version).

Responsibilities:
- deterministic metadata extraction
- LLM-based query expansion + planner JSON generation
- strict schema validation
- deterministic fallback when LLM output is invalid
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from retrieval.accounting_terms import accounting_terms_file_to_llm_digest
from retrieval.query_expansion import expand_query_with_ollama

from agents.contracts import (
    AnalysisTask,
    FilingMetadata,
    FormType,
    OpenIssue,
    PlannerIntent,
    PlannerOutput,
    QueryBundle,
    QualityRequirements,
    Severity,
)


# -----------------------------
# Deterministic helpers
# -----------------------------

_TICKER_STOPWORDS = {
    "SEC", "GAAP", "IFRS", "FY", "FQ", "USD", "EPS", "FCF", "EBITDA", "EBIT", "COGS",
    "YOY", "Q1", "Q2", "Q3", "Q4", "ITEM", "NOTE", "MDA", "CAPEX", "IPO",
    "AI", "ML", "API", "PDF",
}

_TICKER_RE = re.compile(r"""
    (?:
        \$([A-Za-z]{1,5})
        |
        \b([A-Z]{1,5}(?:\.[A-Z])?)\b
    )
""", re.VERBOSE)

_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_FY_RE = re.compile(r"\bFY\s*(19\d{2}|20\d{2})\b", re.IGNORECASE)
_10K_RE = re.compile(r"\b10[-\s]?K\b", re.IGNORECASE)
_10Q_RE = re.compile(r"\b10[-\s]?Q\b", re.IGNORECASE)
_QUARTER_RE = re.compile(r"\b(Q[1-4])\b", re.IGNORECASE)

_MILLION_RE = re.compile(r"\b(in\s+)?millions?\b", re.IGNORECASE)
_BILLION_RE = re.compile(r"\b(in\s+)?billions?\b", re.IGNORECASE)
_THOUSAND_RE = re.compile(r"\b(in\s+)?thousands?\b", re.IGNORECASE)

_POSSESSIVE_NAME_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9&\.\-]*(?:\s+[A-Z][A-Za-z0-9&\.\-]*){0,4})\s*[']s\b"
)
_OF_FOR_NAME_RE = re.compile(
    r"\b(?:of|for)\s+([A-Z][A-Za-z0-9&\.\-]*(?:\s+[A-Z][A-Za-z0-9&\.\-]*){0,4})\b"
)

_DEFAULT_COMPANY_TICKER_MAP: Dict[str, str] = {
    "apple": "AAPL",
    "apple inc": "AAPL",
    "alphabet": "GOOGL",
    "google": "GOOGL",
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
}

_SECTION_HINTS_BY_KEYWORD: List[Tuple[re.Pattern, List[str]]] = [
    (
        re.compile(r"\bdebt|borrow(ing|ings)|notes payable|credit facility|term loan\b", re.IGNORECASE),
        ["Liquidity and Capital Resources", "Debt", "Notes to Consolidated Financial Statements"],
    ),
    (
        re.compile(r"\bcash flow|operating cash|capex|capital expenditures|free cash flow\b", re.IGNORECASE),
        ["Consolidated Statements of Cash Flows", "Liquidity and Capital Resources"],
    ),
    (
        re.compile(r"\bshare(s)?|diluted|basic|eps\b", re.IGNORECASE),
        ["Earnings Per Share", "Equity", "Notes to Consolidated Financial Statements"],
    ),
    (
        re.compile(r"\brevenue|net sales\b", re.IGNORECASE),
        ["Consolidated Statements of Operations", "Revenue Recognition"],
    ),
]


def _normalize_company_key(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"[^\w\s&\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\b(incorporated|inc|corp|corporation|co|company|ltd|limited|plc)\b", "", s).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_company_name(query: str) -> Optional[str]:
    # Normalize curly apostrophes so possessive patterns like "Apple’s" are recognized.
    q = (query or "").replace("’", "'").strip()
    if not q:
        return None
    m = _POSSESSIVE_NAME_RE.search(q)
    if m:
        return m.group(1).strip()
    m2 = _OF_FOR_NAME_RE.search(q)
    if m2:
        return m2.group(1).strip()
    return None


def _resolve_ticker_from_company_name(company_name: Optional[str], mapping: Dict[str, str]) -> Optional[str]:
    if not company_name:
        return None
    key = _normalize_company_key(company_name)
    if not key:
        return None
    if key in mapping:
        return mapping[key]
    for k, v in mapping.items():
        if key == k or key.startswith(k) or k.startswith(key):
            return v
    return None


def _pick_ticker(query: str) -> Optional[str]:
    candidates: List[str] = []
    for m in _TICKER_RE.finditer(query):
        tok = (m.group(1) or m.group(2) or "").strip()
        if not tok:
            continue
        t = tok.upper()
        if t in _TICKER_STOPWORDS:
            continue
        if _YEAR_RE.fullmatch(t):
            continue
        candidates.append(t)

    if "$" in query:
        m = re.search(r"\$([A-Za-z]{1,5})\b", query)
        if m:
            t = m.group(1).upper()
            if t not in _TICKER_STOPWORDS:
                return t

    return candidates[0] if candidates else None


def _pick_fiscal_year(query: str) -> Optional[int]:
    m = _FY_RE.search(query)
    if m:
        return int(m.group(1))
    years = [int(y) for y in _YEAR_RE.findall(query)]
    return max(years) if years else None


def _pick_form_type(query: str) -> FormType:
    if _10Q_RE.search(query) or _QUARTER_RE.search(query):
        return FormType.TEN_Q
    if _10K_RE.search(query):
        return FormType.TEN_K
    if re.search(r"\byear[-\s]?end\b|\byear[-\s]?ended\b|\bat year[-\s]?end\b", query, re.IGNORECASE):
        return FormType.TEN_K
    return FormType.TEN_K


def _pick_fiscal_quarter(query: str) -> Optional[str]:
    m = _QUARTER_RE.search(query)
    return m.group(1).upper() if m else None


def _extract_units_hint(query: str) -> List[str]:
    hints: List[str] = []
    if _MILLION_RE.search(query):
        hints.append("$ in millions")
    if _BILLION_RE.search(query):
        hints.append("$ in billions")
    if _THOUSAND_RE.search(query):
        hints.append("in thousands")
    return hints


def _extract_section_hints(query: str) -> List[str]:
    out: List[str] = []
    for pat, hints in _SECTION_HINTS_BY_KEYWORD:
        if pat.search(query):
            out.extend(hints)
    seen = set()
    deduped: List[str] = []
    for h in out:
        if h not in seen:
            seen.add(h)
            deduped.append(h)
    return deduped


def _guess_metric(query: str) -> str:
    q = query.lower()
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


def _default_must_include(metric: str) -> List[str]:
    m = (metric or "").lower()
    if "debt" in m:
        return ["debt", "borrow", "credit", "notes payable", "long-term", "current portion"]
    if "free cash flow" in m:
        return ["cash flows", "operating activities", "capital expenditures"]
    if "cash flow" in m or "operating" in m:
        return ["cash flows", "operating activities"]
    if "revenue" in m or "net sales" in m:
        return ["revenue", "net sales"]
    if "earnings per share" in m or "eps" in m:
        return ["earnings per share", "diluted", "basic"]
    return []


def _is_reasonable_fiscal_year(fiscal_year: Optional[int]) -> bool:
    if fiscal_year is None:
        return False
    now_year = time.gmtime().tm_year
    return 1990 <= int(fiscal_year) <= (now_year + 1)


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
    if re.search(r"\b(net debt|ratio|margin|growth|change|delta|vs\.?|versus|compare|difference|yoy|qoq|cagr)\b", q):
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


def _merge_unique(a: List[str], b: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in (a or []) + (b or []):
        v = str(x).strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _is_table_focused_compute(plan: PlannerOutput) -> bool:
    if plan.intent != PlannerIntent.FILING_CALC:
        return False
    if (plan.analysis_task.task_type or "").strip().lower() != "compute":
        return False

    expected_artifacts = [str(x).strip().lower() for x in (plan.analysis_task.expected_artifacts or [])]
    doc_types = [str(x).strip().lower() for x in (plan.metadata.doc_types or [])]
    return ("table" in expected_artifacts) or ("table" in doc_types)


class PlannerState(TypedDict, total=False):
    user_query: str
    hints: dict
    issues: list
    metric_guess: str
    intent_hint: str
    task_type_hint: str
    retrieval_needed_hint: bool
    calc_cues: list
    expanded_queries: list
    expansion_error: str
    raw: str
    parsed: dict
    plan: dict
    error: str
    timing_ms: dict


class PlannerAgent:
    """
    Minimal crawl-mode planner built with LangGraph.
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        *,
        model: str = "qwen2.5:14b-instruct",
        temperature: float = 0.0,
        default_doc_types: Optional[List[str]] = None,
        company_ticker_map: Optional[Dict[str, str]] = None,
        enable_query_expansion: bool = True,
        expansion_model: str = "qwen3:4b-instruct",
        accounting_terms_path: Optional[str] = None,
        max_queries: int = 5,
        log_timing: bool = True,
    ) -> None:
        self.llm = llm or ChatOllama(model=model, temperature=temperature)
        self.default_doc_types = default_doc_types or ["table"]
        self.company_ticker_map = {**_DEFAULT_COMPANY_TICKER_MAP, **(company_ticker_map or {})}
        self.enable_query_expansion = enable_query_expansion
        self.expansion_model = expansion_model
        self.max_queries = max(1, min(int(max_queries), 10))
        self.log_timing = bool(log_timing)
        self.last_timing_ms: Dict[str, int] = {}
        self.allowed_line_items: Optional[str] = None
        if self.enable_query_expansion:
            self.allowed_line_items = self._load_accounting_terms_digest(accounting_terms_path)
        self.graph = self._build_graph()

    def plan(self, user_query: str, *, include_trace: bool = True) -> Any:
        t_total = time.perf_counter()
        query = (user_query or "").strip()
        if not query:
            raise ValueError("user_query must be non-empty")
        out = self.graph.invoke({"user_query": query})
        timing = dict(out.get("timing_ms") or {})
        timing["plan_total_ms"] = int((time.perf_counter() - t_total) * 1000)
        self.last_timing_ms = timing
        if self.log_timing:
            ordered = " ".join(f"{k}={timing[k]}" for k in sorted(timing))
            print(f"[planner_timing_ms] {ordered}")
        plan_obj = out["plan"]
        if not include_trace:
            return plan_obj

        trace: Dict[str, Any] = {
            "timing_ms": timing,
            "used_fallback": bool(timing.get("step_fallback_ms")),
        }
        if out.get("error"):
            trace["error"] = str(out["error"])
        if out.get("expansion_error"):
            trace["expansion_error"] = str(out["expansion_error"])

        plan_payload = (
            plan_obj.model_dump(mode="json")
            if hasattr(plan_obj, "model_dump")
            else plan_obj
        )
        return {"plan": plan_payload, "trace": trace}

    def _build_graph(self):
        builder = StateGraph(PlannerState)

        def _timing_update(state: PlannerState, key: str, elapsed_ms: int) -> Dict[str, int]:
            timing = dict(state.get("timing_ms") or {})
            timing[key] = int(elapsed_ms)
            return timing

        def preextract_node(state: PlannerState) -> Dict[str, Any]:
            t0 = time.perf_counter()
            q = state["user_query"]

            ticker = _pick_ticker(q)
            company_name = _extract_company_name(q)
            fy = _pick_fiscal_year(q)
            form = _pick_form_type(q)
            fq = _pick_fiscal_quarter(q)
            units = _extract_units_hint(q)
            sections = _extract_section_hints(q)

            issues: List[OpenIssue] = []
            if company_name:
                issues.append(
                    OpenIssue(
                        code="COMPANY_NAME_DETECTED",
                        message=f"Detected company name '{company_name}'.",
                        severity=Severity.INFO,
                    )
                )

            if ticker is None and company_name:
                resolved = _resolve_ticker_from_company_name(company_name, self.company_ticker_map)
                if resolved:
                    ticker = resolved
                    issues.append(
                        OpenIssue(
                            code="TICKER_INFERRED_FROM_COMPANY_NAME",
                            message=f"Inferred ticker '{ticker}' from company name '{company_name}'. Verify if needed.",
                            severity=Severity.WARNING,
                        )
                    )

            if ticker is None:
                issues.append(
                    OpenIssue(
                        code="TICKER_MISSING",
                        message="No ticker detected (and company name did not resolve to a ticker).",
                        severity=Severity.WARNING,
                    )
                )
            if fy is None:
                issues.append(
                    OpenIssue(
                        code="FISCAL_YEAR_MISSING",
                        message="No fiscal year detected in the query.",
                        severity=Severity.WARNING,
                    )
                )

            hints = FilingMetadata(
                ticker=ticker,
                company_name=company_name,
                fiscal_year=fy,
                form_type=form,
                doc_types=self.default_doc_types,
                fiscal_quarter=fq,
                section_hints=sections,
                units_hint=units,
            )

            return {
                "hints": hints,
                "issues": issues,
                "metric_guess": _guess_metric(q),
                "expanded_queries": [],
                "timing_ms": _timing_update(
                    state, "step_preextract_ms", int((time.perf_counter() - t0) * 1000)
                ),
            }

        def expand_queries_node(state: PlannerState) -> Dict[str, Any]:
            t0 = time.perf_counter()
            if not self.enable_query_expansion:
                return {
                    "expanded_queries": [],
                    "timing_ms": _timing_update(
                        state, "step_expand_queries_ms", int((time.perf_counter() - t0) * 1000)
                    ),
                }
            if not self.allowed_line_items:
                return {
                    "expanded_queries": [],
                    "expansion_error": "QUERY_EXPANSION_TERMS_UNAVAILABLE: accounting terms file not found.",
                    "timing_ms": _timing_update(
                        state, "step_expand_queries_ms", int((time.perf_counter() - t0) * 1000)
                    ),
                }

            t_call = time.perf_counter()
            try:
                expanded = expand_query_with_ollama(
                    state["user_query"],
                    allowed_line_items=self.allowed_line_items,
                    model=self.expansion_model,
                    include_original=False,
                    dedupe=True,
                    max_expansions=8,
                    options={"temperature": 0.0, "num_predict": 512},
                )
            except Exception as e:
                call_ms = int((time.perf_counter() - t_call) * 1000)
                timing = _timing_update(state, "llm_query_expansion_ms", call_ms)
                timing["step_expand_queries_ms"] = int((time.perf_counter() - t0) * 1000)
                return {
                    "expanded_queries": [],
                    "expansion_error": f"QUERY_EXPANSION_FAILED: {e}",
                    "timing_ms": timing,
                }

            call_ms = int((time.perf_counter() - t_call) * 1000)
            expanded = [str(x).strip() for x in (expanded or []) if str(x).strip()]
            timing = _timing_update(state, "llm_query_expansion_ms", call_ms)
            timing["step_expand_queries_ms"] = int((time.perf_counter() - t0) * 1000)
            if not expanded:
                return {
                    "expanded_queries": [],
                    "expansion_error": "QUERY_EXPANSION_EMPTY: expansion returned no terms.",
                    "timing_ms": timing,
                }
            return {
                "expanded_queries": expanded,
                "timing_ms": timing,
            }

        def llm_plan_node(state: PlannerState) -> Dict[str, Any]:
            t0 = time.perf_counter()
            intent_hint, task_type_hint, retrieval_needed_hint, calc_cues = _intent_hint_from_query(
                state["user_query"],
                state["metric_guess"],
            )
            payload = {
                "user_query": state["user_query"],
                "deterministic_hints": state["hints"].model_dump(mode="json"),
                "deterministic_open_issues": [i.model_dump(mode="json") for i in state["issues"]],
                "metric_guess": state["metric_guess"],
                "deterministic_intent_hint": intent_hint.value,
                "deterministic_task_type_hint": task_type_hint,
                "deterministic_retrieval_needed_hint": retrieval_needed_hint,
                "deterministic_calc_cues": calc_cues,
                "taxonomy_expanded_queries": state.get("expanded_queries", []),
                "rules": [
                    "Output JSON only.",
                    "Do not invent fiscal_year.",
                    "Return 1-5 retrieval queries under query_bundle.queries.",
                    "If deterministic_intent_hint is filing_calc and query is not definition, set intent to filing_calc.",
                    "Use planner schema fields exactly.",
                ],
                "required_schema": {
                    "retrieval_needed": "boolean",
                    "intent": ["filing_fact", "filing_calc", "definition", "other"],
                    "metadata": {
                        "ticker": "string|null",
                        "company_name": "string|null",
                        "fiscal_year": "int|null",
                        "form_type": ["10-K", "10-Q"],
                        "doc_types": "list[str]|null",
                        "fiscal_quarter": ["Q1", "Q2", "Q3", "Q4", None],
                        "section_hints": "list[str]",
                        "units_hint": "list[str]",
                    },
                    "query_bundle": {
                        "base_query": "string",
                        "queries": "list[str] (1-5)",
                        "must_include": "list[str]",
                        "nice_to_include": "list[str]",
                        "exclusions": "list[str]",
                    },
                    "analysis_task": {
                        "task_type": ["extract", "compute", "compare", "trend"],
                        "metric": "string",
                        "definition_notes": "list[str]",
                        "expected_artifacts": ["table", "row", "text"],
                        "output_format": ["short_answer", "step_by_step", "table"],
                    },
                    "quality_requirements": {
                        "min_results": "int",
                        "min_total_score": "float",
                        "must_have_provenance": "boolean",
                        "max_context_items": "int",
                        "accept_if_contains_any": "list[str]",
                    },
                    "open_issues": "list[{code,message,severity}]",
                },
            }

            prompt = (
                "You are a planner for an SEC-filings RAG system. "
                "Goal: produce ONE JSON object that matches the required schema."
                "Return one JSON object only, no markdown, no commentary.\n"
                "Never fabricate facts. If a field is unknown, use null."
                "You MUST NOT invent fiscal_year. If not present, set null and add open_issues code='FISCAL_YEAR_MISSING'."
                "Decide intent using the rubric: 1.definition -> conceptual explanation, no filing retrieval needed 2.filing_fact -> ask for factual value from filings 3.filing_calc -> requires computation using filing values 4.other -> anything else"
                "retrieval_needed must be consistent with intent: 1.definition\/other -> usually false 2.filing_fact\/filing_calc -> true (unless missing essential metadata; still can be true but add open_issues)"
                "Quality: 1.query_bundle.queries must contain 1-5 short retrieval queries. 2.Include must_include anchors when the user asks for numeric/financial values."
                + json.dumps(payload, ensure_ascii=False)
            )

            t_call = time.perf_counter()
            try:
                resp = self.llm.invoke(prompt)
                raw = resp.content if hasattr(resp, "content") else str(resp)
            except Exception as e:
                call_ms = int((time.perf_counter() - t_call) * 1000)
                timing = _timing_update(state, "llm_planner_ms", call_ms)
                timing["step_llm_plan_ms"] = int((time.perf_counter() - t0) * 1000)
                return {
                    "raw": "",
                    "parsed": {},
                    "error": f"LLM_CALL_FAILED: {e}",
                    "timing_ms": timing,
                }

            call_ms = int((time.perf_counter() - t_call) * 1000)
            timing = _timing_update(state, "llm_planner_ms", call_ms)
            timing["step_llm_plan_ms"] = int((time.perf_counter() - t0) * 1000)
            parsed = _extract_first_json_object(raw)
            if parsed is None:
                return {
                    "raw": raw,
                    "parsed": {},
                    "error": "PARSE_FAILED",
                    "timing_ms": timing,
                }

            return {"raw": raw, "parsed": parsed, "timing_ms": timing}

        def validate_node(state: PlannerState) -> Dict[str, Any]:
            t0 = time.perf_counter()
            parsed = state.get("parsed") or None
            if parsed is None:
                return {
                    "error": state.get("error") or "NO_PARSED_JSON",
                    "timing_ms": _timing_update(
                        state, "step_validate_ms", int((time.perf_counter() - t0) * 1000)
                    ),
                }

            try:
                plan = PlannerOutput.model_validate(parsed)
            except Exception as e:
                return {
                    "error": f"VALIDATION_FAILED: {e}",
                    "timing_ms": _timing_update(
                        state, "step_validate_ms", int((time.perf_counter() - t0) * 1000)
                    ),
                }

            plan = self._postprocess_plan(
                plan=plan,
                user_query=state["user_query"],
                hints=state["hints"],
                issues=state["issues"],
                expanded_queries=state.get("expanded_queries", []),
                expansion_error=state.get("expansion_error"),
            )
            return {
                "plan": plan,
                "timing_ms": _timing_update(
                    state, "step_validate_ms", int((time.perf_counter() - t0) * 1000)
                ),
            }

        def fallback_node(state: PlannerState) -> Dict[str, Any]:
            t0 = time.perf_counter()
            plan = self._fallback_plan(
                user_query=state["user_query"],
                hints=state["hints"],
                issues=state["issues"],
                expanded_queries=state.get("expanded_queries", []),
                expansion_error=state.get("expansion_error"),
            )
            return {
                "plan": plan,
                "timing_ms": _timing_update(
                    state, "step_fallback_ms", int((time.perf_counter() - t0) * 1000)
                ),
            }

        def route_after_validate(state: PlannerState) -> str:
            return "done" if state.get("plan") is not None else "fallback"

        builder.add_node("preextract", preextract_node)
        builder.add_node("expand_queries", expand_queries_node)
        builder.add_node("llm_plan", llm_plan_node)
        builder.add_node("validate", validate_node)
        builder.add_node("fallback", fallback_node)

        builder.set_entry_point("preextract")
        builder.add_edge("preextract", "expand_queries")
        builder.add_edge("expand_queries", "llm_plan")
        builder.add_edge("llm_plan", "validate")
        builder.add_conditional_edges(
            "validate",
            route_after_validate,
            {"done": END, "fallback": "fallback"},
        )
        builder.add_edge("fallback", END)

        return builder.compile()

    def _postprocess_plan(
        self,
        *,
        plan: PlannerOutput,
        user_query: str,
        hints: FilingMetadata,
        issues: List[OpenIssue],
        expanded_queries: Optional[List[str]] = None,
        expansion_error: Optional[str] = None,
    ) -> PlannerOutput:
        if plan.metadata.company_name is None and hints.company_name is not None:
            plan.metadata.company_name = hints.company_name
        if plan.metadata.ticker is None and hints.ticker is not None:
            plan.metadata.ticker = hints.ticker
        if plan.metadata.fiscal_year is None and hints.fiscal_year is not None:
            plan.metadata.fiscal_year = hints.fiscal_year
        if plan.metadata.fiscal_quarter is None and hints.fiscal_quarter is not None:
            plan.metadata.fiscal_quarter = hints.fiscal_quarter

        if not _is_reasonable_fiscal_year(plan.metadata.fiscal_year):
            replacement = hints.fiscal_year if _is_reasonable_fiscal_year(hints.fiscal_year) else None
            if plan.metadata.fiscal_year is not None:
                plan.open_issues.append(
                    OpenIssue(
                        code="FISCAL_YEAR_UNREASONABLE",
                        message=(
                            f"Planner produced fiscal_year={plan.metadata.fiscal_year}. "
                            f"Replaced with {replacement}."
                        ),
                        severity=Severity.WARNING,
                    )
                )
            plan.metadata.fiscal_year = replacement

        if plan.metadata.doc_types is None:
            plan.metadata.doc_types = self.default_doc_types

        plan.metadata.section_hints = _merge_unique(plan.metadata.section_hints, hints.section_hints)
        plan.metadata.units_hint = _merge_unique(plan.metadata.units_hint, hints.units_hint)

        llm_queries = [str(q).strip() for q in (plan.query_bundle.queries or []) if str(q).strip()]
        taxonomy_queries = [str(q).strip() for q in (expanded_queries or []) if str(q).strip()]
        queries = [user_query] + taxonomy_queries + llm_queries

        deduped_queries: List[str] = []
        seen = set()
        for q in queries:
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped_queries.append(q)

        plan.query_bundle.queries = deduped_queries[: self.max_queries]
        plan.query_bundle.base_query = (plan.query_bundle.base_query or "").strip() or user_query

        metric = (plan.analysis_task.metric or "").strip() or _guess_metric(user_query)
        intent_hint, task_type_hint, retrieval_hint, calc_cues = _intent_hint_from_query(user_query, metric)
        if intent_hint == PlannerIntent.FILING_CALC and plan.intent != PlannerIntent.DEFINITION:
            plan.intent = PlannerIntent.FILING_CALC
            plan.retrieval_needed = True
            plan.analysis_task.task_type = task_type_hint
        elif intent_hint == PlannerIntent.DEFINITION:
            plan.intent = PlannerIntent.DEFINITION
            plan.retrieval_needed = False
            plan.analysis_task.task_type = task_type_hint

        if not plan.query_bundle.must_include:
            plan.query_bundle.must_include = _default_must_include(metric)

        if not plan.quality_requirements.accept_if_contains_any and plan.query_bundle.must_include:
            plan.quality_requirements.accept_if_contains_any = plan.query_bundle.must_include[:8]

        if _is_table_focused_compute(plan):
            plan.quality_requirements.min_results = 1

        if _is_reasonable_fiscal_year(plan.metadata.fiscal_year):
            plan.open_issues = [i for i in (plan.open_issues or []) if i.code != "FISCAL_YEAR_MISSING"]

        existing = {(i.code, i.message) for i in (plan.open_issues or [])}
        for iss in issues:
            if (iss.code, iss.message) not in existing:
                plan.open_issues.append(iss)

        if expansion_error:
            expansion_msg = str(expansion_error).strip()
            if ("QUERY_EXPANSION_NOTICE", expansion_msg) not in existing:
                plan.open_issues.append(
                    OpenIssue(
                        code="QUERY_EXPANSION_NOTICE",
                        message=expansion_msg,
                        severity=Severity.WARNING,
                    )
                )

        if calc_cues and plan.intent == PlannerIntent.FILING_CALC:
            calc_msg = f"Deterministic calc cues: {', '.join(calc_cues)}"
            if ("INTENT_OVERRIDDEN_TO_CALC", calc_msg) not in existing:
                plan.open_issues.append(
                    OpenIssue(
                        code="INTENT_OVERRIDDEN_TO_CALC",
                        message=calc_msg,
                        severity=Severity.INFO,
                    )
                )

        return plan

    def _fallback_plan(
        self,
        *,
        user_query: str,
        hints: FilingMetadata,
        issues: List[OpenIssue],
        expanded_queries: Optional[List[str]] = None,
        expansion_error: Optional[str] = None,
    ) -> PlannerOutput:
        metric = _guess_metric(user_query)
        must = _default_must_include(metric)

        queries: List[str] = [user_query]
        if expanded_queries:
            queries.extend(str(q).strip() for q in expanded_queries if str(q).strip())
        if metric and metric != "filing facts":
            queries.append(f"{metric} balance sheet table")
        if hints.section_hints:
            queries.append(f"{metric} {hints.section_hints[0]}")
        if must:
            queries.append(" ".join(must[:3]) + " table")

        deduped: List[str] = []
        seen = set()
        for q in queries:
            k = q.strip().lower()
            if k and k not in seen:
                seen.add(k)
                deduped.append(q.strip())

        intent, task_type, retrieval_needed, _calc_cues = _intent_hint_from_query(user_query, metric)

        fallback_issues = list(issues)
        if expansion_error:
            fallback_issues.append(
                OpenIssue(
                    code="QUERY_EXPANSION_NOTICE",
                    message=str(expansion_error).strip(),
                    severity=Severity.WARNING,
                )
            )

        qb = QueryBundle(
            base_query=user_query,
            queries=[user_query],
            must_include=must,
            nice_to_include=[],
            exclusions=[],
        )
        qb.queries = deduped[: self.max_queries] or [user_query]

        expected_artifacts = ["table"]
        fallback_min_results = (
            1
            if (
                intent == PlannerIntent.FILING_CALC
                and (task_type or "").strip().lower() == "compute"
                and ("table" in [str(x).strip().lower() for x in expected_artifacts]
                     or "table" in [str(x).strip().lower() for x in (hints.doc_types or [])])
            )
            else 3
        )

        return PlannerOutput(
            retrieval_needed=retrieval_needed,
            intent=intent,
            metadata=hints,
            query_bundle=qb,
            analysis_task=AnalysisTask(
                task_type=task_type,
                metric=metric,
                definition_notes=[],
                expected_artifacts=expected_artifacts,
                output_format="step_by_step",
            ),
            quality_requirements=QualityRequirements(
                min_results=fallback_min_results,
                min_total_score=0.0,
                must_have_provenance=True,
                max_context_items=10,
                accept_if_contains_any=must[:8],
            ),
            open_issues=fallback_issues,
        )

    def _load_accounting_terms_digest(self, accounting_terms_path: Optional[str]) -> Optional[str]:
        candidates: List[Path] = []
        if accounting_terms_path:
            candidates.append(Path(accounting_terms_path))
        candidates.extend(
            [
                Path("../data/config/SEC_accounting_terms.json"),
                Path("data/config/SEC_accounting_terms.json"),
            ]
        )

        for path in candidates:
            if path.exists():
                try:
                    return accounting_terms_file_to_llm_digest(str(path))
                except Exception:
                    return None
        return None
