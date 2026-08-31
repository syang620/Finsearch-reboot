from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Iterable

from mcp_server.tools.sec_metric_registry import METRIC_REGISTRY


class StructuredFactQuestionClass(str, Enum):
    SUPPORTED_DIRECT_METRIC = "supported_direct_metric"
    UNSUPPORTED_DERIVED_METRIC = "unsupported_derived_metric"
    UNSUPPORTED_RATIO = "unsupported_ratio"
    UNSUPPORTED_PER_SHARE = "unsupported_per_share"
    UNSUPPORTED_COMPARISON = "unsupported_comparison"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class StructuredFactCapability:
    metric_id: str
    label: str
    registry_kind: str
    exact_phrases: tuple[str, ...]
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class StructuredFactCapabilityDecision:
    question_class: StructuredFactQuestionClass
    permitted: bool
    matched_metric_ids: tuple[str, ...]
    reason: str


def _normalize_lookup_text(value: Any) -> str:
    text = str(value or "").replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def sanitize_capability_text(value: Any, *, limit: int = 240) -> str | None:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return None
    text = re.sub(
        r"(?i)\b(authorization|api[-_ ]?key|access[-_ ]?token|secret)\b\s*[:=]\s*\S+",
        r"\1=<redacted>",
        text,
    )
    return text[:limit]


_CAPABILITY_SPECS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "total_debt",
        ("total debt", "interest bearing debt"),
        ("borrowings", "debt balance"),
    ),
    ("revenue", ("revenue", "total revenue"), ("net sales", "sales")),
    ("gross_profit", ("gross profit",), ("gross earnings",)),
    ("operating_income", ("operating income", "operating profit"), ("ebit",)),
    ("net_income", ("net income",), ("net earnings",)),
    (
        "cash_and_cash_equivalents",
        ("cash and cash equivalents", "cash equivalents"),
        (),
    ),
    ("total_assets", ("total assets",), ()),
    ("total_liabilities", ("total liabilities",), ()),
    (
        "stockholders_equity",
        ("stockholders equity", "shareholders equity"),
        (),
    ),
    (
        "operating_cash_flow",
        ("operating cash flow", "cash flow from operations", "cash from operations"),
        ("cfo",),
    ),
    (
        "capex",
        ("capital expenditures", "capital expenditure", "capex"),
        (),
    ),
)


def _build_capabilities() -> tuple[StructuredFactCapability, ...]:
    configured_ids = {metric_id for metric_id, _, _ in _CAPABILITY_SPECS}
    registry_ids = set(METRIC_REGISTRY)
    if configured_ids != registry_ids:
        missing = sorted(registry_ids - configured_ids)
        stale = sorted(configured_ids - registry_ids)
        raise RuntimeError(
            "Structured-fact capability policy is out of sync with the metric registry "
            f"(missing={missing}, stale={stale})."
        )
    return tuple(
        StructuredFactCapability(
            metric_id=metric_id,
            label=METRIC_REGISTRY[metric_id].label,
            registry_kind=METRIC_REGISTRY[metric_id].kind,
            exact_phrases=exact_phrases,
            aliases=aliases,
        )
        for metric_id, exact_phrases, aliases in _CAPABILITY_SPECS
    )


_PER_SHARE_PATTERNS = (
    re.compile(r"\bper share\b"),
    re.compile(r"\b(?:basic|diluted)?\s*eps\b"),
    re.compile(r"\bearnings per share\b"),
)
_RATIO_PATTERNS = (
    re.compile(r"\bmargin\b"),
    re.compile(r"\byield\b"),
    re.compile(r"\bratio\b"),
    re.compile(r"\breturn on (?:equity|assets|investment|capital)\b"),
    re.compile(r"\b(?:roe|roa|roic)\b"),
    re.compile(r"\bev\s+ebitda\b"),
    re.compile(r"\bpercentage of\b"),
)
_COMPARISON_PATTERNS = (
    re.compile(r"\bcompare\b"),
    re.compile(r"\bcomparison\b"),
    re.compile(r"\bversus\b"),
    re.compile(r"\bvs\b"),
    re.compile(r"\bdifference between\b"),
    re.compile(r"\b(?:higher|lower|greater|less) than\b"),
)
_DERIVED_PATTERNS = (
    re.compile(r"\b(?:growth|growth rate|cagr)\b"),
    re.compile(r"\b(?:year over year|yoy|quarter over quarter|qoq)\b"),
    re.compile(r"\b(?:change|increase|increased|decrease|decreased)\b"),
    re.compile(r"\bpercentage (?:increase|decrease|change)\b"),
    re.compile(r"\bfree cash flow\b"),
    re.compile(r"\bnet debt\b"),
    re.compile(r"\bebitda\b"),
    re.compile(r"\b(?:balance sheet|financial|financial position) summary\b"),
    re.compile(r"\bkey financial metrics\b"),
)
_AMBIGUOUS_GENERIC_TERMS = frozenset({"cash", "profit", "profitability"})
_EXPLICIT_UNSUPPORTED_CLASSES = frozenset(
    {
        StructuredFactQuestionClass.UNSUPPORTED_DERIVED_METRIC,
        StructuredFactQuestionClass.UNSUPPORTED_RATIO,
        StructuredFactQuestionClass.UNSUPPORTED_PER_SHARE,
        StructuredFactQuestionClass.UNSUPPORTED_COMPARISON,
    }
)


def _contains_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text))


def _matches_any(text: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


class StructuredFactCapabilityPolicy:
    """Classify structured-fact requests without resolving filing execution inputs."""

    def __init__(
        self,
        capabilities: tuple[StructuredFactCapability, ...] | None = None,
    ) -> None:
        self.capabilities = capabilities or _build_capabilities()

    def classify_request(
        self,
        *,
        metric_hint: Any,
        subquestion: Any,
    ) -> StructuredFactCapabilityDecision:
        metric_text = _normalize_lookup_text(metric_hint)
        question_text = _normalize_lookup_text(subquestion)
        combined_text = " ".join(part for part in (metric_text, question_text) if part)

        if _matches_any(combined_text, _PER_SHARE_PATTERNS):
            return self._rejected(
                StructuredFactQuestionClass.UNSUPPORTED_PER_SHARE,
                "Per-share metrics are not executable by the structured-fact lane.",
            )
        if _matches_any(combined_text, _RATIO_PATTERNS):
            return self._rejected(
                StructuredFactQuestionClass.UNSUPPORTED_RATIO,
                "Ratios, margins, and yields require derivation outside the structured-fact lane.",
            )
        if _matches_any(combined_text, _COMPARISON_PATTERNS):
            return self._rejected(
                StructuredFactQuestionClass.UNSUPPORTED_COMPARISON,
                "Comparisons requiring derivation are not executable by the structured-fact lane.",
            )
        if _matches_any(combined_text, _DERIVED_PATTERNS):
            return self._rejected(
                StructuredFactQuestionClass.UNSUPPORTED_DERIVED_METRIC,
                "The requested calculated metric is not a supported registry capability.",
            )

        exact_matches = self._matching_metric_ids(combined_text, use_aliases=False)
        if len(exact_matches) == 1:
            return self._supported(exact_matches[0], "Matched an exact supported metric phrase.")
        if len(exact_matches) > 1:
            return self._ambiguous(exact_matches, "The request names multiple supported metrics.")

        ambiguous_term = None
        if metric_text in _AMBIGUOUS_GENERIC_TERMS:
            ambiguous_term = metric_text
        elif not metric_text:
            ambiguous_term = next(
                (
                    term
                    for term in _AMBIGUOUS_GENERIC_TERMS
                    if _contains_phrase(question_text, term)
                ),
                None,
            )
        if ambiguous_term is not None:
            candidates = self._ambiguous_candidate_ids(ambiguous_term)
            return self._ambiguous(
                candidates,
                "The metric phrase is too broad for deterministic structured execution.",
            )

        alias_matches = self._matching_alias_metric_ids(
            metric_text=metric_text,
            question_text=question_text,
        )
        if len(alias_matches) == 1:
            return self._supported(alias_matches[0], "Matched a supported metric alias.")
        if len(alias_matches) > 1:
            return self._ambiguous(alias_matches, "The metric alias maps to multiple supported metrics.")

        return self._rejected(
            StructuredFactQuestionClass.UNKNOWN,
            "The request does not match an explicit structured-fact capability.",
        )

    def classify_requests(
        self,
        requests: Iterable[dict[str, Any]],
        *,
        original_user_query: Any = None,
    ) -> tuple[StructuredFactCapabilityDecision, ...]:
        request_list = list(requests)
        decisions = tuple(
            self.classify_request(
                metric_hint=request.get("metric_hint"),
                subquestion=request.get("subquestion"),
            )
            for request in request_list
        )
        if not decisions:
            return decisions

        original_text = _normalize_lookup_text(original_user_query)
        if not original_text:
            return decisions
        original_decision = self.classify_request(
            metric_hint=None,
            subquestion=original_text,
        )
        if original_decision.permitted or self._has_independent_conjoined_requests(
            original_text
        ):
            return decisions
        if original_decision.question_class == StructuredFactQuestionClass.UNKNOWN:
            return decisions
        return tuple(original_decision for _request in request_list)

    def prompt_appendix(self) -> str:
        supported = ", ".join(
            phrase
            for capability in self.capabilities
            for phrase in capability.exact_phrases[:1]
        )
        aliases = ", ".join(
            alias
            for capability in self.capabilities
            for alias in capability.aliases
        )
        return (
            "Structured-fact capability policy:\n"
            f"- Supported metric phrases: {supported}.\n"
            f"- Supported aliases: {aliases}.\n"
            "- Registry-derived total debt and capex are supported because the registry owns their execution.\n"
            "- Ratios, margins, yields, per-share metrics, growth/change calculations, derived comparisons, "
            "broad summaries, and unknown concepts must not enter structured execution.\n"
            "- Do not decompose an unsupported request into supported component requests for structured execution.\n"
            "- Generic cash, profit, and profitability requests require metric clarification.\n"
            "- Classify each proposed structured request independently; supported requests may remain in a "
            "hybrid plan while rejected portions are handled by KB retrieval."
        )

    def _matching_metric_ids(self, text: str, *, use_aliases: bool) -> tuple[str, ...]:
        matches: list[str] = []
        for capability in self.capabilities:
            phrases = capability.aliases if use_aliases else capability.exact_phrases
            if any(_contains_phrase(text, phrase) for phrase in phrases):
                matches.append(capability.metric_id)
        return tuple(sorted(set(matches)))

    def _matching_alias_metric_ids(
        self,
        *,
        metric_text: str,
        question_text: str,
    ) -> tuple[str, ...]:
        matches: list[str] = []
        for capability in self.capabilities:
            if metric_text:
                matched = any(metric_text == alias for alias in capability.aliases)
            else:
                matched = any(
                    _contains_phrase(question_text, alias)
                    for alias in capability.aliases
                )
            if matched:
                matches.append(capability.metric_id)
        return tuple(sorted(set(matches)))

    def _has_independent_conjoined_requests(self, text: str) -> bool:
        for conjunction in re.finditer(r"\band\b", text):
            left = text[: conjunction.start()].strip()
            right = text[conjunction.end() :].strip()
            if not left or not right:
                continue
            left_decision = self.classify_request(metric_hint=None, subquestion=left)
            right_decision = self.classify_request(metric_hint=None, subquestion=right)
            left_explicitly_unsupported = (
                left_decision.question_class in _EXPLICIT_UNSUPPORTED_CLASSES
            )
            right_explicitly_unsupported = (
                right_decision.question_class in _EXPLICIT_UNSUPPORTED_CLASSES
            )
            if left_decision.permitted and (
                right_decision.permitted or right_explicitly_unsupported
            ):
                return True
            if right_decision.permitted and (
                left_decision.permitted or left_explicitly_unsupported
            ):
                return True
        return False

    def _ambiguous_candidate_ids(self, text: str) -> tuple[str, ...]:
        if text == "cash":
            return ("cash_and_cash_equivalents", "operating_cash_flow")
        if text in {"profit", "profitability"}:
            return ("gross_profit", "net_income", "operating_income")
        return ()

    @staticmethod
    def _supported(metric_id: str, reason: str) -> StructuredFactCapabilityDecision:
        return StructuredFactCapabilityDecision(
            question_class=StructuredFactQuestionClass.SUPPORTED_DIRECT_METRIC,
            permitted=True,
            matched_metric_ids=(metric_id,),
            reason=reason,
        )

    @staticmethod
    def _ambiguous(
        metric_ids: tuple[str, ...],
        reason: str,
    ) -> StructuredFactCapabilityDecision:
        return StructuredFactCapabilityDecision(
            question_class=StructuredFactQuestionClass.AMBIGUOUS,
            permitted=False,
            matched_metric_ids=tuple(sorted(set(metric_ids))),
            reason=reason,
        )

    @staticmethod
    def _rejected(
        question_class: StructuredFactQuestionClass,
        reason: str,
    ) -> StructuredFactCapabilityDecision:
        return StructuredFactCapabilityDecision(
            question_class=question_class,
            permitted=False,
            matched_metric_ids=(),
            reason=reason,
        )


DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY = StructuredFactCapabilityPolicy()


__all__ = [
    "DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY",
    "StructuredFactCapability",
    "StructuredFactCapabilityDecision",
    "StructuredFactCapabilityPolicy",
    "StructuredFactQuestionClass",
    "sanitize_capability_text",
]
