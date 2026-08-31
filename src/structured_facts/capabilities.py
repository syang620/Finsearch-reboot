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
    re.compile(
        r"\b(?:grow|grew|grown|decline|declined|fall|fell|fallen|"
        r"rise|rose|risen|drop|dropped)\b"
    ),
    re.compile(r"\bpercentage (?:increase|decrease|change)\b"),
    re.compile(r"\bfree cash flow\b"),
    re.compile(r"\bnet debt\b"),
    re.compile(r"\bebitda\b"),
    re.compile(r"\b(?:balance sheet|financial|financial position) summary\b"),
    re.compile(r"\bkey financial metrics\b"),
    re.compile(r"\b(?:sum|average|mean|median)\b"),
    re.compile(
        r"\b(?:add|subtract|minus|multiply|multiplied by|divide|divided by)\b"
    ),
    re.compile(r"\b(?:calculate|compute)\b.*\bplus\b"),
)
_AMBIGUOUS_GENERIC_TERMS = frozenset({"cash", "profit", "profitability"})
_QUESTION_PREFIX_BOUNDARIES = frozenset(
    {
        "a",
        "an",
        "are",
        "did",
        "does",
        "give",
        "how",
        "is",
        "me",
        "much",
        "reported",
        "s",
        "show",
        "the",
        "was",
        "were",
        "what",
    }
)
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
        entity_hints: Iterable[Any] = (),
    ) -> StructuredFactCapabilityDecision:
        metric_text = _normalize_lookup_text(metric_hint)
        question_text = _normalize_lookup_text(subquestion)
        combined_text = " ".join(part for part in (metric_text, question_text) if part)
        raw_text = " ".join(
            part for part in (str(metric_hint or ""), str(subquestion or "")) if part
        )

        if re.search(r"\S\s*[+/]\s*\S", raw_text):
            return self._rejected(
                StructuredFactQuestionClass.UNSUPPORTED_DERIVED_METRIC,
                "Symbolic arithmetic is not executable by the structured-fact lane.",
            )

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

        hint_metric_ids = self._matching_hint_metric_ids(metric_text)
        question_metric_ids = self._matching_question_metric_ids(
            question_text,
            entity_hints=entity_hints,
        )
        if (
            metric_text
            and question_text
            and hint_metric_ids
            and not set(hint_metric_ids).intersection(question_metric_ids)
        ):
            return self._rejected(
                StructuredFactQuestionClass.UNKNOWN,
                "The metric hint does not agree with the structured subquestion.",
            )

        if metric_text:
            if len(hint_metric_ids) == 1:
                return self._supported(
                    hint_metric_ids[0],
                    "Matched a direct supported metric hint.",
                )
            if len(hint_metric_ids) > 1:
                return self._ambiguous(
                    hint_metric_ids,
                    "The metric hint maps to multiple supported metrics.",
                )
            if metric_text in _AMBIGUOUS_GENERIC_TERMS:
                return self._ambiguous(
                    self._ambiguous_candidate_ids(metric_text),
                    "The metric phrase is too broad for deterministic structured execution.",
                )
            return self._rejected(
                StructuredFactQuestionClass.UNKNOWN,
                "The metric hint is not a direct structured-fact capability.",
            )

        question_metric_ids = self._matching_question_metric_ids(
            question_text,
            entity_hints=entity_hints,
        )
        if len(question_metric_ids) == 1:
            return self._supported(
                question_metric_ids[0],
                "Matched a complete supported metric phrase.",
            )
        if len(question_metric_ids) > 1:
            return self._ambiguous(
                question_metric_ids,
                "The question contains multiple supported metric phrases.",
            )

        ambiguous_term = next(
            (
                term
                for term in _AMBIGUOUS_GENERIC_TERMS
                if _contains_phrase(question_text, term)
            ),
            None,
        )
        if ambiguous_term is not None:
            return self._ambiguous(
                self._ambiguous_candidate_ids(ambiguous_term),
                "The metric phrase is too broad for deterministic structured execution.",
            )
        return self._rejected(
            StructuredFactQuestionClass.UNKNOWN,
            "A direct metric hint is required for structured execution.",
        )

    def classify_requests(
        self,
        requests: Iterable[dict[str, Any]],
        *,
        original_user_query: Any = None,
        entity_hints: Iterable[Any] = (),
    ) -> tuple[StructuredFactCapabilityDecision, ...]:
        request_list = list(requests)
        shared_entity_hints = tuple(entity_hints)
        all_entity_hints = shared_entity_hints + tuple(
            request.get("entity_hint") for request in request_list
        )
        decisions = tuple(
            self.classify_request(
                metric_hint=request.get("metric_hint"),
                subquestion=request.get("subquestion"),
                entity_hints=shared_entity_hints + (request.get("entity_hint"),),
            )
            for request in request_list
        )
        if not decisions:
            return decisions

        original_source = str(original_user_query or "").replace(
            ";", " clauseboundary "
        )
        original_text = _normalize_lookup_text(original_source)
        if not original_text:
            return decisions
        original_decision = self.classify_request(
            metric_hint=None,
            subquestion=original_source,
            entity_hints=all_entity_hints,
        )
        if original_decision.permitted:
            return decisions
        if self._has_independent_conjoined_requests(
            original_text,
            entity_hints=all_entity_hints,
        ):
            return self._apply_explicit_clause_rejections(
                original_text,
                request_list,
                decisions,
                entity_hints=all_entity_hints,
            )
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

    def _matching_hint_metric_ids(self, metric_text: str) -> tuple[str, ...]:
        matches = [
            capability.metric_id
            for capability in self.capabilities
            if metric_text in {*capability.exact_phrases, *capability.aliases}
        ]
        return tuple(sorted(set(matches)))

    def _matching_question_metric_ids(
        self,
        text: str,
        *,
        entity_hints: Iterable[Any] = (),
    ) -> tuple[str, ...]:
        entity_token_sequences = tuple(
            tuple(_normalize_lookup_text(value).split())
            for value in entity_hints
            if _normalize_lookup_text(value)
        )
        matches: list[str] = []
        for capability in self.capabilities:
            for phrase in (*capability.exact_phrases, *capability.aliases):
                for match in re.finditer(
                    rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])",
                    text,
                    flags=re.IGNORECASE,
                ):
                    prefix_tokens = text[: match.start()].split()
                    if not self._is_question_metric_prefix(
                        prefix_tokens,
                        entity_token_sequences,
                    ):
                        continue
                    suffix_tokens = text[match.end() :].split()
                    if not self._is_question_metric_suffix(
                        suffix_tokens,
                        entity_token_sequences,
                    ):
                        continue
                    matches.append(capability.metric_id)
                    break
        return tuple(sorted(set(matches)))

    @staticmethod
    def _is_question_metric_prefix(
        tokens: list[str],
        entity_token_sequences: tuple[tuple[str, ...], ...],
    ) -> bool:
        boundary = max(
            (
                index
                for index, token in enumerate(tokens)
                if token.lower() in _QUESTION_PREFIX_BOUNDARIES
            ),
            default=-1,
        )
        metric_prefix = tuple(token.lower() for token in tokens[boundary + 1 :])
        return not metric_prefix or metric_prefix in entity_token_sequences

    @staticmethod
    def _is_question_metric_suffix(
        tokens: list[str],
        entity_token_sequences: tuple[tuple[str, ...], ...],
    ) -> bool:
        if not tokens:
            return True
        first = tokens[0].lower()
        if first in {"is", "was", "were"}:
            tokens = tokens[1:]
            if not tokens:
                return True
            first = tokens[0].lower()
        if first in {"did", "does", "reported"}:
            tokens = tokens[1:]
            for entity_tokens in entity_token_sequences:
                if tuple(token.lower() for token in tokens[: len(entity_tokens)]) == entity_tokens:
                    tokens = tokens[len(entity_tokens) :]
                    break
            if tokens and tokens[0].lower() in {"report", "reported"}:
                tokens = tokens[1:]
            if not tokens:
                return True
            first = tokens[0].lower()
        if first in {"by", "of"}:
            return False
        if first == "as" and len(tokens) > 1 and tokens[1].lower() == "of":
            tokens = tokens[2:]
        elif first in {"at", "during", "for", "from", "in"}:
            tokens = tokens[1:]
        else:
            return False
        if tokens and tokens[0].lower() == "the":
            tokens = tokens[1:]
        if not tokens:
            return False
        temporal = tokens[0].lower()
        return (
            temporal.isdigit()
            or temporal
            in {"date", "fiscal", "fy", "period", "q", "quarter", "year"}
            or bool(re.fullmatch(r"(?:fy|q)\d+", temporal))
        )

    def _has_independent_conjoined_requests(
        self,
        text: str,
        *,
        entity_hints: Iterable[Any] = (),
    ) -> bool:
        for conjunction in re.finditer(
            r"\bas well as\b|\bclauseboundary\b|\bplus\b|\band\b",
            text,
        ):
            left = text[: conjunction.start()].strip()
            right = text[conjunction.end() :].strip()
            if not left or not right:
                continue
            if conjunction.group() in {"and", "plus"} and (
                self._expression_needs_operand(left)
                or (
                    conjunction.group() == "plus"
                    and re.search(r"\b(?:calculate|compute)\b", left)
                )
            ):
                continue
            left_decision = self._classify_original_clause(
                left,
                entity_hints=entity_hints,
            )
            right_decision = self._classify_original_clause(
                right,
                entity_hints=entity_hints,
            )
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

    @staticmethod
    def _expression_needs_operand(left: str) -> bool:
        expression_start = re.search(
            r"\b(?:compare|comparison between|difference between|"
            r"sum(?: of)?|average(?: of)?|mean(?: of)?|median(?: of)?|"
            r"add|subtract|multiply|divide)\b",
            left,
        )
        if expression_start is None:
            return False
        expression_text = left[expression_start.end() :]
        return not re.search(r"\b(?:and|to|versus|vs)\b", expression_text)

    def _apply_explicit_clause_rejections(
        self,
        text: str,
        requests: list[dict[str, Any]],
        decisions: tuple[StructuredFactCapabilityDecision, ...],
        *,
        entity_hints: Iterable[Any] = (),
    ) -> tuple[StructuredFactCapabilityDecision, ...]:
        if "clauseboundary" not in text:
            return decisions
        updated = list(decisions)
        claimed_request_indices: set[int] = set()
        for clause in text.split("clauseboundary"):
            clause_decision = self.classify_request(
                metric_hint=None,
                subquestion=clause,
                entity_hints=entity_hints,
            )
            clause_metric_ids = self._metric_ids_in_text(clause)
            for metric_id in clause_metric_ids:
                index = next(
                    (
                        candidate_index
                        for candidate_index, decision in enumerate(decisions)
                        if candidate_index not in claimed_request_indices
                        and metric_id
                        in (
                            decision.matched_metric_ids
                            or self._matching_hint_metric_ids(
                                _normalize_lookup_text(
                                    requests[candidate_index].get("metric_hint")
                                )
                            )
                        )
                    ),
                    None,
                )
                if index is None:
                    continue
                claimed_request_indices.add(index)
                if (
                    clause_decision.question_class in _EXPLICIT_UNSUPPORTED_CLASSES
                    and updated[index].permitted
                ):
                    updated[index] = clause_decision
        return tuple(updated)

    def _metric_ids_in_text(self, text: str) -> tuple[str, ...]:
        matches = sorted(
            (
                match.start(),
                capability.metric_id,
            )
            for capability in self.capabilities
            for phrase in (*capability.exact_phrases, *capability.aliases)
            for match in [
                re.search(
                    rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])",
                    text,
                )
            ]
            if match is not None
        )
        return tuple(
            dict.fromkeys(
                metric_id
                for _position, metric_id in matches
            )
        )

    def _classify_original_clause(
        self,
        text: str,
        *,
        entity_hints: Iterable[Any] = (),
    ) -> StructuredFactCapabilityDecision:
        decision = self.classify_request(
            metric_hint=None,
            subquestion=text,
            entity_hints=entity_hints,
        )
        if decision.question_class != StructuredFactQuestionClass.UNKNOWN:
            return decision
        matches = self._matching_question_metric_ids(
            text,
            entity_hints=entity_hints,
        )
        if len(matches) == 1:
            return self._supported(
                matches[0],
                "Detected a supported metric only for original-query clause handling.",
            )
        return decision

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
