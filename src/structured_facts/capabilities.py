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
_QUARTERLY_PERIOD_PATTERNS = (
    re.compile(r"\bq\s*[1-4](?:\s+\d{4})?\b"),
    re.compile(r"\b[1-4]q(?:\s*(?:\d{2}|\d{4}))?\b"),
    re.compile(r"\bh\s*[12](?:\s+\d{4})?\b"),
    re.compile(r"\b(?:first|second|1st|2nd) half(?: of)?(?: \d{4})?\b"),
    re.compile(r"\bhalf year\b"),
    re.compile(r"\bquarter(?:ly)?\b"),
    re.compile(r"\b(?:three|six|nine|3|6|9) months?\b"),
    re.compile(r"\binterim\b"),
)
_RATIO_PATTERNS = (
    re.compile(r"\bmargin\b"),
    re.compile(r"\byield\b"),
    re.compile(r"\bratio\b"),
    re.compile(r"\breturn on (?:equity|assets|investment|capital)\b"),
    re.compile(r"\b(?:roe|roa|roic)\b"),
    re.compile(r"\bdebt to equity\b"),
    re.compile(r"\bdebt to (?:assets?|capital)\b"),
    re.compile(r"\b(?:asset|inventory|receivables?|accounts receivable) turnover\b"),
    re.compile(r"\b(?:interest|debt service) coverage\b"),
    re.compile(r"\bequity multiplier\b"),
    re.compile(r"\bdividend payout\b"),
    re.compile(r"\bev\s+ebitda\b"),
    re.compile(r"\b(?:percent|percentage) of\b"),
)
_COMPARISON_PATTERNS = (
    re.compile(r"\bcompare\b"),
    re.compile(r"\bcompared (?:to|with)\b"),
    re.compile(r"\bcomparison\b"),
    re.compile(r"\bversus\b"),
    re.compile(r"\bvs\b"),
    re.compile(r"\bdifference between\b"),
    re.compile(r"\b(?:higher|lower|greater|less) than\b"),
)
_DERIVED_PATTERNS = (
    re.compile(r"\b(?:growth|growth rate|cagr)\b"),
    re.compile(r"\b(?:trend|trajectory)\b"),
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
_NARRATIVE_PATTERNS = (
    re.compile(r"\bwhy\b"),
    re.compile(r"\bexplain\b"),
    re.compile(r"\bwhat drove\b"),
    re.compile(r"\bdrivers?\b"),
    re.compile(r"\breasons?\b"),
)
_AMBIGUOUS_GENERIC_TERMS = frozenset({"cash", "profit", "profitability"})
_QUESTION_PREFIX_BOUNDARIES = frozenset(
    {
        "a",
        "an",
        "are",
        "did",
        "does",
        "drove",
        "driver",
        "drivers",
        "explain",
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
        "why",
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
        fiscal_period: Any = None,
        entity_hints: Iterable[Any] = (),
    ) -> StructuredFactCapabilityDecision:
        shared_entity_hints = tuple(entity_hints)
        metric_text = _normalize_lookup_text(metric_hint)
        question_text = _normalize_lookup_text(subquestion)
        semantic_question_text = question_text
        for value in shared_entity_hints:
            entity_text = _normalize_lookup_text(value)
            if not entity_text:
                continue
            semantic_question_text = re.sub(
                rf"(?<![a-z0-9]){re.escape(entity_text)}(?![a-z0-9])",
                " ",
                semantic_question_text,
            )
        semantic_text = " ".join(
            part for part in (metric_text, semantic_question_text) if part
        )
        raw_text = " ".join(
            part for part in (str(metric_hint or ""), str(subquestion or "")) if part
        )
        period_text = _normalize_lookup_text(fiscal_period)

        if period_text and not re.fullmatch(
            r"(?:fy(?: ?\d{4})?|annual|year|year end|year ended|fiscal year)",
            period_text,
        ):
            return self._rejected(
                StructuredFactQuestionClass.UNSUPPORTED_DERIVED_METRIC,
                "The requested fiscal period is not executable by the annual structured-fact lane.",
            )

        if "%" in raw_text:
            return self._rejected(
                StructuredFactQuestionClass.UNSUPPORTED_RATIO,
                "Percentage ratios require derivation outside the structured-fact lane.",
            )

        if re.search(r"\S\s*[+*/]\s*\S", raw_text) or re.search(
            r"\S\s+-\s+\S",
            raw_text,
        ):
            return self._rejected(
                StructuredFactQuestionClass.UNSUPPORTED_DERIVED_METRIC,
                "Symbolic arithmetic is not executable by the structured-fact lane.",
            )

        if _matches_any(semantic_text, _PER_SHARE_PATTERNS):
            return self._rejected(
                StructuredFactQuestionClass.UNSUPPORTED_PER_SHARE,
                "Per-share metrics are not executable by the structured-fact lane.",
            )
        if _matches_any(semantic_text, _QUARTERLY_PERIOD_PATTERNS):
            return self._rejected(
                StructuredFactQuestionClass.UNSUPPORTED_DERIVED_METRIC,
                "Quarterly periods are not executable by the annual structured-fact lane.",
            )
        if _matches_any(semantic_text, _RATIO_PATTERNS):
            return self._rejected(
                StructuredFactQuestionClass.UNSUPPORTED_RATIO,
                "Ratios, margins, and yields require derivation outside the structured-fact lane.",
            )
        if _matches_any(semantic_text, _COMPARISON_PATTERNS):
            return self._rejected(
                StructuredFactQuestionClass.UNSUPPORTED_COMPARISON,
                "Comparisons requiring derivation are not executable by the structured-fact lane.",
            )
        if _matches_any(semantic_text, _DERIVED_PATTERNS):
            return self._rejected(
                StructuredFactQuestionClass.UNSUPPORTED_DERIVED_METRIC,
                "The requested calculated metric is not a supported registry capability.",
            )

        hint_metric_ids = self._matching_hint_metric_ids(metric_text)
        question_metric_ids = self._matching_question_metric_ids(
            question_text,
            entity_hints=shared_entity_hints,
        )
        if metric_text in _AMBIGUOUS_GENERIC_TERMS:
            return self._ambiguous(
                self._ambiguous_candidate_ids(metric_text),
                "The metric phrase is too broad for deterministic structured execution.",
            )
        if not metric_text and not question_metric_ids:
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
        if _matches_any(semantic_text, _NARRATIVE_PATTERNS):
            return self._rejected(
                StructuredFactQuestionClass.UNSUPPORTED_DERIVED_METRIC,
                "Narrative explanations are not executable by the structured-fact lane.",
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
            return self._rejected(
                StructuredFactQuestionClass.UNKNOWN,
                "The metric hint is not a direct structured-fact capability.",
            )

        question_metric_ids = self._matching_question_metric_ids(
            question_text,
            entity_hints=shared_entity_hints,
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
                fiscal_period=request.get("fiscal_period"),
                entity_hints=shared_entity_hints + (request.get("entity_hint"),),
            )
            for request in request_list
        )
        if not decisions:
            return decisions

        original_source = re.sub(
            r"[;.!?]+|\n+|,\s*(?=also\b)",
            " clauseboundary ",
            str(original_user_query or ""),
            flags=re.IGNORECASE,
        )
        original_text = _normalize_lookup_text(original_source)
        original_text = re.sub(
            r"^(?:clauseboundary )+|(?: clauseboundary)+$",
            "",
            original_text,
        )
        original_decision_source = re.sub(
            r"(?:\s*clauseboundary\s*)+$",
            "",
            original_source,
            flags=re.IGNORECASE,
        )
        if not original_text:
            return decisions
        original_decision = self.classify_request(
            metric_hint=None,
            subquestion=original_decision_source,
            entity_hints=all_entity_hints,
        )
        if original_decision.permitted:
            return decisions
        independent_boundaries = self._independent_conjunctions(
            original_text,
            entity_hints=all_entity_hints,
        )
        if independent_boundaries:
            return self._apply_explicit_clause_rejections(
                original_text,
                request_list,
                decisions,
                independent_boundaries=independent_boundaries,
                entity_hints=all_entity_hints,
            )
        if original_decision.question_class == StructuredFactQuestionClass.UNKNOWN:
            return decisions
        return tuple(original_decision for _request in request_list)

    def classify_uncovered_original_clauses(
        self,
        original_user_query: Any,
        *,
        covered_requests: Iterable[dict[str, Any]] = (),
        entity_hints: Iterable[Any] = (),
    ) -> tuple[tuple[str, StructuredFactCapabilityDecision], ...]:
        original_source = re.sub(
            r"[;.!?]+|\n+|,\s*(?=also\b)",
            " clauseboundary ",
            str(original_user_query or ""),
            flags=re.IGNORECASE,
        )
        original_text = _normalize_lookup_text(original_source)
        original_text = re.sub(
            r"^(?:clauseboundary )+|(?: clauseboundary)+$",
            "",
            original_text,
        )
        if not original_text:
            return ()
        shared_entity_hints = tuple(entity_hints)
        boundaries = self._independent_conjunctions(
            original_text,
            entity_hints=shared_entity_hints,
        )
        if not boundaries:
            return ()
        clauses: list[str] = []
        start = 0
        for boundary in boundaries:
            clauses.append(original_text[start : boundary.start()].strip())
            start = boundary.end()
        clauses.append(original_text[start:].strip())
        covered_request_phrases = tuple(
            (request, phrase)
            for request in covered_requests
            if isinstance(request, dict)
            for phrase in (_normalize_lookup_text(request.get("metric_hint")),)
            if phrase
        )
        entity_phrases = tuple(
            phrase
            for value in shared_entity_hints
            for phrase in (_normalize_lookup_text(value),)
            if phrase
        )
        uncovered_clauses: list[tuple[str, StructuredFactCapabilityDecision]] = []
        for clause in clauses:
            residual = clause
            representation_clause = clause
            for phrase in entity_phrases:
                representation_clause = re.sub(
                    rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])",
                    " ",
                    representation_clause,
                )
            representation_clause = re.sub(
                r"\b(?:at|for|from|in|of)\s*$",
                "",
                representation_clause,
            ).strip()
            for request, phrase in covered_request_phrases:
                if not self._covered_request_represents_clause(
                    representation_clause,
                    request=request,
                    phrase=phrase,
                    entity_hints=shared_entity_hints,
                ):
                    continue
                residual = re.sub(
                    rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])",
                    " ",
                    residual,
                )
            for phrase in entity_phrases:
                residual = re.sub(
                    rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])",
                    " ",
                    residual,
                )
            residual = _normalize_lookup_text(residual)
            meaningful_tokens = {
                token
                for token in residual.split()
                if token not in _QUESTION_PREFIX_BOUNDARIES
                and token
                not in {
                    "and",
                    "as",
                    "at",
                    "clauseboundary",
                    "for",
                    "in",
                    "of",
                    "on",
                    "plus",
                    "to",
                    "well",
                }
                and not token.isdigit()
                and not re.fullmatch(r"fy\d{4}", token)
            }
            if not meaningful_tokens:
                continue
            decision = self._classify_original_clause(
                residual,
                entity_hints=shared_entity_hints,
            )
            if decision.permitted:
                decision = StructuredFactCapabilityDecision(
                    question_class=StructuredFactQuestionClass.UNKNOWN,
                    permitted=False,
                    matched_metric_ids=decision.matched_metric_ids,
                    reason=(
                        "A supported clause omitted from the structured proposal "
                        "requires KB fallback rather than incomplete execution."
                    ),
                )
            uncovered_clauses.append((residual, decision))
        return tuple(uncovered_clauses)

    def _covered_request_represents_clause(
        self,
        clause: str,
        *,
        request: dict[str, Any],
        phrase: str,
        entity_hints: tuple[Any, ...],
    ) -> bool:
        request_decision = self.classify_request(
            metric_hint=request.get("metric_hint"),
            subquestion=request.get("subquestion"),
            fiscal_period=request.get("fiscal_period"),
            entity_hints=entity_hints + (request.get("entity_hint"),),
        )
        connectors = tuple(
            re.finditer(
                r"\bas well as\b|\bclauseboundary\b|\bplus\b|\band\b",
                clause,
            )
        )
        for match in re.finditer(
            rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])",
            clause,
        ):
            segment_start = max(
                (
                    connector.end()
                    for connector in connectors
                    if connector.end() <= match.start()
                ),
                default=0,
            )
            segment_end = min(
                (
                    connector.start()
                    for connector in connectors
                    if connector.start() >= match.end()
                ),
                default=len(clause),
            )
            segment_decision = self.classify_request(
                metric_hint=request.get("metric_hint"),
                subquestion=clause[segment_start:segment_end],
                fiscal_period=request.get("fiscal_period"),
                entity_hints=entity_hints + (request.get("entity_hint"),),
            )
            if (
                segment_decision.question_class == request_decision.question_class
                and segment_decision.permitted == request_decision.permitted
                and segment_decision.matched_metric_ids
                == request_decision.matched_metric_ids
            ):
                return True
        return False

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
        lowered_tokens = tuple(token.lower() for token in tokens)
        for entity_tokens in entity_token_sequences:
            if not entity_tokens or lowered_tokens[-len(entity_tokens) :] != entity_tokens:
                continue
            scaffolding_tokens = lowered_tokens[: -len(entity_tokens)]
            boundary = max(
                (
                    index
                    for index, token in enumerate(scaffolding_tokens)
                    if token in _QUESTION_PREFIX_BOUNDARIES
                ),
                default=-1,
            )
            if not scaffolding_tokens[boundary + 1 :]:
                return True
        boundary = max(
            (
                index
                for index, token in enumerate(tokens)
                if token.lower() in _QUESTION_PREFIX_BOUNDARIES
            ),
            default=-1,
        )
        metric_prefix = lowered_tokens[boundary + 1 :]
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
        temporal_text = " ".join(token.lower() for token in tokens)
        return bool(
            re.fullmatch(
                r"(?:"
                r"(?:\d{4}|fy ?\d+(?: \d{4})?)(?: year end(?:ed)?)?|"
                r"fiscal(?: year)? (?:end(?:ed)? )?(?:\d{4}|fy ?\d+)|"
                r"(?:date|period|year)(?: end(?:ed)?)?(?: \d{4})?"
                r")",
                temporal_text,
            )
        )

    def _independent_conjunctions(
        self,
        text: str,
        *,
        entity_hints: Iterable[Any] = (),
    ) -> tuple[re.Match[str], ...]:
        independent: list[re.Match[str]] = []
        conjunctions = tuple(
            re.finditer(
                r"\bas well as\b|\bclauseboundary\b|\bplus\b|\band\b",
                text,
            )
        )
        for index, conjunction in enumerate(conjunctions):
            left_start = conjunctions[index - 1].end() if index > 0 else 0
            right_end = (
                conjunctions[index + 1].start()
                if index + 1 < len(conjunctions)
                else len(text)
            )
            left = text[left_start : conjunction.start()].strip()
            right = text[conjunction.end() : right_end].strip()
            if not left or not right:
                continue
            if self._conjunction_inside_supported_phrase(text, conjunction):
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
                right_decision.permitted
                or right_explicitly_unsupported
                or right_decision.question_class == StructuredFactQuestionClass.UNKNOWN
                or right_decision.question_class == StructuredFactQuestionClass.AMBIGUOUS
            ):
                independent.append(conjunction)
            elif right_decision.permitted and (
                left_decision.permitted
                or left_explicitly_unsupported
                or left_decision.question_class == StructuredFactQuestionClass.UNKNOWN
                or left_decision.question_class == StructuredFactQuestionClass.AMBIGUOUS
            ):
                independent.append(conjunction)
        return tuple(independent)

    def _conjunction_inside_supported_phrase(
        self,
        text: str,
        conjunction: re.Match[str],
    ) -> bool:
        return any(
            phrase_match.start() < conjunction.start()
            and phrase_match.end() > conjunction.end()
            for capability in self.capabilities
            for phrase in (*capability.exact_phrases, *capability.aliases)
            if " and " in phrase
            for phrase_match in re.finditer(
                rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])",
                text,
            )
        )

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
        independent_boundaries: tuple[re.Match[str], ...],
        entity_hints: Iterable[Any] = (),
    ) -> tuple[StructuredFactCapabilityDecision, ...]:
        clauses: list[str] = []
        start = 0
        for boundary in independent_boundaries:
            clauses.append(text[start : boundary.start()])
            start = boundary.end()
        clauses.append(text[start:])
        updated = list(decisions)
        claimed_request_indices: set[int] = set()
        for clause in clauses:
            clause_decision = self.classify_request(
                metric_hint=None,
                subquestion=clause,
                entity_hints=entity_hints,
            )
            clause_metric_occurrences = self._metric_occurrences_in_text(clause)
            for metric_id, occurrence_year in clause_metric_occurrences:
                matching_indices = [
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
                ]
                if occurrence_year is not None:
                    year_matched_indices = [
                        candidate_index
                        for candidate_index in matching_indices
                        if self._request_fiscal_year(requests[candidate_index])
                        == occurrence_year
                    ]
                    if year_matched_indices:
                        matching_indices = year_matched_indices
                index = next(
                    iter(matching_indices),
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

    @staticmethod
    def _request_fiscal_year(request: dict[str, Any]) -> int | None:
        value = request.get("fiscal_year")
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
        match = re.search(r"\b((?:19|20)\d{2})\b", str(request.get("subquestion") or ""))
        return int(match.group(1)) if match else None

    def _metric_occurrences_in_text(
        self,
        text: str,
    ) -> tuple[tuple[str, int | None], ...]:
        matches = sorted(
            [
                (
                    match.start(),
                    match.end(),
                    capability.metric_id,
                )
                for capability in self.capabilities
                for phrase in (*capability.exact_phrases, *capability.aliases)
                for match in re.finditer(
                    rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])",
                    text,
                )
            ],
            key=lambda item: (item[0], -(item[1] - item[0])),
        )
        selected: list[tuple[int, int, str]] = []
        for position, end, metric_id in matches:
            if any(
                selected_metric_id == metric_id
                and position < selected_end
                and end > selected_start
                for selected_start, selected_end, selected_metric_id in selected
            ):
                continue
            selected.append((position, end, metric_id))
        occurrences: list[tuple[str, int | None]] = []
        for index, (_start, end, metric_id) in enumerate(selected):
            next_start = selected[index + 1][0] if index + 1 < len(selected) else len(text)
            year_match = re.search(r"\b(?:fy\s*)?((?:19|20)\d{2})\b", text[end:next_start])
            occurrences.append(
                (metric_id, int(year_match.group(1)) if year_match else None)
            )
        return tuple(occurrences)

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
