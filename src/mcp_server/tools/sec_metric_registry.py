from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


MetricKind = Literal["atomic", "derived"]
MetricPeriodType = Literal["instant", "duration"]


@dataclass(frozen=True)
class MetricFactCandidate:
    taxonomy: str
    concept_name: str


@dataclass(frozen=True)
class MetricComponentGroup:
    group_id: str
    label: str
    candidates: tuple[MetricFactCandidate, ...]
    required: bool = True


@dataclass(frozen=True)
class MetricDefinition:
    metric_id: str
    label: str
    kind: MetricKind
    period_type: MetricPeriodType
    compute_strategy: Optional[str] = None
    description: str = ""
    unit: Optional[str] = "USD"
    atomic_candidates: tuple[MetricFactCandidate, ...] = field(default_factory=tuple)
    component_groups: tuple[MetricComponentGroup, ...] = field(default_factory=tuple)
    cross_check_candidates: tuple[MetricFactCandidate, ...] = field(default_factory=tuple)
    notes: str = ""


_TOTAL_DEBT_COMPONENT_GROUPS = (
    MetricComponentGroup(
        group_id="current_debt",
        label="Current debt carrying amount",
        candidates=(
            MetricFactCandidate("us-gaap", "LongTermDebtCurrent"),
            MetricFactCandidate("us-gaap", "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths"),
            MetricFactCandidate("us-gaap", "ShortTermBorrowings"),
            MetricFactCandidate("us-gaap", "CommercialPaper"),
        ),
    ),
    MetricComponentGroup(
        group_id="noncurrent_debt",
        label="Noncurrent debt carrying amount",
        candidates=(
            MetricFactCandidate("us-gaap", "LongTermDebtNoncurrent"),
            MetricFactCandidate("us-gaap", "LongTermDebt"),
        ),
    ),
)
_CASH_CANDIDATE = MetricFactCandidate("us-gaap", "CashAndCashEquivalentsAtCarryingValue")


METRIC_REGISTRY: dict[str, MetricDefinition] = {
    "total_debt": MetricDefinition(
        metric_id="total_debt",
        label="Total debt",
        kind="derived",
        period_type="instant",
        compute_strategy="total_debt_carrying_amount",
        description="Interest-bearing debt carrying amount at period end for the anchored annual filing.",
        unit="USD",
        component_groups=_TOTAL_DEBT_COMPONENT_GROUPS,
    ),
    "net_debt": MetricDefinition(
        metric_id="net_debt",
        label="Net debt",
        kind="derived",
        period_type="instant",
        compute_strategy="net_debt_value",
        description="Total debt less cash and cash equivalents at period end.",
        unit="USD",
        component_groups=_TOTAL_DEBT_COMPONENT_GROUPS + (
            MetricComponentGroup(
                group_id="cash_and_cash_equivalents",
                label="Cash and cash equivalents",
                candidates=(_CASH_CANDIDATE,),
            ),
        ),
    ),
    "revenue": MetricDefinition(
        metric_id="revenue",
        label="Revenue",
        kind="atomic",
        period_type="duration",
        description="Annual revenue from the anchored annual filing.",
        unit="USD",
        atomic_candidates=(
            MetricFactCandidate("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
            MetricFactCandidate("us-gaap", "SalesRevenueNet"),
            MetricFactCandidate("us-gaap", "Revenues"),
        ),
    ),
    "gross_profit": MetricDefinition(
        metric_id="gross_profit",
        label="Gross profit",
        kind="atomic",
        period_type="duration",
        description="Annual gross profit from the anchored annual filing.",
        unit="USD",
        atomic_candidates=(MetricFactCandidate("us-gaap", "GrossProfit"),),
    ),
    "operating_income": MetricDefinition(
        metric_id="operating_income",
        label="Operating income",
        kind="atomic",
        period_type="duration",
        description="Annual operating income from the anchored annual filing.",
        unit="USD",
        atomic_candidates=(MetricFactCandidate("us-gaap", "OperatingIncomeLoss"),),
    ),
    "net_income": MetricDefinition(
        metric_id="net_income",
        label="Net income",
        kind="atomic",
        period_type="duration",
        description="Annual net income from the anchored annual filing.",
        unit="USD",
        atomic_candidates=(MetricFactCandidate("us-gaap", "NetIncomeLoss"),),
    ),
    "cash_and_cash_equivalents": MetricDefinition(
        metric_id="cash_and_cash_equivalents",
        label="Cash and cash equivalents",
        kind="atomic",
        period_type="instant",
        description="Cash and cash equivalents at carrying value at period end.",
        unit="USD",
        atomic_candidates=(_CASH_CANDIDATE,),
    ),
    "total_assets": MetricDefinition(
        metric_id="total_assets",
        label="Total assets",
        kind="atomic",
        period_type="instant",
        description="Total assets at period end.",
        unit="USD",
        atomic_candidates=(MetricFactCandidate("us-gaap", "Assets"),),
    ),
    "total_liabilities": MetricDefinition(
        metric_id="total_liabilities",
        label="Total liabilities",
        kind="atomic",
        period_type="instant",
        description="Total liabilities at period end.",
        unit="USD",
        atomic_candidates=(MetricFactCandidate("us-gaap", "Liabilities"),),
    ),
    "stockholders_equity": MetricDefinition(
        metric_id="stockholders_equity",
        label="Stockholders equity",
        kind="atomic",
        period_type="instant",
        description="Stockholders' equity at period end.",
        unit="USD",
        atomic_candidates=(
            MetricFactCandidate("us-gaap", "StockholdersEquity"),
            MetricFactCandidate("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
        ),
    ),
    "operating_cash_flow": MetricDefinition(
        metric_id="operating_cash_flow",
        label="Operating cash flow",
        kind="atomic",
        period_type="duration",
        description="Annual net cash provided by or used in operating activities.",
        unit="USD",
        atomic_candidates=(
            MetricFactCandidate("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
            MetricFactCandidate("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
        ),
    ),
    "capex": MetricDefinition(
        metric_id="capex",
        label="Capital expenditures",
        kind="derived",
        period_type="duration",
        compute_strategy="capex_value",
        description="Cash capital expenditures for the fiscal year.",
        unit="USD",
        component_groups=(
            MetricComponentGroup(
                group_id="primary_cash_capex",
                label="Primary cash capital expenditures",
                candidates=(MetricFactCandidate("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),),
                required=False,
            ),
            MetricComponentGroup(
                group_id="productive_assets_additional",
                label="Additional productive assets cash outflows",
                candidates=(MetricFactCandidate("us-gaap", "PaymentsToAcquireProductiveAssets"),),
                required=False,
            ),
            MetricComponentGroup(
                group_id="fallback_capex_total",
                label="Fallback total capital expenditures",
                candidates=(MetricFactCandidate("us-gaap", "CapitalExpendituresIncurred"),),
                required=False,
            ),
        ),
        notes="Prefer cash payment facts; treat CapitalExpendituresIncurred as a fallback total and avoid summing it with other capex components.",
    ),
}


def get_metric_definition(metric_id: str) -> Optional[MetricDefinition]:
    return METRIC_REGISTRY.get(str(metric_id).strip())
