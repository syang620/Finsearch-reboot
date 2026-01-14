from __future__ import annotations
import re
from typing import Optional, Tuple, Dict

METRIC_ALIASES = {
    "net sales": ["net sales","revenue","revenues","sales"],
    "gross margin": ["gross margin"],
    "operating income": ["operating income","operating profit"],
    "eps": ["earnings per share","eps","diluted earnings per share","diluted eps"]
}

def classify(question: str) -> Dict[str, Optional[str]]:
    q = question.lower()
    scope = "narrative"
    period = None
    pref_form = None
    if re.search(r"\bfy\b|\bfiscal year\b|year ended", q):
        scope = "annual_metric"
        pref_form = "10-K"
    if re.search(r"\bq[1-4]\b|quarter|three months ended", q):
        scope = "quarter_metric"
        pref_form = "10-Q"
    # find year
    m = re.search(r"20\d{2}", q)
    if m: period = m.group(0)
    # find metric
    metric = None
    for key, aliases in METRIC_ALIASES.items():
        for a in aliases:
            if a in q:
                metric = key
                break
        if metric: break
    return {"scope": scope, "period_hint": period, "preferred_form": pref_form, "metric": metric}
