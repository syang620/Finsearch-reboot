from __future__ import annotations

from typing import Any, Iterable, List, Sequence

from retrieval.ollama_client import chat_with_ollama, parse_llm_list_output


# Back-compat with the notebook name.
parse_llm_list = parse_llm_list_output


def build_query_expansion_prompt(
    user_query: str,
    line_items: str,
) -> str:
    return f"""
You are an expert Financial Analyst and RAG Retrieval Assistant.
Your Goal: Map the User Query to the precise "Line Items" found in the provided SEC taxonomy list.

### 1. RULES & CONSTRAINTS
- **Taxonomy Strictness:** You must ONLY output strings that exist exactly in the "Allowed Line Items" list. Do not invent new terms.
- **Ignore Noise:** Ignore specific company names (e.g., "Apple") and years (e.g., "2024") in the query. Focus only on the financial concept.
- **Directionality Check (CRITICAL):**
  - If the user asks for **Liabilities/Debt** (what the company OWES), DO NOT select asset items like "Corporate debt securities" or "Marketable securities".
  - If the user asks for **Assets**, DO NOT select liability items.
- **Balance vs. Flow:**
  - If the user asks for a balance (e.g., "How much debt"), exclude Cash Flow items like "Repayments of..." or "Proceeds from...".
- **Comprehensive Retrieval:** If the user asks for a total (e.g., "Total Debt"), select the explicit total line (e.g., "Total term debt") AND its major components (e.g., "Commercial paper", "Term debt") to ensure coverage.

### 2. EXAMPLES
User Query: "What is the total revenue?"
Output: ["Net sales", "Total net sales", "Net sales: Products", "Net sales: Services"]

User Query: "How much cash do they have?"
Output: ["Cash", "Cash and cash equivalents", "Marketable securities"]

### 3. ALLOWED LINE ITEMS
{line_items}

### 4. CURRENT TASK
User Query: "{user_query}"
Output:
""".strip()


def expand_query_with_ollama(
    user_query: str,
    *,
    allowed_line_items: str | Sequence[str],
    model: str = "qwen3:4b-instruct",
    include_original: bool = True,
    dedupe: bool = True,
    max_expansions: int | None = None,
    options: dict[str, Any] | None = None,
) -> list[str]:
    """
    Expand a user query into a list of allowed SEC taxonomy line items via Ollama.
    """
    if isinstance(allowed_line_items, str):
        line_items_text = allowed_line_items
    else:
        line_items_text = "\n".join(str(x).strip() for x in allowed_line_items if str(x).strip())

    prompt = build_query_expansion_prompt(user_query=user_query, line_items=line_items_text)
    expansions = chat_with_ollama(prompt, model=model, as_list=True, options=options)
    expansions = [x.strip() for x in expansions if isinstance(x, str) and x.strip()]

    if max_expansions is not None:
        expansions = expansions[:max_expansions]

    out: List[str] = []
    if include_original and user_query.strip():
        out.append(user_query.strip())
    out.extend(expansions)

    if not dedupe:
        return out

    seen: set[str] = set()
    deduped: List[str] = []
    for s in out:
        key = s.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped
