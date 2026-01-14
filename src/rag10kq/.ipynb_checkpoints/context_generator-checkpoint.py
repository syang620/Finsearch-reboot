from __future__ import annotations

from typing import Any, Dict, List, Optional

from rag10kq.utils import chat_with_ollama, load_table_jsonl


def build_generator_prompt(
    USER_QUERY: str,
    RETRIEVED_TABLE: str
):
    return f'''
        You are a senior financial analyst. Your task is to answer the user's question using ONLY the provided financial table context.
        
        **Instructions:**
        1. Identify the specific column in the table that matches the fiscal year/date requested. (Be careful not to mix data from the prior year's column).
        2. Identify the relevant rows or line items from the table. 
        3. If the answer is directly available from the table, no calculation is needed. Otherwise, perform calculations as needed. 
        4. **Output Format:**
           - **Part 1 [Scratchpad]:** List the extracted values and show your math: "Value A + Value B = Sum".
           - **Part 2 [Final Answer]:** Answer the user query with your calculation and your analysis in English. 
        
        **Context:**
        User Query: "{USER_QUERY}"
        Retrieved Table:
        """
        {RETRIEVED_TABLE}
        """

        **Required Output Format:**
        [Scratchpad]
        (Show extraction and math)

        [Final Answer]
        '''


def tables_to_llm_texts(
    scored_tables: List[Dict[str, Any]],
    *,
    tables_dir: str = "../data/chunked",
) -> List[str]:
    """
    Load full table text for each entry returned by score_and_select_tables.
    """
    def _get(table: Any, key: str, default: Any = None) -> Any:
        if isinstance(table, dict):
            return table.get(key, default)
        return getattr(table, key, default)

    def _get_payload(table: Any) -> Dict[str, Any]:
        payload = _get(table, "payload", None)
        return payload if isinstance(payload, dict) else {}

    def _parse_doc_id(doc_id: str) -> tuple[Optional[str], Optional[int]]:
        if not doc_id:
            return None, None
        parts = doc_id.split("::")
        try:
            table_idx = parts.index("table")
            doc_prefix = "::".join(parts[:table_idx])
            table_id = int(parts[table_idx + 1])
            return doc_prefix, table_id
        except (ValueError, IndexError):
            return None, None

    table_texts: List[str] = []
    for entry in scored_tables or []:
        table = entry.get("table", entry)
        payload = _get_payload(table)

        doc_prefix = payload.get("prefix")
        table_id = payload.get("table_index")
        if table_id is not None:
            try:
                table_id = int(table_id)
            except (TypeError, ValueError):
                table_id = None

        if doc_prefix is None or table_id is None:
            doc_id = (
                payload.get("rerank_table_doc_id")
                or payload.get("doc_id")
                or _get(table, "doc_id", "")
            )
            doc_prefix, table_id = _parse_doc_id(doc_id)

        if doc_prefix is None or table_id is None:
            continue

        table_obj = load_table_jsonl(tables_dir, doc_prefix, table_id)
        if isinstance(table_obj, dict):
            text = table_obj.get("text")
            if isinstance(text, str) and text.strip():
                table_texts.append(text.strip())

    return table_texts


def generate_table_response(
    user_query: str,
    scored_tables: List[Dict[str, Any]],
    *,
    tables_dir: str = "../data/chunked",
    model: str = "deepseek-r1:14b",
) -> str:
    tables = tables_to_llm_texts(scored_tables, tables_dir=tables_dir)
    retrieved_table = "\n\n".join(tables)
    generator_prompt = build_generator_prompt(user_query, retrieved_table)
    response = chat_with_ollama(
        generator_prompt,
        model=model,
        as_list=False,
        options={"num_predct": 1024, "temperature": 0.1},
    )
    return response
