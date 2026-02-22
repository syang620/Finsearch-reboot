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

def synthesize_answer(
    user_query: str, 
    agent_data: Dict[str, Any], 
    table_name: str, 
    meta: Dict,
    model_name = 'deepseek-r1:14b'
) -> str:
    """
    Synthesizes the final answer using the structured evidence from the agent.
    agent_data contains: 'answer', 'variables', 'reasoning'
    """
    result = agent_data.get("answer")
    variables = agent_data.get("variables", {})
    reasoning = agent_data.get("reasoning", "")
    
    # 1. Format the Evidence for the LLM
    # We create a readable list of what data was used.
    # e.g., "- Net Sales: 14,000\n- Cost: 8,000"
    vars_str = "\n".join([f"- {k}: {v}" for k,v in variables.items()])
    
    unit_hint = "unknown"
    if "millions" in str(meta).lower(): unit_hint = "Millions"
    elif "thousands" in str(meta).lower(): unit_hint = "Thousands"

    prompt = f"""
    You are a Financial Assistant.
    
    User Query: "{user_query}"
    Source Table: "{table_name}"
    
    **Analysis Result:**
    - Calculated Value: {result} (Likely Unit: {unit_hint})
    - Logic Used: {reasoning}
    - Data Points Extracted:
    {vars_str}
    
    **Task:**
    Write a clear, concise response.
    1. State the final answer clearly (add units if applicable).
    2. Provide a "Calculation Breakdown" sentence explaining how it was derived using the Data Points above.
    """
    
    return chat_with_ollama(prompt, model=model_name, as_list=False)
