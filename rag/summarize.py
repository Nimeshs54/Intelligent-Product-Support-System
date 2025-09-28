# rag/summarize.py
"""
LLM-optional answer synthesis.

- If OPENAI_API_KEY is present, uses LangChain + OpenAI to synthesize a response.
- Otherwise, falls back to a deterministic extractive summary (TextRank via sumy),
  so the system works offline and for recruiters without keys.
"""
from __future__ import annotations

import os
import logging
from typing import List, Dict

# -------- Extractive fallback (TextRank) --------
from sumy.parsers import plaintext
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.tokenizers import Tokenizer

_USE_LLM = bool(os.environ.get("OPENAI_API_KEY"))
if _USE_LLM:
    try:
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
    except Exception as e:
        logging.warning(f"[rag.summarize] LangChain not available: {e}")
        _USE_LLM = False

SYSTEM_PROMPT = (
    "You are a senior product support engineer. Write a concise, step-by-step resolution "
    "for the user's ticket using ONLY the provided context. If steps depend on product or version, "
    "say so explicitly. Include: probable cause, steps to validate, steps to fix, and a short "
    "'If this doesn't work' section. Keep it under 250 words. Do not invent facts."
)

def _fallback_extractive(texts: List[str], max_sentences: int = 6) -> str:
    txt = "\n\n".join([t for t in texts if t])
    parser = plaintext.PlaintextParser.from_string(txt, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    sents = summarizer(parser.document, max_sentences)
    return " ".join(str(s) for s in sents)

def _format_context(hits: List[Dict]) -> str:
    blocks = []
    for h in hits:
        blk = f"""[DOC {h.get('doc_id')}]
product={h.get('product')} category={h.get('category')}::{h.get('subcategory')}
resolution_code={h.get('resolution_code')} quality={h.get('quality')}
---
{(h.get('preview') or '').strip()}"""
        blocks.append(blk)
    return "\n\n".join(blocks)

def synthesize(ticket: Dict, hits: List[Dict]) -> Dict:
    """Return {"answer": str, "used_llm": bool}"""
    user_desc = ((ticket.get("subject") or "") + "\n" + (ticket.get("description") or "")).strip()
    context = _format_context(hits)

    if not _USE_LLM:
        answer = _fallback_extractive([user_desc, context], max_sentences=6)
        return {"answer": answer, "used_llm": False}

    # LLM path
    llm = ChatOpenAI(model=os.environ.get("RAG_MODEL", "gpt-4o-mini"), temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Ticket:\n{ticket}\n\nContext:\n{context}\n\nWrite the resolution.")
    ])
    chain = prompt | llm
    resp = chain.invoke({"ticket": user_desc, "context": context})
    content = resp.content if hasattr(resp, "content") else str(resp)
    return {"answer": content, "used_llm": True}
