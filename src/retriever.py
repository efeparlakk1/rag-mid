"""
Aşama 5b — Retriever: tüm arama boru hattının orkestrasyonu

Retrievers.md: "Retriever sistemin tavanıdır." LLM ne kadar iyi olursa olsun,
yanlış context gelirse yanlış cevap üretir. Bu yüzden iki aşamayı birleştiriyoruz:

    sorgu
      │  embed_query (BGE-M3 -> dense + sparse)
      ▼
    [1] Qdrant hibrit arama (dense+sparse, RRF)  ->  top-20 aday   (RECALL)
      ▼
    [2] bge-reranker-v2-m3 cross-encoder         ->  top-5 sonuç   (PRECISION)
      ▼
    LLM'e gidecek nihai bağlam
"""
from __future__ import annotations
from typing import Callable

import config
from src.embedder import Embedder
from src.vector_store import VectorStore, RetrievedDoc
from src.reranker import Reranker

# on_step(stage, detail) -> arayüzün "thinking" adımlarını göstermesi için callback
StepCallback = Callable[[str, dict], None]


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        reranker: Reranker | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.reranker = reranker  # None ise yalnızca hibrit arama yapılır

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        on_step: StepCallback | None = None,
    ) -> list[RetrievedDoc]:
        top_k = top_k or config.TOP_K

        def step(stage: str, detail: dict) -> None:
            if on_step is not None:
                on_step(stage, detail)

        # 1) Sorguyu dense + sparse vektöre çevir
        step("embed", {})
        dense_q, sparse_q = self.embedder.embed_query(query)

        # 2) Hibrit arama: geniş aday havuzu (recall)
        candidates = self.store.hybrid_search(dense_q, sparse_q, limit=config.RRF_LIMIT)
        step("search", {"candidates": len(candidates)})

        # 3) Reranker varsa precision için daralt; yoksa ilk top_k'yı al
        if self.reranker is not None:
            docs = self.reranker.rerank(query, candidates, top_k=top_k)
            step("rerank", {"docs": docs})
            return docs

        docs = candidates[:top_k]
        step("rerank", {"docs": docs})
        return docs
