"""Orta seviye RAG boru hattının modülleri.

Her modül RAG'in bir aşamasına karşılık gelir:
    pdf_loader  -> Aşama 1: PDF'i Markdown'a çevir
    chunker     -> Aşama 2: Markdown'ı anlamlı parçalara böl
    embedder    -> Aşama 3: parçaları dense+sparse vektöre çevir (BGE-M3)
    vector_store-> Aşama 4: Qdrant'a yaz / hibrit ara
    reranker    -> Aşama 5a: cross-encoder ile yeniden sırala
    retriever   -> Aşama 5b: hibrit arama + rerank orkestrasyonu
    llm         -> Aşama 6a: Ollama/Qwen2.5 istemcisi
    chatbot     -> Aşama 6b: prompt + hafıza + query rewrite + citation
"""
