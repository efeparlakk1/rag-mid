"""
chat.py — SOHBET arayüzü (online aşama)

İndexlenmiş bilgi tabanı üzerinde interaktif chatbot:
    soru -> (query rewrite) -> hibrit retrieval -> rerank -> LLM -> citation'lı cevap

Kullanım:
    python chat.py

Komutlar:
    /reset   sohbet hafızasını temizle
    /quit    çıkış
"""
from __future__ import annotations
import sys

import config
from src.embedder import Embedder
from src.vector_store import VectorStore, LockedStoreError
from src.reranker import Reranker
from src.retriever import Retriever
from src.llm import LLM
from src.chatbot import Chatbot


def build_chatbot() -> Chatbot:
    print("Modeller yükleniyor (ilk açılış birkaç saniye sürebilir)...\n")
    try:
        store = VectorStore()
    except LockedStoreError as e:
        print(f"\n! {e}")
        sys.exit(1)

    # Koleksiyon boşsa kullanıcıyı yönlendir
    try:
        n = store.count()
    except Exception:
        n = 0
    if n == 0:
        print("! Bilgi tabanı boş. Önce bir PDF indexle:\n"
              "    python ingest.py data/dosya.pdf")
        sys.exit(1)

    embedder = Embedder()
    reranker = Reranker()
    retriever = Retriever(embedder=embedder, store=store, reranker=reranker)

    llm = LLM()
    if not llm.is_available():
        print(f"! Ollama veya '{config.LLM_MODEL}' modeli bulunamadı.\n"
              f"  Ollama'yı başlat:   ollama serve\n"
              f"  Modeli indir:       ollama pull {config.LLM_MODEL}")
        sys.exit(1)

    print(f"✓ Hazır. Bilgi tabanı: {n} chunk | LLM: {config.LLM_MODEL}\n")
    return Chatbot(retriever=retriever, llm=llm)


def main() -> None:
    bot = build_chatbot()
    print("Sorunu yaz ('/quit' ile çık, '/reset' ile hafızayı temizle).\n")

    try:
        while True:
            try:
                query = input("Sen > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGörüşürüz!")
                break

            if not query:
                continue
            if query == "/quit":
                print("Görüşürüz!")
                break
            if query == "/reset":
                bot.reset()
                print("(hafıza temizlendi)\n")
                continue

            result = bot.ask(query)

            print(f"\nBot > {result.answer}\n")
            # Şeffaflık: hangi kaynaklar kullanıldı?
            if result.sources:
                print("  Kaynaklar:")
                for i, d in enumerate(result.sources, start=1):
                    print(f"   [Doc {i}] {d.section}  (skor={d.score:.3f}, "
                          f"{d.source} #chunk{d.chunk_index})")
            if result.rewritten_query != query:
                print(f"  (yeniden yazılan sorgu: {result.rewritten_query})")
            print()
    finally:
        # Embedded Qdrant kilidini bırak
        bot.retriever.store.close()


if __name__ == "__main__":
    main()
