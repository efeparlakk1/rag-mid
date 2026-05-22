# rag-mid

PDF ve Markdown belgelerden soru-cevap yapan bir RAG (Retrieval-Augmented Generation) sistemi. Belgeleri parçalayıp vektör veritabanına yazar; soruları hibrit arama + yeniden sıralama ile bulup local bir LLM ile yanıtlar ve kaynak gösterir.

## Özellikler

- **PDF ve Markdown** kaynak desteği (`.pdf`, `.md`, `.markdown`)
- **Hibrit arama:** dense + sparse vektör (BGE-M3) ve Qdrant üzerinde RRF füzyonu
- **Yeniden sıralama:** cross-encoder reranker (`bge-reranker-v2-m3`)
- **Local LLM:** Ollama (`qwen3.5:9b`) — veri dışarı çıkmaz
- **Kaynak gösterimi:** her cevapta `[Doc N]` atıfları
- **İki arayüz:** terminal (`chat.py`) ve web (`app.py`, canlı akış + adım göstergeleri)

## İşleyiş

```
İndeksleme:  PDF/MD → Markdown → chunk → embedding (dense+sparse) → Qdrant
Sohbet:      soru → embedding → hibrit arama (RRF, top-20) → reranker (top-5) → LLM → cevap
```

## Gereksinimler

- Python 3.10+
- [Ollama](https://ollama.com) (LLM sunucusu)
- (Opsiyonel) CUDA destekli GPU — yoksa CPU kullanılır, daha yavaş çalışır

## Kurulum

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

ollama pull qwen3.5:9b
```

> İlk indeksleme/sohbet sırasında embedding ve reranker modelleri (BGE-M3, bge-reranker-v2-m3) Hugging Face'ten otomatik indirilir.

## Kullanım

### 1) Belgeleri indeksle

```bash
python ingest.py data/dosya.pdf       # tek dosya
python ingest.py data/                 # klasördeki tüm .pdf + .md
python ingest.py data/yeni.md --keep   # mevcut koleksiyona ekle (sıfırlamadan)
```

### 2) Sohbet et

```bash
python chat.py            # terminal arayüzü
# veya
streamlit run app.py      # web arayüzü → http://localhost:8501
```

`chat.py` komutları: `/reset` (hafızayı temizle), `/quit` (çık).

> Qdrant embedded modu dosya kilidi kullanır; aynı anda yalnızca bir süreç (`ingest.py`, `chat.py` veya `app.py`) çalışabilir.

## Proje Yapısı

| Dosya | Görev |
|---|---|
| `config.py` | Tüm ayarlar (model adları, chunk boyutu, top-k vb.) |
| `ingest.py` | İndeksleme: belge → chunk → embedding → Qdrant |
| `chat.py` | Terminal sohbet arayüzü |
| `app.py` | Streamlit web sohbet arayüzü |
| `src/pdf_loader.py` | Kaynağı Markdown'a çevirir (`.pdf`/`.md`) |
| `src/chunker.py` | Markdown'ı başlık + token bazlı parçalara böler |
| `src/embedder.py` | BGE-M3 ile dense + sparse vektör üretir |
| `src/vector_store.py` | Qdrant koleksiyonu: yazma ve hibrit arama |
| `src/reranker.py` | Cross-encoder ile yeniden sıralama |
| `src/retriever.py` | Hibrit arama + reranker orkestrasyonu |
| `src/llm.py` | Ollama LLM istemcisi |
| `src/chatbot.py` | Prompt, hafıza, query rewrite, citation |

## Yapılandırma

Tüm parametreler `config.py` içindedir:

| Ayar | Varsayılan | Açıklama |
|---|---|---|
| `EMBED_MODEL` | `BAAI/bge-m3` | Embedding modeli |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Reranker modeli |
| `LLM_MODEL` | `qwen3.5:9b` | Ollama LLM |
| `CHUNK_TOKENS` | `512` | Chunk boyutu (token) |
| `CHUNK_OVERLAP` | `75` | Chunk örtüşmesi |
| `RRF_LIMIT` | `20` | Hibrit aramadan dönen aday sayısı |
| `TOP_K` | `5` | Reranker sonrası LLM'e giden chunk sayısı |
| `LLM_TEMPERATURE` | `0.1` | LLM sıcaklığı |

Modeli değiştirmek için `config.py` içindeki ilgili satırı güncelle (gerekiyorsa `ollama pull <model>`).
