"""
app.py — Futuristik RAG sohbet arayüzü (Streamlit)

Çalıştırma:
    streamlit run app.py

Özellikler:
  • Koyu / neon "futuristik" tema (cam efektli balonlar, glow başlık)
  • Canlı token-token cevap akışı (LLM streaming)
  • "Thinking" adımları: query rewrite → embedding → hibrit arama → rerank
  • Her cevabın altında kaynak (citation) chip'leri
"""
from __future__ import annotations
import streamlit as st

import config
from src.embedder import Embedder
from src.vector_store import VectorStore, LockedStoreError
from src.reranker import Reranker
from src.retriever import Retriever
from src.llm import LLM
from src.chatbot import Chatbot

# ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG · Neural Console", page_icon="🛰️", layout="centered")

# ── Futuristik tema (CSS) ─────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Space+Grotesk:wght@300;400;500&display=swap');

:root{
  --bg:#070b14; --bg2:#0c1424;
  --cyan:#00e5ff; --violet:#b14dff; --txt:#dbe7ff; --muted:#7e93b8;
  --line:rgba(0,229,255,.18);
}

/* Arka plan: koyu + ince ızgara + neon halo */
.stApp{
  background:
    radial-gradient(1100px 500px at 80% -10%, rgba(177,77,255,.10), transparent 60%),
    radial-gradient(900px 500px at 0% 0%, rgba(0,229,255,.10), transparent 55%),
    linear-gradient(180deg, var(--bg), var(--bg2));
  color:var(--txt);
  font-family:'Space Grotesk', system-ui, sans-serif;
}
.stApp::before{
  content:""; position:fixed; inset:0; pointer-events:none; opacity:.35;
  background-image:linear-gradient(rgba(0,229,255,.05) 1px,transparent 1px),
                   linear-gradient(90deg,rgba(0,229,255,.05) 1px,transparent 1px);
  background-size:38px 38px;
  mask-image:radial-gradient(circle at 50% 30%, #000 0%, transparent 80%);
}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stToolbar"]{display:none;}

/* Hero başlık */
.hero{ text-align:center; margin:.3rem 0 1.3rem; }
.hero h1{
  font-family:'Orbitron',sans-serif; font-weight:700; letter-spacing:3px;
  font-size:2.05rem; margin:0;
  background:linear-gradient(90deg,var(--cyan),var(--violet));
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
  text-shadow:0 0 26px rgba(0,229,255,.25);
}
.hero p{ color:var(--muted); margin:.35rem 0 0; font-size:.84rem; letter-spacing:1px; }
.hero .pulse{ color:#27e08a; }

/* Sohbet balonları (cam efekti) */
[data-testid="stChatMessage"]{
  background:rgba(13,22,40,.55); backdrop-filter:blur(9px);
  border:1px solid var(--line); border-radius:16px;
  padding:.55rem .9rem; margin:.35rem 0;
  box-shadow:0 0 0 1px rgba(255,255,255,.02), 0 8px 30px rgba(0,0,0,.35);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){
  border-color:rgba(177,77,255,.35);
}

/* Sohbet girişi */
[data-testid="stChatInput"]{
  background:rgba(10,16,30,.85); border:1px solid var(--line);
  border-radius:14px; box-shadow:0 0 22px rgba(0,229,255,.08);
}
[data-testid="stChatInput"] textarea{ color:var(--txt)!important; }
[data-testid="stChatInput"]:focus-within{
  border-color:var(--cyan); box-shadow:0 0 26px rgba(0,229,255,.30);
}

/* Thinking adımları (status / expander) */
[data-testid="stExpander"]{
  border:1px dashed var(--line)!important; border-radius:14px!important;
  background:rgba(8,13,24,.6)!important;
}
.step{ font-size:.86rem; margin:.12rem 0; }
.step b{ color:var(--cyan); }

/* Kaynak chip'leri */
.src{
  display:inline-block; margin:.18rem .3rem .18rem 0; padding:.22rem .55rem;
  font-size:.74rem; border-radius:999px; color:var(--txt);
  border:1px solid var(--line); background:rgba(0,229,255,.06);
}
.src .sc{ color:var(--violet); font-weight:600; }

/* Sidebar */
[data-testid="stSidebar"]{ background:rgba(7,11,20,.92); border-right:1px solid var(--line); }
[data-testid="stSidebar"] *{ color:var(--txt); }
.dot{height:9px;width:9px;border-radius:50%;display:inline-block;background:#27e08a;
     box-shadow:0 0 10px #27e08a;margin-right:6px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ── Sistemi bir kez yükle (modeller cache'lenir) ──────────────────────
@st.cache_resource(show_spinner=False)
def load_system():
    store = VectorStore()
    embedder = Embedder()
    reranker = Reranker()
    retriever = Retriever(embedder=embedder, store=store, reranker=reranker)
    llm = LLM()
    bot = Chatbot(retriever=retriever, llm=llm)
    return bot, store, llm


try:
    with st.spinner("⚙️  Nöral çekirdek başlatılıyor (BGE-M3 · reranker · Qwen2.5)…"):
        bot, store, llm = load_system()
except LockedStoreError as e:
    st.error("🔒 " + str(e))
    st.info("Not: Bu uygulamanın başka bir sekmesi/örneği zaten açık olabilir.")
    st.stop()

try:
    n_chunks = store.count()
except Exception:
    n_chunks = 0


# ── Hero ──────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero"><h1>◢ NEURAL RAG CONSOLE ◣</h1>'
    '<p><span class="pulse">●</span> hibrit retrieval · cross-encoder rerank · local LLM</p></div>',
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⬡ Sistem")
    st.markdown(
        f'<span class="dot"></span>online &nbsp;·&nbsp; **{n_chunks}** chunk',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**Boru hattı**")
    st.caption("Embed → Hibrit (RRF) → Reranker → LLM")
    st.markdown(f"**Embedding**  \n`{config.EMBED_MODEL}`")
    st.markdown(f"**Reranker**  \n`{config.RERANKER_MODEL}`")
    st.markdown(f"**LLM**  \n`{config.LLM_MODEL}`  ·  T={config.LLM_TEMPERATURE}")
    st.caption(f"top-k = {config.TOP_K}  ·  hibrit aday = {config.RRF_LIMIT}")
    st.divider()
    if st.button("🗑  Hafızayı temizle", use_container_width=True):
        bot.reset()
        st.session_state.messages = []
        st.rerun()
    if not llm.is_available():
        st.error(f"Ollama/{config.LLM_MODEL} bulunamadı. `ollama serve` çalışıyor mu?")


# ── Yardımcılar ───────────────────────────────────────────────────────
def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    chips = "".join(
        f'<span class="src">[Doc {i}] {s["section"]} '
        f'<span class="sc">{s["score"]:.2f}</span></span>'
        for i, s in enumerate(sources, start=1)
    )
    with st.expander(f"📎 {len(sources)} kaynak"):
        st.markdown(chips, unsafe_allow_html=True)
        for i, s in enumerate(sources, start=1):
            st.caption(f"[Doc {i}] {s['source']} · #chunk{s['chunk_index']}")


# ── Sohbet geçmişi ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🧑‍🚀" if m["role"] == "user" else "🛰️"):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_sources(m.get("sources", []))


# ── Yeni mesaj ────────────────────────────────────────────────────────
if prompt := st.chat_input("Bilgi tabanına sor…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍🚀"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🛰️"):
        # 1) THINKING adımları
        with st.status("⟡ Nöral işlem hattı çalışıyor…", expanded=True) as status:
            standalone = bot.rewrite_query(prompt)
            if standalone != prompt:
                st.markdown(
                    f'<div class="step">✍️ <b>Query rewrite</b> → '
                    f'<i>{standalone}</i></div>', unsafe_allow_html=True)

            def on_step(stage: str, detail: dict) -> None:
                if stage == "embed":
                    st.markdown('<div class="step">🧬 <b>BGE-M3</b> '
                                'dense + sparse embedding…</div>', unsafe_allow_html=True)
                elif stage == "search":
                    st.markdown(f'<div class="step">🔀 <b>Hibrit arama (RRF)</b> → '
                                f'{detail["candidates"]} aday</div>', unsafe_allow_html=True)
                elif stage == "rerank":
                    docs = detail["docs"]
                    st.markdown(f'<div class="step">🎯 <b>Cross-encoder rerank</b> → '
                                f'en iyi {len(docs)} chunk</div>', unsafe_allow_html=True)

            docs = bot.retriever.retrieve(standalone, on_step=on_step)
            status.update(label="✅ Bağlam hazır — yanıt üretiliyor", state="complete")

        # 2) CEVAP (canlı akış)
        if not docs:
            answer = "Sağlanan belgelerde bu sorunun cevabı bulunmuyor."
            st.markdown(answer)
            sources: list[dict] = []
        else:
            system = bot.system_prompt(docs)
            answer = st.write_stream(bot.llm.generate_stream(system, standalone))
            sources = [
                {"section": d.section.strip("* "), "source": d.source,
                 "chunk_index": d.chunk_index, "score": d.score}
                for d in docs
            ]
            render_sources(sources)

        # 3) Hafıza + geçmiş
        bot.remember(prompt, answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
