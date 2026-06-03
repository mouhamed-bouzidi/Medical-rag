import streamlit as st
from rag.ingestion import ingest
from rag.embeddings import get_vectorstore, build_vectorstore
from rag.chain import ask
import os

st.set_page_config(
    page_title="Chatbot Médical",
    layout="wide"
)

st.title(" Chatbot Médical sur Documents")
st.caption("Posez vos questions sur vos documents médicaux. Les réponses sont toujours sourcées.")

# --- Sidebar : upload de PDFs ---
with st.sidebar:
    st.header("📂 Documents médicaux")

    uploaded_files = st.file_uploader(
        "Uploader vos PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        os.makedirs("data", exist_ok=True)
    for f in uploaded_files:
        dest = f"data/{f.name}"
        if not os.path.exists(dest):  # n'écrase pas si déjà présent
            with open(dest, "wb") as out:
                out.write(f.read())
    st.success(f"{len(uploaded_files)} fichier(s) uploadé(s)")

    
    if st.button(" Indexer les documents", use_container_width=True):
        with st.spinner("Chargement et indexation en cours..."):
            chunks = ingest("data/")
            if chunks:
                build_vectorstore(chunks)
                st.success(f" {len(chunks)} chunks indexés !")
            else:
                st.error("Aucun PDF trouvé dans data/")

    st.divider()
    st.caption("Les réponses sont générées uniquement à partir de vos documents.")

# --- Historique du chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher l'historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📚 Sources utilisées"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['fichier']}** — page {s['page']}")
                    st.caption(s["extrait"])

# --- Input utilisateur ---
question = st.chat_input("Posez votre question médicale...")

if question:
    # Afficher la question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Générer la réponse
    with st.chat_message("assistant"):
        with st.spinner("Recherche dans les documents..."):
            try:
                result = ask(question)
                st.markdown(result["answer"])
                with st.expander("📚 Sources utilisées"):
                    for s in result['sources']:
                        st.markdown(f"**{s['fichier']}** — page {s['page']}")
                        st.caption(s["extrait"])

            except Exception as e:
                st.error(f"Erreur : {str(e)}")