from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

VECTORSTORE_PATH = "vectorstore/"

def get_embeddings():
    """Charge le modèle d'embedding (téléchargé une seule fois)."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )


def build_vectorstore(chunks: list):
    """Crée l'index FAISS à partir des chunks et le sauvegarde."""
    print("⏳ Création de l'index FAISS...")
    
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)
    
    print("✅ Index FAISS sauvegardé.")
    return vectorstore


def load_vectorstore():
    """Charge un index FAISS déjà existant."""
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ Index FAISS chargé.")
    return vectorstore


def get_vectorstore(chunks: list = None):
    """Charge l'index si existe, sinon le crée."""
    if os.path.exists(VECTORSTORE_PATH + "index.faiss"):
        return load_vectorstore()
    
    if chunks is None:
        raise ValueError("Aucun index trouvé. Fournis les chunks pour en créer un.")
    
    return build_vectorstore(chunks)