from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_pdfs(data_dir: str) -> list:
    """Charge tous les PDFs depuis un dossier."""
    documents = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(data_dir, filename)
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            documents.extend(docs)
            print(f"✅ Chargé : {filename} ({len(docs)} pages)")
    
    return documents


def split_documents(documents: list) -> list:
    """Découpe les documents en chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # taille d'un chunk en caractères
        chunk_overlap=50,      # overlap pour ne pas perdre le contexte
        separators=["\n\n", "\n", ".", " "]
    )
    
    chunks = splitter.split_documents(documents)
    print(f"📄 Total chunks créés : {len(chunks)}")
    return chunks


def ingest(data_dir: str) -> list:
    """Pipeline complet : charge + découpe."""
    documents = load_pdfs(data_dir)
    
    if not documents:
        print("⚠️ Aucun PDF trouvé dans le dossier.")
        return []
    
    chunks = split_documents(documents)
    return chunks