from groq import Groq
from rag.embeddings import get_vectorstore
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """Tu es un assistant médical spécialisé.
Réponds en priorité à partir du contexte fourni en citant toujours la source.
Si l'information n'est pas dans le contexte, tu peux utiliser tes connaissances 
médicales générales mais indique clairement :
'⚠️ Cette information ne provient pas de vos documents.'
Ne fais jamais d'inventions médicales.
NE cite PAS les sources directement dans ta réponse (pas de [source: ...] entre crochets).
À la fin de ta réponse, liste les sources utilisées en gras avec le format :
**Sources :**"""


def format_context(docs: list) -> str:
    """Formate les chunks récupérés en contexte lisible pour le LLM."""
    context = ""
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "inconnu")
        page = doc.metadata.get("page", 0) + 1
        context += f"\n--- Document {i+1} [source: {source}, page {page}] ---\n"
        context += doc.page_content
        context += "\n"
    return context


def ask(question: str, k: int = 4) -> dict:
    """
    Pipeline complet :
    1. Recherche les chunks pertinents
    2. Construit le prompt
    3. Envoie à Groq
    4. Retourne la réponse + les sources
    """
    # 1. Retrieval
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(question, k=k)

    # 2. Formatage du contexte
    context = format_context(docs)

    # 3. Construction du prompt
    user_prompt = f"""Contexte extrait des documents médicaux :
{context}

Question : {question}

Réponds en français en citant toujours tes sources."""

    # 4. Appel à Groq
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,  # low = réponses plus précises et moins créatives
    )

    answer = response.choices[0].message.content

    # 5. Retourner réponse + sources utilisées
    sources = []
    for doc in docs:
        sources.append({
            "fichier": doc.metadata.get("source", "inconnu"),
            "page": doc.metadata.get("page", 0) + 1,
            "extrait": doc.page_content[:150] + "..."
        })

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }
