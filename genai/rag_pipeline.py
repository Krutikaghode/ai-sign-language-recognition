import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

ASL_KNOWLEDGE_BASE = [
    {"gloss": "HELLO", "meaning": "A standard greeting in ASL.", "category": "greeting"},
    {"gloss": "THANK YOU", "meaning": "Expression of gratitude.", "category": "courtesy"},
    {"gloss": "HELP", "meaning": "Requesting assistance or aid.", "category": "emergency"},
    {"gloss": "PLEASE", "meaning": "Polite request sign.", "category": "courtesy"},
    {"gloss": "NAME YOU WHAT", "meaning": "Asking for someone's name.", "category": "question"},
    {"gloss": "WHERE BATHROOM", "meaning": "Asking for the location of the restroom.", "category": "question"},
    {"gloss": "NEED WATER", "meaning": "Expressing thirst or requiring water.", "category": "need"},
    {"gloss": "FEEL HAPPY", "meaning": "Expressing positive emotional state.", "category": "emotion"},
    {"gloss": "FEEL SAD", "meaning": "Expressing negative emotional state.", "category": "emotion"},
    {"gloss": "GOOD MORNING", "meaning": "Morning greeting.", "category": "greeting"}
]

class ASLRagRetriever:
    """
    RAG Retriever for ASL knowledge base using Hugging Face sentence-transformers
    (all-MiniLM-L6-v2) and FAISS IndexFlatIP.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", knowledge_base=None):
        self.model_name = model_name
        self.knowledge_base = knowledge_base if knowledge_base is not None else ASL_KNOWLEDGE_BASE
        self.encoder = SentenceTransformer(self.model_name)
        
        self.texts = [f"Gloss: {item['gloss']} | Meaning: {item['meaning']}" for item in self.knowledge_base]
        
        embeddings = self.encoder.encode(self.texts, convert_to_numpy=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype(np.float32))

    def retrieve(self, query: str, top_k: int = 3):
        query_vector = self.encoder.encode([query], convert_to_numpy=True)
        query_vector = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)
        
        distances, indices = self.index.search(query_vector.astype(np.float32), top_k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.knowledge_base):
                item = self.knowledge_base[idx].copy()
                item['score'] = float(dist)
                results.append(item)
        return results

if __name__ == "__main__":
    retriever = ASLRagRetriever()
    res = retriever.retrieve("How to ask for help?")
    print(res)
