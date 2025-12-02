import numpy as np
from typing import List

# Optional imports (activate only if available)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class Embedder:
    def __init__(self, backend="ollama", model_name="nomic-embed-text"):
        """
        backend: "ollama" or "huggingface"
        model_name: embedding model name
        """

        self.backend = backend
        self.model_name = model_name

        if backend == "huggingface":
            if not HF_AVAILABLE:
                raise ImportError("SentenceTransformer not installed.")
            self.model = SentenceTransformer(model_name)

        elif backend == "ollama":
            if not OLLAMA_AVAILABLE:
                raise ImportError("Ollama Python package not installed.")

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.backend == "ollama":
            return self._embed_ollama(texts)

        elif self.backend == "huggingface":
            return self._embed_hf(texts)

        else:
            raise ValueError("Invalid backend selected.")

    # --- Ollama embeddings ---
    def _embed_ollama(self, texts):
        vectors = []
        for t in texts:
            r = ollama.embeddings(model=self.model_name, prompt=t)
            vectors.append(r["embedding"])
        return vectors

    # --- HuggingFace embeddings ---
    def _embed_hf(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()
