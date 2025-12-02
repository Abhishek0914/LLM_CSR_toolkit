import re

def clean_text(text: str) -> str:
    """
    Cleans policy / audit report text for embedding.
    """
    text = text.replace("\u00a0", " ")       # non-breaking spaces
    text = re.sub(r'\s+', ' ', text)         # collapse multiple spaces
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # remove weird unicode  
    return text.strip()


def chunk_text(text: str, chunk_size=500, overlap=50):
    """
    Splits text into small overlapping chunks for embeddings.
    """
    words = text.split()
    chunks = []
    i = 0

    while i < len(words):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap

    return chunks
