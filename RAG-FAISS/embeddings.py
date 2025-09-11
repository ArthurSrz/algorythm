import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from functools import lru_cache

def get_embeddings(texts):
    """ 
    Get the embeddings of the given texts using a sentence transformer model.
    :param texts: The texts to embed. Can be a single string or a list of strings.
    :return: A numpy array of embeddings, one per text.
    """
    # Convert single string to list for consistent processing
    if isinstance(texts, str):
        texts = [texts]
    
    # Compute embeddings with progress bar
    embeddings = []
    for text in tqdm(texts, desc="Computing embeddings"):
        # Use the cached version for single text items
        embedding = get_single_embedding(text)
        embeddings.append(embedding)
    
    return np.array(embeddings)

# Create a cached function for single text items, to avoid re-initializing the model
@lru_cache(maxsize=1024)
def get_single_embedding(text):
    """
    Get embedding for a single text string (can be cached)
    """
    model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    return model.encode(text, show_progress_bar=False)