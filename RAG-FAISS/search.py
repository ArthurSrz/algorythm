import faiss
import numpy as np

def faiss_search(embeddings, query_embedding, k=5):
    """
    Performs FAISS-based L2 search.
    :param embeddings: The embeddings to index and search.
    :param query_embedding: The query embedding.
    :param k: The number of nearest neighbors to retrieve.
    :return: Distances and indices of the top-k nearest neighbors.
    """
    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Perform search
    distances, indices = index.search(np.array([query_embedding]), k)
    
    return distances, indices


def cosine_similarity_search(embeddings, query_embedding, k=5):
    """
    Performs classical cosine similarity search.
    :param embeddings: The embeddings to search.
    :param query_embedding: The query embedding.
    :param k: The number of nearest neighbors to retrieve.
    :return: Indices of the top-k most similar embeddings and their cosine similarities.
    """
    # Normalize embeddings and query embedding
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_query = query_embedding / np.linalg.norm(query_embedding)

    # Compute cosine similarities
    similarities = np.dot(normalized_embeddings, normalized_query)
    
    # Get the indices of available documents
    valid_indices = np.argsort(similarities)[::-1]
    
    # Create arrays of size k, filled with padding values
    padded_indices = np.full(k, -1)  # Fill with -1 like FAISS
    padded_similarities = np.full(k, -np.inf)  # Fill with -inf for similarities
    
    # Fill with actual values up to min(k, number of documents)
    num_valid = min(k, len(valid_indices))
    padded_indices[:num_valid] = valid_indices[:num_valid]
    padded_similarities[:num_valid] = similarities[valid_indices[:num_valid]]
    
    # Return in the same format as FAISS: shape (1, k)
    return np.array([padded_similarities]), np.array([padded_indices])
