from embeddings import get_embeddings
from search import faiss_search, cosine_similarity_search
from llm import generate_response

def rag_pipeline(documents, filenames, query, search_method="faiss", k=5, model="gpt-3.5-turbo"):
    """
    RAG pipeline to retrieve relevant documents and generate a response.
    :param documents: List of documents to search.
    :param filenames: List of filenames corresponding to the documents.
    :param query: The query string.
    :param search_method: The search method to use ('faiss' or 'cosine').
    :param k: The number of nearest neighbors to retrieve.
    :param model: The language model to use for response generation.
    :return: Generated response, distances, and indices.
    """
    # Generate embeddings for documents and query
    doc_embeddings = get_embeddings(documents)
    query_embedding = get_embeddings([query])[0]

    # Perform search based on the specified method
    if search_method == "faiss":
        distances, indices = faiss_search(doc_embeddings, query_embedding, k)
    elif search_method == "cosine":
        distances, indices = cosine_similarity_search(doc_embeddings, query_embedding, k)
    else:
        raise ValueError(f"Unsupported search method: {search_method}")

    # Both methods return indices as a 2D array, so flatten it
    indices = indices.flatten()
    distances = distances.flatten()

    # Print the k best matching documents and their distances
    print("\nTop matching documents:")
    print("-------------------------")
    for idx, (doc_idx, distance) in enumerate(zip(indices, distances), 1):
        if doc_idx != -1:  # Skip padding indices
            print(f"{idx}. {filenames[doc_idx]} (distance: {distance:.4f})")

    # Retrieve relevant documents and create context string
    relevant_docs = [documents[i] for i in indices if i != -1]  # Skip padding indices
    context = "\n\n=====\n\n".join(relevant_docs)  # Join documents with separator

    # Generate response using the language model
    response = generate_response(query, context, model)
    return response, distances, indices