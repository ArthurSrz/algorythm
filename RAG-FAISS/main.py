import unittest
import os
import numpy as np
from rag import rag_pipeline
from llm import generate_response

def main():
    folder_path = os.path.join("data", "jo")
    documents = []
    filenames = []
    
    for filename in os.listdir(folder_path):
        filenames.append(filename)
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                documents.append(file.read())
    
    query = "How can I register as a volunteer for the Paris 2024 Olympic Games?"
    
    print("Query:", query)
    
    # No documents
    print("\n=== No documents (OpenAI)===")
    response = generate_response(query, "", model="gpt-3.5-turbo")
    print("\nResponse:")
    print("-------------------------")
    print(response)
    
    
    # No documents
    print("\n=== No documents (TinyLlama)===")
    response = generate_response(query, "", model="tinyllama")
    print("\nResponse:")
    print("-------------------------")
    print(response)
    
    
    # Test FAISS Search
    print("\n=== FAISS Search with OpenAI ===")
    response, distances, indices = rag_pipeline(documents, filenames, query, search_method="faiss", k=5, model="gpt-3.5-turbo")
    print("\nResponse:")
    print("-------------------------")
    print(response)
    
    
    # Test Cosine Search
    print("\n=== Cosine Search with OpenAI ===")
    response, similarities, indices = rag_pipeline(documents, filenames, query, search_method="cosine", k=5, model="gpt-3.5-turbo")
    print("\nResponse:")
    print("-------------------------")
    print(response)


    # Test FAISS Search with TinyLlama
    # We use a smaller k value (2) to avoid errors due to the smaller context window of TinyLlama
    print("\n=== FAISS Search with TinyLlama ===")
    response, distances, indices = rag_pipeline(documents, filenames, query, search_method="faiss", k=2, model="tinyllama")
    print("\nResponse:")
    print("-------------------------")
    print(response)
    
    
    # Test Cosine Search with TinyLlama
    print("\n=== Cosine Search with TinyLlama ===")
    response, similarities, indices = rag_pipeline(documents, filenames, query, search_method="cosine", k=2, model="tinyllama")
    print("\nResponse:")
    print("-------------------------")
    print(response)
    
    
        
if __name__ == '__main__':
    main()