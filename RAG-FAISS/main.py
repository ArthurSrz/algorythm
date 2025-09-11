#!/usr/bin/env python
import os
import sys
import argparse
from typing import List, Tuple

from rag import rag_pipeline
from llm import generate_response

def read_folder_texts(folder_path: str) -> Tuple[List[str], List[str]]:
    """Charge tous les fichiers texte d’un dossier, retourne (documents, filenames)."""
    documents, filenames = [], []
    if not os.path.isdir(folder_path):
        return documents, filenames
    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    documents.append(f.read())
                    filenames.append(filename)
            except Exception as e:
                print(f"[WARN] Impossible de lire {file_path}: {e}", file=sys.stderr)
    return documents, filenames


def run_no_docs(query: str, model: str):
    """Génère une réponse sans contexte (aucun document)."""
    print("\n=== No documents ===")
    print(f"Model: {model}")
    resp = generate_response(query, "", model=model)
    print("\nResponse:")
    print("-------------------------")
    print(resp)


def run_rag(documents: List[str], filenames: List[str], query: str,
            search_method: str, k: int, model: str):
    """Exécute le pipeline RAG et affiche les top-k et la réponse."""
    label = "distances" if search_method == "faiss" else "similarities"
    print(f"\n=== RAG ({search_method}) ===")
    print(f"Model: {model} | k={k}")
    response, scores, indices = rag_pipeline(
        documents, filenames, query,
        search_method=search_method, k=k, model=model
    )
    print("\nResponse:")
    print("-------------------------")
    print(response)


def parse_args():
    p = argparse.ArgumentParser(
        description="Demo CLI pour RAG (FAISS / cosine) + LLM (OpenAI GPT-3.5 / TinyLlama)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Comportement principal (un seul run à la fois, sauf --demo)
    p.add_argument("--data_dir", type=str, default=os.path.join("data", "jo"),
                   help="Dossier contenant les fichiers texte à indexer.")
    p.add_argument("--query", type=str,
                   default="How can I register as a volunteer for the Paris 2024 Olympic Games?",
                   help="Question posée au système.")
    p.add_argument("--search", choices=["faiss", "cosine"], default="faiss",
                   help="Méthode de recherche vectorielle.")
    p.add_argument("--k", type=int, default=5, help="Nombre de voisins à récupérer.")
    p.add_argument("--model", type=str, default="gpt-3.5-turbo",
                   help="Modèle de génération ('gpt-3.5-turbo' ou 'tinyllama').")
    p.add_argument("--no_docs", action="store_true",
                   help="Génère une réponse sans contexte (n’utilise aucun document).")

    # Mode démo qui reproduit le main.py original (séquence complète)
    p.add_argument("--demo", action="store_true",
                   help="Exécute la séquence de démonstration : "
                        "no-docs GPT, no-docs TinyLlama, RAG FAISS/Cosine GPT (k=5), "
                        "RAG FAISS/Cosine TinyLlama (k=2).")

    return p.parse_args()


def main():
    args = parse_args()

    if args.demo:
        # Réplique la séquence de main.py (dossier, query codés en dur)
        folder_path = os.path.join("data", "jo")
        documents, filenames = read_folder_texts(folder_path)
        query = "How can I register as a volunteer for the Paris 2024 Olympic Games?"
        print("Query:", query)

        # 1) no docs (OpenAI)
        run_no_docs(query, "gpt-3.5-turbo")
        # 2) no docs (TinyLlama)
        run_no_docs(query, "tinyllama")
        # 3) RAG FAISS (OpenAI, k=5)
        if documents:
            run_rag(documents, filenames, query, "faiss", 5, "gpt-3.5-turbo")
        else:
            print("[INFO] Aucun document trouvé pour FAISS (GPT).")
        # 4) RAG Cosine (OpenAI, k=5)
        if documents:
            run_rag(documents, filenames, query, "cosine", 5, "gpt-3.5-turbo")
        else:
            print("[INFO] Aucun document trouvé pour Cosine (GPT).")
        # 5) RAG FAISS (TinyLlama, k=2)
        if documents:
            run_rag(documents, filenames, query, "faiss", 2, "tinyllama")
        else:
            print("[INFO] Aucun document trouvé pour FAISS (TinyLlama).")
        # 6) RAG Cosine (TinyLlama, k=2)
        if documents:
            run_rag(documents, filenames, query, "cosine", 2, "tinyllama")
        else:
            print("[INFO] Aucun document trouvé pour Cosine (TinyLlama).")
        return

    # Mode « simple run » (un seul appel, configurable)
    if args.no_docs:
        run_no_docs(args.query, args.model)
        return

    documents, filenames = read_folder_texts(args.data_dir)
    if not documents:
        print(f"[WARN] Aucun document lisible trouvé dans {args.data_dir}. "
              f"Exécution en mode 'no_docs'.", file=sys.stderr)
        run_no_docs(args.query, args.model)
        return

    # Ajustement automatique pour TinyLlama (fenêtre plus réduite)
    k = args.k
    if "tinyllama" in args.model.lower() and args.k > 2:
        print(f"[INFO] TinyLlama détecté : réduction de k={args.k} -> k=2 pour limiter le contexte.")
        k = 2

    run_rag(documents, filenames, args.query, args.search, k, args.model)


if __name__ == "__main__":
    main()
