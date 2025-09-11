---
layout: post
title: "RAG avec FAISS et Instructor Embeddings"
nav_order: 15
categories: code
---

# RAG avec FAISS et Instructor Embeddings

Cette application illustre un pipeline de **Retrieval-Augmented Generation (RAG)** intégrant :
- des **embeddings** calculés avec Sentence-Transformers,
- une **recherche vectorielle** (FAISS ou similarité cosinus),
- une **génération de texte** par modèle de langage (OpenAI GPT-3.5 ou TinyLlama local).

Le fichier principal (`main.py`) permet de contrôler le corpus, la requête, la méthode de recherche et le modèle, sans modifier le code source.

## Installation

### Prérequis

- **Python** ≥ 3.10 (vérifiez la contrainte indiquée dans `pyproject.toml`).
- **Poetry** pour la gestion des dépendances.
- Une clé OpenAI (si vous utilisez le modèle `gpt-3.5-turbo`).

### Étapes

Pour installer l'application, commencez par copier le dépot du livre ([AI-book sur github][ia-gh]),
soit en recupérant l'archive zip depuis github, soit à l'aide de l'outil git:
```bash
git clone https://github.com/iridia-ulb/AI-book
```

- Exportez votre clé OpenAI si vous souhaitez utiliser GPT-3.5 :

```bash
export OPENAI_API_KEY="sk-..."
```

Alternativement, vous pouvez la renseigner dans le fichier `llm.py` à la ligne 7 (`openai.api_key = "Votre clé"`).

## Utilisation

Affichez l’aide complète :

```bash
poetry run python main.py --help
```

## Exemples

### Démonstration complète (séquence intégrée)

```bash
poetry run python main.py --demo
```

Exécute successivement :
- génération sans documents (GPT-3.5 puis TinyLlama),
- RAG FAISS et Cosine avec GPT-3.5 (k=5),
- RAG FAISS et Cosine avec TinyLlama (k=2).

### Requête RAG simple (FAISS + OpenAI)

```bash
poetry run python main.py --data_dir data/jo --query "Explain volunteer onboarding" --search faiss --k 5 --model gpt-3.5-turbo
```

### Requête RAG avec TinyLlama

```bash
poetry run python main_cli.py --query "What are opening ceremony logistics?" --search cosine --k 4  --model tinyllama
```

Pour TinyLlama, le script ajuste automatiquement k=2 pour respecter sa fenêtre de contexte plus réduite.

### Réponse sans documents

```bash
poetry run python main_cli.py --query "Summarize the volunteer program" --no_docs
```

## Paramètres

- `--data_dir DATA_DIR` : dossier contenant les fichiers texte (par défaut data/jo).

- `--query QUERY` : question posée au système.

- `--search {faiss, cosine}` : méthode de recherche vectorielle.

- `--k K` : nombre de voisins à récupérer (par défaut 5).

- `--model MODEL` : modèle de génération (gpt-3.5-turbo ou tinyllama).

- `--no_docs` : ignore le corpus et génère une réponse directe du LLM.

- `--demo` : exécute la séquence de démonstration (voir ci-dessus).

## Résumé

```bash
usage: main_cli.py [--data_dir DATA_DIR] [--query QUERY]
                   [--search {faiss,cosine}] [--k K]
                   [--model MODEL] [--no_docs] [--demo]
```
