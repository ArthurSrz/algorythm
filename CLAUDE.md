# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational AI repository ("L'IA en pratique avec Python") from IRIDIA-ULB. Fork of `iridia-ulb/AI-book`. Contains 14 standalone Python projects demonstrating AI techniques across search, learning, and optimization.

**User goal**: Learn AI algorithms in depth by building animated visual representations for each project, writing core algorithm code by hand.

## Repository Structure

Each subdirectory is an **independent Python project** with its own `pyproject.toml`, `main.py`, and dependencies. Projects share no code.

| Project | AI Technique | Viz Framework | Entry Point |
|---------|-------------|---------------|-------------|
| 8Puzzle | A*, RL | Pygame (smooth tile sliding) | `main.py` |
| Cat_or_Dog | CNN | PyQt5 (image + convolution preview) | `main.py` |
| Connect4 | Minimax, MCTS | Pygame (grid + piece drop) | `main.py` |
| DecisionTrees | Entropy, Random Forest | Matplotlib/Seaborn (static plots) | `main.py` |
| NanoGPT | Transformer (PyTorch) | Console only | `main.py` |
| RAG-FAISS | Vector search + LLM | Console only | `main.py` |
| Shortest_Path | A* (uni/bidirectional) | Matplotlib + NetworkX (step-through) | `main.py` |
| Snake | A*, Genetic, Hamiltonian | Pygame (grid + pathfinding overlay) | `main.py` |
| SpamDetector | TF-IDF | WordCloud + Matplotlib | `main.py` |
| SpamDetector2 | NLP classification | Console | `main.py` |
| Sudoku | Backtracking, Genetic | Pygame (tile-by-tile solve anim) | `main.py` |
| Tetris/TetrisGA | Genetic Algorithm | Pygame + pygame_gui | `GUI_RunMenu.py` |
| Tetris/TetrisRL | Reinforcement Learning | Pygame | `main.py` |
| nlp | LDA topic modeling | Matplotlib | `LDA.py` |

## Commands

```bash
cd <ProjectName>
poetry install
poetry run python main.py            # Run with defaults
poetry run python main.py --help     # CLI options (algorithm, mode, depth...)
poetry run python train.py           # Train models (Snake, Cat_or_Dog, TetrisRL)
```

No test suite or linting exists across any project. CI/CD (`.github/workflows/main.yml`) deploys only the Jekyll docs site.

## Existing Visualization Patterns

**Pygame projects** use a frame-based game loop with color-coded algorithm state:
- GREEN/RED/YELLOW/CYAN for explored/frontier/path nodes (Snake, Shortest_Path)
- Delay-paced tile updates for step-by-step solving (Sudoku: 63ms, 8Puzzle: interpolated)
- CLI flags control AI vs human mode (`-p` human, `-x` AI, `--algorithm`, `--heuristic`)

**Matplotlib projects** use static plots or NetworkX interactive step-through.

## Visualization Gaps (Opportunities for Animated Representations)

These algorithms currently lack step-by-step visual animation:
- **Minimax/MCTS** (Connect4): no game tree visualization, no pruning animation
- **Genetic algorithms** (Snake, Sudoku, TetrisGA): only generation counter, no fitness distribution or mutation viz
- **Neural network training** (Cat_or_Dog, TetrisRL, NanoGPT): no loss curves, activation heatmaps, or weight viz
- **RL value functions** (8Puzzle, TetrisRL): no Q-value heatmaps or reward curves
- **Decision tree growth** (DecisionTrees): static tree only, no step-by-step split animation

## Learning Workflow Guidelines

The user learns by writing algorithm code themselves. When working on any project:

1. **Scaffold** the surrounding infrastructure (UI, file I/O, imports, data loading)
2. **Request the user write** the core algorithm logic (search functions, fitness evaluation, network forward pass, loss computation)
3. **Explain** the algorithm's mechanics and trade-offs before and after they code
4. **Build animations** that make the algorithm's internal state visible step-by-step

## Key Tech Stack

Python 3.7–3.13 (varies), Poetry, Pygame, TensorFlow, PyTorch, scikit-learn, NLTK, FAISS, sentence-transformers, NetworkX, matplotlib. Documentation is in French.
