# Yote Board Game with Adversarial Search AI

A Python & Pygame implementation of the traditional West African strategy board game **Yoté**, featuring adversarial search algorithms, heuristic evaluations, and interactive UI with performance analysis.

---

## 🎮 Overview

**Yoté** is a classic two-player combinatorial strategy board game originating from West Africa (played on a 5×6 grid). Players alternate turns placing their 12 pieces or moving them horizontally/vertically. Captures are made by jumping over an opponent's piece into an adjacent empty space, with the unique rule that a capturing player also chooses and removes an additional opponent's piece from the board.

---

## ✨ Features

- **Interactive Pygame UI**: Dark premium theme, move indicators, capture selection, and animations.
- **Game Modes**:
  - Human vs AI
  - AI vs AI (with live metrics visualization)
  - Human vs Human
- **AI Search Algorithms**:
  - Minimax
  - Alpha-Beta Pruning
  - Iterative Deepening
  - Move Ordering & Transposition Tables (TT)
  - Quiescence Search
- **Performance Evaluation Suite**:
  - Benchmarks node expansion count, search depth, execution time, and win rates across heuristics and algorithms.
  - Generates CSV/JSON logs and analytical charts in `results/`.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Pygame
- Matplotlib, NumPy, Pandas (for performance benchmarks and plotting)

### Installation

```bash
git clone https://github.com/CodewSanthosh/Yote-Game.git
cd Yote-Game
pip install pygame matplotlib numpy pandas
```

### Running the Game

Launch the Pygame interface:
```bash
python main.py
```

### Running Performance Benchmarks

Run algorithm benchmarks and generate charts:
```bash
python performance.py
```

---

## 📁 Project Structure

```
├── main.py                   # Pygame entry point and user interface
├── game_engine.py            # Yoté rules, board representations, and state transitions
├── move_generator.py         # Legal move and capture generation
├── search_algorithms.py      # Minimax, Alpha-Beta, Iterative Deepening
├── heuristic.py              # Heuristic evaluation functions
├── enhancements.py           # Transposition table, move ordering, quiescence
├── performance.py            # Automated testing and benchmarking script
├── results/                  # Benchmarking results and generated charts
└── .gitignore                # Git ignore configuration
```

---

## 📜 Controls

- **Left-Click**: Select piece / Place piece / Move piece / Choose bonus capture
- **Right-Click**: Deselect / Cancel
- **ESC**: Return to Menu
