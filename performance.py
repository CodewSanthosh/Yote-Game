"""
Module 6: Performance Analysis & Comparison
Benchmarks all algorithms × enhancement configurations across test scenarios.
Generates CSV data and Matplotlib charts.
"""

import os
import time
import csv
import json
from game_engine import GameState, PLAYER1, PLAYER2, ROWS, COLS, EMPTY
from search_algorithms import (alpha_beta_search, negascout_search, mtdf_search,
                                SearchConfig, ALGORITHM_NAMES)


# ─────────────────────────────────────────────
# Predefined Test Boards
# ─────────────────────────────────────────────

def _create_state(board, current_player, p1_hand, p2_hand):
    """Helper to create a GameState from a board configuration."""
    state = GameState()
    state.board = [row[:] for row in board]
    state.current_player = current_player
    state.pieces_in_hand = {PLAYER1: p1_hand, PLAYER2: p2_hand}
    return state


# Opening boards — early game, many pieces in hand
OPENING_BOARDS = [
    _create_state([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], PLAYER1, 11, 11),
    _create_state([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 2, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], PLAYER2, 10, 10),
    _create_state([
        [1, 0, 0, 0, 2],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0],
        [2, 0, 0, 0, 1],
    ], PLAYER1, 9, 9),
]

# Mid-game boards — mixed placement and movement
MIDGAME_BOARDS = [
    _create_state([
        [1, 0, 2, 0, 1],
        [0, 2, 1, 2, 0],
        [1, 0, 0, 0, 2],
        [0, 1, 2, 1, 0],
        [2, 0, 0, 0, 1],
        [0, 2, 1, 0, 0],
    ], PLAYER1, 4, 4),
    _create_state([
        [0, 1, 0, 2, 0],
        [1, 2, 1, 0, 2],
        [0, 0, 2, 1, 0],
        [2, 1, 0, 2, 1],
        [0, 0, 1, 0, 0],
        [0, 2, 0, 1, 0],
    ], PLAYER2, 3, 3),
    _create_state([
        [2, 0, 1, 0, 2],
        [0, 1, 2, 1, 0],
        [0, 0, 0, 0, 0],
        [1, 2, 0, 2, 1],
        [0, 1, 0, 0, 2],
        [0, 0, 2, 1, 0],
    ], PLAYER1, 3, 3),
]

# End-game boards — few pieces remaining
ENDGAME_BOARDS = [
    _create_state([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 1, 0, 2, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 2, 0],
    ], PLAYER1, 0, 0),
    _create_state([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 2, 0, 1, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 2, 0, 0],
    ], PLAYER2, 0, 1),
    _create_state([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 2, 0, 0],
        [0, 0, 1, 2, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 2],
    ], PLAYER1, 1, 0),
]

ALL_BOARDS = {
    'opening': OPENING_BOARDS,
    'midgame': MIDGAME_BOARDS,
    'endgame': ENDGAME_BOARDS,
}


# ─────────────────────────────────────────────
# Enhancement Configurations
# ─────────────────────────────────────────────

CONFIGS = [
    SearchConfig(False, False, False),   # Base
    SearchConfig(True,  False, False),   # +S
    SearchConfig(False, True,  False),   # +TT
    SearchConfig(False, True,  True),    # +TT+HM
    SearchConfig(True,  True,  False),   # +S+TT
    SearchConfig(True,  True,  True),    # +S+TT+HM
]


# ─────────────────────────────────────────────
# Benchmark Runner
# ─────────────────────────────────────────────

def run_benchmark(max_depth=4, time_limit=30.0, results_dir='results'):
    """Run the full benchmark: 3 algorithms × 6 configs × 9 boards.

    Saves results to CSV and JSON.
    """
    os.makedirs(results_dir, exist_ok=True)

    algorithms = {
        'alpha_beta': alpha_beta_search,
        'negascout': negascout_search,
        'mtdf': mtdf_search,
    }

    all_results = []

    total = len(algorithms) * len(CONFIGS) * sum(len(v) for v in ALL_BOARDS.values())
    current = 0

    for algo_key, algo_fn in algorithms.items():
        for config in CONFIGS:
            # MTD-f requires TT
            if algo_key == 'mtdf' and not config.use_tt:
                effective_config = SearchConfig(config.use_sorting, True, config.use_hash_move)
            else:
                effective_config = SearchConfig(config.use_sorting, config.use_tt, config.use_hash_move)

            for stage_name, boards in ALL_BOARDS.items():
                stage_nodes = []
                stage_times = []
                stage_depths = []
                stage_cutoffs = []
                stage_tt_hits = []

                for board_idx, board_state in enumerate(boards):
                    current += 1
                    label = f"{ALGORITHM_NAMES[algo_key]}{config.label()}"
                    print(f"[{current}/{total}] {label} | {stage_name} board {board_idx + 1}")

                    test_state = board_state.copy()
                    result = algo_fn(test_state, max_depth, effective_config, time_limit)

                    stage_nodes.append(result.nodes_expanded)
                    stage_times.append(result.time_ms)
                    stage_depths.append(result.depth_reached)
                    stage_cutoffs.append(result.ab_cutoffs)
                    stage_tt_hits.append(result.tt_hits)

                    all_results.append({
                        'algorithm': ALGORITHM_NAMES[algo_key],
                        'config': config.label(),
                        'stage': stage_name,
                        'board_index': board_idx,
                        'nodes_expanded': result.nodes_expanded,
                        'time_ms': round(result.time_ms, 1),
                        'depth_reached': result.depth_reached,
                        'ab_cutoffs': result.ab_cutoffs,
                        'tt_hits': result.tt_hits,
                        'branching_factor': round(result.branching_factor(), 2),
                        'best_move': str(result.best_move),
                        'value': result.value,
                    })

                # Print stage averages
                n = len(boards)
                avg_nodes = sum(stage_nodes) / n
                avg_time = sum(stage_times) / n
                avg_depth = sum(stage_depths) / n
                print(f"  → Avg: {avg_nodes:.0f} nodes, {avg_time:.1f}ms, depth {avg_depth:.1f}")

    # Save to CSV
    csv_path = os.path.join(results_dir, 'performance_data.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to {csv_path}")

    # Save to JSON
    json_path = os.path.join(results_dir, 'performance_data.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {json_path}")

    return all_results


def generate_comparison_table(results):
    """Generate a formatted comparison table from benchmark results."""
    # Group by algorithm + config
    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(list))

    for r in results:
        key = f"{r['algorithm']} {r['config']}"
        grouped[key][r['stage']].append(r)

    print("\n" + "=" * 100)
    print("PERFORMANCE COMPARISON TABLE")
    print("=" * 100)
    print(f"{'Algorithm + Config':<30} {'Stage':<10} {'Avg Nodes':>12} {'Avg Time(ms)':>14} {'Avg Depth':>10} {'Avg BF':>8}")
    print("-" * 100)

    for key in sorted(grouped.keys()):
        for stage in ['opening', 'midgame', 'endgame']:
            if stage in grouped[key]:
                entries = grouped[key][stage]
                avg_nodes = sum(e['nodes_expanded'] for e in entries) / len(entries)
                avg_time = sum(e['time_ms'] for e in entries) / len(entries)
                avg_depth = sum(e['depth_reached'] for e in entries) / len(entries)
                avg_bf = sum(e['branching_factor'] for e in entries) / len(entries)
                print(f"{key:<30} {stage:<10} {avg_nodes:>12.1f} {avg_time:>14.1f} {avg_depth:>10.1f} {avg_bf:>8.2f}")
        print()


def generate_charts(results, results_dir='results'):
    """Generate Matplotlib comparison charts."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Matplotlib not available. Skipping chart generation.")
        return

    charts_dir = os.path.join(results_dir, 'charts')
    os.makedirs(charts_dir, exist_ok=True)

    from collections import defaultdict

    # Group results
    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        key = f"{r['algorithm']}\n{r['config']}"
        grouped[key][r['stage']].append(r)

    # --- Chart 1: Average Nodes Expanded ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Average Nodes Expanded (Depth 4)', fontsize=16, fontweight='bold')

    for idx, stage in enumerate(['opening', 'midgame', 'endgame']):
        ax = axes[idx]
        labels = []
        values = []
        for key in sorted(grouped.keys()):
            if stage in grouped[key]:
                entries = grouped[key][stage]
                labels.append(key)
                values.append(sum(e['nodes_expanded'] for e in entries) / len(entries))

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))
        bars = ax.barh(range(len(labels)), values, color=colors)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(stage.capitalize(), fontsize=13)
        ax.set_xlabel('Nodes Expanded')

    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'nodes_expanded.png'), dpi=150)
    plt.close()

    # --- Chart 2: Average Execution Time ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Average Execution Time (ms) — Depth 4', fontsize=16, fontweight='bold')

    for idx, stage in enumerate(['opening', 'midgame', 'endgame']):
        ax = axes[idx]
        labels = []
        values = []
        for key in sorted(grouped.keys()):
            if stage in grouped[key]:
                entries = grouped[key][stage]
                labels.append(key)
                values.append(sum(e['time_ms'] for e in entries) / len(entries))

        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(labels)))
        bars = ax.barh(range(len(labels)), values, color=colors)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(stage.capitalize(), fontsize=13)
        ax.set_xlabel('Time (ms)')

    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'execution_time.png'), dpi=150)
    plt.close()

    # --- Chart 3: Best Performing Summary ---
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Best Configurations by Stage (Fewest Nodes)', fontsize=16, fontweight='bold')

    stages = ['opening', 'midgame', 'endgame']
    best_data = []
    for stage in stages:
        best_key = None
        best_nodes = float('inf')
        for key in grouped:
            if stage in grouped[key]:
                entries = grouped[key][stage]
                avg = sum(e['nodes_expanded'] for e in entries) / len(entries)
                if avg < best_nodes:
                    best_nodes = avg
                    best_key = key
        best_data.append((stage, best_key, best_nodes))

    colors = ['#ffd700', '#00e5ff', '#ff6b6b']
    bars = ax.bar([d[0].capitalize() for d in best_data],
                  [d[2] for d in best_data], color=colors, edgecolor='black')

    for bar, data in zip(bars, best_data):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                data[1].replace('\n', ' '), ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Average Nodes Expanded')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'best_configs.png'), dpi=150)
    plt.close()

    print(f"Charts saved to {charts_dir}/")


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  YOTE GAME — Performance Benchmark")
    print("  3 Algorithms × 6 Configs × 9 Boards")
    print("=" * 60)
    print()

    results = run_benchmark(max_depth=4, time_limit=30.0)
    generate_comparison_table(results)
    generate_charts(results)

    print("\nBenchmark complete!")
