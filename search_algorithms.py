"""
Module 3: Adversarial Search Algorithms
Implements Alpha-Beta Pruning, NegaScout, and MTD-f.

All algorithms use the Negamax framework and support iterative deepening
with a configurable time limit (default 30 s per depth).
"""

import time
from game_engine import GameState
from move_generator import generate_moves
from heuristic import evaluate
from enhancements import (TranspositionTable, order_moves,
                           EXACT, LOWERBOUND, UPPERBOUND)

INF = float('inf')


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

class SearchResult:
    """Aggregated result of a search operation."""

    def __init__(self):
        self.best_move = None
        self.value = 0
        self.nodes_expanded = 0
        self.depth_reached = 0
        self.time_ms = 0.0
        self.ab_cutoffs = 0
        self.tt_hits = 0
        self.tt_stores = 0

    def branching_factor(self):
        """Compute the effective branching factor."""
        if self.depth_reached <= 0 or self.nodes_expanded <= 1:
            return 0.0
        return self.nodes_expanded ** (1.0 / self.depth_reached)

    def __repr__(self):
        return (f"SearchResult(move={self.best_move}, val={self.value}, "
                f"nodes={self.nodes_expanded}, depth={self.depth_reached}, "
                f"time={self.time_ms:.1f}ms, cuts={self.ab_cutoffs}, "
                f"tt_hits={self.tt_hits})")


class SearchConfig:
    """Configuration for search enhancements."""

    def __init__(self, use_sorting=False, use_tt=False, use_hash_move=False):
        self.use_sorting = use_sorting
        self.use_tt = use_tt
        self.use_hash_move = use_hash_move

    def label(self):
        parts = []
        if self.use_sorting:
            parts.append("S")
        if self.use_tt:
            parts.append("TT")
        if self.use_hash_move:
            parts.append("HM")
        return "+" + "+".join(parts) if parts else "(base)"

    def __repr__(self):
        return f"SearchConfig({self.label()})"


# ─────────────────────────────────────────────
# Internal: Ordered Move Retrieval
# ─────────────────────────────────────────────

def _get_ordered_moves(state, config, tt):
    """Generate moves with optional ordering applied."""
    moves = generate_moves(state)
    if not moves:
        return moves

    if config.use_sorting:
        moves = order_moves(moves, state,
                            tt if config.use_hash_move else None,
                            config.use_hash_move)
    elif config.use_hash_move and tt:
        hm = tt.get_hash_move(state)
        if hm and hm in moves:
            moves.remove(hm)
            moves.insert(0, hm)

    return moves


# ═════════════════════════════════════════════
# ALGORITHM 1: Alpha-Beta Pruning (Negamax)
# ═════════════════════════════════════════════

def _ab_negamax(state, depth, alpha, beta, config, tt, stats, deadline):
    """Recursive Negamax with Alpha-Beta pruning."""
    stats[0] += 1  # nodes_expanded

    if time.time() >= deadline:
        return evaluate(state, state.current_player)

    if state.game_over or depth == 0:
        return evaluate(state, state.current_player)

    # Transposition table lookup
    orig_alpha = alpha
    hash_move = None
    if config.use_tt and tt:
        found, tt_val, hash_move = tt.lookup(state, depth, alpha, beta)
        if found:
            return tt_val

    # Generate and order moves
    moves = generate_moves(state)
    if not moves:
        return -10000  # No legal moves = loss

    if config.use_sorting:
        moves = order_moves(moves, state,
                            tt if config.use_hash_move else None,
                            config.use_hash_move)
    elif config.use_hash_move and hash_move and hash_move in moves:
        moves.remove(hash_move)
        moves.insert(0, hash_move)

    best_value = -INF
    best_move = moves[0]

    for move in moves:
        child = state.apply_move_and_switch(move)
        val = -_ab_negamax(child, depth - 1, -beta, -alpha,
                           config, tt, stats, deadline)

        if val > best_value:
            best_value = val
            best_move = move

        alpha = max(alpha, val)
        if alpha >= beta:
            stats[1] += 1  # ab_cutoffs
            break

    # Transposition table store
    if config.use_tt and tt:
        if best_value <= orig_alpha:
            flag = UPPERBOUND
        elif best_value >= beta:
            flag = LOWERBOUND
        else:
            flag = EXACT
        tt.store(state, depth, best_value, flag, best_move)

    return best_value


def alpha_beta_search(state, max_depth, config=None, time_limit=30.0):
    """Iterative-deepening Alpha-Beta search.

    Returns:
        SearchResult with best move and performance statistics.
    """
    if config is None:
        config = SearchConfig()

    tt = TranspositionTable() if config.use_tt else None
    result = SearchResult()
    start = time.time()
    deadline = start + time_limit

    for depth in range(1, max_depth + 1):
        if time.time() >= deadline:
            break

        # Root-level search: iterate over root moves to track best_move
        moves = generate_moves(state)
        if not moves:
            break

        if config.use_sorting:
            moves = order_moves(moves, state,
                                tt if config.use_hash_move else None,
                                config.use_hash_move)
        elif config.use_hash_move and tt:
            hm = tt.get_hash_move(state)
            if hm and hm in moves:
                moves.remove(hm)
                moves.insert(0, hm)

        # stats: [nodes_expanded, ab_cutoffs]
        stats = [0, 0]
        best_value = -INF
        best_move = moves[0]

        for move in moves:
            child = state.apply_move_and_switch(move)
            stats[0] += 1
            val = -_ab_negamax(child, depth - 1, -INF, -best_value,
                               config, tt, stats, deadline)
            if val > best_value:
                best_value = val
                best_move = move

        if config.use_tt and tt:
            tt.store(state, depth, best_value, EXACT, best_move)

        result.nodes_expanded += stats[0]
        result.ab_cutoffs += stats[1]
        result.depth_reached = depth
        result.value = best_value
        result.best_move = best_move

    result.time_ms = (time.time() - start) * 1000
    if tt:
        result.tt_hits = tt.hits
        result.tt_stores = tt.stores
    return result


# ═════════════════════════════════════════════
# ALGORITHM 2: NegaScout (Principal Variation Search)
# ═════════════════════════════════════════════

def _negascout(state, depth, alpha, beta, config, tt, stats, deadline):
    """Recursive NegaScout search."""
    stats[0] += 1

    if time.time() >= deadline:
        return evaluate(state, state.current_player)

    if state.game_over or depth == 0:
        return evaluate(state, state.current_player)

    # TT lookup
    orig_alpha = alpha
    hash_move = None
    if config.use_tt and tt:
        found, tt_val, hash_move = tt.lookup(state, depth, alpha, beta)
        if found:
            return tt_val

    moves = generate_moves(state)
    if not moves:
        return -10000

    # Ordering (critical for NegaScout)
    if config.use_sorting:
        moves = order_moves(moves, state,
                            tt if config.use_hash_move else None,
                            config.use_hash_move)
    elif config.use_hash_move and hash_move and hash_move in moves:
        moves.remove(hash_move)
        moves.insert(0, hash_move)

    # Search first move with full window
    child = state.apply_move_and_switch(moves[0])
    best_value = -_negascout(child, depth - 1, -beta, -alpha,
                              config, tt, stats, deadline)
    best_move = moves[0]

    if best_value >= beta:
        stats[1] += 1
        if config.use_tt and tt:
            tt.store(state, depth, best_value, LOWERBOUND, best_move)
        return best_value

    if best_value > alpha:
        alpha = best_value

    # Remaining moves with null window, re-search if needed
    for i in range(1, len(moves)):
        move = moves[i]
        child = state.apply_move_and_switch(move)

        # Null-window search
        val = -_negascout(child, depth - 1, -alpha - 1, -alpha,
                          config, tt, stats, deadline)

        # Failed high → re-search with full window
        if alpha < val < beta:
            val = -_negascout(child, depth - 1, -beta, -val,
                              config, tt, stats, deadline)

        if val > best_value:
            best_value = val
            best_move = move

        alpha = max(alpha, val)
        if alpha >= beta:
            stats[1] += 1
            break

    # TT store
    if config.use_tt and tt:
        if best_value <= orig_alpha:
            flag = UPPERBOUND
        elif best_value >= beta:
            flag = LOWERBOUND
        else:
            flag = EXACT
        tt.store(state, depth, best_value, flag, best_move)

    return best_value


def negascout_search(state, max_depth, config=None, time_limit=30.0):
    """Iterative-deepening NegaScout search."""
    if config is None:
        config = SearchConfig()

    tt = TranspositionTable() if config.use_tt else None
    result = SearchResult()
    start = time.time()
    deadline = start + time_limit

    for depth in range(1, max_depth + 1):
        if time.time() >= deadline:
            break

        moves = generate_moves(state)
        if not moves:
            break

        if config.use_sorting:
            moves = order_moves(moves, state,
                                tt if config.use_hash_move else None,
                                config.use_hash_move)
        elif config.use_hash_move and tt:
            hm = tt.get_hash_move(state)
            if hm and hm in moves:
                moves.remove(hm)
                moves.insert(0, hm)

        stats = [0, 0]

        # First root move — full window
        child = state.apply_move_and_switch(moves[0])
        stats[0] += 1
        best_value = -_negascout(child, depth - 1, -INF, INF,
                                  config, tt, stats, deadline)
        best_move = moves[0]
        alpha = best_value

        # Remaining root moves — null window, then re-search
        for i in range(1, len(moves)):
            move = moves[i]
            child = state.apply_move_and_switch(move)
            stats[0] += 1

            val = -_negascout(child, depth - 1, -alpha - 1, -alpha,
                              config, tt, stats, deadline)
            if val > alpha:
                val = -_negascout(child, depth - 1, -INF, -val,
                                  config, tt, stats, deadline)

            if val > best_value:
                best_value = val
                best_move = move
                alpha = val

        if config.use_tt and tt:
            tt.store(state, depth, best_value, EXACT, best_move)

        result.nodes_expanded += stats[0]
        result.ab_cutoffs += stats[1]
        result.depth_reached = depth
        result.value = best_value
        result.best_move = best_move

    result.time_ms = (time.time() - start) * 1000
    if tt:
        result.tt_hits = tt.hits
        result.tt_stores = tt.stores
    return result


# ═════════════════════════════════════════════
# ALGORITHM 3: MTD-f (Memory-enhanced Test Driver)
# ═════════════════════════════════════════════

def _mtdf(state, f, depth, config, tt, stats, deadline):
    """Single MTD-f iteration: converge on minimax value via null-window searches."""
    g = f
    upper = INF
    lower = -INF

    while lower < upper:
        if time.time() >= deadline:
            break

        beta = max(g, lower + 1)
        g = _ab_negamax(state, depth, beta - 1, beta,
                        config, tt, stats, deadline)

        if g < beta:
            upper = g
        else:
            lower = g

    return g


def mtdf_search(state, max_depth, config=None, time_limit=30.0):
    """Iterative-deepening MTD-f search.

    MTD-f requires a transposition table — it is forced on.
    """
    if config is None:
        config = SearchConfig(use_tt=True)
    config.use_tt = True  # MTD-f REQUIRES TT

    tt = TranspositionTable()
    result = SearchResult()
    start = time.time()
    deadline = start + time_limit

    first_guess = 0

    for depth in range(1, max_depth + 1):
        if time.time() >= deadline:
            break

        stats = [0, 0]
        value = _mtdf(state, first_guess, depth, config, tt, stats, deadline)
        first_guess = value

        # Retrieve best move from TT (MTD-f stores it during search)
        best_move = tt.get_hash_move(state)

        result.nodes_expanded += stats[0]
        result.ab_cutoffs += stats[1]
        result.depth_reached = depth
        result.value = value
        if best_move:
            result.best_move = best_move

    result.time_ms = (time.time() - start) * 1000
    result.tt_hits = tt.hits
    result.tt_stores = tt.stores
    return result


# ─────────────────────────────────────────────
# Unified Search Interface
# ─────────────────────────────────────────────

ALGORITHMS = {
    'alpha_beta': alpha_beta_search,
    'negascout': negascout_search,
    'mtdf': mtdf_search,
}

ALGORITHM_NAMES = {
    'alpha_beta': 'Alpha-Beta',
    'negascout': 'NegaScout',
    'mtdf': 'MTD-f',
}


def search(state, algorithm='alpha_beta', max_depth=4, config=None, time_limit=30.0):
    """Unified search interface.

    Args:
        state:      Current GameState
        algorithm:  'alpha_beta', 'negascout', or 'mtdf'
        max_depth:  Maximum search depth
        config:     SearchConfig (optional)
        time_limit: Time limit in seconds

    Returns:
        SearchResult
    """
    search_fn = ALGORITHMS.get(algorithm, alpha_beta_search)
    return search_fn(state, max_depth, config, time_limit)
