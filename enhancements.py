"""
Module 4: Search Enhancements
Move Ordering, Transposition Table (Zobrist hashing), and Hash Move heuristic.
"""

import random
from game_engine import ROWS, COLS, PLAYER1, PLAYER2, EMPTY

# ─────────────────────────────────────────────
# Transposition Table Flags
# ─────────────────────────────────────────────
EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2


# ─────────────────────────────────────────────
# Zobrist Hashing
# ─────────────────────────────────────────────

class ZobristHash:
    """Zobrist hashing for O(1) incremental board hashing."""

    def __init__(self, seed=42):
        rng = random.Random(seed)
        # Random bitstrings for (row, col, piece_type)
        self.piece_table = [[[rng.getrandbits(64) for _ in range(3)]
                             for _ in range(COLS)] for _ in range(ROWS)]
        # Random bitstrings for pieces-in-hand counts (0..12)
        self.hand_hash = {
            PLAYER1: [rng.getrandbits(64) for _ in range(13)],
            PLAYER2: [rng.getrandbits(64) for _ in range(13)],
        }
        # Random bitstring for side to move
        self.side_hash = rng.getrandbits(64)

    def compute(self, state):
        """Compute the full Zobrist hash for a game state."""
        h = 0
        for r in range(ROWS):
            for c in range(COLS):
                h ^= self.piece_table[r][c][state.board[r][c]]
        h ^= self.hand_hash[PLAYER1][state.pieces_in_hand[PLAYER1]]
        h ^= self.hand_hash[PLAYER2][state.pieces_in_hand[PLAYER2]]
        if state.current_player == PLAYER2:
            h ^= self.side_hash
        return h


# ─────────────────────────────────────────────
# Transposition Table
# ─────────────────────────────────────────────

class TranspositionTable:
    """Hash table storing previously evaluated game positions."""

    def __init__(self, max_size=2_000_000):
        self.table = {}
        self.max_size = max_size
        self.zobrist = ZobristHash()
        self.hits = 0
        self.stores = 0

    def _key(self, state):
        return self.zobrist.compute(state)

    def lookup(self, state, depth, alpha, beta):
        """Look up a state in the transposition table.

        Returns:
            (found, value, best_move)
            - found=True  → value is usable directly
            - found=False → best_move may still be set (hash move)
        """
        entry = self.table.get(self._key(state))
        if entry is None:
            return False, 0, None

        stored_depth, stored_value, flag, best_move = entry

        if stored_depth >= depth:
            if flag == EXACT:
                self.hits += 1
                return True, stored_value, best_move
            elif flag == LOWERBOUND and stored_value >= beta:
                self.hits += 1
                return True, stored_value, best_move
            elif flag == UPPERBOUND and stored_value <= alpha:
                self.hits += 1
                return True, stored_value, best_move

        # Depth insufficient, but hash move is still useful for ordering
        return False, 0, best_move

    def store(self, state, depth, value, flag, best_move=None):
        """Store an evaluation result. Uses depth-preferred replacement."""
        key = self._key(state)
        existing = self.table.get(key)
        if existing is None or existing[0] <= depth:
            self.table[key] = (depth, value, flag, best_move)
            self.stores += 1
        # Simple size management
        if len(self.table) > self.max_size:
            # Remove ~10% of entries (oldest first via iterator)
            to_remove = self.max_size // 10
            keys = list(self.table.keys())[:to_remove]
            for k in keys:
                del self.table[k]

    def get_hash_move(self, state):
        """Retrieve the stored best move for a state, if any."""
        entry = self.table.get(self._key(state))
        return entry[3] if entry else None

    def clear(self):
        """Clear all entries and reset statistics."""
        self.table.clear()
        self.hits = 0
        self.stores = 0


# ─────────────────────────────────────────────
# Move Ordering
# ─────────────────────────────────────────────

def order_moves(moves, state, tt=None, use_hash_move=True):
    """Order moves to maximize pruning efficiency.

    Priority (highest first):
        1. Hash Move from transposition table
        2. Capture moves (material gain)
        3. Center placements
        4. Normal movements toward center

    Args:
        moves:         List of moves to order
        state:         Current game state
        tt:            TranspositionTable (optional)
        use_hash_move: Whether to prioritize the hash move

    Returns:
        Reordered list of moves.
    """
    hash_move = None
    if tt and use_hash_move:
        hash_move = tt.get_hash_move(state)

    scored = [(_score_move(m, hash_move), i, m) for i, m in enumerate(moves)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, _, m in scored]


def _score_move(move, hash_move=None):
    """Assign a priority score to a move for ordering purposes."""
    # Hash move gets absolute priority
    if hash_move is not None and move == hash_move:
        return 100000

    move_type = move[0]

    if move_type == 'capture':
        return 10000  # Captures are very high priority

    if move_type == 'place':
        _, r, c = move
        # Prefer center placements
        center_r, center_c = (ROWS - 1) / 2.0, (COLS - 1) / 2.0
        dist = abs(r - center_r) + abs(c - center_c)
        return 5000 - int(dist * 100)

    if move_type == 'move':
        _, fr, fc, tr, tc = move
        # Slightly prefer moving toward center
        center_r, center_c = (ROWS - 1) / 2.0, (COLS - 1) / 2.0
        old_dist = abs(fr - center_r) + abs(fc - center_c)
        new_dist = abs(tr - center_r) + abs(tc - center_c)
        return 1000 + int((old_dist - new_dist) * 100)

    return 0
