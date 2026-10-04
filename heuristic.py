"""
Module 5: Heuristic Evaluation Function
Estimates the quality of a Yote board position at the search depth limit.

Features:
    1. Piece Count     — material advantage (highest weight)
    2. Mobility        — number of legal moves
    3. Positional Value — center control vs edge/corner
    4. Capture Potential — available capture opportunities
"""

from game_engine import ROWS, COLS, EMPTY, GameState

# Orthogonal directions for mobility/capture checks
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Feature weights (tunable)
W_PIECE_COUNT = 10
W_MOBILITY = 3
W_POSITIONAL = 2
W_CAPTURE = 5

# Positional value table — center cells are strategically stronger
POSITION_VALUES = [
    [1, 2, 2, 2, 1],
    [2, 3, 4, 3, 2],
    [2, 4, 5, 4, 2],
    [2, 4, 5, 4, 2],
    [2, 3, 4, 3, 2],
    [1, 2, 2, 2, 1],
]

# Terminal evaluation bounds
WIN_SCORE = 10000
LOSS_SCORE = -10000


def evaluate(state, player):
    """Evaluate the board from the perspective of `player`.

    Returns:
        Positive = favorable, Negative = unfavorable.
    """
    opponent = GameState.opponent(player)

    # --- Terminal checks ---
    p_total = state.total_pieces(player)
    o_total = state.total_pieces(opponent)

    if p_total <= 2:
        return LOSS_SCORE
    if o_total <= 2:
        return WIN_SCORE

    score = 0

    # 1. Piece count difference
    score += W_PIECE_COUNT * (p_total - o_total)

    # 2. Mobility difference
    p_mob = _count_mobility(state, player)
    o_mob = _count_mobility(state, opponent)
    score += W_MOBILITY * (p_mob - o_mob)

    # 3. Positional advantage
    score += W_POSITIONAL * _positional_score(state, player, opponent)

    # 4. Capture potential difference
    p_cap = _count_captures(state, player)
    o_cap = _count_captures(state, opponent)
    score += W_CAPTURE * (p_cap - o_cap)

    return score


def _count_mobility(state, player):
    """Count the number of legal moves available to a player."""
    count = 0

    # Placement moves
    if state.pieces_in_hand[player] > 0:
        for r in range(ROWS):
            for c in range(COLS):
                if state.board[r][c] == EMPTY:
                    count += 1

    # Movement + capture moves
    opponent = GameState.opponent(player)
    for r, c in state.get_player_positions(player):
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if GameState.is_valid_pos(nr, nc):
                if state.board[nr][nc] == EMPTY:
                    count += 1  # Normal move
                elif state.board[nr][nc] == opponent:
                    tr, tc = r + 2 * dr, c + 2 * dc
                    if GameState.is_valid_pos(tr, tc) and state.board[tr][tc] == EMPTY:
                        count += 1  # Capture

    return count


def _positional_score(state, player, opponent):
    """Sum of positional values for the player minus opponent."""
    score = 0
    for r in range(ROWS):
        for c in range(COLS):
            cell = state.board[r][c]
            if cell == player:
                score += POSITION_VALUES[r][c]
            elif cell == opponent:
                score -= POSITION_VALUES[r][c]
    return score


def _count_captures(state, player):
    """Count the number of capture opportunities for a player."""
    opponent = GameState.opponent(player)
    count = 0
    for r, c in state.get_player_positions(player):
        for dr, dc in DIRECTIONS:
            mr, mc = r + dr, c + dc
            if not GameState.is_valid_pos(mr, mc) or state.board[mr][mc] != opponent:
                continue
            tr, tc = r + 2 * dr, c + 2 * dc
            if GameState.is_valid_pos(tr, tc) and state.board[tr][tc] == EMPTY:
                count += 1
    return count
