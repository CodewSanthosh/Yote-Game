"""
Module 2: Move Generation & State Management
Generates all legal moves from a given Yote game state.

Move types:
    ('place', row, col)                                    — place a piece from hand
    ('move', from_r, from_c, to_r, to_c)                  — move to adjacent empty cell
    ('capture', from_r, from_c, to_r, to_c, extra_r, extra_c) — capture + extra removal
"""

from game_engine import ROWS, COLS, EMPTY, GameState

# Orthogonal direction vectors
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def generate_moves(state):
    """Generate all legal moves for the current player.

    Ordering: Captures first, then placements, then normal moves.
    This natural ordering already provides a small pruning benefit.

    Returns:
        List of move tuples.
    """
    player = state.current_player
    opponent = GameState.opponent(player)
    moves = []

    # 1. Capture moves (highest priority — material gain)
    moves.extend(_generate_captures(state, player, opponent))

    # 2. Placement moves (if pieces remain in hand)
    if state.pieces_in_hand[player] > 0:
        moves.extend(_generate_placements(state))

    # 3. Normal movement moves
    moves.extend(_generate_movements(state, player))

    return moves


def _generate_placements(state):
    """Generate all valid piece-placement moves."""
    moves = []
    for r in range(ROWS):
        for c in range(COLS):
            if state.board[r][c] == EMPTY:
                moves.append(('place', r, c))
    return moves


def _generate_movements(state, player):
    """Generate all valid movement moves for pieces on the board."""
    moves = []
    for r, c in state.get_player_positions(player):
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if GameState.is_valid_pos(nr, nc) and state.board[nr][nc] == EMPTY:
                moves.append(('move', r, c, nr, nc))
    return moves


def _generate_captures(state, player, opponent):
    """Generate all valid capture moves.

    Each physical capture (jump) generates N separate moves,
    one for each possible extra opponent piece to remove.
    """
    moves = []
    for r, c in state.get_player_positions(player):
        for dr, dc in DIRECTIONS:
            # Adjacent cell must contain an opponent piece
            mr, mc = r + dr, c + dc
            if not GameState.is_valid_pos(mr, mc) or state.board[mr][mc] != opponent:
                continue
            # Cell beyond the opponent must be empty
            tr, tc = r + 2 * dr, c + 2 * dc
            if not GameState.is_valid_pos(tr, tc) or state.board[tr][tc] != EMPTY:
                continue

            # Valid capture — now enumerate extra removal targets
            opp_positions = state.get_player_positions(opponent)
            extra_targets = [(er, ec) for er, ec in opp_positions if (er, ec) != (mr, mc)]

            if extra_targets:
                for er, ec in extra_targets:
                    moves.append(('capture', r, c, tr, tc, er, ec))
            else:
                # Opponent had only the one jumped piece on the board
                moves.append(('capture', r, c, tr, tc, -1, -1))

    return moves


def generate_successors(state):
    """Generate all (move, successor_state) pairs.

    Returns:
        List of (move, new_state) tuples with turns already switched.
    """
    return [(move, state.apply_move_and_switch(move)) for move in generate_moves(state)]


def has_legal_moves(state):
    """Efficiently check whether the current player has at least one legal move."""
    player = state.current_player

    # Placement moves
    if state.pieces_in_hand[player] > 0:
        for r in range(ROWS):
            for c in range(COLS):
                if state.board[r][c] == EMPTY:
                    return True

    # Movement and capture moves
    opponent = GameState.opponent(player)
    for r, c in state.get_player_positions(player):
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if GameState.is_valid_pos(nr, nc):
                if state.board[nr][nc] == EMPTY:
                    return True
                if state.board[nr][nc] == opponent:
                    tr, tc = r + 2 * dr, c + 2 * dc
                    if GameState.is_valid_pos(tr, tc) and state.board[tr][tc] == EMPTY:
                        return True

    return False
