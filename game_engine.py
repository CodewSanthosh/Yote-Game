"""
Module 1: Yote Game Engine
Core game logic — board management, rules, turn management, and game state.

Yote Rules:
- 6×5 board, 2 players, 12 pieces each
- Placement phase: place a piece on any empty cell
- Movement phase: move a piece to an adjacent empty cell (orthogonal only)
- Capture: jump over adjacent opponent to empty cell beyond, PLUS remove one extra opponent piece
- Win: opponent has ≤2 total pieces OR no legal moves
"""

# Board dimensions
ROWS = 6
COLS = 5

# Cell values
EMPTY = 0
PLAYER1 = 1  # Human (Gold)
PLAYER2 = 2  # AI (Cyan)

# Starting pieces per player
PIECES_PER_PLAYER = 12


class GameState:
    """Complete state of a Yote game."""

    __slots__ = ['board', 'current_player', 'pieces_in_hand', 'winner', 'game_over']

    def __init__(self):
        self.board = [[EMPTY] * COLS for _ in range(ROWS)]
        self.current_player = PLAYER1
        self.pieces_in_hand = {PLAYER1: PIECES_PER_PLAYER, PLAYER2: PIECES_PER_PLAYER}
        self.winner = None
        self.game_over = False

    def copy(self):
        """Create a deep copy of the game state."""
        new = GameState.__new__(GameState)
        new.board = [row[:] for row in self.board]
        new.current_player = self.current_player
        new.pieces_in_hand = {PLAYER1: self.pieces_in_hand[PLAYER1],
                               PLAYER2: self.pieces_in_hand[PLAYER2]}
        new.winner = self.winner
        new.game_over = self.game_over
        return new

    def pieces_on_board(self, player):
        """Count pieces on the board for a given player."""
        count = 0
        for row in self.board:
            for cell in row:
                if cell == player:
                    count += 1
        return count

    def total_pieces(self, player):
        """Total pieces (in hand + on board) for a given player."""
        return self.pieces_in_hand[player] + self.pieces_on_board(player)

    @staticmethod
    def opponent(player):
        """Get the opponent of a player."""
        return PLAYER2 if player == PLAYER1 else PLAYER1

    @staticmethod
    def is_valid_pos(r, c):
        """Check if a position is within board bounds."""
        return 0 <= r < ROWS and 0 <= c < COLS

    def get_player_positions(self, player):
        """Get all board positions occupied by a player's pieces."""
        positions = []
        for r in range(ROWS):
            for c in range(COLS):
                if self.board[r][c] == player:
                    positions.append((r, c))
        return positions

    def apply_move(self, move):
        """Apply a move and return a NEW GameState (does NOT switch turns).

        Move formats:
            ('place', row, col)
            ('move', from_r, from_c, to_r, to_c)
            ('capture', from_r, from_c, to_r, to_c, extra_r, extra_c)
        """
        new_state = self.copy()
        move_type = move[0]

        if move_type == 'place':
            _, r, c = move
            new_state.board[r][c] = self.current_player
            new_state.pieces_in_hand[self.current_player] -= 1

        elif move_type == 'move':
            _, fr, fc, tr, tc = move
            new_state.board[tr][tc] = self.current_player
            new_state.board[fr][fc] = EMPTY

        elif move_type == 'capture':
            _, fr, fc, tr, tc, er, ec = move
            # Move the capturing piece
            new_state.board[tr][tc] = self.current_player
            new_state.board[fr][fc] = EMPTY
            # Remove the jumped piece (midpoint between from and to)
            jr, jc = (fr + tr) // 2, (fc + tc) // 2
            new_state.board[jr][jc] = EMPTY
            # Remove the extra piece (player's choice)
            if er >= 0 and ec >= 0:
                new_state.board[er][ec] = EMPTY

        return new_state

    def apply_move_and_switch(self, move):
        """Apply a move, check game over, and switch turns."""
        new_state = self.apply_move(move)
        new_state.check_game_over()
        if not new_state.game_over:
            new_state.current_player = GameState.opponent(self.current_player)
        return new_state

    def check_game_over(self):
        """Check win/loss conditions. Returns True if game is over."""
        for player in [PLAYER1, PLAYER2]:
            if self.total_pieces(player) <= 2:
                self.game_over = True
                self.winner = GameState.opponent(player)
                return True
        return False

    def board_tuple(self):
        """Return a hashable tuple representation of the board."""
        return tuple(tuple(row) for row in self.board)

    def __hash__(self):
        return hash((self.board_tuple(), self.current_player,
                     self.pieces_in_hand[PLAYER1], self.pieces_in_hand[PLAYER2]))

    def __eq__(self, other):
        if not isinstance(other, GameState):
            return False
        return (self.board == other.board and
                self.current_player == other.current_player and
                self.pieces_in_hand == other.pieces_in_hand)

    def __repr__(self):
        symbols = {EMPTY: '.', PLAYER1: 'G', PLAYER2: 'C'}
        lines = []
        for r in range(ROWS):
            lines.append(' '.join(symbols[self.board[r][c]] for c in range(COLS)))
        lines.append(f"Turn: P{self.current_player} | Hand: P1={self.pieces_in_hand[PLAYER1]}, P2={self.pieces_in_hand[PLAYER2]}")
        return '\n'.join(lines)
