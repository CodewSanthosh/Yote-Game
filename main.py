"""
Yote Game — Main Application (Pygame UI)
A premium-looking Yote board game with AI opponents.

Controls:
    Left-click: Select piece / Place piece / Move piece / Choose capture target
    Right-click: Deselect / Cancel
    ESC: Return to menu
"""

import pygame
import sys
import math
import time
import threading
from game_engine import GameState, ROWS, COLS, EMPTY, PLAYER1, PLAYER2
from move_generator import generate_moves, has_legal_moves, DIRECTIONS
from search_algorithms import (search, SearchConfig, SearchResult,
                                ALGORITHM_NAMES)
from heuristic import evaluate

# ─────────────────────────────────────────────
# Window & Layout Constants
# ─────────────────────────────────────────────
WIDTH, HEIGHT = 1200, 800
FPS = 60

CELL_SIZE = 85
BOARD_PADDING = 40
BOARD_X = 60
BOARD_Y = 120
BOARD_W = COLS * CELL_SIZE
BOARD_H = ROWS * CELL_SIZE

PANEL_X = BOARD_X + BOARD_W + 80
PANEL_W = WIDTH - PANEL_X - 40

PIECE_RADIUS = 32

# ─────────────────────────────────────────────
# Color Palette (Dark Premium Theme)
# ─────────────────────────────────────────────
BG_PRIMARY = (10, 10, 30)
BG_SECONDARY = (18, 18, 48)
BOARD_BG = (45, 30, 10)
BOARD_BORDER = (80, 60, 30)
CELL_LIGHT = (198, 156, 109)
CELL_DARK = (178, 136, 89)
CELL_HOVER = (220, 190, 140)
CELL_SELECTED = (255, 215, 80)
CELL_VALID = (80, 200, 120)
CELL_CAPTURE = (255, 100, 100)
CELL_EXTRA_REMOVE = (255, 50, 50)

# Player colors
P1_COLOR = (255, 215, 0)       # Gold
P1_DARK = (200, 160, 0)
P1_GLOW = (255, 230, 100)
P2_COLOR = (0, 213, 255)       # Cyan
P2_DARK = (0, 150, 200)
P2_GLOW = (100, 230, 255)

# UI Colors
TEXT_PRIMARY = (230, 230, 240)
TEXT_SECONDARY = (160, 160, 200)
TEXT_ACCENT = (255, 215, 0)
PANEL_BG = (20, 20, 55)
PANEL_BORDER = (50, 50, 100)
BUTTON_BG = (40, 40, 90)
BUTTON_HOVER = (60, 60, 130)
BUTTON_ACTIVE = (80, 80, 180)
BUTTON_TEXT = (220, 220, 240)
SUCCESS_COLOR = (80, 200, 120)
WARNING_COLOR = (255, 180, 50)
ERROR_COLOR = (255, 80, 80)

# ─────────────────────────────────────────────
# Game Modes & States
# ─────────────────────────────────────────────
MODE_MENU = 'menu'
MODE_PLAYING = 'playing'
MODE_GAME_OVER = 'game_over'

PHASE_SELECT = 'select'            # Waiting for player action
PHASE_MOVE_TARGET = 'move_target'  # Piece selected, choosing destination
PHASE_EXTRA_REMOVE = 'extra_remove'  # After capture, choosing extra piece to remove
PHASE_AI_THINKING = 'ai_thinking'


class Button:
    """Simple clickable button."""

    def __init__(self, x, y, w, h, text, font, active=False):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.font = font
        self.active = active
        self.hovered = False

    def draw(self, surface):
        if self.active:
            color = BUTTON_ACTIVE
        elif self.hovered:
            color = BUTTON_HOVER
        else:
            color = BUTTON_BG

        # Draw button with rounded corners
        pygame.draw.rect(surface, color, self.rect, border_radius=8)
        pygame.draw.rect(surface, PANEL_BORDER, self.rect, 2, border_radius=8)

        # Text
        text_surf = self.font.render(self.text, True,
                                     TEXT_ACCENT if self.active else BUTTON_TEXT)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def update(self, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)

    def clicked(self, mouse_pos):
        return self.rect.collidepoint(mouse_pos)


class YoteGame:
    """Main game application."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("YOTE — AI Board Game")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_title = pygame.font.SysFont('segoeui', 36, bold=True)
        self.font_heading = pygame.font.SysFont('segoeui', 22, bold=True)
        self.font_body = pygame.font.SysFont('segoeui', 16)
        self.font_small = pygame.font.SysFont('segoeui', 13)
        self.font_button = pygame.font.SysFont('segoeui', 15, bold=True)
        self.font_large = pygame.font.SysFont('segoeui', 48, bold=True)

        # Game state
        self.mode = MODE_MENU
        self.game_state = None
        self.phase = PHASE_SELECT
        self.selected_piece = None
        self.valid_moves = []
        self.capture_pending = None  # Stores partial capture before extra removal
        self.hover_cell = None
        self.last_move = None
        self.move_history = []
        self.status_message = ""
        self.ai_result = None
        self.ai_thread = None

        # Animation
        self.animation_piece = None
        self.animation_start = None
        self.animation_from = None
        self.animation_to = None
        self.animation_duration = 0.25

        # Settings
        self.ai_algorithm = 'alpha_beta'
        self.ai_depth = 4
        self.ai_config = SearchConfig(use_sorting=True, use_tt=True, use_hash_move=True)
        self.game_mode = 'human_vs_ai'  # 'human_vs_ai' or 'ai_vs_ai'

        # Last AI result for display
        self.last_ai_result = None

        # Menu buttons
        self._create_menu_buttons()

    def _create_menu_buttons(self):
        """Create menu screen buttons."""
        cx = WIDTH // 2
        cy = HEIGHT // 2

        self.menu_buttons = {
            'hvai': Button(cx - 150, cy - 40, 300, 50, "Human vs AI", self.font_button),
            'aivai': Button(cx - 150, cy + 30, 300, 50, "AI vs AI", self.font_button),
        }

        # Algorithm buttons (for settings panel)
        self.algo_buttons = {}
        self.config_buttons = {}
        self.depth_buttons = {}

    def _create_panel_buttons(self):
        """Create side panel buttons during gameplay."""
        y = BOARD_Y + 10

        # Algorithm selection
        self.algo_buttons = {}
        for i, (key, name) in enumerate(ALGORITHM_NAMES.items()):
            btn = Button(PANEL_X, y + i * 38, PANEL_W, 32, name, self.font_button,
                         active=(key == self.ai_algorithm))
            self.algo_buttons[key] = btn

        y += len(ALGORITHM_NAMES) * 38 + 20

        # Depth selection
        self.depth_buttons = {}
        for i, d in enumerate([2, 3, 4, 5, 6]):
            btn = Button(PANEL_X + i * 55, y, 48, 32, str(d), self.font_button,
                         active=(d == self.ai_depth))
            self.depth_buttons[d] = btn

        y += 55

        # Enhancement toggles
        self.config_buttons = {
            'sorting': Button(PANEL_X, y, PANEL_W, 32, "Move Ordering",
                              self.font_button, active=self.ai_config.use_sorting),
            'tt': Button(PANEL_X, y + 38, PANEL_W, 32, "Transposition Table",
                         self.font_button, active=self.ai_config.use_tt),
            'hm': Button(PANEL_X, y + 76, PANEL_W, 32, "Hash Move",
                          self.font_button, active=self.ai_config.use_hash_move),
        }

    # ─────────────────────────────────────────
    # Game Logic
    # ─────────────────────────────────────────

    def start_game(self):
        """Initialize a new game."""
        self.game_state = GameState()
        self.mode = MODE_PLAYING
        self.phase = PHASE_SELECT
        self.selected_piece = None
        self.valid_moves = []
        self.capture_pending = None
        self.last_move = None
        self.move_history = []
        self.status_message = "Player 1's turn — Place or move a piece"
        self.last_ai_result = None
        self._create_panel_buttons()

        # If AI vs AI, or if AI goes first
        if self.game_mode == 'ai_vs_ai':
            self._start_ai_turn()
        elif self.game_mode == 'human_vs_ai' and self.game_state.current_player == PLAYER2:
            self._start_ai_turn()

    def cell_at_pixel(self, mx, my):
        """Convert pixel coordinates to board cell (row, col) or None."""
        if (BOARD_X <= mx < BOARD_X + BOARD_W and
                BOARD_Y <= my < BOARD_Y + BOARD_H):
            c = (mx - BOARD_X) // CELL_SIZE
            r = (my - BOARD_Y) // CELL_SIZE
            if 0 <= r < ROWS and 0 <= c < COLS:
                return (r, c)
        return None

    def cell_center(self, r, c):
        """Get the pixel center of a cell."""
        x = BOARD_X + c * CELL_SIZE + CELL_SIZE // 2
        y = BOARD_Y + r * CELL_SIZE + CELL_SIZE // 2
        return (x, y)

    def _get_valid_destinations(self, r, c):
        """Get valid destination cells for a selected piece."""
        player = self.game_state.current_player
        moves = generate_moves(self.game_state)
        destinations = {}  # (tr, tc) → list of moves

        for move in moves:
            if move[0] == 'move' and move[1] == r and move[2] == c:
                dest = (move[3], move[4])
                destinations.setdefault(dest, []).append(move)
            elif move[0] == 'capture' and move[1] == r and move[2] == c:
                dest = (move[3], move[4])
                destinations.setdefault(dest, []).append(move)

        return destinations

    def _handle_human_click(self, cell):
        """Handle a human player's click on a cell."""
        r, c = cell
        state = self.game_state
        player = state.current_player

        if self.phase == PHASE_SELECT:
            # Check if clicking on own piece to select it
            if state.board[r][c] == player:
                self.selected_piece = (r, c)
                self.valid_moves = self._get_valid_destinations(r, c)
                self.phase = PHASE_MOVE_TARGET
                self.status_message = f"Piece selected at ({r},{c}) — Click destination"
                return

            # Check if clicking empty cell to place a piece
            if state.board[r][c] == EMPTY and state.pieces_in_hand[player] > 0:
                move = ('place', r, c)
                all_moves = generate_moves(state)
                if move in all_moves:
                    self._execute_move(move)
                    return

        elif self.phase == PHASE_MOVE_TARGET:
            # Clicking on another own piece → reselect
            if state.board[r][c] == player:
                self.selected_piece = (r, c)
                self.valid_moves = self._get_valid_destinations(r, c)
                self.status_message = f"Piece selected at ({r},{c}) — Click destination"
                return

            dest = (r, c)
            if dest in self.valid_moves:
                moves_to_dest = self.valid_moves[dest]

                # If all moves to this dest are normal moves
                normal_moves = [m for m in moves_to_dest if m[0] == 'move']
                capture_moves = [m for m in moves_to_dest if m[0] == 'capture']

                if normal_moves and not capture_moves:
                    self._execute_move(normal_moves[0])
                    return

                if capture_moves:
                    # Need to choose extra removal target
                    # Get unique extra removal targets
                    extra_targets = set()
                    for m in capture_moves:
                        if m[5] >= 0 and m[6] >= 0:
                            extra_targets.add((m[5], m[6]))

                    if not extra_targets:
                        # No extra piece to remove (opponent only had the jumped piece)
                        self._execute_move(capture_moves[0])
                        return

                    # Enter extra removal phase
                    self.capture_pending = {
                        'from': (self.selected_piece[0], self.selected_piece[1]),
                        'to': dest,
                        'moves': capture_moves,
                        'targets': extra_targets,
                    }
                    self.phase = PHASE_EXTRA_REMOVE
                    self.status_message = "Capture! Click an opponent piece to remove"
                    return

            # Clicked invalid destination
            self.selected_piece = None
            self.valid_moves = {}
            self.phase = PHASE_SELECT
            self.status_message = "Invalid destination — try again"

        elif self.phase == PHASE_EXTRA_REMOVE:
            target = (r, c)
            if target in self.capture_pending['targets']:
                # Find the matching capture move
                for m in self.capture_pending['moves']:
                    if m[5] == r and m[6] == c:
                        self._execute_move(m)
                        self.capture_pending = None
                        return
            self.status_message = "Click an opponent piece to remove!"

    def _execute_move(self, move):
        """Execute a move, update game state, and check for game over."""
        self.last_move = move
        self.move_history.append(move)
        self.game_state = self.game_state.apply_move_and_switch(move)

        # Reset selection
        self.selected_piece = None
        self.valid_moves = {}
        self.phase = PHASE_SELECT

        # Check game over
        if self.game_state.game_over:
            self.mode = MODE_GAME_OVER
            winner = self.game_state.winner
            self.status_message = f"Game Over! Player {winner} ({'Gold' if winner == 1 else 'Cyan'}) wins!"
            return

        # Check if next player has legal moves
        if not has_legal_moves(self.game_state):
            self.game_state.game_over = True
            self.game_state.winner = GameState.opponent(self.game_state.current_player)
            self.mode = MODE_GAME_OVER
            winner = self.game_state.winner
            self.status_message = f"Game Over! Player {winner} ({'Gold' if winner == 1 else 'Cyan'}) wins!"
            return

        current = self.game_state.current_player
        self.status_message = f"Player {current}'s turn ({'Gold' if current == 1 else 'Cyan / AI'})"

        # Start AI turn if needed
        if self.game_mode == 'human_vs_ai' and current == PLAYER2:
            self._start_ai_turn()
        elif self.game_mode == 'ai_vs_ai':
            self._start_ai_turn()

    def _start_ai_turn(self):
        """Launch AI search in a background thread."""
        self.phase = PHASE_AI_THINKING
        self.status_message = f"AI ({ALGORITHM_NAMES.get(self.ai_algorithm, 'Alpha-Beta')}) thinking..."
        self.ai_result = None

        def run_search():
            config = SearchConfig(
                use_sorting=self.ai_config.use_sorting,
                use_tt=self.ai_config.use_tt,
                use_hash_move=self.ai_config.use_hash_move,
            )
            result = search(self.game_state, self.ai_algorithm,
                            self.ai_depth, config, time_limit=30.0)
            self.ai_result = result

        self.ai_thread = threading.Thread(target=run_search, daemon=True)
        self.ai_thread.start()

    def _check_ai_result(self):
        """Check if AI has finished thinking."""
        if self.ai_result is not None and self.phase == PHASE_AI_THINKING:
            result = self.ai_result
            self.last_ai_result = result
            self.ai_result = None

            if result.best_move:
                self._execute_move(result.best_move)
            else:
                # AI found no move — shouldn't happen normally
                self.status_message = "AI could not find a move!"
                self.phase = PHASE_SELECT

    # ─────────────────────────────────────────
    # Drawing
    # ─────────────────────────────────────────

    def _draw_background(self):
        """Draw gradient background."""
        for y in range(HEIGHT):
            t = y / HEIGHT
            r = int(BG_PRIMARY[0] * (1 - t) + BG_SECONDARY[0] * t)
            g = int(BG_PRIMARY[1] * (1 - t) + BG_SECONDARY[1] * t)
            b = int(BG_PRIMARY[2] * (1 - t) + BG_SECONDARY[2] * t)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (WIDTH, y))

    def _draw_board(self):
        """Draw the game board with cells and pieces."""
        # Board background with shadow
        shadow_rect = pygame.Rect(BOARD_X + 4, BOARD_Y + 4, BOARD_W, BOARD_H)
        pygame.draw.rect(self.screen, (0, 0, 0), shadow_rect, border_radius=6)
        board_rect = pygame.Rect(BOARD_X - 4, BOARD_Y - 4, BOARD_W + 8, BOARD_H + 8)
        pygame.draw.rect(self.screen, BOARD_BORDER, board_rect, border_radius=6)
        inner_rect = pygame.Rect(BOARD_X, BOARD_Y, BOARD_W, BOARD_H)
        pygame.draw.rect(self.screen, BOARD_BG, inner_rect, border_radius=4)

        # Draw cells
        for r in range(ROWS):
            for c in range(COLS):
                x = BOARD_X + c * CELL_SIZE
                y = BOARD_Y + r * CELL_SIZE
                cell_rect = pygame.Rect(x + 2, y + 2, CELL_SIZE - 4, CELL_SIZE - 4)

                # Cell color based on state
                color = CELL_LIGHT if (r + c) % 2 == 0 else CELL_DARK

                if self.phase == PHASE_EXTRA_REMOVE and self.capture_pending:
                    if (r, c) in self.capture_pending['targets']:
                        color = CELL_EXTRA_REMOVE
                    elif self.hover_cell == (r, c) and (r, c) in self.capture_pending['targets']:
                        color = (255, 120, 120)

                elif self.selected_piece == (r, c):
                    color = CELL_SELECTED
                elif (r, c) in self.valid_moves:
                    # Check if capture or move
                    moves = self.valid_moves[(r, c)]
                    has_capture = any(m[0] == 'capture' for m in moves)
                    color = CELL_CAPTURE if has_capture else CELL_VALID
                elif self.hover_cell == (r, c) and self.phase in (PHASE_SELECT, PHASE_MOVE_TARGET):
                    color = CELL_HOVER

                # Highlight last move
                if self.last_move:
                    mt = self.last_move[0]
                    if mt == 'place' and (r, c) == (self.last_move[1], self.last_move[2]):
                        color = tuple(min(255, ch + 30) for ch in color)
                    elif mt in ('move', 'capture'):
                        if (r, c) in ((self.last_move[1], self.last_move[2]),
                                      (self.last_move[3], self.last_move[4])):
                            color = tuple(min(255, ch + 30) for ch in color)

                pygame.draw.rect(self.screen, color, cell_rect, border_radius=6)
                pygame.draw.rect(self.screen, BOARD_BORDER, cell_rect, 1, border_radius=6)

                # Draw piece
                piece = self.game_state.board[r][c]
                if piece != EMPTY:
                    cx, cy = self.cell_center(r, c)
                    self._draw_piece(cx, cy, piece)

                # Draw valid move indicator
                if self.phase == PHASE_MOVE_TARGET and (r, c) in self.valid_moves:
                    cx, cy = self.cell_center(r, c)
                    if self.game_state.board[r][c] == EMPTY:
                        pygame.draw.circle(self.screen, CELL_VALID, (cx, cy), 8)
                        pygame.draw.circle(self.screen, (60, 160, 100), (cx, cy), 8, 2)

        # Column labels
        for c in range(COLS):
            x = BOARD_X + c * CELL_SIZE + CELL_SIZE // 2
            label = self.font_small.render(str(c), True, TEXT_SECONDARY)
            self.screen.blit(label, (x - label.get_width() // 2, BOARD_Y - 20))
            self.screen.blit(label, (x - label.get_width() // 2, BOARD_Y + BOARD_H + 6))

        # Row labels
        for r in range(ROWS):
            y = BOARD_Y + r * CELL_SIZE + CELL_SIZE // 2
            label = self.font_small.render(str(r), True, TEXT_SECONDARY)
            self.screen.blit(label, (BOARD_X - 18, y - label.get_height() // 2))

    def _draw_piece(self, cx, cy, player, scale=1.0):
        """Draw a game piece with gradient and glow."""
        radius = int(PIECE_RADIUS * scale)

        if player == PLAYER1:
            color = P1_COLOR
            dark = P1_DARK
            glow = P1_GLOW
        else:
            color = P2_COLOR
            dark = P2_DARK
            glow = P2_GLOW

        # Shadow
        pygame.draw.circle(self.screen, (20, 20, 20), (cx + 2, cy + 3), radius)

        # Outer ring (dark)
        pygame.draw.circle(self.screen, dark, (cx, cy), radius)

        # Inner fill (lighter)
        pygame.draw.circle(self.screen, color, (cx, cy), radius - 3)

        # Highlight (top-left)
        highlight_x = cx - radius // 4
        highlight_y = cy - radius // 4
        pygame.draw.circle(self.screen, glow, (highlight_x, highlight_y), radius // 3)

        # Border
        pygame.draw.circle(self.screen, dark, (cx, cy), radius, 2)

    def _draw_panel(self):
        """Draw the side control panel."""
        # Panel background
        panel_rect = pygame.Rect(PANEL_X - 15, BOARD_Y - 30, PANEL_W + 30, BOARD_H + 60)
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, PANEL_BORDER, panel_rect, 2, border_radius=12)

        y = BOARD_Y - 15

        # Algorithm section
        heading = self.font_heading.render("Algorithm", True, TEXT_ACCENT)
        self.screen.blit(heading, (PANEL_X, y))
        y += 30

        for key, btn in self.algo_buttons.items():
            btn.rect.y = y
            btn.draw(self.screen)
            y += 38

        y += 10

        # Depth section
        heading = self.font_heading.render("Search Depth", True, TEXT_ACCENT)
        self.screen.blit(heading, (PANEL_X, y))
        y += 30

        for i, (d, btn) in enumerate(self.depth_buttons.items()):
            btn.rect.x = PANEL_X + i * 55
            btn.rect.y = y
            btn.draw(self.screen)

        y += 50

        # Enhancements section
        heading = self.font_heading.render("Enhancements", True, TEXT_ACCENT)
        self.screen.blit(heading, (PANEL_X, y))
        y += 30

        for key, btn in self.config_buttons.items():
            btn.rect.y = y
            btn.draw(self.screen)
            y += 38

        y += 15

        # Piece counts
        heading = self.font_heading.render("Pieces", True, TEXT_ACCENT)
        self.screen.blit(heading, (PANEL_X, y))
        y += 28

        if self.game_state:
            p1_hand = self.game_state.pieces_in_hand[PLAYER1]
            p1_board = self.game_state.pieces_on_board(PLAYER1)
            p2_hand = self.game_state.pieces_in_hand[PLAYER2]
            p2_board = self.game_state.pieces_on_board(PLAYER2)

            # Player 1
            self._draw_piece(PANEL_X + 15, y + 10, PLAYER1, 0.5)
            txt = self.font_body.render(f"Gold:  Board {p1_board}  |  Hand {p1_hand}  |  Total {p1_hand + p1_board}",
                                        True, P1_COLOR)
            self.screen.blit(txt, (PANEL_X + 35, y + 2))
            y += 28

            # Player 2
            self._draw_piece(PANEL_X + 15, y + 10, PLAYER2, 0.5)
            txt = self.font_body.render(f"Cyan:  Board {p2_board}  |  Hand {p2_hand}  |  Total {p2_hand + p2_board}",
                                        True, P2_COLOR)
            self.screen.blit(txt, (PANEL_X + 35, y + 2))
            y += 35

        # Last AI stats
        if self.last_ai_result:
            heading = self.font_heading.render("Last AI Search", True, TEXT_ACCENT)
            self.screen.blit(heading, (PANEL_X, y))
            y += 25

            res = self.last_ai_result
            stats = [
                f"Nodes: {res.nodes_expanded:,}",
                f"Depth: {res.depth_reached}",
                f"Time: {res.time_ms:.0f} ms",
                f"Cutoffs: {res.ab_cutoffs:,}",
                f"TT Hits: {res.tt_hits:,}",
                f"BF: {res.branching_factor():.1f}",
            ]
            for stat in stats:
                txt = self.font_small.render(stat, True, TEXT_SECONDARY)
                self.screen.blit(txt, (PANEL_X + 5, y))
                y += 18

    def _draw_header(self):
        """Draw the title bar."""
        title = self.font_title.render("YOTE", True, TEXT_ACCENT)
        self.screen.blit(title, (BOARD_X, 20))

        subtitle = self.font_body.render("AI Board Game — Adversarial Search", True, TEXT_SECONDARY)
        self.screen.blit(subtitle, (BOARD_X + 110, 35))

        # Current player indicator
        if self.game_state and self.mode == MODE_PLAYING:
            cp = self.game_state.current_player
            indicator_color = P1_COLOR if cp == PLAYER1 else P2_COLOR
            indicator_text = "Gold's Turn" if cp == PLAYER1 else "Cyan's Turn"
            txt = self.font_heading.render(indicator_text, True, indicator_color)
            self.screen.blit(txt, (BOARD_X, 65))

            # Config label
            cfg = f"{ALGORITHM_NAMES.get(self.ai_algorithm, 'Alpha-Beta')} {self.ai_config.label()} | Depth {self.ai_depth}"
            cfg_txt = self.font_small.render(cfg, True, TEXT_SECONDARY)
            self.screen.blit(cfg_txt, (BOARD_X, 92))

    def _draw_status_bar(self):
        """Draw the bottom status bar."""
        bar_y = HEIGHT - 40
        pygame.draw.rect(self.screen, PANEL_BG, (0, bar_y, WIDTH, 40))
        pygame.draw.line(self.screen, PANEL_BORDER, (0, bar_y), (WIDTH, bar_y))

        # Status message
        msg_color = TEXT_PRIMARY
        if "wins" in self.status_message.lower():
            msg_color = SUCCESS_COLOR
        elif "thinking" in self.status_message.lower():
            msg_color = WARNING_COLOR
        elif "invalid" in self.status_message.lower():
            msg_color = ERROR_COLOR

        txt = self.font_body.render(self.status_message, True, msg_color)
        self.screen.blit(txt, (20, bar_y + 10))

        # Move count
        move_txt = self.font_small.render(f"Moves: {len(self.move_history)}", True, TEXT_SECONDARY)
        self.screen.blit(move_txt, (WIDTH - 120, bar_y + 12))

    def _draw_menu(self):
        """Draw the main menu screen."""
        self._draw_background()

        # Title
        title = self.font_large.render("YOTE", True, TEXT_ACCENT)
        title_rect = title.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 160))
        self.screen.blit(title, title_rect)

        subtitle = self.font_heading.render("AI-Powered Board Game", True, TEXT_SECONDARY)
        sub_rect = subtitle.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 110))
        self.screen.blit(subtitle, sub_rect)

        desc = self.font_body.render("Adversarial Search: Alpha-Beta · NegaScout · MTD-f",
                                     True, TEXT_SECONDARY)
        desc_rect = desc.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 80))
        self.screen.blit(desc, desc_rect)

        # Decorative line
        pygame.draw.line(self.screen, PANEL_BORDER,
                         (WIDTH // 2 - 200, HEIGHT // 2 - 60),
                         (WIDTH // 2 + 200, HEIGHT // 2 - 60), 2)

        # Menu buttons
        for btn in self.menu_buttons.values():
            btn.draw(self.screen)

        # Footer
        footer = self.font_small.render("CS23702 – Artificial Intelligence | Team 18",
                                        True, TEXT_SECONDARY)
        footer_rect = footer.get_rect(center=(WIDTH // 2, HEIGHT - 40))
        self.screen.blit(footer, footer_rect)

    def _draw_game_over(self):
        """Draw game over overlay."""
        # Semi-transparent overlay
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))

        # Winner announcement
        if self.game_state and self.game_state.winner:
            winner = self.game_state.winner
            color = P1_COLOR if winner == PLAYER1 else P2_COLOR
            name = "Gold" if winner == PLAYER1 else "Cyan"

            txt = self.font_large.render(f"{name} Wins!", True, color)
            txt_rect = txt.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 40))
            self.screen.blit(txt, txt_rect)

        # Instructions
        instr = self.font_body.render("Press ESC for menu or SPACE to play again",
                                      True, TEXT_SECONDARY)
        instr_rect = instr.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 30))
        self.screen.blit(instr, instr_rect)

        # Show final stats
        if self.last_ai_result:
            stats = self.font_small.render(
                f"Last AI: {self.last_ai_result.nodes_expanded:,} nodes, "
                f"{self.last_ai_result.time_ms:.0f}ms, depth {self.last_ai_result.depth_reached}",
                True, TEXT_SECONDARY)
            stats_rect = stats.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 70))
            self.screen.blit(stats, stats_rect)

    # ─────────────────────────────────────────
    # Main Loop
    # ─────────────────────────────────────────

    def run(self):
        """Main game loop."""
        running = True

        while running:
            mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.mode = MODE_MENU
                    elif event.key == pygame.K_SPACE and self.mode == MODE_GAME_OVER:
                        self.start_game()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self._handle_left_click(mouse_pos)
                    elif event.button == 3:  # Right click
                        self._handle_right_click()

            # Update hover
            if self.mode == MODE_PLAYING:
                self.hover_cell = self.cell_at_pixel(*mouse_pos)
                self._check_ai_result()

                # Update button hover states
                for btn in self.algo_buttons.values():
                    btn.update(mouse_pos)
                for btn in self.depth_buttons.values():
                    btn.update(mouse_pos)
                for btn in self.config_buttons.values():
                    btn.update(mouse_pos)
            elif self.mode == MODE_MENU:
                for btn in self.menu_buttons.values():
                    btn.update(mouse_pos)

            # Draw
            self._draw()
            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()

    def _handle_left_click(self, mouse_pos):
        """Handle left mouse click."""
        if self.mode == MODE_MENU:
            if self.menu_buttons['hvai'].clicked(mouse_pos):
                self.game_mode = 'human_vs_ai'
                self.start_game()
            elif self.menu_buttons['aivai'].clicked(mouse_pos):
                self.game_mode = 'ai_vs_ai'
                self.start_game()

        elif self.mode == MODE_PLAYING and self.phase != PHASE_AI_THINKING:
            # Check panel button clicks
            for key, btn in self.algo_buttons.items():
                if btn.clicked(mouse_pos):
                    self.ai_algorithm = key
                    for k, b in self.algo_buttons.items():
                        b.active = (k == key)
                    return

            for d, btn in self.depth_buttons.items():
                if btn.clicked(mouse_pos):
                    self.ai_depth = d
                    for dd, b in self.depth_buttons.items():
                        b.active = (dd == d)
                    return

            if self.config_buttons.get('sorting') and self.config_buttons['sorting'].clicked(mouse_pos):
                self.ai_config.use_sorting = not self.ai_config.use_sorting
                self.config_buttons['sorting'].active = self.ai_config.use_sorting
                return
            if self.config_buttons.get('tt') and self.config_buttons['tt'].clicked(mouse_pos):
                self.ai_config.use_tt = not self.ai_config.use_tt
                self.config_buttons['tt'].active = self.ai_config.use_tt
                return
            if self.config_buttons.get('hm') and self.config_buttons['hm'].clicked(mouse_pos):
                self.ai_config.use_hash_move = not self.ai_config.use_hash_move
                self.config_buttons['hm'].active = self.ai_config.use_hash_move
                return

            # Board click
            cell = self.cell_at_pixel(*mouse_pos)
            if cell and (self.game_mode == 'human_vs_ai' and
                         self.game_state.current_player == PLAYER1):
                self._handle_human_click(cell)

    def _handle_right_click(self):
        """Handle right mouse click (deselect)."""
        if self.mode == MODE_PLAYING:
            self.selected_piece = None
            self.valid_moves = {}
            self.capture_pending = None
            if self.phase in (PHASE_MOVE_TARGET, PHASE_EXTRA_REMOVE):
                self.phase = PHASE_SELECT
                self.status_message = "Deselected — Click a piece or empty cell"

    def _draw(self):
        """Main draw dispatcher."""
        if self.mode == MODE_MENU:
            self._draw_menu()
        elif self.mode in (MODE_PLAYING, MODE_GAME_OVER):
            self._draw_background()
            self._draw_header()
            self._draw_board()
            self._draw_panel()
            self._draw_status_bar()

            # AI thinking indicator
            if self.phase == PHASE_AI_THINKING:
                self._draw_thinking_indicator()

            if self.mode == MODE_GAME_OVER:
                self._draw_game_over()

    def _draw_thinking_indicator(self):
        """Draw a pulsing indicator while AI is thinking."""
        t = time.time()
        pulse = (math.sin(t * 4) + 1) / 2  # 0 to 1

        cx = BOARD_X + BOARD_W // 2
        cy = BOARD_Y + BOARD_H // 2

        # Pulsing circle
        radius = int(40 + pulse * 15)
        alpha = int(100 + pulse * 80)
        color = (*WARNING_COLOR[:3], alpha)

        surf = pygame.Surface((radius * 2 + 10, radius * 2 + 10), pygame.SRCALPHA)
        pygame.draw.circle(surf, color, (radius + 5, radius + 5), radius)
        self.screen.blit(surf, (cx - radius - 5, cy - radius - 5))

        txt = self.font_heading.render("AI Thinking...", True, WARNING_COLOR)
        txt_rect = txt.get_rect(center=(cx, cy))
        self.screen.blit(txt, txt_rect)


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    game = YoteGame()
    game.run()
