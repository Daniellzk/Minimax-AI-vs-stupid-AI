import random

import numpy as np
import pygame
import sys
import math

BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

ROW_COUNT = 6
COLUMN_COUNT = 7

PLAYER = 0
AI = 1

EMPTY_PIECE = 0
PLAYER_PIECE = 1
AI_PIECE = 2

WINDOW_LENGTH = 4


def create_board():
    board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    return board


def drop_piece(board, row, col, piece):
    board[row][col] = piece


def is_valid_location(board, col):
    return board[ROW_COUNT - 1][col] == 0


def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r


def print_board(board):
    print(np.flip(board, 0))


def winning_move(board, piece):
    # Check horizontal locations for win
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][
                c + 3] == piece:
                return True

    # Check vertical locations for win
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][
                c] == piece:
                return True

    # Check positively sloped diagonals
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece and board[r + 3][
                c + 3] == piece:
                return True

    # Check negatively sloped diagonals
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and board[r - 3][
                c + 3] == piece:
                return True


def draw_board(board):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (
                int(c * SQUARESIZE + SQUARESIZE / 2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)

    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == PLAYER_PIECE:
                pygame.draw.circle(screen, RED, (
                    int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)),
                                   RADIUS)
            elif board[r][c] == AI_PIECE:
                pygame.draw.circle(screen, YELLOW, (
                    int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)),
                                   RADIUS)
    pygame.display.update()


def score_evaluation(window, piece):
    score = 0
    opponent_piece = PLAYER_PIECE
    if piece == PLAYER_PIECE:
        opponent_piece = AI_PIECE
    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY_PIECE) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(EMPTY_PIECE) == 2:
        score += 2

    if window.count(opponent_piece) == 3 and window.count(EMPTY_PIECE) == 1:
        score -= 4
    return score


def position_score(board, piece):
    score = 0

    # Check if it's center
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    score += center_array.count(piece) * 3
    # check horizontal window score
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMN_COUNT - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += score_evaluation(window, piece)

    # check vertical window score
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROW_COUNT - 3):
            window = col_array[r: r + WINDOW_LENGTH]
            score += score_evaluation(window, piece)

    # check positive slope window score
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += score_evaluation(window, piece)

    # check negative slope window score
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + 3 - i][c + i] for i in range(WINDOW_LENGTH)]
            score += score_evaluation(window, piece)

    return score


def is_terminal_node(board):
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0


def minimax(board, depth, alpha, beta, maximizing_player):
    valid_locations = get_valid_locations(board)
    if depth == 0 or is_terminal_node(board):
        if is_terminal_node(board):
            if winning_move(board, AI_PIECE):
                return None, 100000000000000
            elif winning_move(board, PLAYER_PIECE):
                return None, -100000000000000
            else:
                return None, 0
        else:
            return None, position_score(board, AI_PIECE)
    if maximizing_player:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            board_copy = board.copy()
            drop_piece(board_copy, row, col, AI_PIECE)
            new_score = minimax(board_copy, depth - 1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value
    else:
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            board_copy = board.copy()
            drop_piece(board_copy, row, col, PLAYER_PIECE)
            new_score = minimax(board_copy, depth - 1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value



def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations


def pick_best_col(board, piece):
    valid_locations = get_valid_locations(board)
    best_score = -9999999999
    best_col = random.choice(valid_locations)
    score_array = []
    for col in valid_locations:
        row = get_next_open_row(board, col)
        temp_board = board.copy()
        drop_piece(temp_board, row, col, piece)
        score = position_score(temp_board, piece)
        score_array.append(score)
        if score > best_score:
            best_score = score
            best_col = col
    print('\n')
    print(score_array)
    return best_col


board = create_board()
print_board(board)
game_over = False
turn = 1

pygame.init()

SQUARESIZE = 100

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 2) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE / 2 - 5)

screen = pygame.display.set_mode(size)
draw_board(board)
pygame.display.update()

myfont = pygame.font.SysFont("monospace", 75)

while not game_over:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()



        # if event.type == pygame.MOUSEBUTTONDOWN:
        #     pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
        #     pygame.draw.rect(screen, BLACK, (22, 710, width, SQUARESIZE))
        #     # print(event.pos)
        #     # Ask for Player 1 Input
    if turn == PLAYER:
        col = pick_best_col(board, PLAYER_PIECE)

        if is_valid_location(board, col):
            pygame.time.wait(700)
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, PLAYER_PIECE)

            if winning_move(board, 1):
                label = myfont.render("Player 1 wins!!", 1, RED)
                screen.blit(label, (22, 710))
                game_over = True

            turn += 1
            turn = turn % 2
            print('\n')
            print_board(board)
            draw_board(board)

    # # Waiting for AI Input
    if turn == AI and not game_over:
        col = minimax(board, 5, -math.inf, math.inf, True)[0]

        if is_valid_location(board, col):
            pygame.time.wait(700)
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, AI_PIECE)

            if winning_move(board, 2):
                label = myfont.render("Player 2 wins!!", 1, YELLOW)
                screen.blit(label, (22, 710))
                game_over = True

            print('\n')
            print_board(board)
            draw_board(board)

            turn += 1
            turn = turn % 2

    if game_over:
        pygame.time.wait(5000)
        sys.exit()
