"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    if board[0].count(EMPTY)+board[1].count(EMPTY)+board[2].count(EMPTY) % 2 == 1 :
        return X
    return O

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    raise NotImplementedError


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    raise NotImplementedError


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    if is_three_board(board,O):
        return O
    if is_three_board(board,X):
        return X
    return None

def is_three_board(board, player):
    diag1 = [board[0][0],board[1][1],board[2][2]]
    diag2 = [board[0][2],board[1][1],board[2][0]]
    return any(is_three_row(row,player) for row in board) or is_three_row(diag1,player) or is_three_row(diag2,player) 

def is_three_row(row, player):
    return row.count(player) == 3

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    return winner(board) != None or not(any(r.count(EMPTY)>0 for r in board))


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    if winner(board) == O:
        return -1
    return 0

def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    raise NotImplementedError
