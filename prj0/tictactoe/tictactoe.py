"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None

class InvalidPosition(Exception):
    "Position is not empty"
    pass

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
    if (board[0].count(EMPTY)+board[1].count(EMPTY)+board[2].count(EMPTY)) % 2 == 1 :
        return X
    return O

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = []
    for i in range(3):
        for j in range(3):
            if board[i][j]==EMPTY:
                actions.append((i,j))
    return actions

def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if board[action[0]][action[1]] != EMPTY:
        raise InvalidPosition
    
    new_board = copy.deepcopy(board)
    new_board[action[0]][action[1]] = player(new_board)
    return new_board

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
    diag = [[board[0][0],board[1][1],board[2][2]],[board[0][2],board[1][1],board[2][0]]]
    vert = [[board[0][0],board[1][0],board[2][0]],[board[0][1],board[1][1],board[2][1]],[board[0][2],board[1][2],board[2][2]]]
    return any(is_three_row(row,player) for row in [*board,*diag,*vert]) 

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

def max_value(board,cur_min):
    if terminal(board):
        return utility(board)
    v=-2
    for action in actions(board):
        v = max(v,min_value(result(board,action),v))
        if cur_min!=None and v>= cur_min:
            return v
    return v

def min_value(board,cur_max):
    if terminal(board):
        return utility(board)
    v=2
    for action in actions(board):
        v = min(v,max_value(result(board,action),v))
        if cur_max!=None and v<=cur_max: 
            return v
    return v

def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    acs = actions(board)
    if player(board) == X: # MAX
        min_values_for_actions=list(map(lambda a:(min_value(result(board,a),None),a),acs))
        highest=max([t[0] for t in min_values_for_actions])
        for t in min_values_for_actions:
            if t[0] == highest:
                return t[1]
    if player(board) == O: # MIN
        max_values_for_actions=list(map(lambda a:(max_value(result(board,a),None),a),acs))
        smallest=min([t[0] for t in max_values_for_actions])
        for t in max_values_for_actions:
            if t[0] == smallest:
                return t[1]
    return None

