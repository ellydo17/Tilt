import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from fontTools.misc.py23 import xrange
from tabulate import tabulate

"""A program to generate a condensation and fundamental matrix for each card."""

def countGreenSliders(board):
    count_Green = 0
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == "G":
                count_Green += 1
    return count_Green


def tiltRight(board):
    # tilt to the right
    board = board.copy()
    stop = False
    checking = True
    tempBoard = board.copy()
    for i in range(len(board)):
        row = board[i]
        index = len(row)
        blocker = index
        hole = index // 2 + 1
        for j in range(len(row) - 1, -1, -1):
            elem = row[j]
            if i == hole - 1:
                if elem == "B" and j < hole - 1 < blocker:  # blue node cannot go into the hole
                    stop = True
                    board = tempBoard
                    break
                elif elem == "G" and j < hole - 1 < blocker:  # green node goes into the hole
                    row[j] = "-"
                    count_Green -= 1
                    continue
            if elem == "-" and j == len(row) - 1:
                index = j  # last index of the row
            elif elem == "-" and row[j + 1] != "-":
                index = j
            else:
                if (elem == "B" or elem == "G") and (blocker > index):  # handle the blocker
                    row[index] = elem
                    row[j] = "-"
                    index -= 1
                elif elem == "I":
                    blocker = j
        count_Green = countGreenSliders(board)
        if count_Green != 0:
            stop_temp = False
        else:
            stop_temp = True
        if stop == True or stop_temp == True:
            checking = False
            break
    # print(board)
    return board

def tiltLeft(board):
    # tilt to the left
    board = board.copy()
    checking = True
    stop = False
    tempBoard = board.copy()

    for i in range(len(board)):
        # loop through each row in the board
        row = board[i]
        index = -1
        blocker = index
        hole = len(row) // 2 + 1
        for j in range(len(row)):
            elem = row[j]
            if i == hole - 1:
                if elem == "B" and j > hole - 1 > blocker:
                    stop = True
                    board = tempBoard
                    break
                elif elem == "G" and j > hole - 1 > blocker:
                    row[j] = "-"
                    count_Green -= 1
                    continue
            if elem == "-" and j == 0:
                index = j
            elif elem == "-" and row[j - 1] != "-":
                index = j
            else:
                if (elem == "B" or elem == "G") and (blocker < index):
                    row[index] = elem
                    row[j] = "-"
                    index += 1
                elif elem == "I":
                    blocker = j
        count_Green = countGreenSliders(board)
        if count_Green != 0:
            stop_temp = False
        else:
            stop_temp = True
        if stop == True or stop_temp == True:
            checking = False
            break
    # print(board)
    return board

def tiltDown(board):
    # tilt down
    board = board.copy()

    stop = False
    checking = True
    tempBoard = board.copy()

    for i in range(len(board)):
        column = board[:, i]
        index = len(column)
        blocker = index
        hole = index // 2 + 1
        for j in range(len(column) - 1, -1, -1):
            elem = column[j]
            if i == hole - 1:
                if elem == "B" and j < hole - 1 < blocker:
                    stop = True
                    board = tempBoard
                    break
                elif elem == "G" and j < hole - 1 < blocker:
                    column[j] = "-"
                    count_Green -= 1
                    continue
            if elem == "-" and j == len(column) - 1:
                index = j
            elif elem == "-" and column[j + 1] != "-":
                index = j
            else:
                if (elem == "B" or elem == "G") and (blocker > index):
                    column[index] = elem
                    column[j] = "-"
                    index -= 1
                elif elem == "I":
                    blocker = j
        count_Green = countGreenSliders(board)
        if count_Green != 0:
            stop_temp = False
        else:
            stop_temp = True
        if stop == True or stop_temp == True:
            checking = False
            break
    # print(board)
    return board

def tiltUp(board):
    # tilt up
    board = board.copy()

    stop = False
    checking = True
    tempBoard = board.copy()

    for i in range(len(board)):
        # loop through each row in the board
        column = board[:, i]
        index = -1
        blocker = index
        hole = len(column) // 2 + 1
        for j in range(len(column)):
            elem = column[j]
            if i == hole - 1:
                if elem == "B" and j > hole - 1 > blocker:
                    stop = True
                    board = tempBoard
                    break
                elif elem == "G" and j > hole - 1 > blocker:
                    column[j] = "-"
                    count_Green -= 1
                    continue
            if elem == "-" and j == 0:
                index = j
            elif elem == "-" and column[j - 1] != "-":
                index = j
            else:
                if (elem == "B" or elem == "G") and (blocker < index):
                    column[index] = elem
                    column[j] = "-"
                    index += 1
                elif elem == "I":
                    blocker = j
        count_Green = countGreenSliders(board)
        if count_Green != 0:
            stop_temp = False
        else:
            stop_temp = True
        if stop == True or stop_temp == True:
            checking = False
            break
    # print(board)
    return board

def green(board):
    count_Green = countGreenSliders(board)
    if count_Green == 0:
        return True
    else:
        return False


def findMoves(board, moves):
    for i in moves:
        if np.array_equal(i, board):
            return True
    return False

def returnBoardIndex(board, moves):
    """Return the index of a certain board in the array. Return "board doesn't exist" if the board
    configuration can't be found."""
    for i in range(len(moves)):
        if np.array_equal(moves[i], board):
            return i
    return "board doesn't exist"

def create_graph(moves, g, board_direction, board_original, title_edge, node_end):
    """Create a graph for all possible game moves. node_end is to store the end nodes that have a loop."""
    if not np.array_equal(board_direction, board_original):
        if not findMoves(board_direction, moves):
            if not green(board_direction):
                moves.append(board_direction)
                g.add_node(returnBoardIndex(board_direction, moves))
                # add the edge between the board_Left and the current board
                g.add_edge(returnBoardIndex(board_original, moves), returnBoardIndex(board_direction, moves), title=title_edge)
                tiltRecursive(board_direction, moves, g, node_end)
            elif green(board_direction):
                moves.append(board_direction)
                g.add_node(returnBoardIndex(board_direction, moves))
                # add the edge between the board_Right and the current board
                g.add_edge(returnBoardIndex(board_original, moves), returnBoardIndex(board_direction, moves), title=title_edge)
                g.add_edge(returnBoardIndex(board_direction, moves), returnBoardIndex(board_direction, moves), title=title_edge)
                node_end.append(str(returnBoardIndex(board_direction, moves)))
        elif findMoves(board_direction, moves):
            # add the node of the board_Left
            g.add_node(returnBoardIndex(board_direction, moves))
            # add the edge between the board_Left and the current board
            g.add_edge(returnBoardIndex(board_original, moves), returnBoardIndex(board_direction, moves), title=title_edge)


def tiltRecursive(board, moves, g, node_end):
    """Tilt the board to all possible directions."""
    board_Left = tiltLeft(board)
    board_Right = tiltRight(board)
    board_Up = tiltUp(board)
    board_Down = tiltDown(board)

    create_graph(moves, g, board_Left, board, "L", node_end)

    create_graph(moves, g, board_Right, board, "R", node_end)

    create_graph(moves, g, board_Up, board, "U", node_end)

    create_graph(moves, g, board_Down, board, "D", node_end)

def make_table(dict):
    """Make a table for the board configurations. One column is the node number,
    one column is the corresponding board configuration."""
    # return tabulate(dict, headers='keys')
    table = [["Number", "Condensation"]]
    for things in dict:
        table.append([things, dict[things]])
    return tabulate(table, headers='firstrow', tablefmt='fancy_grid')

def transition_matrix(g):
    """A method that generates a transition matrix for a graph of a game card. In a transition matrix,
    each entry at a position (i,j) corresponding to a probability to transition from node i to node j.
    Source: https://stackoverflow.com/questions/37311651/get-node-list-from-random-walk-in-networkx"""   
    # let networkx return the adjacency matrix A
    A = nx.adjacency_matrix(g)
    A = A.todense()
    A = np.array(A, dtype = np.float64)

    # let's evaluate the degree matrix D
    D = np.diag(np.sum(A, axis=1))

    # ...and the transition matrix T. T = D^(-1) A. 
    T = np.dot(np.linalg.inv(D),A)
    return T

def fundamental_matrix(Q):
    """Calculate the fundamental matrix. Source: 
    https://stackoverflow.com/questions/11705733/best-way-to-calculate-the-fundamental-matrix-of-an-absorbing-markov-chain"""
    I = np.identity(Q.shape[0]) #get the identity matrix that has the same shape as the matrix Q.
    # o = np.ones(Q.shape[0])
    F = np.linalg.inv(I-Q)
    return F

def main():
    board = [np.array([["G", "I", "-", "-", "-"],
                       ["-", "-", "-", "-", "-"],
                       ["-", "-", "X", "I", "-"],
                       ["-", "-", "-", "-", "-"],
                       ["-", "-", "I", "-", "-"]]),
             np.array([["I", "G", "B", "-", "-"],
                       ["-", "-", "-", "-", "-"],
                       ["-", "-", "X", "-", "-"],
                       ["-", "-", "-", "-", "-"],
                       ["-", "-", "-", "-", "-"]]),
             np.array([["B", "G", "I", "-", "-"],
                       ["-", "-", "-", "-", "-"],
                       ["-", "-", "X", "-", "-"],
                       ["-", "-", "-", "-", "-"],
                       ["B", "G", "-", "-", "-"]]),
             np.array([["I", "-", "-", "-", "G"],
                       ["-", "-", "-", "-", "-"],
                       ["-", "-", "X", "-", "-"],
                       ["-", "-", "-", "-", "-"],
                       ["B", "-", "-", "-", "G"]]),
             np.array([["B", "G", "B", "-", "-"],
                       ["B", "I", "G", "-", "-"],
                       ["-", "-", "X", "-", "-"],
                       ["-", "-", "-", "-", "-"],
                       ["-", "-", "-", "-", "-"]]),
             np.array([["-", "I", "B", "-", "G"],
                       ["-", "I", "-", "-", "-"],
                       ["I", "I", "X", "-", "-"],
                       ["-", "-", "-", "-", "-"],
                       ["-", "-", "-", "-", "G"]]),
             np.array([["I", "-", "I", "I", "-"],
                       ["-", "-", "-", "-", "-"],
                       ["-", "-", "X", "-", "-"],
                       ["G", "-", "-", "-", "-"],
                       ["B", "-", "-", "-", "-"]]),
             np.array([["G", "-", "I", "-", "-"],
                       ["G", "I", "I", "-", "-"],
                       ["B", "I", "X", "-", "-"],
                       ["B", "-", "-", "-", "-"],
                       ["B", "-", "-", "-", "-"]]),
             np.array([["-", "-", "I", "B", "B"],
                       ["-", "-", "-", "-", "-"],
                       ["-", "-", "X", "-", "-"],
                       ["-", "-", "-", "-", "-"],
                       ["G", "B", "-", "-", "-"]]),
             np.array([["-", "-", "-", "-", "-"],
                       ["-", "-", "-", "-", "-"],
                       ["-", "-", "X", "B", "B"],
                       ["-", "-", "I", "I", "I"],
                       ["-", "-", "-", "B", "G"]]),
             np.array([["B", "I", "-", "-", "-"],
                       ["G", "G", "I", "-", "I"],
                       ["-", "-", "X", "-", "-"],
                       ["-", "-", "-", "-", "-"],
                       ["-", "-", "-", "-", "-"]]),
             np.array([["-", "I", "-", "-", "-"],
                       ["-", "G", "I", "G", "-"],
                       ["-", "-", "X", "I", "-"],
                       ["-", "-", "-", "-", "-"],
                       ["-", "I", "-", "-", "-"]]),
             np.array([["I", "I", "I", "-", "-"],
                       ["-", "I", "I", "-", "-"],
                       ["-", "-", "X", "-", "-"],
                       ["-", "-", "-", "-", "-"],
                       ["G", "G", "-", "-", "B"]]),
             np.array([["-", "-", "-", "-", "-"],
                       ["-", "-", "-", "-", "-"],
                       ["-", "-", "X", "-", "-"],
                       ["-", "-", "I", "I", "-"],
                       ["-", "-", "B", "G", "I"]]),
             np.array([["I", "B", "-", "-", "B"],
                       ["-", "-", "-", "-", "G"],
                       ["-", "-", "X", "-", "I"],
                       ["-", "-", "-", "-", "-"],
                       ["-", "-", "-", "-", "-"]]),
             np.array([["I", "-", "-", "I", "-"],
                       ["G", "-", "I", "-", "-"],
                       ["B", "-", "X", "-", "-"],
                       ["-", "-", "-", "-", "-"],
                       ["-", "-", "-", "-", "-"]]),
             np.array([["-", "-", "-", "-", "-"],
                       ["-", "-", "-", "-", "-"],
                       ["-", "I", "X", "-", "-"],
                       ["-", "-", "-", "-", "-"],
                       ["I", "G", "B", "-", "-"]]),
             np.array([["B", "I", "-", "-", "-"],
                       ["B", "-", "-", "I", "-"],
                       ["I", "I", "X", "I", "-"],
                       ["G", "-", "-", "-", "-"],
                       ["-", "-", "-", "-", "-"]]),
             np.array([["-", "-", "I", "G", "G"],
                       ["-", "-", "B", "B", "B"],
                       ["-", "-", "X", "-", "-"],
                       ["-", "I", "-", "-", "-"],
                       ["-", "-", "-", "-", "-"]]),
             np.array([["I", "B", "-", "-", "-"],
                       ["G", "-", "I", "-", "-"],
                       ["B", "-", "X", "-", "-"],
                       ["-", "-", "I", "-", "-"],
                       ["-", "-", "I", "-", "-"]]),
             np.array([["I", "-", "I", "-", "I"],
                       ["-", "-", "-", "-", "-"],
                       ["-", "I", "X", "-", "-"],
                       ["-", "-", "I", "-", "G"],
                       ["-", "-", "I", "-", "B"]]),
             np.array([["-", "-", "I", "G", "B"],
                       ["-", "I", "-", "-", "G"],
                       ["-", "-", "X", "-", "-"],
                       ["-", "-", "-", "-", "-"],
                       ["-", "-", "-", "-", "B"]]),
             np.array([["-", "-", "I", "-", "-"],
                       ["G", "-", "-", "-", "-"],
                       ["I", "-", "X", "-", "-"],
                       ["-", "-", "-", "-", "-"],
                       ["I", "B", "-", "-", "I"]]),
             np.array([["I", "-", "-", "-", "-"],
                       ["-", "-", "I", "G", "G"],
                       ["-", "-", "X", "I", "I"],
                       ["-", "-", "-", "B", "B"],
                       ["-", "-", "-", "-", "-"]]),
             np.array([["I", "-", "-", "-", "-"],
                       ["-", "-", "-", "-", "I"],
                       ["-", "-", "X", "I", "G"],
                       ["-", "-", "I", "B", "G"],
                       ["-", "-", "-", "B", "B"]]),
             np.array([["I", "-", "I", "-", "-"],
                       ["-", "-", "-", "-", "-"],
                       ["-", "-", "X", "-", "-"],
                       ["-", "-", "B", "B", "B"],
                       ["-", "-", "I", "G", "B"]]),
             np.array([["I", "I", "-", "-", "-"],
                       ["-", "-", "I", "-", "-"],
                       ["B", "-", "X", "-", "-"],
                       ["I", "I", "-", "-", "-"],
                       ["B", "G", "-", "-", "-"]]),
             np.array([["-", "I", "B", "-", "B"],
                       ["-", "I", "B", "-", "G"],
                       ["-", "-", "X", "I", "B"],
                       ["-", "I", "I", "I", "-"],
                       ["-", "-", "-", "-", "-"]]),
             np.array([["I", "B", "G", "B", "I"],
                       ["-", "I", "I", "-", "-"],
                       ["-", "-", "X", "-", "-"],
                       ["-", "-", "-", "-", "-"],
                       ["-", "B", "I", "-", "-"]]),
             np.array([["-", "I", "I", "-", "-"],
                       ["-", "-", "-", "-", "-"],
                       ["-", "I", "X", "-", "-"],
                       ["I", "B", "I", "-", "-"],
                       ["B", "G", "B", "-", "I"]]),
             np.array([["-", "I", "I", "-", "-"],
                       ["-", "-", "-", "-", "-"],
                       ["-", "-", "X", "-", "-"],
                       ["-", "G", "I", "B", "-"],
                       ["-", "I", "B", "G", "-"]]),
             np.array([["I", "I", "-", "-", "I"],
                       ["G", "G", "-", "-", "-"],
                       ["B", "-", "X", "-", "-"],
                       ["-", "-", "-", "-", "-"],
                       ["-", "-", "I", "-", "-"]]),
             np.array([["-", "-", "-", "-", "-"],
                       ["-", "-", "-", "-", "-"],
                       ["-", "-", "X", "I", "-"],
                       ["-", "-", "I", "G", "B"],
                       ["I", "-", "I", "I", "G"]]),
             np.array([["-", "I", "-", "I", "-"],
                       ["-", "-", "-", "-", "-"],
                       ["-", "I", "X", "-", "-"],
                       ["-", "G", "B", "I", "-"],
                       ["-", "G", "B", "I", "-"]]),
             np.array([["-", "G", "I", "B", "-"],
                       ["G", "B", "I", "-", "-"],
                       ["I", "-", "X", "-", "-"],
                       ["B", "-", "-", "-", "-"],
                       ["-", "-", "-", "-", "-"]]),
             np.array([["-", "I", "-", "B", "G"],
                       ["-", "I", "-", "G", "B"],
                       ["-", "-", "X", "I", "B"],
                       ["-", "-", "-", "-", "-"],
                       ["-", "-", "-", "I", "-"]]),
             np.array([["-", "-", "I", "B", "G"],
                       ["-", "I", "-", "B", "G"],
                       ["-", "I", "X", "-", "-"],
                       ["-", "-", "I", "-", "-"],
                       ["-", "-", "-", "-", "-"]]),
             np.array([["I", "B", "I", "-", "-"],
                       ["G", "-", "-", "-", "-"],
                       ["G", "-", "X", "-", "I"],
                       ["-", "-", "-", "-", "I"],
                       ["-", "-", "I", "-", "-"]]),
             np.array([["I", "-", "I", "G", "G"],
                       ["-", "-", "-", "B", "I"],
                       ["-", "-", "X", "I", "-"],
                       ["-", "-", "I", "-", "-"],
                       ["-", "-", "-", "-", "-"]]),
             np.array([["I", "-", "-", "-", "G"],
                       ["G", "-", "I", "-", "B"],
                       ["B", "-", "X", "I", "-"],
                       ["-", "-", "I", "-", "-"],
                       ["-", "I", "-", "-", "-"]])]

    board_num = 3
    moves = [board[board_num - 1]]
    g = nx.DiGraph()
    g.add_node(0, color='#00ff1e')
    node_end = []
    tiltRecursive(board[board_num - 1], moves, g, node_end)
    
    #fundamental matrix
    print(f"{node_end}")

    #condensation
    scc = list(nx.strongly_connected_components(g))
    c = nx.condensation(g, scc)
    dict_node_condense = {}
    n = len(scc)
    i = 0
    while i < n:
        #create a dictionary of nodes as a keys and condensation as values
        dict_node_condense[i] = scc[i]
        i += 1
    # print(dict_node_condense)

    #make a table: one column is number 0, 1, 2,...; and the other column is the condensation
    # print(make_table(dict_node_condense))
    # with open(f'table condensation {board_num}.txt', 'w') as f:
    #     f.write(make_table(dict_node_condense))

    #fundamental matrix
    np.set_printoptions(linewidth=np.inf)
    t = transition_matrix(g)
    print(t)
    # drop the absorbing state
    for i in node_end:
        q = np.delete(t, int(i), 0)
        q = np.delete(q, int(i), 1)
    # print(q)
    with open(f'q matrix {board_num}.txt', 'w') as f:
        f.write(str(q))
    #calculate the fundamental matrix
    # f = fundamental_matrix(q)
    # print(f)

    #condensation graph
    # nx.draw(c, with_labels=True, font_weight='bold')
    # plt.show()
    # while (True):
    #     num = int(input("What is the configuration for this node? "))
    #     if input != "n":
    #         print(scc[num])
    #         continue
    #     else:
    #         break

if __name__ == '__main__':
    main()
