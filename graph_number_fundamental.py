from collections import defaultdict
from heapq import *
from http.client import NETWORK_AUTHENTICATION_REQUIRED
from platform import node
from pyvis import network as net
import numpy as np
from numpy import linalg as LA
from tabulate import tabulate

# g = nx.Graph("800px", "1100px", directed=True)

"""A program to generate graphs where each node represent a game move. The label of each node
is 1, 2, 3,... instead of the whole board configuration. This also generates a fundamental
matrix for each graph."""

def countGreenSliders(board):
    """Count the current number of green sliders on the board."""
    count_Green = 0
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == "G":
                count_Green += 1
    return count_Green


def tiltRight(board):
    """Obtain the new board configuration after a right tilt."""
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
    """Obtain the new board configuration after a left tilt."""
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
    """Obtain the new board configuration after a down tilt."""
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
    """Obtain the new board configuration after a up tilt."""
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
    """Check whether there is any green slider on the board. Return True if there is. Return False if there isn't."""
    count_Green = countGreenSliders(board)
    if count_Green == 0:
        return True
    else:
        return False


def findMoves(board, moves):
    """Check whether the board configuration exists. Return True if it does. Return False if there it doesn't."""
    for i in moves:
        if np.array_equal(i, board):
            return True
    return False

# def getBoardIndices(board):
#     i = 0
#     for i in range(len(moves)):
#         if np.array_equal(moves[i], board):
#             return i

def returnBoardIndex(board, moves):
    """Return the index of a certain board in the array. Return "board doesn't exist" if the board
    configuration can't be found."""
    for i in range(len(moves)):
        if np.array_equal(moves[i], board):
            return i
    return "board doesn't exist"

def get_key(dict, val):
    """Return key for a value in the list"""
    for key, value in dict.items():
        if val == value:
            return key

    return "key doesn't exist"

def create_graph(dict, moves, g, board_direction, board_original, title_edge, edges, node_end):
    """Create a graph for all possible game moves. Here, we add edges with weight = 1 to
    the edges array for the Dijkstra's algorithm.
    dict =  dictionary for number as a key and board configuration as a value.
    moves = array that stores all the board configurations.
    board_direction =  board configuration after a move (up, down, left, right).
    title_edge = "U, D, L, E".
    edges =  array that stores all edges.
    node_end = the winning state (stored as a string)."""
    if not np.array_equal(board_direction, board_original):
        if not findMoves(board_direction, moves):
            if not green(board_direction):
                moves.append(board_direction)
                dict[f"{returnBoardIndex(board_direction, moves)}"] = str(board_direction)
                g.add_node(returnBoardIndex(board_direction, moves))
                # add the edge between the board_Left and the current board
                g.add_edge(returnBoardIndex(board_original, moves), returnBoardIndex(board_direction, moves), title=title_edge)
                edges.append((str(returnBoardIndex(board_original, moves)), str(returnBoardIndex(board_direction, moves)), 1))
                tiltRecursive(dict, board_direction, moves, g, edges, node_end)
            elif green(board_direction):
                moves.append(board_direction)
                dict[f"{returnBoardIndex(board_direction, moves)}"] = str(board_direction)
                g.add_node(returnBoardIndex(board_direction, moves))
                # add the edge between the board_Right and the current board
                g.add_edge(returnBoardIndex(board_original, moves), returnBoardIndex(board_direction, moves), title=title_edge)
                g.add_edge(returnBoardIndex(board_direction, moves), returnBoardIndex(board_direction, moves), title=title_edge)
                edges.append((str(returnBoardIndex(board_original, moves)), str(returnBoardIndex(board_direction, moves)), 1))
                edges.append((str(returnBoardIndex(board_direction, moves)), str(returnBoardIndex(board_direction, moves)), 1))
                node_end.append(str(returnBoardIndex(board_direction, moves)))
        elif findMoves(board_direction, moves):
            # add the node of the board_Left
            g.add_node(returnBoardIndex(board_direction, moves))
            # add the edge between the board_Left and the current board
            g.add_edge(returnBoardIndex(board_original, moves), returnBoardIndex(board_direction, moves), title=title_edge)
            edges.append((str(returnBoardIndex(board_original, moves)), str(returnBoardIndex(board_direction, moves)), 1))

def tiltRecursive(dict, board, moves, g, edges, node_end):
    """Tilt the board to all possible directions."""
    board_Left = tiltLeft(board)
    board_Right = tiltRight(board)
    board_Up = tiltUp(board)
    board_Down = tiltDown(board)

    create_graph(dict, moves, g, board_Left, board, "L", edges, node_end)

    create_graph(dict, moves, g, board_Right, board, "R", edges, node_end)

    create_graph(dict, moves, g, board_Up, board, "U", edges, node_end)

    create_graph(dict, moves, g, board_Down, board, "D", edges, node_end)

#Source: https://gist.github.com/kachayev/5990802?permalink_comment_id=2339584
def dijkstra(edges, f, t):
    g = defaultdict(list)
    for l, r, c in edges:
        g[l].append((c, r))

    q, seen, mins = [(0, f, [])], set(), {f: 0}
    while q:
        (cost, v1, path) = heappop(q)
        if v1 not in seen:
            seen.add(v1)
            path = [v1] + path
            if v1 == t:
                return (cost, path)

            for c, v2 in g.get(v1, ()):
                if v2 in seen:
                    continue
                prev = mins.get(v2, None)
                next = cost + c
                if prev is None or next < prev:
                    mins[v2] = next
                    heappush(q, (next, v2, path))

    return (float("inf"), [])

def make_table(dict):
    """Make a table for the board configurations. One column is the node number,
    one column is the corresponding board configuration."""
    # return tabulate(dict, headers='keys')
    table = [["Number", "Nodes"]]
    for things in dict:
        table.append([things, dict[things]])
    return tabulate(table, headers='firstrow', tablefmt='fancy_grid')

def make_table_dijkstra(tuple):
    """Make a table for the board configurations. One column is the node number,
    one column is the corresponding board configuration."""
    # return tabulate(dict, headers='keys')
    table = [["Order", "Path"]]
    for i in range(len(tuple[1])):
        table.append([i, tuple[1][len(tuple[1]) - 1 - i]])
    return tabulate(table, headers='firstrow', tablefmt='fancy_grid')

def convert_list_to_matrix(adje_list):
    """Convert a list in this form {0: {1}, 1: {0, 2}} to an adjacency matrix"""
    dict_len = len(adje_list)
    mat = np.zeros((dict_len,dict_len), dtype=int) #initialize a zero matrix
    # print(mat)
    for key in adje_list:
        values = adje_list[key]
        # print(key)
        # print(values)
        list_values = list(values)
        for i in range(len(list_values)):
            mat[key][list_values[i]] = 1
    return mat

def transition_matrix(adj_matrix):
    """A method that generates a transition matrix for a graph of a game card. In a transition matrix,
    each entry at a position (i,j) corresponding to a probability to transition from node i to node j.
    Source: https://stackoverflow.com/questions/37311651/get-node-list-from-random-walk-in-networkx
    Source: https://math.stackexchange.com/questions/1070681/how-to-define-a-transition-matrix-mathematically"""   

    # let's evaluate the degree matrix D
    D = np.diag(np.sum(adj_matrix, axis=1))
    # print(f"The degree matrix is \n{D}.")
    # ...and the transition matrix T. T = D^(-1) A. 
    T = np.dot(np.linalg.inv(D),adj_matrix)
    return T

def fundamental_matrix(q):
    """Calculate the fundamental matrix with matrix q as the parameter. Source: 
    https://stackoverflow.com/questions/11705733/best-way-to-calculate-the-fundamental-matrix-of-an-absorbing-markov-chain"""
    I = np.identity(q.shape[0]) #get the identity matrix that has the same shape as the matrix Q.
    # o = np.ones(Q.shape[0])
    F = np.linalg.inv(I-q)
    return F

def delete_column_q(num_column, absorbing_rows_cols_to_delete):
    """Delete the column of the transition matrix to get the matrix R"""
    nonabsorbing_to_delete = []
    for i in range(num_column):
        nonabsorbing_to_delete.append(i)
    for itemRow in absorbing_rows_cols_to_delete:
        for itemCol in nonabsorbing_to_delete:
            if itemRow == itemCol:
                nonabsorbing_to_delete.remove(itemRow)
    return nonabsorbing_to_delete

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

    board_num = int(input("Enter the board number: "))
    moves = [board[board_num - 1]]
    #Dijkstra's algorithms
    edges = []
    node_end = []

    # -----------------------------------------------------------
    # Generate graphs for all cards
    # -----------------------------------------------------------
    dict_nodes_edges = {'0': str(board[board_num - 1])}
    g = net.Network("800px", "1100px", directed=True)
    g.add_node(0, color='#00ff1e')
    tiltRecursive(dict_nodes_edges, board[board_num - 1], moves, g, edges, node_end)
    # print(f"{node_end}")

    # -----------------------------------------------------------
    # Dijkstra's algorithms to find the shortest path that leads to victory. The output matches
    # the solutions behind each challenge card. 
    # -----------------------------------------------------------
    # print("Starting -> Winning: \n", end="")
    # dijkstra_tuple = (dijkstra(edges, "0", f"{int(node_end[0])}"))
    # print(dijkstra_tuple)
    # print(make_table_dijkstra(dijkstra_tuple))
 
    # print(len(moves))
    # print(dict_nodes_edges)

    # -----------------------------------------------------------
    # 1. Get the adjacency list. 
    # 2. Then, get the adjacency matrix.
    # 3. Get the matrix Q by dropping the absorbing states.
    # -----------------------------------------------------------
    
    # get adjacency list. This is to add a loop to any node that does not have an outgoing edge.    
    adj_list = g.get_adj_list()

    # get adjacency matrix
    adj_matrix = convert_list_to_matrix(adj_list)
    # print(f"The adjacency matrix is {adj_list}")
    for key in adj_list:
        values = adj_list[key]
        if len(values) == 0:
            # print(f"{key} is valid")
            g.add_edge(key, key, title="L")
            adj_matrix[key][key] = 1 #change the value of end nodes that are just added a loop to. 
    print(f"The adjacency matrix of card {board_num} is: \n{adj_matrix}") 
    # g.show(f"card #{board_num}.html")

    # calculate the fundamental matrix
    np.set_printoptions(linewidth=np.inf)
    t = transition_matrix(adj_matrix)
    # debug degree matrix
    D = np.diag(np.sum(adj_matrix, axis=1))
    num_column = D.shape[1]
    print(f"The degree matrix of card {board_num} is \n{D}.")
    # for i in range(num_column):
    #     print(f"The index for {i} is {D[i][i]}");
    t_round = t
    print(f"The transition matrix of card {board_num} is: \n{t_round}")

    # -----------------------------------------------------------
    # For cards have NO losing absorbing states
    # -----------------------------------------------------------
    # Drop the absorbing state to get the matrix Q, for cards have NO losing absorbing states. 
    if board_num in [1, 2, 3, 4, 5, 6, 7, 8, 10, 14, 15, 16, 17, 20, 21, 23, 26, 27, 29, 30, 33]:
        node_end_int = list(map(int, node_end))

        matrix_Q = np.delete(t_round, node_end_int, 0)
        matrix_Q = np.delete(matrix_Q, node_end_int, 1)
        print(f"The matrix Q of card {board_num} is: \n{matrix_Q}")

        # Get the matrix R
        t_delete_row = np.delete(t_round, node_end_int, 0)
        nonabsorbing_to_delete = delete_column_q(num_column, node_end_int)
        matrix_R = np.delete(t_delete_row, nonabsorbing_to_delete, 1)
        print(f"The matrix R of card {board_num} is: \n{matrix_R}")

    # -----------------------------------------------------------
    # For cards HAVE losing absorbing states
    # -----------------------------------------------------------
    # Edit the fundamental matrix for cards have losing absorbing states,
    # first delete rows and columns of nodes in the absorbing states
    # -----------------------------------------------------------
    else:
        if board_num == 40:
            # Determine the rows and columns of the vertices that need to be removed from the matrix T to get the matrix Q.
            absorbing_rows_cols_to_delete = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                             23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                                             43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                                             63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
                                             83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
                                             102, 103, 104, 106, 105, 107, 108, 261, 262, 263, 264, 265, 266, 267, 268, 269,
                                             270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285,
                                             286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301,
                                             302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317,
                                             318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333,
                                             334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 365, 366,
                                             363]
            matrix_Q_columns_deleted = np.delete(t_round, absorbing_rows_cols_to_delete, 1)
            matrix_Q = np.delete(matrix_Q_columns_deleted, absorbing_rows_cols_to_delete, 0)

        elif board_num == 39:
            # Determine the rows and columns of the vertices that need to be removed from the matrix T to get the matrix Q.
            absorbing_rows_cols_to_delete = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                             41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                                             122, 181, 182, 180, 183]
            matrix_Q_columns_deleted = np.delete(t_round, absorbing_rows_cols_to_delete, 1)
            matrix_Q = np.delete(matrix_Q_columns_deleted, absorbing_rows_cols_to_delete, 0)

        elif board_num == 38:
            # Determine the rows and columns of the vertices that need to be removed from the matrix T to get the matrix Q.
            absorbing_rows_cols_to_delete = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 79]
            matrix_Q_columns_deleted = np.delete(t_round, absorbing_rows_cols_to_delete, 1)
            matrix_Q = np.delete(matrix_Q_columns_deleted, absorbing_rows_cols_to_delete, 0)

        elif board_num == 37:
            # Determine the rows and columns of the vertices that need to be removed from the matrix T to get the matrix Q.
            absorbing_rows_cols_to_delete = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                             32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                             49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 128, 127, 182]
            matrix_Q_columns_deleted = np.delete(t_round, absorbing_rows_cols_to_delete, 1)
            matrix_Q = np.delete(matrix_Q_columns_deleted, absorbing_rows_cols_to_delete, 0)

        elif board_num == 36:
            # Determine the rows and columns of the vertices that need to be removed from the matrix T to get the matrix Q.
            absorbing_rows_cols_to_delete = [2, 3, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 44, 45,
                                             46, 47, 48, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                                             62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                                             76, 77, 78, 79, 80]
            matrix_Q_columns_deleted = np.delete(t_round, absorbing_rows_cols_to_delete, 1)
            matrix_Q = np.delete(matrix_Q_columns_deleted, absorbing_rows_cols_to_delete, 0)

        elif board_num == 35:
            # Determine the rows and columns of the vertices that need to be removed from the matrix T to get the matrix Q.
            absorbing_rows_cols_to_delete = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 72, 73, 75, 76, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
            101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 139, 140, 155, 156, 174, 175, 193,
            194, 138, 185]
            matrix_Q_columns_deleted = np.delete(t_round, absorbing_rows_cols_to_delete, 1)
            matrix_Q = np.delete(matrix_Q_columns_deleted, absorbing_rows_cols_to_delete, 0)

        elif board_num == 34:
            # Determine the rows and columns of the vertices that need to be removed from the matrix T to get the matrix Q.
            absorbing_rows_cols_to_delete = [21, 33, 34, 35, 47]
            matrix_Q_columns_deleted = np.delete(t_round, absorbing_rows_cols_to_delete, 1)
            matrix_Q = np.delete(matrix_Q_columns_deleted, absorbing_rows_cols_to_delete, 0)

        elif board_num == 32:
            # Determine the rows and columns of the vertices that need to be removed from the matrix T to get the matrix Q.
            absorbing_rows_cols_to_delete = [6, 7, 14, 15, 16, 35, 36, 37, 27]
            matrix_Q_columns_deleted = np.delete(t_round, absorbing_rows_cols_to_delete, 1)
            matrix_Q = np.delete(matrix_Q_columns_deleted, absorbing_rows_cols_to_delete, 0)

        elif board_num == 31:
            # Determine the rows and columns of the vertices that need to be removed from the matrix T to get the matrix Q.
            absorbing_rows_cols_to_delete = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                                             39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                                             59, 60, 61, 62, 63, 64, 65, 17, 18, 12, 13, 14, 15, 16, 66, 11, 77, 78, 69, 70,
                                             71, 72, 73, 74, 75, 76, 79, 80, 81, 82, 67, 68, 9, 10, 115, 116, 123, 124, 102,
                                             103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 117, 118, 119, 120,
                                             121, 122, 125, 126, 127, 128, 100, 101, 132, 133, 134, 135, 130, 131, 97]
            matrix_Q_columns_deleted = np.delete(t_round, absorbing_rows_cols_to_delete, 1)
            matrix_Q = np.delete(matrix_Q_columns_deleted, absorbing_rows_cols_to_delete, 0)

        elif board_num == 28:
            # Determine the rows and columns of the vertices that need to be removed from the matrix T to get the matrix Q.
            absorbing_rows_cols_to_delete = [20, 21, 73, 74, 75, 89, 93, 98, 99, 100, 91]
            matrix_Q_columns_deleted = np.delete(t_round, absorbing_rows_cols_to_delete, 1)
            matrix_Q = np.delete(matrix_Q_columns_deleted, absorbing_rows_cols_to_delete, 0)

        elif board_num == 25:
            # Determine the rows and columns of the vertices that need to be removed from the matrix T to get the matrix Q.
            absorbing_rows_cols_to_delete = [45, 46, 47, 36, 37, 38, 39, 40, 41, 42, 58, 59, 29]
            matrix_Q_columns_deleted = np.delete(t_round, absorbing_rows_cols_to_delete, 1)
            matrix_Q = np.delete(matrix_Q_columns_deleted, absorbing_rows_cols_to_delete, 0)

        elif board_num == 24:
            # Determine the rows and columns of the vertices that need to be removed from the matrix T to get the matrix Q.
            absorbing_rows_cols_to_delete = [36, 37, 38, 44, 45, 46, 32, 33, 34, 35, 39, 40, 41, 42, 43, 74]
            matrix_Q_columns_deleted = np.delete(t_round, absorbing_rows_cols_to_delete, 1)
            matrix_Q = np.delete(matrix_Q_columns_deleted, absorbing_rows_cols_to_delete, 0)

        elif board_num == 22:
            # Determine the rows and columns of the vertices that need to be removed from the matrix T to get the matrix Q.
            absorbing_rows_cols_to_delete = [10, 11, 9, 12, 13, 14, 15, 17, 18, 23, 24, 16, 19, 20, 21, 22, 7]
            matrix_Q_columns_deleted = np.delete(t_round, absorbing_rows_cols_to_delete, 1) # first delete the columns
            matrix_Q = np.delete(matrix_Q_columns_deleted, absorbing_rows_cols_to_delete, 0) # then delete the rows to get the matrix Q

        elif board_num == 19:
            # Determine the rows and columns of the vertices that need to be removed from the matrix T to get the matrix Q.
            absorbing_rows_cols_to_delete = [14, 15, 16, 17, 18, 19, 12]
            matrix_Q_columns_deleted = np.delete(t_round, absorbing_rows_cols_to_delete, 1)
            matrix_Q = np.delete(matrix_Q_columns_deleted, absorbing_rows_cols_to_delete, 0)

        elif board_num == 18:
            # Determine the rows and columns of the vertices that need to be removed from the matrix T to get the matrix Q.
            absorbing_rows_cols_to_delete = [80, 84, 66, 68, 69, 70, 71, 75, 76, 77]
            matrix_Q_columns_deleted = np.delete(t_round, absorbing_rows_cols_to_delete, 1) #drop the absorbing state
            matrix_Q = np.delete(matrix_Q_columns_deleted, absorbing_rows_cols_to_delete, 0) #drop the absorbing state

        elif board_num == 13:
            # Determine the rows and columns of the vertices that need to be removed from the matrix T to get the matrix Q.
            absorbing_rows_cols_to_delete = [1, 2, 16, 17, 18, 15, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 36]
            matrix_Q_columns_deleted = np.delete(t_round, absorbing_rows_cols_to_delete, 1)
            matrix_Q = np.delete(matrix_Q_columns_deleted, absorbing_rows_cols_to_delete, 0)

        elif board_num == 12:
            # Determine the rows and columns of the vertices that need to be removed from the matrix T to get the matrix Q.
            absorbing_rows_cols_to_delete = [5, 6, 7, 8, 11, 12, 32, 33, 31, 9, 10, 28]
            matrix_Q_columns_deleted = np.delete(t_round, absorbing_rows_cols_to_delete, 1)
            matrix_Q = np.delete(matrix_Q_columns_deleted, absorbing_rows_cols_to_delete, 0)

        elif board_num == 11:
            # Determine the rows and columns of the vertices that need to be removed from the matrix T to get the matrix Q.
            absorbing_rows_cols_to_delete = [16, 17, 18, 19, 20, 21, 26]
            matrix_Q_columns_deleted = np.delete(t_round, absorbing_rows_cols_to_delete, 1)
            matrix_Q = np.delete(matrix_Q_columns_deleted, absorbing_rows_cols_to_delete, 0)

        elif board_num == 9:
            # Determine the rows and columns of the vertices that need to be removed from the matrix T to get the matrix Q.
            absorbing_rows_cols_to_delete = [4, 5, 6, 7, 16]
            matrix_Q_columns_deleted = np.delete(t_round, absorbing_rows_cols_to_delete, 1)
            matrix_Q = np.delete(matrix_Q_columns_deleted, absorbing_rows_cols_to_delete, 0)

        print(f"The matrix Q of card {board_num} is \n{matrix_Q}")
        # -----------------------------------------------------------
        # Calculate the matrix R for cards with losing absorbing states.
        # -----------------------------------------------------------
        losing_indices = []
        matrix_R = []
        winning_indices = []
        winning_index = 0

        # first, get the matrix that resembles matrix R in the canonical form by deleting the absorbing rows
        # and the nonabsorbing cols. But note that here, the losing absorbing cols are not summed up yet
        delete_absorbing_rows_to_get_R = np.delete(t_round, absorbing_rows_cols_to_delete, 0) #delete the absorbing rows
        nonabsorbing_to_delete = (delete_column_q(num_column, absorbing_rows_cols_to_delete))
        delete_nonabsorbing_cols_to_get_R = np.delete(delete_absorbing_rows_to_get_R, nonabsorbing_to_delete, 1) #delete the nonabsorbing cols
        # print(f"The number of column of delete_absorbing_rows_to_get_R is {delete_absorbing_rows_to_get_R.shape[1]}")
        # print(f"The number of column of delete_nonabsorbing_cols_to_get_R is {delete_nonabsorbing_cols_to_get_R.shape[1]}")
        print(f"The deleted column for card {board_num} is {nonabsorbing_to_delete}")
        print(f"The matrix that resembles matrix R but the losing absorbing states are not summed up yet of card {board_num} is \n{delete_nonabsorbing_cols_to_get_R}")

        # get the columns of the losing absorbing states from the matrix above and then sum all columns of losing absorbing states
        # print(f"The length of the deleted row is {len(absorbing_rows_cols_to_delete)}")

        # find the winning column indices by seeing which column indices that are smaller than the real column indices in the transition
        # matrix
        if board_num == 9:
            winning_index = 16
        elif board_num == 11:
            winning_index = 26
        elif board_num == 12:
            winning_index = 28
        elif board_num == 13:
            winning_index = 36
        elif board_num == 19:
            winning_index = 12
        elif board_num == 22:
            winning_index = 7
        elif board_num == 24:
            winning_index = 74
        elif board_num == 25:
            winning_index = 29
        elif board_num == 28:
            winning_index = 91
        elif board_num == 31:
            winning_index = 97
        elif board_num == 32:
            winning_index = 27
        elif board_num == 34:
            winning_index = 47
        elif board_num == 35:
            winning_index = 185
        elif board_num == 36:
            winning_index = 44
        elif board_num == 37:
            winning_index = 182
        elif board_num == 38:
            winning_index = 79
        elif board_num == 39:
            winning_index = 122
        elif board_num == 40:
            winning_index = 363

        if board_num == 18:
            for i in range(0, 8): # because card 18 has multiple winning states
                winning_indices.append(i)
        else:
            winning_index_deep_copy = winning_index # make a copy so that in the index is not updated in the comparison in the loop.
            for i in nonabsorbing_to_delete:
                if i < winning_index_deep_copy:
                    winning_index = winning_index - 1
                else:
                    pass
            winning_indices.append(winning_index) #get the winning columns
        print(f"The winning column index for card {board_num} is {winning_indices}")
        losing_cols_to_get_R = np.delete(delete_nonabsorbing_cols_to_get_R, winning_indices, 1)
        # print(f"The number of column of losing_cols_to_get_R is {losing_cols_to_get_R.shape[1]}")
        # print(f"losing_cols_to_get_R is \n{losing_cols_to_get_R}")
        sum_losing_cols_to_get_R=np.sum(losing_cols_to_get_R,axis=1)
        sum_losing_cols_to_get_R_transpose=sum_losing_cols_to_get_R.reshape((-1,1)) #tranpose because sum_losing_cols_to_get_R is horizontal
        print(f"The sum of all column of losing absorbing states of card {board_num} is \n{sum_losing_cols_to_get_R_transpose}")

        # get the column of the winning absorbing states from the matrix above
        if board_num == 18:
            for i in range(len(absorbing_rows_cols_to_delete)):
                if i in winning_indices:
                    pass
                else:
                    losing_indices.append(i)
        else:
            for i in range(len(absorbing_rows_cols_to_delete)):
                if i == winning_index:
                    pass
                else:
                    losing_indices.append(i)
        winning_cols_to_get_R = np.delete(delete_nonabsorbing_cols_to_get_R, losing_indices, 1)
        print(f"The losing column indices of card {board_num} are {losing_indices}")
        print(f"The winning column for card {board_num} is \n{winning_cols_to_get_R}")

        # append the losing and winning states together to have the matrix R
        matrix_R = np.append(sum_losing_cols_to_get_R_transpose, winning_cols_to_get_R, 1)

        # -----------------------------------------------------------
        # Calculate the fundamental matrix for cards
        # -----------------------------------------------------------
        print(f"The matrix after dropping absorbing states of card {board_num} is: \n{matrix_Q}")
        print(f"The matrix R of card {board_num} is \n{matrix_R}")

    f = fundamental_matrix(matrix_Q)
    f_round = f
    print(f"The fundamental matrix of card {board_num} is \n {f_round}")
    print(f"The condition number of fundamental matrix of card {board_num} is \n {LA.cond(f)}")

    # -----------------------------------------------------------
    # Calculate the expected step to the absorbing state
    """The expected number of steps before being absorbed when starting in 
    transient state i is the ith entry of the vector t=N1, where 1 is a 
    length-t column vector whose entries are all 1. Source: 
    https://en.wikipedia.org/wiki/Absorbing_Markov_chain"""
    num_row = f.shape[0]
    arr_ones = np.ones((num_row, 1)) #create an array of ones

    # Array to calculate expected number of steps
    E = np.matmul(f, arr_ones)
    print(f"Matrix to calculate the expected number of steps for card {board_num} is:\n {E}")

    # -----------------------------------------------------------
    # Calculate the probability of absoprtion given that the process starts in nonabsorbing
    # state. B = F.R
    # -----------------------------------------------------------
    """B = F.R"""
    B = np.matmul(f_round, matrix_R)
    print(f"The matrix B of card {board_num} is \n {B}")

    # # to check whether the new edge is added
    # adj_list = g.get_adj_list()
    # print(adj_list)

    # -----------------------------------------------------------
    # Make a table: one column is number 0, 1, 2,...; and the other column is the board configuration
    # -----------------------------------------------------------
    # print(make_table(dict_nodes_edges))
    # with open(f'table {board_num}.txt', 'w') as f:
    #     f.write(make_table(dict_nodes_edges))

    # -----------------------------------------------------------
    # Make a table: one column is order 0, 1, 2,...; and the other column is the shortest path to the 
    # winning state
    # -----------------------------------------------------------
    # print(make_table(dict_nodes_edges))
    # with open(f'table {board_num}.txt', 'w') as f:
    #     f.write(make_table(dict_nodes_edges))

    # with open(f'table shortest path {board_num}.txt', 'w') as f:
    #     f.write(make_table_dijkstra(dijkstra_tuple))

    # while (True):
    #     num = int(input("What is the configuration for this node? "))
    #     if input != "n":
    #         print(moves[num])
    #         continue
    #     else:
    #         break

if __name__ == '__main__':
    main()
