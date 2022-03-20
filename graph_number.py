from collections import defaultdict
from heapq import *
from http.client import NETWORK_AUTHENTICATION_REQUIRED
from platform import node
from pyvis import network as net
import numpy as np
from tabulate import tabulate

# g = nx.Graph("800px", "1100px", directed=True)

"""A program to generate graphs where each node represent a game move. The label of each node
is 1, 2, 3,... instead of the whole board configuration."""

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

    board_num = 1
    moves = [board[board_num - 1]]
    #Dijkstra's algorithms
    edges = []
    node_end = []

    #Generate graphs for all cards
    dict_nodes_edges = {'0': str(board[board_num - 1])}
    g = net.Network("800px", "1100px", directed=True)
    g.add_node(0, color='#00ff1e')
    tiltRecursive(dict_nodes_edges, board[board_num - 1], moves, g, edges, node_end)
    print(f"{node_end}")

    #Dijkstra's algorithms
    print("Starting -> Winning: \n", end="")
    dijkstra_tuple = (dijkstra(edges, "0", f"{int(node_end[0])}"))
    print(dijkstra_tuple)
    print(make_table_dijkstra(dijkstra_tuple))
 
    # print(len(moves))
    # print(dict_nodes_edges)

    #make a table: one column is number 0, 1, 2,...; and the other column is the board configuration
    # print(make_table(dict_nodes_edges))
    # with open(f'table {board_num}.txt', 'w') as f:
    #     f.write(make_table(dict_nodes_edges))

    #make a table: one column is order 0, 1, 2,...; and the other column is the shortest path to the 
    #winning state
    # print(make_table(dict_nodes_edges))
    # with open(f'table {board_num}.txt', 'w') as f:
    #     f.write(make_table(dict_nodes_edges))

    # with open(f'table shortest path {board_num}.txt', 'w') as f:
    #     f.write(make_table_dijkstra(dijkstra_tuple))

    # g.show(f"card #{board_num}.html")

    # while (True):
    #     num = int(input("What is the configuration for this node? "))
    #     if input != "n":
    #         print(moves[num])
    #         continue
    #     else:
    #         break

if __name__ == '__main__':
    main()
