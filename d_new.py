from collections import defaultdict
from heapq import *
import numpy as np

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

def getBoardIndices(board):
    i = 0
    for i in range(len(moves)):
        if np.array_equal(moves[i], board):
            return i

def tiltRecursiveEdge(board, moves, edges):
    board_Left = tiltLeft(board)
    board_Right = tiltRight(board)
    board_Up = tiltUp(board)
    board_Down = tiltDown(board)

    if not np.array_equal(board_Left, board):
        if not findMoves(board_Left, moves):
            if not green(board_Left):
                moves.append(board_Left)
                # add the node of the board_Left
                # g.add_node(str(board_Left))
                # add the edge between the board_Left and the current board
                edges.append((str(getBoardIndices(board)), str(getBoardIndices(board_Left)), 1))
                tiltRecursiveEdge(board_Left, moves, edges)
            elif green(board_Left):
                moves.append(board_Left)
                # add the node of the board_Right
                # g.add_node(str(board_Left))
                # add the edge between the board_Right and the current board
                edges.append((str(getBoardIndices(board)), str(getBoardIndices(board_Left)), 1))
                edges.append((str(getBoardIndices(board_Left)), str(getBoardIndices(board_Left)), 1))
                print(((str(getBoardIndices(board_Left)), str(getBoardIndices(board_Left)), 1)))
        elif findMoves(board_Left, moves):
            # add the node of the board_Left
            # g.add_node(str(board_Left))
            # add the edge between the board_Left and the current board
            edges.append((str(getBoardIndices(board)), str(getBoardIndices(board_Left)), 1))

    if not np.array_equal(board_Right, board):
        if not findMoves(board_Right, moves):
            if not green(board_Right):
                moves.append(board_Right)
                # add the node of the board_Right
                # g.add_node(str(board_Right))
                # add the edge between the board_Right and the current board
                edges.append((str(getBoardIndices(board)), str(getBoardIndices(board_Right)), 1))
                edges.append((str(getBoardIndices(board)), str(getBoardIndices(board_Right)), 1))
                tiltRecursiveEdge(board_Right, moves, edges)
            elif green(board_Right):
                moves.append(board_Right)
                # add the node of the board_Right
                # g.add_node(str(board_Right))
                # add the edge between the board_Right and the current board
                edges.append((str(getBoardIndices(board)), str(getBoardIndices(board_Right)), 1))
                edges.append((str(getBoardIndices(board_Right)), str(getBoardIndices(board_Right)), 1))
                print(((str(getBoardIndices(board_Right)), str(getBoardIndices(board_Right)), 1)))
        elif findMoves(board_Right, moves):
            # add the node of the board_Right
            # g.add_node(str(board_Right))
            # add the edge between the board_Right and the current board
            edges.append(((str(getBoardIndices(board)), str(getBoardIndices(board_Right)), 1)))

    if not np.array_equal(board_Up, board):
        if not findMoves(board_Up, moves):
            if not green(board_Up):
                moves.append(board_Up)
                # add the node of the board_Up
                # g.add_node(str(board_Up))
                # add the edge between the board_Up and the current board
                edges.append((str(getBoardIndices(board)), str(getBoardIndices(board_Up)), 1))
                tiltRecursiveEdge(board_Up, moves, edges)
            elif green(board_Up):
                moves.append(board_Up)
                # add the node of the board_Right
                # g.add_node(str(board_Up))
                # add the edge between the board_Right and the current board
                edges.append((str(getBoardIndices(board)), str(getBoardIndices(board_Up)), 1))
                edges.append((str(getBoardIndices(board_Up)), str(getBoardIndices(board_Up)), 1))
                print(((str(getBoardIndices(board_Up)), str(getBoardIndices(board_Up)), 1)))
        elif findMoves(board_Up, moves):
            # add the node of the board_Up
            # g.add_node(str(board_Up))
            # add the edge between the board_Up and the current board
            edges.append((str(getBoardIndices(board)), str(getBoardIndices(board_Up)), 1))

    if not np.array_equal(board_Down, board):
        if not findMoves(board_Down, moves):
            if not green(board_Down):
                moves.append(board_Down)
                # add the node of the board_Down
                # g.add_node(str(board_Down))
                # add the edge between the board_Down and the current board
                edges.append((str(getBoardIndices(board)), str(getBoardIndices(board_Down)), 1))
                tiltRecursiveEdge(board_Down, moves, edges)
            elif green(board_Down):
                moves.append(board_Down)
                # add the node of the board_Right
                # g.add_node(str(board_Down))
                # add the edge between the board_Right and the current board
                edges.append((str(getBoardIndices(board)), str(getBoardIndices(board_Down)), 1))
                edges.append((str(getBoardIndices(board_Down)), str(getBoardIndices(board_Down)), 1))
                print(((str(getBoardIndices(board_Down)), str(getBoardIndices(board_Down)), 1)))
        elif findMoves(board_Down, moves):
            # add the node of the board_Down
            # g.add_node(str(board_Down))
            # add the edge between the board_Down and the current board
            edges.append((str(getBoardIndices(board)), str(getBoardIndices(board_Down)), 1))

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

if __name__ == "__main__":
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
    board_num = 0
    moves = [board[board_num]]
    edges = []
    tiltRecursiveEdge(board[board_num], moves, edges)

    # print("=== Dijkstra ===")
    # print(*edges, sep = "\n")
    print("Starting -> Winning: ", end="")
    print(dijkstra(edges, "0", "8"))

    # while (True):
    #     num = int(input("What is the configuration for this node? "))
    #     if input != "n":
    #         print(moves[num])
    #         continue
    #     else:
    #         break