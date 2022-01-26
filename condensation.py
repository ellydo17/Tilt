import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# g = nx.Graph("800px", "1100px", directed=True)

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

def tiltRecursive(board, moves, g):
    board_Left = tiltLeft(board)
    board_Right = tiltRight(board)
    board_Up = tiltUp(board)
    board_Down = tiltDown(board)

    if not np.array_equal(board_Left, board):
        if not findMoves(board_Left, moves):
            if not green(board_Left):
                moves.append(board_Left)
                # add the node of the board_Left
                g.add_node(str(board_Left))
                # add the edge between the board_Left and the current board
                g.add_edge(str(board), str(board_Left), title="L")
                tiltRecursive(board_Left, moves, g)
            elif green(board_Left):
                moves.append(board_Left)
                # add the node of the board_Right
                g.add_node(str(board_Left))
                # add the edge between the board_Right and the current board
                g.add_edge(str(board), str(board_Left), title="L")
                g.add_edge(str(board_Left), str(board_Left), title="L")
        elif findMoves(board_Left, moves):
            # add the node of the board_Left
            g.add_node(str(board_Left))
            # add the edge between the board_Left and the current board
            g.add_edge(str(board), str(board_Left), title="L")

    if not np.array_equal(board_Right, board):
        if not findMoves(board_Right, moves):
            if not green(board_Right):
                moves.append(board_Right)
                # add the node of the board_Right
                g.add_node(str(board_Right))
                # add the edge between the board_Right and the current board
                g.add_edge(str(board), str(board_Right), title="R")
                g.add_edge(str(board), str(board_Right), title="R")
                tiltRecursive(board_Right, moves, g)
            elif green(board_Right):
                moves.append(board_Right)
                # add the node of the board_Right
                g.add_node(str(board_Right))
                # add the edge between the board_Right and the current board
                g.add_edge(str(board), str(board_Right), title="R")
                g.add_edge(str(board_Right), str(board_Right), title="R")
        elif findMoves(board_Right, moves):
            # add the node of the board_Right
            g.add_node(str(board_Right))
            # add the edge between the board_Right and the current board
            g.add_edge(str(board), str(board_Right), title="R")

    if not np.array_equal(board_Up, board):
        if not findMoves(board_Up, moves):
            if not green(board_Up):
                moves.append(board_Up)
                # add the node of the board_Up
                g.add_node(str(board_Up))
                # add the edge between the board_Up and the current board
                g.add_edge(str(board), str(board_Up), title="U")
                tiltRecursive(board_Up, moves, g)
            elif green(board_Up):
                moves.append(board_Up)
                # add the node of the board_Right
                g.add_node(str(board_Up))
                # add the edge between the board_Right and the current board
                g.add_edge(str(board), str(board_Up), title="U")
                g.add_edge(str(board_Up), str(board_Up), title="U")
        elif findMoves(board_Up, moves):
            # add the node of the board_Up
            g.add_node(str(board_Up))
            # add the edge between the board_Up and the current board
            g.add_edge(str(board), str(board_Up), title="U")

    if not np.array_equal(board_Down, board):
        if not findMoves(board_Down, moves):
            if not green(board_Down):
                moves.append(board_Down)
                # add the node of the board_Down
                g.add_node(str(board_Down))
                # add the edge between the board_Down and the current board
                g.add_edge(str(board), str(board_Down), title="D")
                tiltRecursive(board_Down, moves, g)
            elif green(board_Down):
                moves.append(board_Down)
                # add the node of the board_Right
                g.add_node(str(board_Down))
                # add the edge between the board_Right and the current board
                g.add_edge(str(board), str(board_Down), title="D")
                g.add_edge(str(board_Down), str(board_Down), title="D")
        elif findMoves(board_Down, moves):
            # add the node of the board_Down
            g.add_node(str(board_Down))
            # add the edge between the board_Down and the current board
            g.add_edge(str(board), str(board_Down), title="D")

def main():
    # card #1
    # board = np.array([["G", "I", "-", "-", "-"],
    #               ["-", "-", "-", "-", "-"],
    #               ["-", "-", "X", "I", "-"],
    #               ["-", "-", "-", "-", "-"],
    #               ["-", "-", "I", "-", "-"]])
    # card #2
    # board = np.array([["I", "G", "B", "-", "-"],
    #               ["-", "-", "-", "-", "-"],
    #               ["-", "-", "X", "-", "-"],
    #               ["-", "-", "-", "-", "-"],
    #               ["-", "-", "-", "-", "-"]])
    # # card #3
    # board = np.array([["B", "G", "I", "-", "-"],
    #                   ["-", "-", "-", "-", "-"],
    #                   ["-", "-", "X", "-", "-"],
    #                   ["-", "-", "-", "-", "-"],
    #                   ["B", "G", "-", "-", "-"]])
    # card #4
    # board = np.array([["I", "-", "-", "-", "G"],
    #                   ["-", "-", "-", "-", "-"],
    #                   ["-", "-", "X", "-", "-"],
    #                   ["-", "-", "-", "-", "-"],
    #                   ["B", "-", "-", "-", "G"]])
    # card #5
    board = np.array([["B", "G", "B", "-", "-"],
                      ["B", "I", "G", "-", "-"],
                      ["-", "-", "X", "-", "-"],
                      ["-", "-", "-", "-", "-"],
                      ["-", "-", "-", "-", "-"]])
    # card #6
    # board = np.array([["-", "I", "B", "-", "G"],
    #                   ["-", "I", "-", "-", "-"],
    #                   ["I", "I", "X", "-", "-"],
    #                   ["-", "-", "-", "-", "-"],
    #                   ["-", "-", "-", "-", "G"]])
    # card #7
    # board = np.array([["I", "-", "I", "I", "-"],
    #                   ["-", "-", "-", "-", "-"],
    #                   ["-", "-", "X", "-", "-"],
    #                   ["G", "-", "-", "-", "-"],
    #                   ["B", "-", "-", "-", "-"]])
    # card #8
    # board = np.array([["G", "-", "I", "-", "-"],
    #                   ["G", "I", "I", "-", "-"],
    #                   ["B", "I", "X", "-", "-"],
    #                   ["B", "-", "-", "-", "-"],
    #                   ["B", "-", "-", "-", "-"]])
    # card #9
    # board = np.array([["-", "-", "I", "B", "B"],
    #                   ["-", "-", "-", "-", "-"],
    #                   ["-", "-", "X", "-", "-"],
    #                   ["-", "-", "-", "-", "-"],
    #                   ["G", "B", "-", "-", "-"]])
    # # card #10
    # board = np.array([["-", "-", "-", "-", "-"],
    #                   ["-", "-", "-", "-", "-"],
    #                   ["-", "-", "X", "B", "B"],
    #                   ["-", "-", "I", "I", "I"],
    #                   ["-", "-", "-", "B", "G"]])
    # card #11
    # board = np.array([["B", "I", "-", "-", "-"],
    #                   ["G", "G", "I", "-", "I"],
    #                   ["-", "-", "X", "-", "-"],
    #                   ["-", "-", "-", "-", "-"],
    #                   ["-", "-", "-", "-", "-"]])
    # # card #12
    # board = np.array([["-", "I", "-", "-", "-"],
    #                   ["-", "G", "I", "G", "-"],
    #                   ["-", "-", "X", "I", "-"],
    #                   ["-", "-", "-", "-", "-"],
    #                   ["-", "I", "-", "-", "-"]])
    # card #13
    # board = np.array([["I", "I", "I", "-", "-"],
    #                   ["-", "I", "I", "-", "-"],
    #                   ["-", "-", "X", "-", "-"],
    #                   ["-", "-", "-", "-", "-"],
    #                   ["G", "G", "-", "-", "B"]])
    # card #14
    # board = np.array([["-", "-", "-", "-", "-"],
    #                   ["-", "-", "-", "-", "-"],
    #                   ["-", "-", "X", "-", "-"],
    #                   ["-", "-", "I", "I", "-"],
    #                   ["-", "-", "B", "G", "I"]])
    # card #15
    # board = np.array([["I", "B", "-", "-", "B"],
    #                   ["-", "-", "-", "-", "G"],
    #                   ["-", "-", "X", "-", "I"],
    #                   ["-", "-", "-", "-", "-"],
    #                   ["-", "-", "-", "-", "-"]])
    # card #16
    # board = np.array([["I", "-", "-", "I", "-"],
    #                   ["G", "-", "I", "-", "-"],
    #                   ["B", "-", "X", "-", "-"],
    #                   ["-", "-", "-", "-", "-"],
    #                   ["-", "-", "-", "-", "-"]])
    # card #17
    # board = np.array([["-", "-", "-", "-", "-"],
    #                   ["-", "-", "-", "-", "-"],
    #                   ["-", "I", "X", "-", "-"],
    #                   ["-", "-", "-", "-", "-"],
    #                   ["I", "G", "B", "-", "-"]])
    # card #18
    # board = np.array([["B", "I", "-", "-", "-"],
    #                   ["B", "-", "-", "I", "-"],
    #                   ["I", "I", "X", "I", "-"],
    #                   ["G", "-", "-", "-", "-"],
    #                   ["-", "-", "-", "-", "-"]])
    # card #19
    # board = np.array([["-", "-", "I", "G", "G"],
    #                   ["-", "-", "B", "B", "B"],
    #                   ["-", "-", "X", "-", "-"],
    #                   ["-", "I", "-", "-", "-"],
    #                   ["-", "-", "-", "-", "-"]])
    # card #20
    # board = np.array([["I", "B", "-", "-", "-"],
    #                   ["G", "-", "I", "-", "-"],
    #                   ["B", "-", "X", "-", "-"],
    #                   ["-", "-", "I", "-", "-"],
    #                   ["-", "-", "I", "-", "-"]])
    # card #21
    # board = np.array([["I", "-", "I", "-", "I"],
    #                   ["-", "-", "-", "-", "-"],
    #                   ["-", "I", "X", "-", "-"],
    #                   ["-", "-", "I", "-", "G"],
    #                   ["-", "-", "I", "-", "B"]])
    # card #22
    # board = np.array([["-", "-", "I", "G", "B"],
    #                   ["-", "I", "-", "-", "G"],
    #                   ["-", "-", "X", "-", "-"],
    #                   ["-", "-", "-", "-", "-"],
    #                   ["-", "-", "-", "-", "B"]])
    # card #23
    # board = np.array([["-", "-", "I", "-", "-"],
    #                   ["G", "-", "-", "-", "-"],
    #                   ["I", "-", "X", "-", "-"],
    #                   ["-", "-", "-", "-", "-"],
    #                   ["I", "B", "-", "-", "I"]])
    # card #24
    # board = np.array([["I", "-", "-", "-", "-"],
    #                   ["-", "-", "I", "G", "G"],
    #                   ["-", "-", "X", "I", "I"],
    #                   ["-", "-", "-", "B", "B"],
    #                   ["-", "-", "-", "-", "-"]])
    # card #25
    # board = np.array([["I", "-", "-", "-", "-"],
    #                   ["-", "-", "-", "-", "I"],
    #                   ["-", "-", "X", "I", "G"],
    #                   ["-", "-", "I", "B", "G"],
    #                   ["-", "-", "-", "B", "B"]])
    # card #26
    # board = np.array([["I", "-", "I", "-", "-"],
    #                   ["-", "-", "-", "-", "-"],
    #                   ["-", "-", "X", "-", "-"],
    #                   ["-", "-", "B", "B", "B"],
    #                   ["-", "-", "I", "G", "B"]])
    # card #27
    # board = np.array([["I", "I", "-", "-", "-"],
    #                   ["-", "-", "I", "-", "-"],
    #                   ["B", "-", "X", "-", "-"],
    #                   ["I", "I", "-", "-", "-"],
    #                   ["B", "G", "-", "-", "-"]])
    # card #28
    # board = np.array([["-", "I", "B", "-", "B"],
    #                   ["-", "I", "B", "-", "G"],
    #                   ["-", "-", "X", "I", "B"],
    #                   ["-", "I", "I", "I", "-"],
    #                   ["-", "-", "-", "-", "-"]])
    # card #29
    # board = np.array([["I", "B", "G", "B", "I"],
    #                   ["-", "I", "I", "-", "-"],
    #                   ["-", "-", "X", "-", "-"],
    #                   ["-", "-", "-", "-", "-"],
    #                   ["-", "B", "I", "-", "-"]])
    # card #30
    # board = np.array([["-", "I", "I", "-", "-"],
    #                   ["-", "-", "-", "-", "-"],
    #                   ["-", "I", "X", "-", "-"],
    #                   ["I", "B", "I", "-", "-"],
    #                   ["B", "G", "B", "-", "I"]])
    # card #31
    # board = np.array([["-", "I", "I", "-", "-"],
    #                   ["-", "-", "-", "-", "-"],
    #                   ["-", "-", "X", "-", "-"],
    #                   ["-", "G", "I", "B", "-"],
    #                   ["-", "I", "B", "G", "-"]])
    # card #32
    # board = np.array([["I", "I", "-", "-", "I"],
    #                   ["G", "G", "-", "-", "-"],
    #                   ["B", "-", "X", "-", "-"],
    #                   ["-", "-", "-", "-", "-"],
    #                   ["-", "-", "I", "-", "-"]])
    # card #33
    # board = np.array([["-", "-", "-", "-", "-"],
    #                   ["-", "-", "-", "-", "-"],
    #                   ["-", "-", "X", "I", "-"],
    #                   ["-", "-", "I", "G", "B"],
    #                   ["I", "-", "I", "I", "G"]])
    # card #34
    # board = np.array([["-", "I", "-", "I", "-"],
    #                   ["-", "-", "-", "-", "-"],
    #                   ["-", "I", "X", "-", "-"],
    #                   ["-", "G", "B", "I", "-"],
    #                   ["-", "G", "B", "I", "-"]])
    # card #35
    # board = np.array([["-", "G", "I", "B", "-"],
    #                   ["G", "B", "I", "-", "-"],
    #                   ["I", "-", "X", "-", "-"],
    #                   ["B", "-", "-", "-", "-"],
    #                   ["-", "-", "-", "-", "-"]])
    # card #36
    # board = np.array([["-", "I", "-", "B", "G"],
    #                   ["-", "I", "-", "G", "B"],
    #                   ["-", "-", "X", "I", "B"],
    #                   ["-", "-", "-", "-", "-"],
    #                   ["-", "-", "-", "I", "-"]])
    # card #37
    # board = np.array([["-", "-", "I", "B", "G"],
    #                   ["-", "I", "-", "B", "G"],
    #                   ["-", "I", "X", "-", "-"],
    #                   ["-", "-", "I", "-", "-"],
    #                   ["-", "-", "-", "-", "-"]])
    # card #38
    # board = np.array([["I", "B", "I", "-", "-"],
    #                   ["G", "-", "-", "-", "-"],
    #                   ["G", "-", "X", "-", "I"],
    #                   ["-", "-", "-", "-", "I"],
    #                   ["-", "-", "I", "-", "-"]])
    # card #39
    # board = np.array([["I", "-", "I", "G", "G"],
    #                   ["-", "-", "-", "B", "I"],
    #                   ["-", "-", "X", "I", "-"],
    #                   ["-", "-", "I", "-", "-"],
    #                   ["-", "-", "-", "-", "-"]])
    # card #40
    # board = np.array([["I", "-", "-", "-", "G"],
    #                   ["G", "-", "I", "-", "B"],
    #                   ["B", "-", "X", "I", "-"],
    #                   ["-", "-", "I", "-", "-"],
    #                   ["-", "I", "-", "-", "-"]])

    # tempBoard = board.copy()
    moves = [board]
    # g = Network("800px", "1100px", directed=True)
    g = nx.DiGraph()
    g.add_node(str(moves[0]), color='#00ff1e')
    tiltRecursive(board, moves, g)
    print(len(moves))

    #condensation
    scc = list(nx.strongly_connected_components(g))
    c = nx.condensation(g, scc)
    nx.draw(c, with_labels=True, font_weight='bold')
    plt.show()
    while (True):
        num = int(input("What is the configuration for this node? "))
        if input != "n":
            print(scc[num])
            continue
        else:
            break

if __name__ == '__main__':
    main()