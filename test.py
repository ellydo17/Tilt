from collections import namedtuple
from itertools import combinations

from pyvis.network import Network
import numpy as np

g = Network("800px", "1100px", directed=True)


# for i in range(1, 37):
#     g.add_node(i);

board = np.array([["B", "G", "B", "-", "-"],
                  ["B", "I", "G", "-", "-"],
                  ["-", "-", "X", "-", "-"],
                  ["-", "-", "-", "-", "-"],
                  ["-", "-", "-", "-", "-"]])


tempBoard = board.copy()

directions = np.array(["L", "R", "U", "D"])

def countGreenSliders(board):
    count_Green = 0;
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == "G":
                count_Green += 1
    return count_Green

def tiltRight(board):
    # tilt to the right
    count_Green = countGreenSliders(board)
    stop = True

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
                elif elem == "G" and (j == hole - 2 or j == hole - 3):  # green node goes into the hole
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
        if count_Green != 0:
            stop = False
        else:
            stop = True
        if stop == True:
            break
    print(board)
    return stop

def tiltLeft(board):
    # tilt to the left
    count_Green = countGreenSliders(board)
    stop = True

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
                elif elem == "G" and (j == len(row) or j == len(row) - 1):
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
        if count_Green != 0:
            stop = False
        else:
            stop = True
        if stop == True:
            break
    print(board)
    return stop

def tiltDown(board):
    # tilt down
    count_Green = countGreenSliders(board)
    stop = True

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
                elif elem == "G" and (j == hole - 2 or j == hole - 3):
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
            stop = False
        else:
            stop = True
        if stop == True:
            break
    print(board)
    return stop

def tiltUp(board):
    # tilt up
    count_Green = countGreenSliders(board)
    stop = True

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
                elif elem == "G" and (j == len(column) or j == len(column) - 1):
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
            stop = False
        else:
            stop = True
        if stop == True:
            break
    print(board)
    return stop

def tilt():
    stop = False
    while (stop == False):
        print("Enter #: 0. Quit    L. Tilt Left   R. Tilt Right    U. Tilt Up    D. Tilt Down")
        num = input()
        if num == "0":
            print("Exiting the game...")
        elif num == "L":
            stop = tiltLeft(board)
        elif num == "R":
            stop = tiltRight(board)
        elif num == "U":
            stop = tiltUp(board)
        elif num == "D":
            stop = tiltDown(board)
        else:
            print("Invalid option.")
            continue
    print("You won the game!")

tilt()