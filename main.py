from collections import namedtuple
from itertools import combinations

from pyvis.network import Network
import numpy as np

g = Network("800px", "1100px", directed=True)


# for i in range(1, 37):
#     g.add_node(i);

# board = np.array([["-", "-", "-", "-", "-"],
#                       ["-", "-", "-", "-", "-"],
#                       ["-", "-", "X", "-", "-"],
#                       ["-", "-", "-", "-", "-"],
#                       ["-", "B", "B", "G", "B"]])

ggg = np.array([["B", "G", "G", "-", "-"],
                      ["B", "I", "B", "-", "-"],
                      ["-", "-", "X", "-", "-"],
                      ["-", "-", "-", "-", "-"],
                      ["-", "-", "-", "-", "-"]])

# board = np.array([["B", "G", "G", "-", "-"],
#                   ["B", "I", "B", "-", "-"],
#                   ["-", "-", "X", "-", "-"],
#                   ["-", "-", "-", "-", "-"],
#                   ["-", "-", "-", "-", "-"]])

# board = np.array([["-", "-", "-", "G", "B"],
#                   ["-", "I", "-", "-", "B"],
#                   ["-", "-", "X", "-", "B"],
#                   ["-", "-", "-", "-", "-"],
#                   ["-", "-", "-", "-", "-"]])


tempBoard = ggg.copy()

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
                    print("cscscssc")
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
        count_Green = countGreenSliders(board)
        print(count_Green)
        if count_Green != 0:
            stop_temp = False
        else:
            stop_temp = True
        if stop == True or stop_temp == True:
            checking = False
            break

    if checking == False:
        board = tempBoard

    print(board)
    return stop, board, stop_temp

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
        count_Green = countGreenSliders(board)
        print(count_Green)
        if count_Green != 0:
            stop_temp = False
        else:
            stop_temp = True
        if stop == True or stop_temp == True:
            checking = False
            break

    if checking == False:
        board = tempBoard

    print(board)
    return stop, board, stop_temp

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
        print(count_Green)
        if count_Green != 0:
            stop_temp = False
        else:
            stop_temp = True
        if stop == True or stop_temp == True:
            print(stop, stop_temp)
            checking = False
            break

    if checking == False:
        board = tempBoard
    print(board)
    return stop, board, stop_temp

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
        print(count_Green)
        if count_Green != 0:
            stop_temp = False
        else:
            stop_temp = True
        if stop == True or stop_temp == True:
            checking = False
            break

    if checking == False:
        board = tempBoard
    print(board)
    return stop, board, stop_temp

def tilt():
    ggg = np.array([["-", "-", "I", "B", "B"],
                      ["-", "-", "-", "-", "-"],
                      ["-", "-", "X", "-", "-"],
                      ["-", "-", "-", "-", "-"],
                      ["G", "B", "-", "-", "-"]])

    # board = np.array([["-", "-", "-", "-", "-"],
    #                      ["-", "-", "-", "-", "-"],
    #                      ["-", "-", "X", "-", "-"],
    #                      ["-", "-", "-", "-", "-"],
    #                      ["-", "B", "B", "G", "B"]])

    # board = np.array([["-", "-", "-", "-", "-"],
    #                   ["B", "-", "-", "-", "-"],
    #                   ["B", "-", "X", "-", "-"],
    #                   ["G", "-", "-", "-", "-"],
    #                   ["B", "-", "-", "-", "-"]])

    stop_temp = False
    temp = False
    while (temp == False):
        print("Enter #: 0. Quit    L. Tilt Left   R. Tilt Right    U. Tilt Up    D. Tilt Down")
        num = input()
        if num == "0":
            print("Exiting the game...")
        elif num == "L":
            stop_temp, ggg, temp = tiltLeft(ggg)
        elif num == "R":
            stop_temp, ggg, temp = tiltRight(ggg)
        elif num == "U":
            stop_temp, ggg, temp = tiltUp(ggg)
        elif num == "D":
            stop_temp, ggg, temp = tiltDown(ggg)
        else:
            print("Invalid option.")
            continue
    print("You won the game!")

def main():
    tilt()


if __name__ == '__main__':
    main()