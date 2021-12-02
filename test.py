from collections import namedtuple
from itertools import combinations

import Arrays as Arrays
from pyvis.network import Network
import numpy as np

g = Network("800px", "1100px", directed=True)


# for i in range(1, 37):
#     g.add_node(i);

oriBoard = np.array([["B", "G", "B", "-", "-"],
                  ["B", "I", "G", "-", "-"],
                  ["-", "-", "X", "-", "-"],
                  ["-", "-", "-", "-", "-"],
                  ["-", "-", "-", "-", "-"]])


tempBoard = oriBoard.copy()

# directions = np.array(["L", "R", "U", "D"])

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
    return board

def tiltLeft(board):
    # tilt to the left
    board = board.copy()
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
    return board

def tiltDown(board):
    # tilt down
    board = board.copy()
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
    return board

def tiltUp(board):
    # tilt up
    board = board.copy()
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
    return board

# def tilt():
#     print("Enter #: 0. Quit    L. Tilt Left   R. Tilt Right    U. Tilt Up    D. Tilt Down")
#     num = input()
#     if num == "0":
#         print("Exiting the game...")
#     elif num == "L":
#         print(tiltLeft(board))
#     elif num == "R":
#         print(tiltRight(board))
#     elif num == "U":
#         print(tiltUp(board))
#     elif num == "D":
#         print(tiltDown(board))
#     else:
#         print("Invalid option.")

def tilt(board):
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

def tiltRecursive(board, moves):
    print("Original \n", board)
    # print("Moves: \n", moves)

    board_Left = tiltLeft(board)
    board_Right= tiltRight(board)
    board_Up = tiltUp(board)
    board_Down = tiltDown(board)

    # print("Left \n", board_Left)
    # print("Right \n", board_Right)
    # print("Up \n", board_Up)
    # print("Down \n", board_Down)
    #make a function that compares two 2-D arrays.
    print(not np.array_equal(board_Right, board))
    print(not findMoves(board_Right, moves))
    print(not tilt(board_Right))
    if not np.array_equal(board_Left,board) and not findMoves(board_Left, moves) and not tilt(board_Left):
        moves.append(board_Left)
        tiltRecursive(board_Left, moves)
        print("got out1!")
        # g.add_node()
    print(not np.array_equal(board_Right, board))
    print(not findMoves(board_Right, moves))
    print(not tilt(board_Right))
    if not np.array_equal(board_Right,board) and not findMoves(board_Right, moves) and not tilt(board_Right):
        moves.append(board_Right)
        tiltRecursive(board_Right, moves)
        print("got out2!")
        # g.add_node()
    print(not np.array_equal(board_Right, board))
    print(not findMoves(board_Right, moves))
    print(not tilt(board_Right))
    if not np.array_equal(board_Up,board) and not findMoves(board_Up, moves) and not tilt(board_Up):
        moves.append(board_Up)
        tiltRecursive(board_Up, moves)
        print("got out3!")
        # g.add_node()
    print(not np.array_equal(board_Right, board))
    print(not findMoves(board_Right, moves))
    print(not tilt(board_Right))
    if not np.array_equal(board_Down,board) and not findMoves(board_Down, moves) and not tilt(board_Down):
        moves.append(board_Down)
        tiltRecursive(board_Down, moves)
        print("got out4!")
        # g.add_node()


def main():
    moves = [oriBoard]
    tiltRecursive(oriBoard, moves)

if __name__ == '__main__':
    main()