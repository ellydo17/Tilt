from collections import namedtuple
from itertools import combinations

from pyvis.network import Network
import numpy as np

g = Network("800px", "1100px", directed=True)
for i in range(1, 37):
    g.add_node(i);

# rows, cols = (5, 5)
# board = [[0] * cols] * rows
# board[0][0] = "B"
# board[4][0] = "0"
# board[4][1] = "G"
# board[0][2] = "X"
# board[4][2] = "X"

board = np.array([["G", "-", "I", "B", "-"],
                  ["-", "-", "-", "-", "-"],
                  ["G", "-", "X", "-", "-"],
                  ["-", "-", "-", "-", "-"],
                  ["G", "B", "-", "-", "-"]])

tempBoard = board.copy()

directions = np.array(["L", "R", "U", "D"])

for d in directions:
    if d == "R":
        stop = False
        for i in range(len(board)):
            # loop through each row in the board
            row = board[i]
            # loop through each entry of each row in the board
            # the reason why I have to use row[0] is because board[i] is actually
            # ['-' '-' 'I' 'B' 'B'] (if i is 0), which is an array of one element which is the
            # array ['-' '-' 'I' 'B' 'B']. len(row[0]) accesses ['-' '-' 'I' 'B' 'B']
            # and returns 5.
            index = len(row)
            blocker = index
            hole = index // 2 + 1
            for j in range(len(row) - 1, -1, -1):
                # row[0][j] accesses each element in ['-' '-' 'I' 'B' 'B'] (j = 0).
                elem = row[j]
                if i == hole - 1:
                    if elem == "B" and j < hole - 1 < blocker:
                        stop = True
                        board = tempBoard
                        break
                    elif elem == "G" and (j == hole - 2 or j == hole - 3):
                        print(j)
                        row[j] = "-"
                        continue
                if elem == "-" and j == len(row) - 1:
                    index = j
                elif elem == "-" and row[j + 1] != "-":
                    index = j
                else:
                    if (elem == "B" or elem == "G") and (blocker > index):
                        row[index] = elem
                        row[j] = "-"
                        index -= 1
                    elif elem == "I":
                        blocker = j

            if stop == True:
                break

    if d == "L":
        stop = False
        for i in range(len(board)):
            # loop through each row in the board
            row = board[i]
            # loop through each entry of each row in the board
            # the reason why I have to use row[0] is because board[i] is actually
            # ['-' '-' 'I' 'B' 'B'] (if i is 0), which is an array of one element which is the
            # array ['-' '-' 'I' 'B' 'B']. len(row[0]) accesses ['-' '-' 'I' 'B' 'B']
            # and returns 5.
            index = len(row)
            blocker = index
            hole = index // 2 + 1
            for j in range(len(row) - 1, -1, -1):
                # row[0][j] accesses each element in ['-' '-' 'I' 'B' 'B'] (j = 0).
                elem = row[j]
                if i == hole - 1:
                    if elem == "B" and j < hole - 1 < blocker:
                        stop = True
                        board = tempBoard
                        break
                    elif elem == "G" and (j == hole - 2 or j == hole - 3):
                        print(j)
                        row[j] = "-"
                        continue
                if elem == "-" and j == len(row) - 1:
                    index = j
                elif elem == "-" and row[j + 1] != "-":
                    index = j
                else:
                    if (elem == "B" or elem == "G") and (blocker > index):
                        row[index] = elem
                        row[j] = "-"
                        index -= 1
                    elif elem == "I":
                        blocker = j

            if stop == True:
                break

print(board)

# g.show('intermediate.html')


# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
