# Title: Dijkstra's Algorithm for finding single source shortest path from scratch
# Author: Shubham Malik
# References: https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
# Part of Cosmos by OpenGenus Foundation
import math
import sys
import numpy as np

class PriorityQueue:
    # Based on Min Heap
    def __init__(self):
        self.cur_size = 0
        self.array = []
        self.pos = {}   # To store the pos of node in array
    def isEmpty(self):
        return self.cur_size == 0
    def min_heapify(self, idx):
        lc = self.left(idx)
        rc = self.right(idx)
        if lc < self.cur_size and self.array(lc)[0] < self.array(idx)[0]:
            smallest = lc
        else:
            smallest = idx
        if rc < self.cur_size and self.array(rc)[0] < self.array(smallest)[0]:
            smallest = rc
        if smallest != idx:
            self.swap(idx, smallest)
            self.min_heapify(smallest)
    def insert(self, tup):
        # Inserts a node into the Priority Queue
        self.pos[tup[1]] = self.cur_size
        self.cur_size += 1
        self.array.append((sys.maxsize, tup[1]))
        self.decrease_key((sys.maxsize, tup[1]), tup[0])
    def extract_min(self):
        # Removes and returns the min element at top of priority queue
        min_node = self.array[0][1]
        self.array[0] = self.array[self.cur_size - 1]
        self.cur_size -= 1
        self.min_heapify(1)
        del self.pos[min_node]
        return min_node
    def left(self, i):
        # returns the index of left child
        return 2 * i + 1
    def right(self, i):
        # returns the index of right child
        return 2 * i + 2
    def par(self, i):
        # returns the index of parent
        return math.floor(i / 2)
    def swap(self, i, j):
        # swaps array elements at indices i and j
        # update the pos{}
        self.pos[self.array[i][1]] = j
        self.pos[self.array[j][1]] = i
        temp = self.array[i]
        self.array[i] = self.array[j]
        self.array[j] = temp
    def decrease_key(self, tup, new_d):
        idx = self.pos[tup[1]]
        # assuming the new_d is atmost old_d
        self.array[idx] = (new_d, tup[1])
        while idx > 0 and self.array[self.par(idx)][0] > self.array[idx][0]:
            self.swap(idx, self.par(idx))
            idx = self.par(idx)
class Graph:
    def __init__(self, num):
        self.adjList = {}   # To store graph: u -> (v,w)
        self.num_nodes = num    # Number of nodes in graph
        # To store the distance from source vertex
        self.dist = [0] * self.num_nodes
        self.par = [-1] * self.num_nodes  # To store the path
    def add_edge(self, u, v, w):
        #  Edge going from node u to v and v to u with weight w
        # u (w)-> v, v (w) -> u
        # Check if u already in graph
        if u in self.adjList.keys():
            self.adjList[u].append((v, w))
        else:
            self.adjList[u] = [(v, w)]
        # Assuming undirected graph
        # if v in self.adjList.keys():
        #     self.adjList[v].append((u, w))
        # else:
        #     self.adjList[v] = [(u, w)]
    def show_graph(self):
        # u -> v(w)
        for u in self.adjList:
            print(u, '->', ' -> '.join(str("{}({})".format(v, w))
                                      for v, w in self.adjList[u]))
    def dijkstra(self, src):
        # Flush old junk values in par[]
        self.par = [-1] * self.num_nodes
        # src is the source node
        self.dist[src] = 0
        Q = PriorityQueue()
        Q.insert((0, src))  # (dist from src, node)
        for u in self.adjList.keys():
            if u != src:
                self.dist[u] = sys.maxsize  # Infinity
                self.par[u] = -1
        while not Q.isEmpty():
            u = Q.extract_min()  # Returns node with the min dist from source
            # Update the distance of all the neighbours of u and
            # if their prev dist was INFINITY then push them in Q
            for v, w in self.adjList[u]:
                new_dist = self.dist[u] + w
                if self.dist[v] > new_dist:
                    if self.dist[v] == sys.maxsize:
                        Q.insert((new_dist, v))
                    else:
                        Q.decrease_key((self.dist[v], v), new_dist)
                    self.dist[v] = new_dist
                    self.par[v] = u
        # Show the shortest distances from src
        self.show_distances(src)
    def show_distances(self, src):
        print("Distance from node: {}".format(src))
        for u in range(self.num_nodes):
            print('Node {} has distance: {}'.format(u, self.dist[u]))
    def show_path(self, src, dest):
        # To show the shortest path from src to dest
        # WARNING: Use it *after* calling dijkstra
        path = []
        cost = 0
        temp = dest
        # Backtracking from dest to src
        while self.par[temp] != -1:
            path.append(temp)
            if temp != src:
                for v, w in self.adjList[temp]:
                    if v == self.par[temp]:
                        cost += w
                        break
            temp = self.par[temp]
        path.append(src)
        path.reverse()
        print('----Path to reach {} from {}----'.format(dest, src))
        for u in path:
            print('{}'.format(u), end=' ')
            if u != dest:
                print('-> ', end='')
        print('\nTotal cost of path: ', cost)

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
                elif elem == "G" and (j == hole or j == hole + 1):
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
                elif elem == "G" and (j == hole or j == hole + 1):
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

# def tiltRecursiveNode(board, moves):
#     board_Left = tiltLeft(board)
#     board_Right = tiltRight(board)
#     board_Up = tiltUp(board)
#     board_Down = tiltDown(board)
#
#     if not np.array_equal(board_Left, board):
#         if not findMoves(board_Left, moves):
#             if not green(board_Left):
#                 moves.append(board_Left)
#                 tiltRecursiveNode(board_Left, moves)
#             elif green(board_Left):
#                 moves.append(board_Left)
#
#     if not np.array_equal(board_Right, board):
#         if not findMoves(board_Right, moves):
#             if not green(board_Right):
#                 moves.append(board_Right)
#                 tiltRecursiveNode(board_Right, moves)
#             elif green(board_Right):
#                 moves.append(board_Right)
#
#     if not np.array_equal(board_Up, board):
#         if not findMoves(board_Up, moves):
#             if not green(board_Up):
#                 moves.append(board_Up)
#                 tiltRecursiveNode(board_Up, moves)
#             elif green(board_Up):
#                 moves.append(board_Up)
#
#     if not np.array_equal(board_Down, board):
#         if not findMoves(board_Down, moves):
#             if not green(board_Down):
#                 moves.append(board_Down)
#                 tiltRecursiveNode(board_Down, moves)
#             elif green(board_Down):
#                 moves.append(board_Down)

def tiltRecursiveEdge(board, moves, g):
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
                g.add_edge(getBoardIndices(board), getBoardIndices(board_Left), 1)
                tiltRecursiveEdge(board_Left, moves, g)
            elif green(board_Left):
                moves.append(board_Left)
                # add the node of the board_Right
                # g.add_node(str(board_Left))
                # add the edge between the board_Right and the current board
                g.add_edge(getBoardIndices(board), getBoardIndices(board_Left), 1)
                g.add_edge(getBoardIndices(board_Left), getBoardIndices(board_Left), 1)
        elif findMoves(board_Left, moves):
            # add the node of the board_Left
            # g.add_node(str(board_Left))
            # add the edge between the board_Left and the current board
            g.add_edge(getBoardIndices(board), getBoardIndices(board_Left), 1)

    if not np.array_equal(board_Right, board):
        if not findMoves(board_Right, moves):
            if not green(board_Right):
                moves.append(board_Right)
                # add the node of the board_Right
                # g.add_node(str(board_Right))
                # add the edge between the board_Right and the current board
                g.add_edge(getBoardIndices(board), getBoardIndices(board_Right), 1)
                g.add_edge(getBoardIndices(board), getBoardIndices(board_Right), 1)
                tiltRecursiveEdge(board_Right, moves, g)
            elif green(board_Right):
                moves.append(board_Right)
                # add the node of the board_Right
                # g.add_node(str(board_Right))
                # add the edge between the board_Right and the current board
                g.add_edge(getBoardIndices(board), getBoardIndices(board_Right), 1)
                g.add_edge(getBoardIndices(board_Right), getBoardIndices(board_Right), 1)
        elif findMoves(board_Right, moves):
            # add the node of the board_Right
            # g.add_node(str(board_Right))
            # add the edge between the board_Right and the current board
            g.add_edge(getBoardIndices(board), getBoardIndices(board_Right), 1)

    if not np.array_equal(board_Up, board):
        if not findMoves(board_Up, moves):
            if not green(board_Up):
                moves.append(board_Up)
                # add the node of the board_Up
                # g.add_node(str(board_Up))
                # add the edge between the board_Up and the current board
                g.add_edge(getBoardIndices(board), getBoardIndices(board_Up), 1)
                tiltRecursiveEdge(board_Up, moves, g)
            elif green(board_Up):
                moves.append(board_Up)
                # add the node of the board_Right
                # g.add_node(str(board_Up))
                # add the edge between the board_Right and the current board
                g.add_edge(getBoardIndices(board), getBoardIndices(board_Up), 1)
                g.add_edge(getBoardIndices(board_Up), getBoardIndices(board_Up), 1)
        elif findMoves(board_Up, moves):
            # add the node of the board_Up
            # g.add_node(str(board_Up))
            # add the edge between the board_Up and the current board
            g.add_edge(getBoardIndices(board), getBoardIndices(board_Up), 1)

    if not np.array_equal(board_Down, board):
        if not findMoves(board_Down, moves):
            if not green(board_Down):
                moves.append(board_Down)
                # add the node of the board_Down
                # g.add_node(str(board_Down))
                # add the edge between the board_Down and the current board
                g.add_edge(getBoardIndices(board), getBoardIndices(board_Down), 1)
                tiltRecursiveEdge(board_Down, moves, g)
            elif green(board_Down):
                moves.append(board_Down)
                # add the node of the board_Right
                # g.add_node(str(board_Down))
                # add the edge between the board_Right and the current board
                g.add_edge(getBoardIndices(board), getBoardIndices(board_Down), 1)
                g.add_edge(getBoardIndices(board_Down), getBoardIndices(board_Down), 1)
        elif findMoves(board_Down, moves):
            # add the node of the board_Down
            # g.add_node(str(board_Down))
            # add the edge between the board_Down and the current board
            g.add_edge(getBoardIndices(board), getBoardIndices(board_Down), 1)

if __name__ == '__main__':
    board = np.array([["-", "-", "-", "-", "-"],
                      ["-", "-", "-", "-", "-"],
                      ["-", "-", "X", "-", "-"],
                      ["-", "-", "B", "B", "B"],
                      ["-", "-", "I", "G", "B"]])
    moves = [board]
    # tiltRecursiveNode(board, moves)
    graph = Graph(10)
    tiltRecursiveEdge(board, moves, graph)

    graph.show_graph()
    print("\n ---")
    graph.dijkstra(0)
    print("\n ---")
    graph.show_path(0, 4)
    input("What is the configuration for this node? ")
