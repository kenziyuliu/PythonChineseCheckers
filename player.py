import numpy as np
import board_utils
from config import *
import copy
import os
import random
from MCTS import *
from board import *
from model import *

"""
Both players must support decide_move(self, board, verbose) method
"""

class HumanPlayer:
    def __init__(self, player_num):
        self.player_num = player_num
        # TODO: include more members if needed

    def decide_move(self, board, verbose=True, total_moves=None):
        """
        Given current board, return a move to play.
        :type board: Class Board
        :rtype A list of 2 tuples, specifying the move's FROM and TO.
        """

        # First print game info
        os.system('clear')
        board.visualise(cur_player=self.player_num)

        valid_moves = board.get_valid_moves(self.player_num)
        human_valid_moves = board_utils.convert_np_to_human_moves(valid_moves)

        if verbose:
            for checker in human_valid_moves:
                print("Checker {} can move to: {}".format(checker, sorted(human_valid_moves[checker])))

        print()
        (from_i, from_j), (to_i, to_j) = (-1, -1), (-1, -1)
        while 1:
            if verbose:
                # x = the row number on visualised board, y = the position of the checker in that row from left
                print('You should specify position by row number and the count from left.')
                print('Please input your move with format: start_row start_col end_row end_col')

            try:
                human_from_row, human_from_col, human_to_row, human_to_col = map(int, input().split())
            except ValueError:
                print("\nInvalid Move Format! Try again!")
                continue

            (from_i, from_j), (to_i, to_j) = board_utils.human_coord_to_np_index((human_from_row, human_from_col)), \
                                             board_utils.human_coord_to_np_index((human_to_row, human_to_col))

            if (from_i, from_j) in valid_moves and (to_i, to_j) in valid_moves[(from_i, from_j)]:
                break

            print("\nInvalid Move! Try again!")

        return (from_i, from_j), (to_i, to_j)



class GreedyPlayer:
    def __init__(self, player_num):
        self.player_num = player_num

    def decide_move(self, board, verbose=False, training=False, total_moves=None):
        valid_moves = board.get_valid_moves(self.player_num)
        human_valid_moves = board_utils.convert_np_to_human_moves(valid_moves)

        best_moves = []
        max_dist = float('-inf')
        for start in human_valid_moves:
            for end in human_valid_moves[start]:
                dist = end[0] - start[0]    # Evaluate distance by how many steps forward
                if self.player_num == 1:    # Revert distance as player1 moves up
                    dist = -dist
                if dist > max_dist:
                    max_dist = dist
                    best_moves = [(start, end)]
                elif dist == max_dist:
                    best_moves.append((start, end))

        # When there are many possible moves, pick the one that's the last
        last_checker, _ = max(best_moves, key=lambda x: (x[0][0] if self.player_num == PLAYER_ONE else -x[0][0]))
        # Take away staying-move, and get all moves that is for the last checker
        filtered_best_moves = [move for move in best_moves if move[0][0] == last_checker[0]]

        if training:
            return filtered_best_moves

        # Then randomly sample a move
        pick_start, pick_end = random.choice(filtered_best_moves)

        if verbose:
            board.visualise(cur_player = self.player_num)
            print('GreedyPlayer moved from {} to {}\n'.format(pick_start, pick_end))

        return board_utils.human_coord_to_np_index(pick_start), \
               board_utils.human_coord_to_np_index(pick_end)



class AiPlayer:
    def __init__(self, player_num, model):
        self.player_num = player_num
        self.model = model

    def decide_move(self, board, verbose=False, total_moves=None):
        """
        Given current board, return a move to play.
        :type board: Class Board
        :rtype A list of 2 tuples, specifying the move's FROM and TO.
        """
        if verbose:
            board.visualise(cur_player = self.player_num)
            print('Facing the board above, Ai Version {} is thinking.'.format(self.model.version))

        node = Node(board, self.player_num)

        tree_tau = TREE_TAU
        if total_moves is not None and total_moves > TOTAL_MOVES_TILL_TAU0:
            tree_tau = 0.01

        tree = MCTS(node, self.model, tree_tau=tree_tau)
        pi, sampled_edge = tree.search()
        if verbose:
            print('Ai Version {} moved from {} to {}\n'.format(
                self.model.version, sampled_edge.fromPos, sampled_edge.toPos))

        return sampled_edge.fromPos, sampled_edge.toPos


if __name__ == "__main__":
    ai = AiPlayer(1)
    board = Board()
    print(ai.decide_move(board))
