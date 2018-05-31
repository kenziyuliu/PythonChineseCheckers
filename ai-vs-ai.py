import re
import sys

from game import Game
from config import *
from model import *
from utils import find_version_given_filename
"""
Run this file with argument specifying the models from terminal if you
want to play ai-vs-ai game
e.g. python3 ai-vs-ai.py saved-models/version0000.h5 saved-models/version0033.h5
"""

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python3 ai-vis-ai.py <model1_path> <model2_path>')
        exit()

    count = { PLAYER_ONE : 0, PLAYER_TWO : 0 }
    model1 = ResidualCNN()
    filename1 = sys.argv[1]
    version_num1 = utils.find_version_given_filename(filename1)
    model1.version = version_num1
    print("\nLoading model1 from path {}".format(filename1))
    # model1.load(filename1)
    model1.load_weights(filename1)
    print("Model1 is loaded sucessfully\n")

    model2 = ResidualCNN()
    filename2 = sys.argv[2]
    version_num2 = utils.find_version_given_filename(filename2)
    model2.version = version_num2
    print("Loading model2 from path {}".format(filename2))
    # model2.load(filename2)
    model2.load_weights(filename2)
    print("Model2 is loaded sucessfully\n")
    for i in range(1):
        print(i, end=' ')
        game = Game(p1_type='ai', p2_type='ai', verbose=True, model1=model1, model2=model2)
        winner = game.start()
        count[winner] += 1

    print('AiPlayer {} wins {} matches'.format(PLAYER_ONE, count[PLAYER_ONE]))
    print('AiPlayer {} wins {} matches'.format(PLAYER_TWO, count[PLAYER_TWO]))
