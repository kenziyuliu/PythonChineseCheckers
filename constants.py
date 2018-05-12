''' Player '''
# Fixed 2 player
PLAYER_ONE = 1
PLAYER_TWO = 2

''' Board/Game '''
ROWS_OF_CHECKERS = 3
NUM_CHECKERS = (1 + ROWS_OF_CHECKERS) * ROWS_OF_CHECKERS // 2
NUM_DIRECTIONS = 6
BOARD_WIDTH = BOARD_HEIGHT = ROWS_OF_CHECKERS * 2 + 1
NUM_HIST_MOVES = 3      # Number of history moves to keep
TYPES_OF_PLAYERS = ['h', 'g', 'a']
PLAYER_ONE_DISTANCE_OFFSET = 70
PLAYER_TWO_DISTANCE_OFFSET = -14

''' MCTS and RL '''
PROGRESS_MOVE_LIMIT = 50
TREE_TAU = 1
REWARD = {"lose" : -200, "draw" : 0, "win" : 200}
C_PUCT = 1
MCTS_SIMULATIONS = 125
EPSILON = 1e-5

''' Train '''
NUM_WORKERS = 12        # For generating self plays in parallel
NUM_SELF_PLAY = 120
SAVE_MODELS_DIR = 'saved-models/'
