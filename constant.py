import numpy as np

#board
EMPTY = 0
BLACK = 1
WHITE = 2
DRAW = 3

SIZE = 15
NUMBER = 5

#hashing
BOARD_HASHING = []
n = 2
while len(BOARD_HASHING) < SIZE**2:
    if len(BOARD_HASHING) == 0:
        BOARD_HASHING.append(n)
        continue
    while True:
        n += 1
        for factor in BOARD_HASHING:
            if n % factor == 0:
                break
        else:
            BOARD_HASHING.append(n)
            break
del n
del factor

#preprocess
HISTORY_STEPS = 1

#supervised learning
VALUE_WEIGHT = 0.1

#MCTS
C_PUCT = 5.0
VIRTUAL_LOSS = 1.0
VIRTUAL_VISIT = 1.0

PROCESS_SLEEP_TIME = 0.0

DIRICHLET_ALPHA = 0.03
