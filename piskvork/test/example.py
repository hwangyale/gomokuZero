import random

from gomokuZero.board.board import Board
from gomokuZero.model.mcts import MCTS

import pisqpipe as pp
from pisqpipe import DEBUG_EVAL, DEBUG

pp.infotext = 'name="pbrain-mcts", author="You Huang", version="1.0", country="China", www="https://github.com/hwangyale/gomokuZero"'

board = Board()

nn_path = 'data/pre_train/yixin_version_nn_config.json'
mcts_config = {
    'rollout_time': 500, 'max_thread': 50
}
mcts = MCTS(
    PolicyValueNetwork.load_model(nn_path), **mcts_config
)


def brain_init():
	if pp.width != 15 or pp.height != 15:
		pp.pipeOut("ERROR size of the board")
		return
	pp.pipeOut("OK")

def brain_restart():
	global board
	board = Board()
	pp.pipeOut("OK")

def isFree(x, y):
	return (x, y) in board.legal_positions

def brain_my(x, y):
	if isFree(x, y):
		board.move((x, y))
	else:
		pp.pipeOut("ERROR my move [{},{}]".format(x, y))

def brain_opponents(x, y):
	if isFree(x, y):
		board.move((x, y))
	else:
		pp.pipeOut("ERROR opponents's move [{},{}]".format(x, y))

def brain_block(x, y):
	pp.pipeOut("Do not support")
	# if isFree(x,y):
	# 	board[x][y] = 3
	# else:
	# 	pp.pipeOut("ERROR winning move [{},{}]".format(x, y))

def brain_takeback(x, y):
	pp.pipeOut("Do not support")
	# if x >= 0 and y >= 0 and x < pp.width and y < pp.height and board[x][y] != 0:
	# 	board[x][y] = 0
	# 	return 0
	# return 2

def brain_turn():
	if pp.terminateAI:
		return
	x, y = mcts.get_positions(board, Tau=0.0, verbose=2)
	pp.do_mymove(x, y)

def brain_end():
	pass

def brain_about():
	pp.pipeOut(pp.infotext)

if DEBUG_EVAL:
	import win32gui
	def brain_eval(x, y):
		# TODO check if it works as expected
		wnd = win32gui.GetForegroundWindow()
		dc = win32gui.GetDC(wnd)
		rc = win32gui.GetClientRect(wnd)
		c = str(board[x][y])
		win32gui.ExtTextOut(dc, rc[2]-15, 3, 0, None, c, ())
		win32gui.ReleaseDC(wnd, dc)

# "overwrites" functions in pisqpipe module
pp.brain_init = brain_init
pp.brain_restart = brain_restart
pp.brain_my = brain_my
pp.brain_opponents = brain_opponents
pp.brain_block = brain_block
pp.brain_takeback = brain_takeback
pp.brain_turn = brain_turn
pp.brain_end = brain_end
pp.brain_about = brain_about
if DEBUG_EVAL:
	pp.brain_eval = brain_eval

def main():
	pp.main()

if __name__ == "__main__":
	main()
