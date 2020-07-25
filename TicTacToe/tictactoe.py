import numpy as np
import pickle

BOARD_ROWS = 3
BOARD_COLS = 3

class State:
	def __init__(self, p1, p2):
		self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
		self.p1 = p1
		self.p2 = p2
		self.isEnd = False
		self.boardHash = None
		self.playerSymbol = 1

	def getHash(self):
		self.boardHash = str(self.board.reshape(BOARD_ROWS * BOARD_COLS))
		return self.boardHash

	def availablePositions(self):
		positions = []
		for i in range(BOARD_ROWS):
			for j in range(BOARD_COLS):
				if self.board[i, j] == 0:
					positions.append((i, j))
		return positions

	def winner(self):
		for i in range(BOARD_ROWS):
			if sum(self.board[i, : ]) == 3:
				self.isEnd = True
				return 1
			if sum(self.board[i, : ]) == -3:
				self.isEnd = True
				return -1

		for j in range(BOARD_COLS):
			if sum(self.board[ : , j]) == 3:
				self.isEnd = True
				return 1
			if sum(self.board[ : , j]) == -3:
				self.isEnd = True
				return -1

		diag_sum1 = sum([self.board[i, i] for i in range(BOARD_ROWS)])
		diag_sum2 = sum([self.board[i, BOARD_COLS - i - 1] for i in range(BOARD_COLS)])
		diag_sum = max(abs(diag_sum1), abs(diag_sum2))

		if diag_sum == 3:
			self.isEnd = True
			if diag_sum1 == 3 or diag_sum2 == 3:
				return 1
			else:
				return -1

		if len(self.availablePositions()) == 0:
			self.isEnd = True
			return 0

		self.isEnd = False
		return None

	def updateState(self, position):
		self.board[position] = self.playerSymbol
		self.playerSymbol = -1 if self.playerSymbol == 1 else 1

	def giveReward(self):
		result = self.winner()
		if result == 1:
			self.p1.feedReward(1)
			self.p2.feedReward(-1)
		elif result == -1:
			self.p1.feedReward(-1)
			self.p2.feedReward(1)
		else:
			self.p1.feedReward(0.1)
			self.p2.feedReward(0.5)

	def reset(self):
		self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
		self.boardHash = None
		self.isEnd = False
		self.playerSymbol = 1

	def play(self, rounds=100, train_mode=False):
		if train_mode:
			print("Training")
		for i in range(rounds):
			if not train_mode:
				self.showBoard()

			if i % 1000 == 0:
				print("Rounds {}".format(i))

			while not self.isEnd:
				#Player 1	
				positions = self.availablePositions()
				p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
				self.updateState(p1_action)
				if not train_mode:
					self.showBoard()
				board_hash = self.getHash()
				self.p1.addState(board_hash)

				win = self.winner()
				if win is not None:
					if not train_mode:
						if win == 1:
							print(self.p1.name, "wins!")
						else:
							print("Tie!")
					self.giveReward()
					self.p1.reset()
					self.p2.reset()
					self.reset()
					break

				else:
					#Player 2
					positions = self.availablePositions()
					p2_action = self.p2.chooseAction(positions, self.board, self.playerSymbol)
					self.updateState(p2_action)
					if not train_mode:
						self.showBoard()
					board_hash = self.getHash()
					self.p2.addState(board_hash)

					win = self.winner()
					if win is not None:
						if not train_mode:
							if win == -1:
								print(self.p2.name, "wins!")
							else:
								print("Tie!")
						self.giveReward()
						self.p1.reset()
						self.p2.reset()
						self.reset()
						break


	def showBoard(self):
		# p1 : X p2 : O
		for i in range(BOARD_ROWS):
			print('-------------')
			out = '| '
			for j in range(BOARD_COLS):
				if self.board[i, j] == 1:
					token = 'X'
				if self.board[i, j] == -1:
					token = 'O'
				if self.board[i, j] == 0:
					token = ' '
				out += token + ' | '
			print(out)
		print('-------------')

class Player:
	def __init__(self, name, exp_rate=0.2, human=False):
		self.human = human
		if self.human:
			self.name = name
		else:
			self.name = name
			self.states = []
			self.lr = 0.2
			self.exp_rate = exp_rate
			self.decay_gamma = 0.9
			self.states_value = {}

	def getHash(self, board):
		boardHash = str(board.reshape(BOARD_ROWS * BOARD_COLS))
		return boardHash

	def chooseAction(self, positions, current_board, symbol):
		if self.human:
			while True:
				action = eval(input("Enter action: "))
				if action in positions:
					return action
		else:
			if np.random.uniform(0, 1) <= self.exp_rate:
				idx = np.random.choice(len(positions))
				action = positions[idx]
			else:
				value_max = -1
				for p in positions:
					next_board = current_board.copy()
					next_board[p] = symbol
					next_boardHash = self.getHash(next_board)
					value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash)
					if value >= value_max:
						value_max = value
						action = p
		#print("{} takes action {}".format(self.name, action))
		return action

	def addState(self, state):
		if not self.human:
			self.states.append(state)

	def feedReward(self, reward):
		if self.human:
			pass
		else:
			for s in reversed(self.states):
				if self.states_value.get(s) is None:
					self.states_value[s] = 0
				self.states_value[s] += self.lr * (self.decay_gamma * reward - self.states_value[s])
				reward = self.states_value[s]
			#print(self.states_value)

	def reset(self):
		if self.human:
			pass
		else:
			self.states = []

	def savePolicy(self):
		fw = open('policy_' + str(self.name), 'wb')
		pickle.dump(self.states_value, fw)
		fw.close()

	def loadPolicy(self, file):
		fr = open(file, 'rb')
		self.states_value = pickle.load(fr)
		fr.close()

if __name__ == '__main__':
	p1 = Player("p1")
	p2 = Player("p2", human=True)
	p1.loadPolicy("policy_p1")

	st = State(p1, p2)
	st.play(10)
	
