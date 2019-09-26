import keyboard
from Connection import Connection

class PyUnityCommander():
	client = None

	def __init__(self):
		self.client = Connection().get_connection()

	def walk_forward(self):
		self.client.send(bytes([119])) # forward byte
		self.client.send(bytes([101])) # step byte

	def walk_backward(self):
		self.client.send(bytes([115])) # backward byte
		self.client.send(bytes([101])) # step byte

	def walk_right(self):
		self.client.send(bytes([100])) # right byte
		self.client.send(bytes([101])) # step byte

	def walk_left(self):
		self.client.send(bytes([97])) # left byte
		self.client.send(bytes([101])) # step byte

	def init_hotkeys(self):
		keyboard.add_hotkey('w', self.walk_forward)
		keyboard.add_hotkey('a', self.walk_left)
		keyboard.add_hotkey('s', self.walk_backward)
		keyboard.add_hotkey('d', self.walk_right)