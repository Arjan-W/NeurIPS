import keyboard
import socket
import numpy as np
import PIL.Image

class PyUnityCommands():

	def __init__(self):
		self.create_connection()
	
	def create_connection(self):
		ip = "127.0.0.1" # ip as defined in the application
		port = 13000 # port as defined in the application
		self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # instantiate a client object
		self.client.connect((ip, port)) # connect the client object to the server
		
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
	
	
		
		
if __name__ == '__main__':
	commander = PyUnityCommands()
	keyboard.add_hotkey('w', commander.walk_forward)
	keyboard.add_hotkey('a', commander.walk_left)
	keyboard.add_hotkey('s', commander.walk_backward)
	keyboard.add_hotkey('d', commander.walk_right)
	while True:
		x = 1
		
		
		
		
		
		
		
		
		
		
		
		