import socket

class Connection():
	IP_ADDRESS = "127.0.0.1"
	PORT = 13000

	def __init__(self):
		self._create_connection()

	def _create_connection(self):
		self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.client.connect((self.IP_ADDRESS, self.PORT))

	def get_connection(self):
		return self.client