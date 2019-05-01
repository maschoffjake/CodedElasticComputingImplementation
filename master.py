# Class used for creating a 'master' node

class Master:

	# Constructor, pass in the communcation for the master, since the rank is 0
    def __init__(self, comm):
    	self.comm = comm

    # Helper used for sending data from the master node
 	def send(self, dest, data, tag=None):
 		if tag == None:
 			self.comm.Send(data, dest=dest)
 		else:
 			self.comm.Send(data, dest=dest, tag=tag)
	