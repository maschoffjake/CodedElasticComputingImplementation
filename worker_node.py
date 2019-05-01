# Class used for creating a 'worker' node

class Worker:

	# Constructor for the worker node. Takes in rank since it changes for worker nodes
    def __init__(self, comm, rank):
    	self.comm = comm
    	self.rank = rank

    # No need to send a destination, since worker nodes will only be communicating with the master node
    def send(self, data, tag=None):
    	if tag == None:
    		self.comm.Send(data, dest=0)
    	else:
    		self.comm.Send(data, dest=0, tag=tag)