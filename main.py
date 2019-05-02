from mpi4py import MPI
from worker_node import Worker
from master import Master
from svm import SVM
import numpy as np


def main():
    svm_model = SVM()

    # Setup the communcation framework
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #print(MPI.Get_processor_name(), str(rank))

    # Get the training date, and initialize the weights for the system 
    #train_data = svm_model.get_train_data()
    #print(train_data)
    #np.split(train_data, int(size)-1)

    train_data = np.arange(13)

    split_data_features = split_data(train_data, size - 1)

    number_of_data_features = 219
    weights = np.full(number_of_data_features, np.random.uniform(low=-0.01, high=0.01))


    # Compute the gradients for the dataset
    if rank == 0:
    	compute_gradients(comm, split_data_features)
    else:
    	compute_gradients(comm)

    #trained_weights = svm_model.train_and_eval(weights, train_data)

# Returns an array of arrays. Splits the array in 'num_splits' arrays that contain original data 
# of 'data'
def split_data(data, num_splits):

	# Return list
	return_list = []

	# Size of each array
	size_of_arrays = len(data) // num_splits

	# How to keep track of how to subsection array
	start = 0
	end = size_of_arrays

	for x in range(num_splits-1):
		return_list.append(data[start:end])
		start += size_of_arrays
		end += size_of_arrays

	# Add the last split
	return_list.append(data[start:])
	return return_list

def compute_gradients(comm, data=None, master=None):

	# Send each worker a chunk of data
	scatter_data_to_workers(data)

	# Calculate the gradients on each of the worker nodes, recieve them, and update the weights on the master node
	calculate_gradients()

	# Receive gradients from all of the worker nodes, and add average it into the master node
	receive_gradients()


# Function used to send data from the master node to all of the woder nodes
def scatter_data_to_workers(data):

	# Get rank
	rank = comm.Get_rank()

	# Scatter the data to work nodes
	if rank == 0:
		for i in range(1, 5):
			comm.send(data[i-1], dest=i)

# Calculate gradients on each of the worker nodes to send back to the master node to update
def calculate_gradients():

	if rank != 0:
		scattered_data = comm.recv(source=0)

# Receive the gradients from all of the worker nodes
def receive_gradients():

	# Receive messages from all of the worker nodes
	for i in range(1,5):
		data = comm.recv(source=i)
		print(data)
		

if __name__ == '__main__':
    main()
