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

    train_data = np.arange(12)

    split_data_features = split_data(train_data, 4)

    print("Length:", len(split_data_features))
    print(split_data_features)

    number_of_data_features = 219
    weights = np.full(number_of_data_features, np.random.uniform(low=-0.01, high=0.01))

    # Create the master node
    if rank == 0:
    	master = Master(comm)

    # Compute the gradients for the dataset
    # if rank == 0:
    # 	compute_gradients(comm, train_data)
    # else:
    # 	compute_gradients(comm)

    #trained_weights = svm_model.train_and_eval(weights, train_data)


# if rank == 0:
#     print("Master doing work.")
# elif rank == 1:
#     print("Worker 1")
# elif rank == 2:
#     print("Worker 2")
# elif rank == 3:
#     print("Worker 3")
# elif rank == 4:
# 	print("Worker 4")

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

	# Scatter the data
	scattered_data = comm.scatter(data, root=0)


	print(scattered_data, str(comm.Get_rank()))

if __name__ == '__main__':
    main()
