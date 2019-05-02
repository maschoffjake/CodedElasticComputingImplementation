from mpi4py import MPI
from worker_node import Worker
from master import Master
from svm import SVM


def main():
    svm_model = SVM()

    # Setup the communcation framework
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #print(MPI.Get_processor_name(), str(rank))

    # Get the training date, and initialize the weights for the system 
    train_data = svm_model.get_train_data()
    number_of_data_features = 219
    weights = np.full(number_of_data_features, np.random.uniform(low=-0.01, high=0.01))

    # Create the master node
    Master master = new Master(comm)

    # Compute the gradients for the dataset
    compute_gradients(comm, master, train_data)

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

def compute_gradients(comm, master, data):

	# Scatter the data
	master.scatter(data)

	print(str(comm.Get_rank()), data)

if __name__ == '__main__':
    main()
