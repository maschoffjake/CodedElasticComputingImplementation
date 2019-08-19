from mpi4py import MPI
from svm import SVM
import numpy as np
import sys
import random


# Create the SVM model
svm_model = SVM()

def main():

    print(sys.argv[1])

    # Grab preemption max number
    if len(sys.argv) == 2:
        if sys.argv[1].isdigit():

            # Take in a commandline argument to preempt machines... IE not send data to them if they
            # are randomly choosen. Pass in the maximum amount of machines that can be preemptied in an iteration
            preemption = int(sys.argv[1])
        else:
            print("Make sure the preemption value is an integer. Try again.\n")
            return -1
    else:
        preemption = 0

    if len(sys.argv) > 2:
        print("Only pass in one positive integer to represent max preemptions.\n")
        return -1

    # Setup the communcation framework  
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Get the training date, and initialize the weights for the system 
    train_data = svm_model.get_train_data()
    test_data = svm_model.get_test_data()

    # Split the data features into seperate data sectors for number of nodes in network (minus master since it
    # is doing no computations in this architecture)
    split_data_features = split_data(train_data, size-1)

    # Initialize the weights, there are 219 in the data set we used
    number_of_data_features = 219
    weights = np.full(number_of_data_features, np.random.uniform(low=-0.01, high=0.01))

    # Compute the gradients for the dataset
    if rank == 0:
        avg_weights = compute_gradients(comm, data=split_data_features, weights=weights, max_preemptions=preemption)
        current_results = svm_model.predict(avg_weights, test_data, debug=False)
        print(current_results)
    else:
        compute_gradients(comm)


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

    for x in range(num_splits - 1):
        return_list.append(data[start:end])
        start += size_of_arrays
        end += size_of_arrays

    # Add the last split
    return_list.append(data[start:])
    return return_list


def compute_gradients(comm, data=None, weights=None, max_preemptions=0):

    # Send each worker a chunk of data
    nodes_to_skip = scatter_data_to_workers(comm, data, weights, max_preemptions=max_preemptions)

    # Calculate the gradients on each of the worker nodes, recieve them, and update the weights on the master node
    calculate_gradients(comm, nodes_to_skip)

    # Receive gradients from all of the worker nodes, and add average it into the master node
    avg_weights = receive_gradients(comm, nodes_to_skip)
    return avg_weights


# Function used to send data from the master node to all of the woder nodes
def scatter_data_to_workers(comm, data, weights, max_preemptions=0):

    # Get rank
    rank = comm.Get_rank()

    nodes_preemptied = {}

    # Scatter the data to work nodes
    if rank == 0:

        # Nodes count which are not sent to 
        nodes_preemptied_count = 0

        for i in range(1, 5):

            # See if this node should not be sent data
            flip = random.randint(0, 1)

            # See if we can skip this node, if so skip and try to send to the next node
            if flip == 1 and nodes_preemptied_count != max_preemptions:
                nodes_preemptied_count += 1
                nodes_preemptied[i] = 1
                print("Not sending to node", str(i))
                data_to_send = {
                    'use': 0
                }
                comm.send(data_to_send, dest=i)
                continue

            data_to_send = {
                'weights': weights,
                'data': data[i - 1],
                'use': 1
            }
            comm.send(data_to_send, dest=i)
            print("Sent to node", str(i))

    return nodes_preemptied

# Calculate gradients on each of the worker nodes to send back to the master node to update
def calculate_gradients(comm, nodes_to_skip):
    # Get rank
    rank = comm.Get_rank()

    if rank != 0:
        data = comm.recv(source=0)

        # Check to see if there shouldn't be a gradient sent back
        if data['use'] == 0:
            return

        # Otherwise send back the gradients from this node
        trained_weights = svm_model.train_and_eval(data['weights'], data['data'])
        comm.send(trained_weights, dest=0)


# Receive the gradients from all of the worker nodes
def receive_gradients(comm, nodes_to_skip):
    # Get rank
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Take in the weights from each worker
    avg_weights = np.zeros(219)

    # Number of gradients recieved from workers 
    gradients_recieved = 0

    if rank == 0:
        # Receive messages from all of the worker nodes
        for i in range(1, 5):

            if nodes_to_skip.get(i) != None:
                continue

            data = comm.recv(source=i)
            avg_weights += data
            gradients_recieved += 1
        avg_weights /= gradients_recieved
    return avg_weights



if __name__ == '__main__':
    main()
