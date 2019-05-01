from mpi4py import MPI
from worker_node import Worker
from master import Master
from svm import SVM


def main():
    svm_modal = SVM()

    # Setup the communcation framework
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    print(MPI.Get_processor_name(), str(rank))

    train_data = svm_modal.get_train_data()

    trained_weights = svm_modal.train_and_eval(weights, train_data)


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


if __name__ == '__main__':
    main()
