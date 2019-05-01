from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print(MPI.Get_processor_name(), str(rank))

if rank == 0:
    print("Master doing work.")
elif rank == 1:
    print("Worker 1")
elif rank == 2:
    print("Worker 2")
elif rank == 3:
    print("Worker 3")
elif rank == 4:
	print("Worker 4")
