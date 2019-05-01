from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print(MPI.Get_processor_name(), str(rank))
