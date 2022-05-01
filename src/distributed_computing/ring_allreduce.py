import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def allreduce(send, recv):
   rank = dist.get_rank()
   size = dist.get_world_size()
   send_buff = torch.zeros(send.size())
   recv_buff = torch.zeros(send.size())
   accum = torch.zeros(send.size())
   accum[:] = send[:]

   left = ((rank - 1) + size) % size
   right = (rank + 1) % size

   for i in range(size - 1):
       if i % 2 == 0:
           # Send send_buff
           send_req = dist.isend(send_buff, right)
           dist.recv(recv_buff, left)
           accum[:] += recv[:]
       else:
           # Send recv_buff
           send_req = dist.isend(recv_buff, right)
           dist.recv(send_buff, left)
           accum[:] += send[:]
       send_req.wait()
   recv[:] = accum[:]

def run_allreduce(rank, sz):

    data = torch.rand(sz // 4, dtype=torch.float32)
    recv = torch.zeros_like(data)
    print("start profiling")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU],
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./result_gloo'),
        record_shapes=True
    ) as p:
        allreduce(send=data, recv=recv)
        p.step()

    print(recv)


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29600'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 4
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run_allreduce))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()