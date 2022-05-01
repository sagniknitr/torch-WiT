import torch.multiprocessing as mp
import time
import argparse
import torch
import logging
import os

from torch import distributed as dist
""" Implementation of a recursive-allreduce with addition. """


def reduceScatter(x, left, right):
        if (left == right): return
        size = right - left + 1
        mid = (left + right)//2
        if dist.get_rank() <= mid:
            partner = dist.get_rank() + size//2
        else:
            partner = dist.get_rank() - size//2

        if dist.get_rank() <= mid:
            send_req = dist.isend(x[mid:right + 1], partner)
            tmp = torch.zeros_like(x[left:mid + 1])
            dist.recv(tmp, partner)
            x[left:mid + 1] = x[left:mid + 1] + tmp
        else:
            send_req = dist.isend(x[left:mid + 1], partner)
            tmp = torch.zeros_like(x[mid:right + 1])
            dist.recv(tmp, partner)
            x[mid:right + 1] = x[mid:right + 1] + tmp

        send_req.wait()

        if dist.get_rank() <= mid:
            reduceScatter(x, left, mid)
        else:
            reduceScatter(x, mid+1, right)



def allGather(x, left, right):
        if (left == right): return
        size = right - left + 1
        mid = (left + right)//2
        if dist.get_rank() <= mid:
            partner = dist.get_rank() + size//2
        else:
            partner = dist.get_rank() - size//2

        if dist.get_rank() <= mid:
            allGather(x, left, mid)
        else:
            allGather(x, mid+1, right)

        if dist.get_rank() <= mid:
            send_req = dist.isend(x[left:mid + 1], partner)
            dist.recv(x[mid: right + 1], partner)
        else:
            send_req = dist.isend(x[mid:right + 1], partner)
            dist.recv(x[left: mid + 1], partner)
        send_req.wait()



def run_allreduce(rank, size):
    tot_time = 0

    data = torch.rand(1, 1024, dtype=torch.float32)
    recv = torch.zeros_like(data)
    print(data)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU],
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./result_recursive'),
        record_shapes=True
    ) as p:
        reduceScatter(data, 0, size-1)
        allGather(data, 0, size-1)
        #p.step()

    print(data)






def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29600'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    size = 16
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run_allreduce))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()