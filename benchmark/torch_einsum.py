# to run: OMP_NUM_THREADS=1 python benchmark/torch_einsum.py
import time, timeit
import numpy as np
import torch

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

def bfunc_gpu_star(N):
    ar = torch.rand(N,N)
    ca = ar.cuda(0)
    z = torch.einsum('ji,kl,li->jkl', ca, ca, ca)
    torch.cuda.synchronize() # wait for mm to finish

def bfunc_gpu_matmul(N):
    ar = torch.rand(N,N)
    ca = ar.cuda(0)
    z = torch.einsum('ij,jk->ik', ca, ca)
    torch.cuda.synchronize() # wait for mm to finish

def bfunc_cpu_star(N):
    ar = torch.rand(N,N)
    z = torch.einsum('ji,kl,li->jkl', ar, ar, ar)

def bfunc_cpu_matmul(N):
    ar = torch.rand(N,N)
    z = torch.einsum('ij,jk->ik', ar, ar)

def bfunc_npcpu_star(N):
    ar = np.array(np.random.randn(N,N),dtype=np.float32)
    z = np.einsum('ji,kl,li->jkl', ar, ar, ar)

def bfunc_npcpu_matmul(N):
    ar = np.array(np.random.randn(N,N),dtype=np.float32)
    z = np.einsum('ij,jk->ik', ar, ar)

bfunc_gpu_star(100)
bfunc_gpu_matmul(100)

N1 = 10000
N2 = 100
#a = time.perf_counter()
#for i in range(N1):
#    bfunc_gpu_matmul(100)
#b = time.perf_counter()
#print('GPU: Matmul 100 {:.02e}s'.format((b - a)/N1))
#
#a = time.perf_counter()
#for i in range(N2):
#    bfunc_gpu_star(100)
#b = time.perf_counter()
#print('GPU: Star 100 {:.02e}s'.format((b - a)/N2))

t1 = timeit.timeit("bfunc_gpu_matmul(100)", setup="from __main__ import bfunc_gpu_matmul", number=N1)
print('GPU: Matmul 100 {:.02e}s'.format(t1/N1))
t2 = timeit.timeit("bfunc_gpu_star(100)", setup="from __main__ import bfunc_gpu_star", number=N2)
print('GPU: Star 100 {:.02e}s'.format(t2/N2))

t1 = timeit.timeit("bfunc_cpu_matmul(100)", setup="from __main__ import bfunc_cpu_matmul", number=N1)
print('CPU: Matmul 100 {:.02e}s'.format(t1/N1))
t2 = timeit.timeit("bfunc_cpu_star(100)", setup="from __main__ import bfunc_cpu_star", number=N2)
print('CPU: Star 100 {:.02e}s'.format(t2/N2))


t1 = timeit.timeit("bfunc_npcpu_matmul(100)", setup="from __main__ import bfunc_npcpu_matmul", number=N1)
print('NPCPU: Matmul 100 {:.02e}s'.format(t1/N1))
t2 = timeit.timeit("bfunc_npcpu_star(100)", setup="from __main__ import bfunc_npcpu_star", number=N2)
print('NPCPU: Star 100 {:.02e}s'.format(t2/N2))
