import torch, time, timeit
import numpy as np

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
ar = torch.rand(100,100)
ca = ar.cuda()

z = torch.einsum('ij,ik,il->jkl', ar, ar, ar)
z = torch.einsum('ij,jk->ik', ar, ar)
z = torch.einsum('ij,ik,il->jkl', ca, ca, ca)
z = torch.einsum('ij,jk->ik', ca, ca)

torch.cuda.synchronize()

N1 = 10000
a = time.perf_counter()
for i in range(N1):
    z = torch.einsum('ij,jk->ik', ca, ca)
    torch.cuda.synchronize() # wait for mm to finish
b = time.perf_counter()
print('GPU: Matmul 100 {:.02e}s'.format((b - a)/N1))

N2 = 100
a = time.perf_counter()
for i in range(N2):
    z = torch.einsum('ij,ik,il->jkl', ca, ca, ca)
    torch.cuda.synchronize() # wait for mm to finish
b = time.perf_counter()
print('GPU: Star 100 {:.02e}s'.format((b - a)/N2))

t1 = timeit.timeit("torch.einsum('ij,jk->ik', ar, ar)", setup="from __main__ import torch, ar, np", number=N1)
print('CPU: Matmul 100 {:.02e}s'.format(t1/N1))
t2 = timeit.timeit("torch.einsum('ij,ik,il->jkl', ar, ar, ar)", setup="from __main__ import torch, ar, np", number=N2)
print('CPU: Star 100 {:.02e}s'.format(t2/N2))


t1 = timeit.timeit("np.einsum('ij,jk->ik', ar, ar)", setup="from __main__ import torch, ar, np", number=N1)
print('NPCPU: Matmul 100 {:.02e}s'.format(t1/N1))
t2 = timeit.timeit("np.einsum('ij,ik,il->jkl', ar, ar, ar)", setup="from __main__ import torch, ar, np", number=N2)
print('NPCPU: Star 100 {:.02e}s'.format(t2/N2))
