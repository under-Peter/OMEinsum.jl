import torch, time
torch.set_num_threads(1)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

def bfunc_gpu_star(N):
    ar = torch.rand(N,N)
    ca = ar.cuda(0)
    z = torch.einsum('ji,ki,li->jkl', ca, ca, ca)
    torch.cuda.synchronize() # wait for mm to finish

N2 = 100
ar = torch.rand(100,100)
ca = ar.cuda(0)
z = torch.einsum('ij,ik,il->jkl', ca, ca, ca)
torch.cuda.synchronize() # wait for mm to finish
a = time.perf_counter()
for i in range(N2):
    bfunc_gpu_star(300)
b = time.perf_counter()
print('GPU: Star {:.02e}s'.format((b - a)/N2))
