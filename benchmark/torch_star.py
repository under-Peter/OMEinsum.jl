import torch, time

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

N2 = 10
a = time.perf_counter()
for i in range(N2):
    ar = torch.rand(800,800)
    ca = ar.cuda(1)
    z = torch.einsum('ij,ik,il->jkl', ca, ca, ca)
    torch.cuda.synchronize() # wait for mm to finish
b = time.perf_counter()
print('GPU: Star {:.02e}s'.format((b - a)/N2))
