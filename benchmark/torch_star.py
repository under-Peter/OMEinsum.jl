import torch, time
torch.set_num_threads(1)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

N2 = 100
ar = torch.rand(100,100)
ca = ar.cuda(2)
z = torch.einsum('ij,ik,il->jkl', ca, ca, ca)
torch.cuda.synchronize() # wait for mm to finish
a = time.perf_counter()
for i in range(N2):
    ar = torch.rand(800,800)
    ca = ar.cuda(2)
    z = torch.einsum('ji,ki,li->jkl', ca, ca, ca)
    torch.cuda.synchronize() # wait for mm to finish
b = time.perf_counter()
print('GPU: Star {:.02e}s'.format((b - a)/N2))
