import torch, time

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
ca = torch.rand(100,100).cuda()

z = torch.einsum('ij,ik,il->jkl', ca, ca, ca)

torch.cuda.synchronize()
torch.cuda.synchronize()

a = time.perf_counter()
for i in range(10):
    z = torch.einsum('ij,ik,il->jkl', ca, ca, ca)
    torch.cuda.synchronize() # wait for mm to finish
b = time.perf_counter()
print('batch GPU {:.02e}s'.format(b - a))
