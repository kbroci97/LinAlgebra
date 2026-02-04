import torch 

x = torch.tensor(3.0, requires_grad = True)
y = torch.tensor(1.0, requires_grad = True)
z = torch.tensor(1.0, requires_grad = True)

f = -4*z*x**3*y**2 + 5*z**3 + y*x**3*z**2
f.backward()

print(x.grad)
print(y.grad)
print(z.grad)