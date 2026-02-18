import torch
import pandas as pd 

df = pd.read_csv('data.csv')
X = torch.tensor(df.drop('Y', axis = 1).to_numpy()).float()
Y = torch.tensor(df['Y'].to_numpy()).float().reshape(-1,1)

w = torch.tensor([
    [0.0]
]).float()

b = torch.tensor([
    [0.0]
])

Yhat = X@w + b

r = Yhat - Y

SSE = r.T@r

loss = SSE/3

print(Yhat)
print(r)
print(SSE)
print(loss)