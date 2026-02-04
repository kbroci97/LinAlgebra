import torch
import pandas as pd 

df = pd.read_csv('data.csv')
X = torch.tensor(df.drop('Y', axis = 1).to_numpy()).float()
Y = torch.tensor(df['Y'].to_numpy()).float().reshape(-1,1)

w = torch.tensor([
    [1.9],
    [-.8],
    [1.8],
    [.6],
    [-.4]
]).float()

b = torch.tensor([
    [2.4]
])

Yhat = X@w + b

r = Yhat - Y

SSE = r.T@r

loss = SSE/20

print(Yhat)
print(r)
print(SSE)
print(loss)