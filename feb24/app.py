import torch
import torch.nn as nn
import gradio as gr 

model_data = torch.load('model.pth')
fm = model_data['fm']
fs = model_data['fs']
tm = model_data['tm']
ts = model_data['ts']
parameters = model_data['parameters']

model = nn.Linear(2,1)
model.load_state_dict(parameters)

def f(weight, engineSize):
    features = torch.tensor([
        [weight, engineSize]
    ]).float()
    X = (features -fm)/fs
    Yhat = model(X)
    prediction = Yhat * ts + tm 
    return(prediction.item())

with gr.Blocks() as iface:
    weightBox = gr.Number(label = 'Provide Weight of Vehicle')
    engineBox = gr.Number(label = 'Provide Engine Size')
    mpgBox = gr.Number(label = 'MPG Prediction')
    weightBox.change(fn = f, inputs = [weightBox, engineBox], outputs = [mpgBox])
    engineBox.change(fn = f, inputs = [weightBox, engineBox], outputs = [mpgBox])

iface.launch() 