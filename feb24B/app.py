import torch
import torch.nn as nn 
import gradio as gr

model_data = torch.load('model.pth')

fm = model_data['fm']
fs = model_data['fs']
parameters = model_data['parameters']

linear = nn.Linear(1,1)
linear.load_state_dict(parameters)

model = nn.Sequential(
    linear,
    nn.Sigmoid()
)

def f(tumorSize):
    features = torch.tensor([
        [tumorSize]
    ]).float()
    X = (features - fm)/fs
    result = model(X)
    if(result >= .5):
        classification = 'Malignant'
    else:
        classification = 'Benign'
    return classification

with gr.Blocks() as iface:
    sizeBox = gr.Number(label = 'Provide Size of Tumor (cm)')
    diagnosisBox = gr.Text(label = 'Predicted Diagnosis')
    sizeBox.change(fn = f, inputs = [sizeBox], outputs = [diagnosisBox])

iface.launch() 