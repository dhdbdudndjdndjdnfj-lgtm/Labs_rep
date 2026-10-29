import torch
import torch.nn as nn

import math

#Обучающая выборка
x = torch.linspace(-math.pi, math.pi, 100).unsqueeze(1)
y = torch.sin(x)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.trainiga = nn.Sequential(
            nn.Linear(1,16),
            nn.ReLU(),
            nn.Linear(16,1)
        )
        self.loss_fn = nn.MSELoss()
        self.optimiser = torch.optim.Adam(self.parameters(), lr = 0.001)
        
    def forward(self, x):
        y_pred = self.trainiga(x) #То, что предсказывает модель
        return y_pred
        
model = CNN()
epochs = 5000

for i in range(epochs):
    y_pred = model(x)
    loss = model.loss_fn(y_pred, y)
    
    model.optimiser.zero_grad()
    loss.backward()
    model.optimiser.step()
    
print(model(torch.tensor([0.9])).item(), torch.sin(torch.tensor([0.9])).item(), sep = "\t")
    
    
    
    
        
    