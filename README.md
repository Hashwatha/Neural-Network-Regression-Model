# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The objective of this project is to develop a Neural Network Regression Model that can accurately predict a target variable based on input features. The model will leverage deep learning techniques to learn intricate patterns from the dataset and provide reliable predictions.

## Neural Network Model

<img width="1088" height="465" alt="image" src="https://github.com/user-attachments/assets/059e738e-cfba-4e74-996d-7b532866b480" />

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Hashwatha M
### Register Number: 212223240051
```

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

X = torch.linspace(1,70,70).reshape(-1,1)

torch.manual_seed(71)
e = torch.randint(-8,9,(70,1),dtype=torch.float)

y = 2*X + 1 + e
print(y.shape)

plt.scatter(X.numpy(), y.numpy(),color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Data for Linear Regression')
plt.show()
torch.manual_seed(59)
```
```

class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


torch.manual_seed(59)
model = Model(1, 1)
print('Weight:', model.linear.weight.item())
print('Bias:  ', model.linear.bias.item())

loss_function = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

epochs = 50
losses = []

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_function(y_pred, y)
    losses.append(loss.item())

    loss.backward()
    optimizer.step()


    print(f'epoch: {epoch:2}  loss: {loss.item():10.8f}  '
          f'weight: {model.linear.weight.item():10.8f}  '
          f'bias: {model.linear.bias.item():10.8f}')

plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('epoch');
plt.show()


x1 = torch.tensor([X.min().item(), X.max().item()])


w1, b1 = model.linear.weight.item(), model.linear.bias.item()


y1 = x1 * w1 + b1


print(f'Final Weight: {w1:.8f}, Final Bias: {b1:.8f}')
print(f'X range: {x1.numpy()}')
print(f'Predicted Y values: {y1.numpy()}')


plt.scatter(X.numpy(), y.numpy(), label="Original Data")
plt.plot(x1.numpy(), y1.numpy(), 'r', label="Best-Fit Line")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trained Model: Best-Fit Line')
plt.legend()
plt.show()

```
## Dataset Information

<img width="844" height="593" alt="image" src="https://github.com/user-attachments/assets/b710a16d-dd20-4b74-8d2e-2dc8387213d5" />

## OUTPUT

### Training Loss Vs Iteration Plot

<img width="1048" height="543" alt="image" src="https://github.com/user-attachments/assets/1dc75b6e-fd5f-4f1b-926c-0c01c6dcffad" />

### Best Fit line plot

<img width="928" height="582" alt="image" src="https://github.com/user-attachments/assets/e0c9a11d-9250-4584-af1c-3966e2051a7e" />

### New Sample Data Prediction

<img width="1103" height="108" alt="image" src="https://github.com/user-attachments/assets/4e4369c7-62f4-48a4-9c31-b8dba9cd6efb" />

## RESULT

The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
