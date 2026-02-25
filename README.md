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
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,2)
        self.fc2 = nn.Linear(2,5)
        self.fc3 = nn.Linear(5,2)
        self.fc4 = nn.Linear(2,1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.relu(self.fc3(x))
    x = self.fc4(x)

    return x



# Initialize the Model, Loss Function, and Optimizer

ai_world = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_world.parameters(),lr=0.001)


def train_model(ai_world, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_world(X_train), y_train)
        loss.backward()
        optimizer.step()

        ai_world.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


```
## Dataset Information

<img width="223" height="543" alt="image" src="https://github.com/user-attachments/assets/474181d7-cc78-440b-8d3a-21db3dced7aa" />

## OUTPUT

### Training Loss Vs Iteration Plot

<img width="754" height="552" alt="image" src="https://github.com/user-attachments/assets/9a31313f-e9c9-44cb-8d53-58e0e03da39c" />


### New Sample Data Prediction

<img width="939" height="287" alt="image" src="https://github.com/user-attachments/assets/5be7300c-7677-4b1e-ba5c-68dc5af1a103" />


## RESULT

The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
