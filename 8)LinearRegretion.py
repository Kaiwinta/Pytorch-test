#Design model (input, output size , froward pass)
#2) construct loss and optimizer
#3) Training Loop:
#       -frward pass: compute prediction
#       -backward pass: gradients
#       -update weigths

import torch
import torch.nn as nn       #Stands for neural networks
import numpy as np
from sklearn import datasets    
import matplotlib.pyplot as plt


#Prepare data

x_numpy, y_numpy = datasets.make_regression(n_samples=100 , n_features=1, noise = 20 , random_state=1)

X = torch.from_numpy(x_numpy.astype(np.float32))
Y = torch.from_numpy(y_numpy.astype(np.float32))

#Reshaping 
Y = Y.view(Y.shape[0],1)

n_samples, n_features = X.shape


#Model
input_size = n_features
output_size = 1
model = nn.Linear(input_size,output_size)

#Loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)


#Training loop
num_epoch = 1000
for epoch in range(num_epoch):
    #forward pass an d loss
    y_predicted = model(X)
    loss = criterion(y_predicted,Y)

    #Bakcward pass
    loss.backward()

    #Update
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1) %100 == 0:
        print(f"epoch : {epoch+1}, loss = {loss.item():.3f}")

predicted = model(X).detach()           #On detache du graphe ==> requires grad == false
plt.plot(x_numpy,y_numpy,'ro')
plt.plot(x_numpy,predicted,'b')
plt.show()