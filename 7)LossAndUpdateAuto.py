import torch
import torch.nn as nn

#Design model (input, output size , froward pass)
#2) construct loss and optimizer
#3) Training Loop:
#       -frward pass: compute prediction
#       -backward pass: gradients
#       -update weigths



#f = w* x
#f = 2*x

x = torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
y = torch.tensor([[2],[4],[6],[8]],dtype=torch.float32)

TestTensor = torch.tensor  ([5],dtype = torch.float32)

n_samples , n_features = x.shape
print(n_samples , n_features)

inpout_size = n_features
output_size = n_features

model = nn.Linear(inpout_size,output_size)

"""class LinearRegression(nn.Module):

    def __init__ (self, input_dim , output_dim):
        super(LinearRegression,self).__init__()

        #define layers
        self.lin = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        return self.lin(x)"""



print(f'Prediction before training : f(5) = {model(TestTensor).item():.3f}')

#Training
learning_rate = 0.01
#L'utilisation de backward demande plus d'iteration mais automatise le sclacul de gradient
n_iters = 1000

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr =  learning_rate)

for epoch in range(n_iters):
    #prediction(forward pass)
    y_pred = model(x)

    #loss
    l = loss(y, y_pred)

    #gradient = backward pass
    l.backward()        #Calculate dl/dw automaticly

    #Update weight
    optimizer.step()
    
    #zero gradient
    optimizer.zero_grad()

    if epoch % 100 == 0:
        [w, b ]=model.parameters()
        print(f'epoch {epoch+1}: w {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training : f(5) = {model(TestTensor).item():.3f}')