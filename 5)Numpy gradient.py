import numpy as np

"""
    Premier vraie IA

    w évolue au fur et à mesure en augmentant
    Le loss lui diminue entre chaque learning
    Et les résultats se rapproche donc plus au fur et à mesure

    Là tout est fait manuellemtn, l'objectif est de faire automatiquement
"""

#f = w* x
#f = 2*x

x = np.array([1,2,3,4],dtype=np.float32)
y = np.array([2,4,6,8],dtype=np.float32)

#On initialise un poid
w = 0.0

#Calculate model prediction
def frowward(x):
    return w * x

#Calculate loss =MSe 
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

#Gradient
#Mse  =1/N * (w*x - y )**2
#dj/dw = 1/n éx (w*x -y)

def gradient(x,y,y_predicted):
    return np.dot(2*x,y_predicted-y).mean()

print(f'Prediction before training : f(5) = {frowward(5):.3f}')

#Training
learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):
    #prediction(forward pass)
    y_pred = frowward(x)

    #loss
    l = loss(y, y_pred)

    #gradient
    dw = gradient(x,y,y_pred)

    #Update weight
    w -= learning_rate * dw

    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training : f(5) = {frowward(5):.3f}')