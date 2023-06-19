import torch

"""
    Premier vraie IA

    w évolue au fur et à mesure en augmentant
    Le loss lui diminue entre chaque learning
    Et les résultats se rapproche donc plus au fur et à mesure

    Là torch est utilisé donc on peu automatisé
    L'on automatise uniquement le calcul de gradient actuellement
"""

#f = w* x
#f = 2*x

x = torch.tensor([1,2,3,4],dtype=torch.float32)
y = torch.tensor([2,4,6,8],dtype=torch.float32)

#On initialise un poid qui demandera un gradient
w = torch.tensor(0.0, dtype=torch.float32 , requires_grad=True)

#Calculate model predictionf
def frowward(x):
    return w * x

#Calculate loss =MSe 
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()



print(f'Prediction before training : f(5) = {frowward(5):.3f}')

#Training
learning_rate = 0.01
#L'utilisation de backward demande plus d'iteration mais automatise le sclacul de gradient
n_iters = 100

for epoch in range(n_iters):
    #prediction(forward pass)
    y_pred = frowward(x)

    #loss
    l = loss(y, y_pred)

    #gradient = backward pass
    l.backward()        #Calculate dl/dw automaticly

    #Update weight
    with torch.no_grad():
        w -= learning_rate * w.grad
    
    #zero gradient
    w.grad.zero_()

    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training : f(5) = {frowward(5):.3f}')