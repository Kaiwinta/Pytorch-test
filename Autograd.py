import torch

def premier_gradiant():
    x = torch.randn(3 , requires_grad=True)     #randn is like rand but with a different distribution of 0 and 1
    print(x)

    y = x+2
    print(y)                #L'on voit grad_fn AddBackward      L'ordinateur sais que l'on a éffectuer une addition

    z = y * y *2

    print(z)                #L'on voit MulBackward              L'ordinateur sait que l'on a fait une multiplication


    z = z.mean()    #L'on calcul la moyenne du tensor afin de n'avoir qu un élément
    z.backward()    #Calcul dz/dx      Marche uniquement sur un élément un Scalar Values
    print(x.grad)   #Affiche le gradient de x

    #Dans le cas ou l'on veut le faire pour plusieurs éléments:

    v = torch.zeros(3,dtype=torch.float32)  #La valeur est la taille de y
    y.backward(v)   #L'on ajoute un paramètre qui est un tensor dit vecteur
    print(x.grad)

def No_Grad():
    """
        In this chapter we want to stop pytorch from tracking the computer calculation
        There's many way so us only one of them
    
    """
    x = torch.randn(3 , requires_grad=True)     #randn is like rand but with a different distribution of 0 and 1
    print(x)

    #1
    y = x.detach()              #Créer un nouveau tensor avec les même valeurs mais sans requires_grad
    print(y)
    #2
    x.requires_grad_(False)     #Permet de ne plus traquer les requires_grad
    print(x)
    #3
    with torch.no_grad():       #Permet de fare des calculs sans traquer l'historqie des calculs
        y = x+2
        print(y)

def Reset():
     #The objective is only to show that ne need to reset the gard at every iteration
     weight = torch.ones(4,requires_grad=True)

     for epoch in range(3):
        model_output = (weight*3).sum() #Just a dummy training wich is the sum of all the values of weigth times 3 so 1*3 + 1*3 + 1*3 + 1*3 + 1*3

        print(model_output)

        model_output.backward()

        print(weight.grad)              #We show the gradient wich all are 3

        weight.grad.zero_()             #We reset it otherwiseit will be 3 then 6, then 9

Reset()