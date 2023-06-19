"""
    Cours théorique
Voir Backpropagation.pdf 
    
Si une valeur x ==> y ==> z 
                a(x)  b(y)

Alors dz/dx = dz/dy * dy/dx

C'es la Chain Rule Sert à trouver le final gradient

Python forme des graphes pour chaque opération

Ex :  voir Computational_Graph.png

Le local grdient permettent de mieux obtenir le gradient final

3 étapes:

    Forward pass:
        Aplly all the function and compute the loss 

    Compute Local Gradient:
        We callulate the local gradient at each node

    Backward pass:
        Compute the gradient of the loss
        Using the Chain rule
        dLoss /dWeight

We want to minimize our loss
"""