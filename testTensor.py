import torch

#A tensor is a neested list with many dimention it can be more than 3

x = torch.empty(2,3)            #Permet de créer un tensor en deux dimensiion avec toutes la valeurs vides
x = torch.rand(2,2)             #Permet ce créer avec des valeur aléatoires entre 0 et 1
x = torch.zeros(2,2)            #Rempli que de 0
x = torch.ones(2,2)             #Rempli avec des valeur 1.
x = torch.tesnor([2.5,1.3])     #Rempli avec une liste python

x = torch.ones(2,2 , dtype = torch.int)     #dtype permet d'avoir des valeurs du type demandé dans le tensor
                                            #float16 , float32, int , double

print(x.dtype)                  #Permet de voir le type
print(x.size())                 #Permet d'avoir la taille  !Fonction
print(x)                        #Permet d'afficher le tensor
