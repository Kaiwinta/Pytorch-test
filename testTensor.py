import torch

#A tensor is a neested list with many dimention it can be more than 3

x = torch.empty(2,3)            #Permet de créer un tensor en deux dimensiion avec toutes la valeurs vides
x = torch.rand(2,2)             #Permet ce créer avec des valeur aléatoires entre 0 et 1 j'ai l'impression 
x = torch.zeros(2,2)            #Rempli que de 0
x = torch.ones(2,2)             #Rempli avec des valeur 1.
x = torch.tensor([2.5,1.3])     #Rempli avec une liste python

x = torch.ones(2,2 , dtype = torch.int)     #dtype permet d'avoir des valeurs du type demandé dans le tensor
                                            #float16 , float32, int , double
print("\n==================")
print(x.dtype)                  #Permet de voir le type
print(x.size())                 #Permet d'avoir la taille  !Fonction
print(x)                        #Permet d'afficher le tensor
print("==================\n")

x = torch.rand(2,2)
y = torch.rand(2,2)

z = x+y                         #L'addition de tensor marche avec +
z = torch.add(x,y)              #fonction de Torch qui fait la même chose

z = x-y                         #Soustraction de tensor
z = torch.sub(x,y)              #Equivalent plus propre

z = x*y                         #multiplication de tensor
z = torch.mul(x,y)              #Equivalent plus propre

z = x/y                         #Division de tensor
z = torch.div(x,y)              #Equivalent plus propre

print("\n==================")
print(x)
print(y,"\n")
print(z)
print("==================\n")

y.add_(x)                       #Additionne x à y et stocke dans y
y.sub_(x)                       #Soustrait x à y et stocke dans y 
y.mul_(x)                       #Multiplie x à y et stocke dans y 
y.div_(x)                       #Divise par x et stocke dans y
                                #Dans pytorch, si une fonction à un _ c'est que c'est une modification sur place
                                #Exemple add_ modifie sur place

