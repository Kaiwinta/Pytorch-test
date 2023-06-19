import torch

"""
    This code is all the Basis about Tensor
    The code is divised in a few chapter that are the different function
    I documented the most i can in both languages but if anything is false plese provide me a correction
"""

def Initialisation():

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

def Basic_operation():

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

def Slicing():
    x = torch.rand(5,3)
    print("\n==================")
    print(x)
    print(x[: , 0])         #Premier parametre des crochet = ligne , deuxième = slice des colonnes, l'on affiche toute les ligne et leur première colonne
    print(x[0 , :])         #Affiche toutes les colonnes de la première ligne
    print(x[1,1])           #Affiche un unique élement

    print(x[1,1].item())    #Affiche la valeur réelle, ne marche qu'avec un élément
    print("==================\n")

def Reshape_Tensor():

    x = torch.rand(4,4)

    y =x.view(16)           #La fonction .view permet de mettre les élément dans une autre forme
                            #Le nombre d'élément total doit être conservé

    z = x.view(8,2)         #Nombre total conservé et mise en page encore différente
                            #D'abord l'on lis toute la première ligne puis toute la deuxième

    t = x.viex(-1,8)        #L'on choisi une seule dimension la deuxième se fera automatiquemeny
                            #Permet de ne pas perdre de donnée et d'avoir la bonne taille 

    print("\n==================")
    print(x)
    print(y) 
    print(z)
    print(t)
    print("==================\n")

def Numpy_to_Tensor():

    import numpy as np

    a = torch.ones(5,5)
    print(a)
    b = a.numpy()           #Convertis en un numpy array
    print(b)

    #Attention, si l'on est sur le CPU, modifier le tensor modifiera l'array car même adresse mémoire et vice versa

    a = np.ones(5)
    b = torch.from_numpy(a) #Convertis en un Tensor
    print(a)
    print(b)

    #Attention, si l'on est sur le CPU, modifier le tensor modifiera l'array car même adresse mémoire et vice versa

    #Permet de créer un tensor sur la carte graphique
    #Avantage: Calcul plus rapide et pas de partage d'adresse mémoire avec des numpy arrays
    #Desavantage : L'on ne peut pas utiliser numpy sur la gpu
    if torch.cuda.is_available():
        device = torch.device("cuda")       #L'on définis cuda comme device
        x = torch.ones(5, device = device)  #Créer un tensor sur le Gpu

        print("==============")
        print(b.device)                     #Affiche la device de b (cpu)
        print(x.device)                     #Affiche la device de x (cuda:0)

        y = torch.ones(5)
        print("==============")
        print(y.device)     
        y = y.to(device)                     #Deplace y du cpu au gpu
        print(y.device)

        z = y+x                             #Calculs plus rapides sur le cpu
        z = z.to('cpu')                     #Deplace z au cpu pour pouvoir utiliser numpy

        t = z.numpy()
        print(t)

    x = torch.ones(5, requires_grad=True)   #Paramètre bonus qui sera utiliser en cas d'optimisation en vue
    print(x)