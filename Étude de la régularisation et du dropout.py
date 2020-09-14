# ##############################################################################
#                   Étude de la régularisation et du dropout                   #
# ##############################################################################

# Le code s'articule en 5 parties :
#   I - la préparation des données
#   II - la construction du classifieur (celui présenté en 1.3 du rapport)
#   III - les fonctions de création d'adversaire
#   IV - les codes pour évaluer la distorsion des adversaires
#   V - les résultats

# Nous nous sommes intéressés à l'apprentissage non ciblé,
# ie: on veut trouver un adversaire, peu importe sa cible.





import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import time
from random import shuffle
import scipy.optimize as so




# I - Préparation des données
# ###########################

batch_size= 64
(x_train, y_train), (x_test, y_test)=tf.keras.datasets.mnist.load_data()
x_train=(x_train.reshape(-1, 784)/255).astype(np.float32)
x_test= (x_test.reshape(-1,  784)/255).astype(np.float32)
train_ds=tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)



# II - Construction du classifieur
# ################################

# Construction  d'un réseau à 2 couches cachées de 100 neurones chacune, dont le but est de classifier la base de données MNIST

def FC100_100_10(lambda0 = (10**(-5), 10**(-5), 10**(-6)), couches = (100,100)):
    Nh1, Nh2 = couches
    lambda1, lambda2, lambda3 = lambda0
    model = models.Sequential([
        layers.Dense(units=Nh1, activation='sigmoid',input_shape=(784,),
        kernel_regularizer=tf.keras.regularizers.l2(lambda1/100)),
        #layers.Dropout(0.9),
        layers.Dense(units=Nh2, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(lambda2/100)),
        #layers.Dropout(0.9),
        layers.Dense(units=10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(lambda3/10))
    ])
    
    print("Structure du modele")
    model.summary()
    
    return(model)


optimizer=tf.keras.optimizers.Adam()
loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
train_loss=tf.keras.metrics.Mean()
train_accuracy=tf.keras.metrics.SparseCategoricalAccuracy()
test_loss=tf.keras.metrics.Mean()
test_accuracy=tf.keras.metrics.SparseCategoricalAccuracy()

# ici "@tf.function" perm d'exécute + rapidement les instructions
@tf.function
def train_step(images, labels, model):
  with tf.GradientTape() as tape:
    predictions=model(images)
    loss=loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_loss(loss) # + bas dans "def train" on comprend l'intérêt de cette instruction (édition des résultats
  train_accuracy(labels, predictions) # idem

def train(train_ds, nbr_entrainement, model):
  for entrainement in range(nbr_entrainement):
    start=time.time()
    for images, labels in train_ds:
      train_step(images, labels, model)
    message='Entrainement {:04d}, loss: {:6.4f}, accuracy: {:7.4f}%, temps: {:7.4f}'
    print(message.format(entrainement+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         time.time()-start))
    train_loss.reset_states()
    train_accuracy.reset_states()

def test(xt,yt,model):
  start=time.time()
  predictions=model(xt)
  t_loss=loss_object(yt, predictions)
  test_loss(t_loss)
  test_accuracy(yt, predictions)
  message='Loss: {:6.4f}, accuracy: {:7.4f}%, temps: {:7.4f}'
  print(message.format(test_loss.result(),
                       test_accuracy.result()*100,
                       time.time()-start))



def gradientX2(model, image, chiffre_cible):
  with tf.GradientTape() as tape:
    tape.watch(image)
    predictions=model(image)
    output_adversial = tf.math.log(1-predictions[0,chiffre_cible]) # le log de la composante correspondant au chiffre_cible noté "tk" dans la note
    # le log pour obtenir l'équivalent de la "loss function"
    # indice 0 car prediction.shape = (1,10) et 1e premier indice correspondant à la taille du batch (ici 1 car une seule image fournie)
  gradients = tape.gradient(output_adversial, image)
  # gradients.shape = (1,784)
  return gradients
    
def creation_Adversaire2(m, model):
    # Sélection de l'image à perturber
    image_original = x_test[m]
    imReelle = image_original.reshape(1,784)

    imAdversaire = tf.constant(imReelle) # on doit faire un "cast" en tenseur de la structure np.array de imReelle
                                         # pour utiliser les fonctionnalités de TensorFlow (modele + gradient)
    # Delta = imReelle - imAdversaire
    # EQM =  np.mean(Delta*Delta)
    # print(EQM)
    
    n=0
    alpha = 0.01
    chiffre_reconnu = np.argmax(model(imAdversaire))
    chiffre_original = y_test[m]
    # Le + simle des algo : "iterative gradient" (on se limite à 100 itérations)
    while n < 1000 and chiffre_reconnu == chiffre_original:
        n=n+1
        delta = alpha * gradientX2(model, imAdversaire,chiffre_original)
        imAdversaire = tf.clip_by_value(imAdversaire + delta, clip_value_min=0, clip_value_max=1)
        chiffre_reconnu = np.argmax(model(imAdversaire))
        
    imAdversaire = imAdversaire.numpy() # On transforme le tf de TensorFlow en np.array de NumPy
    Delta = imReelle - imAdversaire
    EQM =  np.mean(Delta*Delta)
    
    if n == 1000:
         #print("ca marche pas :( ")
         EQM = -1
    
    # # On edite le carré de la norme de Frobenius de la perturbation (moyenné par le nombre de pixels)
    # print("EQM = ",np.mean(Delta * Delta)) # pour ce cas EQM = 0.0015074897
    
    #print(EQM)
    return(imAdversaire, imReelle, EQM)
    


    
def gradientX42(model, image, n):
    chiffre_original = y_test[n]
    with tf.GradientTape() as tape:
        tape.watch(image)
        predictions=model(image)
        loss_val = loss_object(chiffre_original, 1-predictions)
        output_adversial = loss_val
    gradients = tape.gradient(output_adversial, image)
    return gradients
    
def gradientX52(mode, image, n):
    chiffre_original = y_test[n]
    with tf.GradientTape() as tape:
        tape.watch(image)
        predictions=model(image)
        output_adversial = predictions[chiffre_original]
    gradients = tape.gradient(output_adversial, image)
    # print(type(gradients), type(predictions))
    # gradients = -gradients/(predictions)
    return gradients
    
def func(x, model, c, n):
    ad = x.reshape(1,784)
    imAdversaire = tf.constant(ad)
    predictions=model(imAdversaire)
    
    # on calcule la fonction que l'on souahite minimiser
    loss_val = loss_object(y_test[n], 1-predictions)
    loss_val = loss_val.numpy()
    y = np.linalg.norm(ad-x_test[n])*c + loss_val
    
    #on calcule son gradient
    gra = gradientX42(model, imAdversaire, n)
    gra = (gra.numpy()+c*(2*ad[0]-2*x_test[n])).flatten()
    return y, gra

def IntriguingAdversersarialSansButPrecis(n, model1):
    """ on va chercher à créer un adversaire, pour le moment on ne renvoie rien !!!
    
    n = 1000 l'id de l'image que l'on veut trafiquer
    l = le chiffre que l'on souhaiterait trouver
    
    reseau, lambda0, epochs, plot_precision = sont des paramètres pour l'algorithme d'apprentissage
    
    On pourrait ajouter une précision à atteindre !!!
    On pourrait aussir proposer de trouver le premier adversaire que l'on peut, indépendamment de d'un objectif (ce sont des modifs faciles à faire
    """

    #on convertit plein de choses en tenseur pour pouvoir calculer lo et g
    l = y_test[n]


    #il s'agit désormais de trouver un bon c
    c = 10
    bounds = [(0,1)]*784
    
    predict = model1.predict(np.array([x_test[n]]))
    pic = x_test[n]
    asample = pic.reshape((1,28,28))
    ploc = pic.reshape((1,28,28))

    ad = so.minimize(func, x_test[n], method = 'L-BFGS-B', jac = True,args=(model1, 0, n), bounds = bounds)
    
    asample = ad.x.reshape((1,28,28))
    ploc = x_test[n].reshape((1,28,28))
    predict1 = model1.predict(np.array([ad.x]))
    
    
    possible = (np.argmax(predict1) != l)
    
    if ad.success == False:
        print(n, "raté")
    
    if not(np.argmax(predict) == l):
        #print("coucou")
        return(42,0)
    
    elif not(possible):
        return(42,-1)
    
    
    while np.argmax(predict) == l:
        #on s'arrête dès que l'on atteint un label différent du bon

        #on minimise la fonction
        ad = so.minimize(func, x_test[n], method = 'L-BFGS-B', jac = True,args=(model1, c, n), bounds = bounds)
        
        asample = ad.x.reshape((1,28,28))
        ploc = x_test[n].reshape((1,28,28))
        predict = model1.predict(np.array([ad.x]))
        c = c/1.5
    
    Delta = asample-ploc
    # print("c = ", c)
    #print(n, "EQM = ",np.mean(Delta * Delta))
    # print("prediction image : ", np.argmax(predict))
    # securite = model1.predict(asample.reshape(1,784))
    # print("prediction : ", predict)
    # print("securite : ", securite)
    # pic = asample[0]*255
    # plt.imshow(pic, cmap="gray")
    # plt.show()
    return(asample, np.mean(Delta * Delta))   
    
# ###################

nbr_entrainement= 10
lambda0 = (0,0,0)
#lambda0 = (10**(2), 10**(2), 10**(1))
model = FC100_100_10(lambda0 = lambda0, couches = (100,100))
model.compile(optimizer = optimizer, loss = loss_object, metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train,
          batch_size=64,
          epochs=nbr_entrainement,
)

# On fait appls aux fonctions "keras" très pratique
optimizer=tf.keras.optimizers.Adam()
loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
train_loss=tf.keras.metrics.Mean()
train_accuracy=tf.keras.metrics.SparseCategoricalAccuracy()
test_loss=tf.keras.metrics.Mean()
test_accuracy=tf.keras.metrics.SparseCategoricalAccuracy()

# model.fit
# print("Entrainement")
print(lambda0, nbr_entrainement, "dropout :", 0.9)
# train(train_ds, nbr_entrainement, model)

##
print("Jeu de test")
test(x_test,y_test, model)
    
minimisons = [x for x in range(len(y_test))]
shuffle(minimisons)
liste1 = []
liste2 = []
cbn = 0
for i in minimisons[:1000]:
    cbn += 1
    print(cbn)
    retour_adversaire1 = creation_Adversaire2(i, model)
    retour_adversaire2 = IntriguingAdversersarialSansButPrecis(i, model)
    liste1.append(retour_adversaire1[-1])
    liste2.append(retour_adversaire2[-1])

liste1 = [i for i in liste1 if i != 0]
a1 = len(liste1)
print("Il y a : ", 1000-a1, " prédictions fausses au départ, donc inutile de leur chercher un adversaire")
liste2 = [i for i in liste2 if i != 0]
a2 = len(liste2)
print(1000 - a2, " normalement on devrait tourver la même chose, donc on vérifie")

liste1 = [i for i in liste1 if i!= -1]
b1 = len(liste1)
print("Il y a : ", a1 - b1, " entrées qui n'ont pas d'adversaires avec la méthode de l'iterative gradient")
liste2 = [i for i in liste2 if i != -1]
b2 = len(liste2)
print("Il y a : ", a2 - b2, " entrées qui n'ont pas d'adversaires avec la méthode de l'article Intriguing properties")

liste1 = [i for i in liste1 if not(np.isnan(i))]
print("Il y a : ", b1 - len(liste1), " entrées qui admettent nan avec la méthode de l'iterative gradient")
liste2 = [i for i in liste2 if not(np.isnan(i))]
print("Il y a : ", b2 - len(liste2), " entrées qui admettent nan avec la méthode de l'article Intriguing properties")

distortion_moyenne1 = sum([np.sqrt(i) for i in liste1])/len(liste1)
distortion_moyenne2 = sum([np.sqrt(i) for i in liste2])/len(liste2)

print("La distortion moyenne dans le cas de l'iterative gradient est : ", distortion_moyenne1)
print("La distortion moyenne dans le cas de Intriguing properties est : ", distortion_moyenne2)

##

import matplotlib.pyplot as plt

plt.close()
#n = np.random.randint(0,10000,1)[0] # on va perturber la 1000 ième image de la base de test qui correspond au chiffre 9
n = 1000
X = x_test
Y = to_categorical(y_test)
# Sélection de l'image à perturber
label_oiginal  = Y[n]  # un "9"
image_original = X[n]

'''
Calcul du gradient du log de la vraisemblance du chiffre cible (proposé par l'adversaire)
par rapport aux entrées (image) >  vecteur de 784 composantes
'''
  

imReelle = image_original.reshape(1,784)
# plt.figure(figsize=(2, 2))
# plt.imshow(imReelle.reshape([28, 28]),cmap = "gray")
# plt.show()

# On modifie l'image du "9" de telle sorte que le réseau la confonde avec celle d'un 0 (crapy crapy shity code n'est ce pas ?)
# imAdversaire = tf.constant(imReelle) # on doit faire un "cast" en tenseur de la structure np.array de imReelle
#                                      # pour utiliser les fonctionnalités de TensorFlow (modele + gradient)
# n=0
# alpha = 0.01

# Le + simle des algo : "iterative gradient" (on se limite à 100 itérations)

imAdversaire2 = IntriguingAdversersarialSansButPrecis(n, model)[0][0]
imAdversaire =  creation_Adversaire2(n, model)[0][0]


imAdversaire = np.array([imAdversaire]) # On transforme le tf de TensorFlow en np.array de NumPy
Delta = imReelle - imAdversaire
imAdversaire2 = np.array(imAdversaire2).reshape(1,784) # On transforme le tf de TensorFlow en np.array de NumPy
Delta2 = imReelle - imAdversaire2


plt.figure(figsize=(9, 3))

plt.subplot(1,3,1)
plt.imshow(imReelle.reshape([28, 28]),cmap = "gray")
plt.title("Chiffre = " + np.str(np.argmax(model(imReelle))))
plt.xlabel("confidence = " + np.str(np.max(model(imReelle)))[:5])
# Perturbation adverse
plt.subplot(1,3,2)
plt.imshow(Delta2.reshape([28, 28]),cmap = "gray")
plt.title("Perturbation adversaire")
# Image modifiée par l'adversaire
plt.subplot(1,3,3)
plt.imshow(imAdversaire2.reshape([28, 28]),cmap = "gray")
plt.title("Chiffre = " + np.str(np.argmax(model(imAdversaire2))))
plt.xlabel("confidence = " + np.str(np.max(model(imAdversaire2)))[:5])

plt.show()

# On edite le carré de la norme de Frobenius de la perturbation (moyenné par le nombre de pixels)
print("EQM = ",np.mean(Delta * Delta)) # pour ce cas EQM = 0.0015074897
print("EQM2= ", np.mean(Delta2 * Delta2))


################# RÉSULTATS 20 EPOCHS RÉGULARISATION :


# loss: 0.0075 - accuracy: 0.9985
# (0, 0, 0) 20
# 
# >>> (executing lines 335 to 371 of "Creation d adversaire.py")
# Jeu de test
# Loss: 0.0883, accuracy: 97.5900%, temps:  0.4062
# Il y a :  25  prédictions fausses au départ, donc inutile de leur chercher un adversaire
# 25  normalement on devrait tourver la même chose, donc on vérifie
# Il y a :  1  entrées qui n'ont pas d'adversaires avec la méthode de l'iterative gradient
# Il y a :  47  entrées qui n'ont pas d'adversaires avec la méthode de l'article Intriguing properties
# Il y a :  46  entrées qui admettent nan avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui admettent nan avec la méthode de l'article Intriguing properties
# La distortion moyenne dans le cas de l'iterative gradient est :  0.024838614573096055
# La distortion moyenne dans le cas de Intriguing properties est :  0.021458593299112184


# loss: 0.0066 - accuracy: 0.9988
# (1e-10, 1e-10, 1e-11) 20
# 
# >>> (executing lines 335 to 371 of "Creation d adversaire.py")
# Jeu de test
# Loss: 0.0881, accuracy: 97.8200%, temps:  0.0411
# Il y a :  19  prédictions fausses au départ, donc inutile de leur chercher un adversaire
# 19  normalement on devrait tourver la même chose, donc on vérifie
# Il y a :  0  entrées qui n'ont pas d'adversaires avec la méthode de l'iterative gradient
# Il y a :  69  entrées qui n'ont pas d'adversaires avec la méthode de l'article Intriguing properties
# Il y a :  69  entrées qui admettent nan avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui admettent nan avec la méthode de l'article Intriguing properties
# La distortion moyenne dans le cas de l'iterative gradient est :  0.025070281658707626
# La distortion moyenne dans le cas de Intriguing properties est :  0.021481298053300596
# 

# loss: 0.0065 - accuracy: 0.9989
# (1e-07, 1e-07, 1e-08) 20
# 
# >>> (executing lines 335 to 371 of "Creation d adversaire.py")
# Jeu de test
# Loss: 0.0866, accuracy: 97.6600%, temps:  0.0392
# Il y a :  22  prédictions fausses au départ, donc inutile de leur chercher un adversaire
# 22  normalement on devrait tourver la même chose, donc on vérifie
# Il y a :  1  entrées qui n'ont pas d'adversaires avec la méthode de l'iterative gradient
# Il y a :  55  entrées qui n'ont pas d'adversaires avec la méthode de l'article Intriguing properties
# Il y a :  54  entrées qui admettent nan avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui admettent nan avec la méthode de l'article Intriguing properties
# La distortion moyenne dans le cas de l'iterative gradient est :  0.026168675480369087
# La distortion moyenne dans le cas de Intriguing properties est :  0.02288504460431849

# 
# loss: 0.0080 - accuracy: 0.9984
# (1e-05, 1e-05, 1e-06) 20
# 
# >>> (executing lines 335 to 371 of "Creation d adversaire.py")
# Jeu de test
# Loss: 0.0883, accuracy: 97.8700%, temps:  0.0406
# Il y a :  24  prédictions fausses au départ, donc inutile de leur chercher un adversaire
# 24  normalement on devrait tourver la même chose, donc on vérifie
# Il y a :  0  entrées qui n'ont pas d'adversaires avec la méthode de l'iterative gradient
# Il y a :  43  entrées qui n'ont pas d'adversaires avec la méthode de l'article Intriguing properties
# Il y a :  43  entrées qui admettent nan avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui admettent nan avec la méthode de l'article Intriguing properties
# La distortion moyenne dans le cas de l'iterative gradient est :  0.029019919092071193
# La distortion moyenne dans le cas de Intriguing properties est :  0.02595129261911148


# loss: 0.0444 - accuracy: 0.9963
# (0.001, 0.001, 0.0001) 20
# 
# >>> (executing lines 335 to 371 of "Creation d adversaire.py")
# Jeu de test
# Loss: 0.0705, accuracy: 97.8200%, temps:  0.0497
# 9375 raté
# Il y a :  17  prédictions fausses au départ, donc inutile de leur chercher un adversaire
# 17  normalement on devrait tourver la même chose, donc on vérifie
# Il y a :  0  entrées qui n'ont pas d'adversaires avec la méthode de l'iterative gradient
# Il y a :  3  entrées qui n'ont pas d'adversaires avec la méthode de l'article Intriguing properties
# Il y a :  3  entrées qui admettent nan avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui admettent nan avec la méthode de l'article Intriguing properties
# La distortion moyenne dans le cas de l'iterative gradient est :  0.037655337063633665
# La distortion moyenne dans le cas de Intriguing properties est :  0.03685985198224255


# loss: 1.9483 - accuracy: 0.6547
# (1, 1, 0.1) 20
# 
# >>> (executing lines 335 to 371 of "Creation d adversaire.py")
# Jeu de test
# Loss: 1.1801, accuracy: 66.2400%, temps:  0.0514
# Il y a :  322  prédictions fausses au départ, donc inutile de leur chercher un adversaire
# 322  normalement on devrait tourver la même chose, donc on vérifie
# Il y a :  150  entrées qui n'ont pas d'adversaires avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui n'ont pas d'adversaires avec la méthode de l'article Intriguing properties
# Il y a :  0  entrées qui admettent nan avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui admettent nan avec la méthode de l'article Intriguing properties
# La distortion moyenne dans le cas de l'iterative gradient est :  0.04891241357249369
# La distortion moyenne dans le cas de Intriguing properties est :  0.06621344592165468


# loss: 2.3023 - accuracy: 0.1124
# (100, 100, 10) 20
# 
# >>> (executing lines 335 to 374 of "Creation d adversaire.py")
# Jeu de test
# Loss: 2.3010, accuracy: 11.3500%, temps:  0.0626
# Il y a :  883  prédictions fausses au départ, donc inutile de leur chercher un adversaire
# 883  normalement on devrait tourver la même chose, donc on vérifie
# Il y a :  117  entrées qui n'ont pas d'adversaires avec la méthode de l'iterative gradient
# Il y a :  117  entrées qui n'ont pas d'adversaires avec la méthode de l'article Intriguing properties
# Il y a :  0  entrées qui admettent nan avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui admettent nan avec la méthode de l'article Intriguing properties


# loss: 2.3681 - accuracy: 0.1124
# (100000, 100000, 10000) 20
# 
# >>> (executing lines 335 to 371 of "Creation d adversaire.py")
# Jeu de test
# Loss: 2.3010, accuracy: 11.3500%, temps:  0.0426
# Il y a :  883  prédictions fausses au départ, donc inutile de leur chercher un adversaire
# 883  normalement on devrait tourver la même chose, donc on vérifie
# Il y a :  117  entrées qui n'ont pas d'adversaires avec la méthode de l'iterative gradient
# Il y a :  117  entrées qui n'ont pas d'adversaires avec la méthode de l'article Intriguing properties
# Il y a :  0  entrées qui admettent nan avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui admettent nan avec la méthode de l'article Intriguing properties
# Traceback (most recent call last):
#   File "/home/matthieu/Creation d adversaire.py", line 367, in <module>
#     distortion_moyenne1 = sum([np.sqrt(i) for i in liste1])/len(liste1)
# ZeroDivisionError: division by zero


################# RÉSULTATS 20 EPOCHS DROPOUT :

# loss: 0.0068 - accuracy: 0.9987
# (0, 0, 0) 20 dropout : 0
# 
# >>> (executing lines 335 to 374 of "Creation d adversaire.py")
# Jeu de test
# Loss: 0.0814, accuracy: 97.8500%, temps:  0.3821
# Il y a :  20  prédictions fausses au départ, donc inutile de leur chercher un adversaire
# 20  normalement on devrait tourver la même chose, donc on vérifie
# Il y a :  0  entrées qui n'ont pas d'adversaires avec la méthode de l'iterative gradient
# Il y a :  67  entrées qui n'ont pas d'adversaires avec la méthode de l'article Intriguing properties
# Il y a :  67  entrées qui admettent nan avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui admettent nan avec la méthode de l'article Intriguing properties
# La distortion moyenne dans le cas de l'iterative gradient est :  0.0259436574203002
# La distortion moyenne dans le cas de Intriguing properties est :  0.022615262119048334


# loss: 0.0403 - accuracy: 0.9871
# (0, 0, 0) 20 dropout : 0.1
# 
# >>> (executing lines 335 to 374 of "Creation d adversaire.py")
# Jeu de test
# Loss: 0.0803, accuracy: 97.5100%, temps:  0.0427
# Il y a :  38  prédictions fausses au départ, donc inutile de leur chercher un adversaire
# 38  normalement on devrait tourver la même chose, donc on vérifie
# Il y a :  0  entrées qui n'ont pas d'adversaires avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui n'ont pas d'adversaires avec la méthode de l'article Intriguing properties
# Il y a :  0  entrées qui admettent nan avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui admettent nan avec la méthode de l'article Intriguing properties
# La distortion moyenne dans le cas de l'iterative gradient est :  0.030846176826198567
# La distortion moyenne dans le cas de Intriguing properties est :  0.028143506931058555


# loss: 0.0617 - accuracy: 0.9801
# (0, 0, 0) 20 dropout : 0.2
# 
# >>> (executing lines 335 to 374 of "Creation d adversaire.py")
# Jeu de test
# Loss: 0.0753, accuracy: 97.7400%, temps:  0.0485
# Il y a :  20  prédictions fausses au départ, donc inutile de leur chercher un adversaire
# 20  normalement on devrait tourver la même chose, donc on vérifie
# Il y a :  0  entrées qui n'ont pas d'adversaires avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui n'ont pas d'adversaires avec la méthode de l'article Intriguing properties
# Il y a :  0  entrées qui admettent nan avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui admettent nan avec la méthode de l'article Intriguing properties
# La distortion moyenne dans le cas de l'iterative gradient est :  0.03293430821932092
# La distortion moyenne dans le cas de Intriguing properties est :  0.030188230727808605
# 
# 
# 
# loss: 0.0924 - accuracy: 0.9715
# (0, 0, 0) 20 dropout : 0.3
# 
# >>> (executing lines 335 to 374 of "Creation d adversaire.py")
# Jeu de test
# Loss: 0.0782, accuracy: 97.6300%, temps:  0.0497
# Il y a :  16  prédictions fausses au départ, donc inutile de leur chercher un adversaire
# 16  normalement on devrait tourver la même chose, donc on vérifie
# Il y a :  0  entrées qui n'ont pas d'adversaires avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui n'ont pas d'adversaires avec la méthode de l'article Intriguing properties
# Il y a :  0  entrées qui admettent nan avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui admettent nan avec la méthode de l'article Intriguing properties
# La distortion moyenne dans le cas de l'iterative gradient est :  0.03443599793358661
# La distortion moyenne dans le cas de Intriguing properties est :  0.031512614544388876
# 
# 
# 
# loss: 0.1274 - accuracy: 0.9618
# (0, 0, 0) 20 dropout : 0.4
# 
# >>> (executing lines 335 to 374 of "Creation d adversaire.py")
# Jeu de test
# Loss: 0.0880, accuracy: 97.2900%, temps:  0.0552
# Il y a :  26  prédictions fausses au départ, donc inutile de leur chercher un adversaire
# 26  normalement on devrait tourver la même chose, donc on vérifie
# Il y a :  0  entrées qui n'ont pas d'adversaires avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui n'ont pas d'adversaires avec la méthode de l'article Intriguing properties
# Il y a :  0  entrées qui admettent nan avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui admettent nan avec la méthode de l'article Intriguing properties
# La distortion moyenne dans le cas de l'iterative gradient est :  0.03285560151048146
# La distortion moyenne dans le cas de Intriguing properties est :  0.02994906419907654
# 
# 
# 
# loss: 0.1801 - accuracy: 0.9460
# (0, 0, 0) 20 dropout : 0.5
# 
# >>> (executing lines 335 to 374 of "Creation d adversaire.py")
# Jeu de test
# Loss: 0.1048, accuracy: 96.8400%, temps:  0.0722
# Il y a :  26  prédictions fausses au départ, donc inutile de leur chercher un adversaire
# 26  normalement on devrait tourver la même chose, donc on vérifie
# Il y a :  0  entrées qui n'ont pas d'adversaires avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui n'ont pas d'adversaires avec la méthode de l'article Intriguing properties
# Il y a :  0  entrées qui admettent nan avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui admettent nan avec la méthode de l'article Intriguing properties
# La distortion moyenne dans le cas de l'iterative gradient est :  0.03440587814919769
# La distortion moyenne dans le cas de Intriguing properties est :  0.03151895336270796
# 
# 
# 
# loss: 0.2606 - accuracy: 0.9243
# (0, 0, 0) 20 dropout : 0.6
# 
# >>> (executing lines 335 to 374 of "Creation d adversaire.py")
# Jeu de test
# Loss: 0.1377, accuracy: 95.7300%, temps:  0.0849
# Il y a :  49  prédictions fausses au départ, donc inutile de leur chercher un adversaire
# 49  normalement on devrait tourver la même chose, donc on vérifie
# Il y a :  0  entrées qui n'ont pas d'adversaires avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui n'ont pas d'adversaires avec la méthode de l'article Intriguing properties
# Il y a :  0  entrées qui admettent nan avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui admettent nan avec la méthode de l'article Intriguing properties
# La distortion moyenne dans le cas de l'iterative gradient est :  0.032215500074550395
# La distortion moyenne dans le cas de Intriguing properties est :  0.030174339051692187
# 
# 
# 
# loss: 0.3875 - accuracy: 0.8903
# (0, 0, 0) 20 dropout : 0.7
# 
# >>> (executing lines 335 to 374 of "Creation d adversaire.py")
# Jeu de test
# Loss: 0.1771, accuracy: 94.7000%, temps:  0.0840
# Il y a :  47  prédictions fausses au départ, donc inutile de leur chercher un adversaire
# 47  normalement on devrait tourver la même chose, donc on vérifie
# Il y a :  0  entrées qui n'ont pas d'adversaires avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui n'ont pas d'adversaires avec la méthode de l'article Intriguing properties
# Il y a :  0  entrées qui admettent nan avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui admettent nan avec la méthode de l'article Intriguing properties
# La distortion moyenne dans le cas de l'iterative gradient est :  0.03185072798138496
# La distortion moyenne dans le cas de Intriguing properties est :  0.030230008567159777



# loss: 0.6445 - accuracy: 0.8097
# (0, 0, 0) 20 dropout : 0.8
# 
# >>> (executing lines 335 to 374 of "Creation d adversaire.py")
# Jeu de test
# Loss: 0.2685, accuracy: 92.4700%, temps:  0.0432
# Il y a :  85  prédictions fausses au départ, donc inutile de leur chercher un adversaire
# 85  normalement on devrait tourver la même chose, donc on vérifie
# Il y a :  7  entrées qui n'ont pas d'adversaires avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui n'ont pas d'adversaires avec la méthode de l'article Intriguing properties
# Il y a :  0  entrées qui admettent nan avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui admettent nan avec la méthode de l'article Intriguing properties
# La distortion moyenne dans le cas de l'iterative gradient est :  0.030822289055318464
# La distortion moyenne dans le cas de Intriguing properties est :  0.03200462577532104



# loss: 1.2830 - accuracy: 0.5212
# (0, 0, 0) 20 dropout : 0.9
# 
# >>> (executing lines 335 to 374 of "Creation d adversaire.py")
# Jeu de test
# Loss: 0.6823, accuracy: 87.0400%, temps:  0.0526
# Il y a :  132  prédictions fausses au départ, donc inutile de leur chercher un adversaire
# 132  normalement on devrait tourver la même chose, donc on vérifie
# Il y a :  108  entrées qui n'ont pas d'adversaires avec la méthode de l'iterative gradient
# Il y a :  2  entrées qui n'ont pas d'adversaires avec la méthode de l'article Intriguing properties
# Il y a :  0  entrées qui admettent nan avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui admettent nan avec la méthode de l'article Intriguing properties
# La distortion moyenne dans le cas de l'iterative gradient est :  0.027597826553602843
# La distortion moyenne dans le cas de Intriguing properties est :  0.04050762489556116