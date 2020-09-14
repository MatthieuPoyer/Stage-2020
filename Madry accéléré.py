# ##############################################################################
#           Méthode d'apprentissage robuste : Madry accéléré                   #
# ##############################################################################

# Le code s'articule en 5 parties :
#   I - la préparation des données
#   II - la construction du classifieur ou discriminant (terminologie GAN)
#   III - la construction du générateur
#   IV - apprentissage par mise en compétition du classifieur et du générateur
#   V - les résultats

# Nous nous sommes intéressés à l'apprentissage non ciblé,
# ie: on veut trouver un adversaire, peu importe sa cible.



import tensorflow as tf
from tensorflow import keras
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
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)


# II - Construction du classifieur
# ################################

# Construction  d'un réseau à 2 couches cachées de 100 neurones chacune, dont le but est de classifier la base de données MNIST

def FC100_100_10(couches = (100,100)):
    Nh1, Nh2 = couches
    model = models.Sequential([
        layers.Dense(units=Nh1, activation='sigmoid',input_shape=(784,)),
        layers.Dense(units=Nh2, activation='sigmoid'),
        layers.Dense(units=10, activation='softmax')
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
  train_loss(loss)
  train_accuracy(labels, predictions)

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
  return(test_accuracy.result()*100)

# III - Construction du générateur
# ################################

def FC500_500_10_ReLU(couches = (500,500)):
    Nh1, Nh2 = couches
    model = models.Sequential([
        layers.Dense(units=Nh1, activation='relu',input_shape=(784,)),
        layers.Dense(units=Nh2, activation='relu'),
        layers.Dense(units=784, activation='sigmoid')
    ])
    
    print("Structure du modele")
    model.summary()
    return(model)
    
train_loss_gene=tf.keras.metrics.Mean()

def loss_object_gene(images_originales, images_generees, labels):
  loss = c*tf.reduce_mean(tf.square(images_originales - images_generees))
  label_genere = discriminator(images_generees)
  a = loss_object(labels, 1-label_genere)
  loss += a        
  return(loss)
  
# Test de la norme infini
  
# def loss_object_gene(images_originales, images_generees, labels):
#   loss = 10**(0)*tf.math.reduce_max(tf.math.abs(images_originales - images_generees))
#   label_genere = discriminator(images_generees)
#   a = loss_object(labels, 1-label_genere)
#   #a = tf.equal(tf.argmax(discriminator(images_originales), axis = 1),tf.argmax(discriminator(images_generees), axis = 1))         #dans l'idéal il faudrait remplacer predict_images originales par le vrai label
#   #a = tf.cast(a, tf.float32)
#   loss += a        
#   return(loss)
  

@tf.function
def train_step_gene(images_originales, labels, model):
  with tf.GradientTape() as tape:
    images_generees=model(images_originales)
    loss=loss_object_gene(images_originales, images_generees, labels)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_loss_gene(loss)

def train_gene(test_ds, nbr_entrainement, model):
  for entrainement in range(nbr_entrainement):
    start=time.time()
    for images_originales, labels in test_ds:
      train_step_gene(images_originales, labels, model)
    message='Entrainement {:04d}, loss: {:6.4f}, temps: {:7.4f}'
    
    print(message.format(entrainement+1,
                        train_loss_gene.result(),
                        time.time()-start))
    train_loss_gene.reset_states()

    
# IV - Apprentissage par mise en compétition du classifieur et du générateur
# ##########################################################################

# 1 - initialisation des paramètes

c = 100
x_t = x_train
y_t = y_train
bdd_test = x_test
nbr_entrainement_disc= 20
nbr_entrainement_gene= 50

discriminator = FC100_100_10(couches = (100,100))
generator = FC500_500_10_ReLU()

nbre_etapes_gene_disc = 2

liste_dist_test = []
liste_dist = []
liste_precision = []


# 2 - la boucle / la mise en compétition

for i in range(nbre_etapes_gene_disc):
  
  t_ds=tf.data.Dataset.from_tensor_slices((x_t, y_t)).batch(batch_size)
  
  print("Entrainement Disc", i)
  train(t_ds, nbr_entrainement_disc, discriminator)
  
  discriminator.trainable = False
  
  # nous permet de vérifier la précision de note classifieur
  print("Jeu de test")
  liste_precision.append(test(x_test,y_test, discriminator))
  
  print("Entrainement Géné", i)
  train_gene(train_ds, nbr_entrainement_gene, generator)
  
  # nous permet de connaître la proportion d'images générées mal classifiées
  print("TRAIN, ERROR RATE")
  sol = generator(x_train)
  part123 = sum(np.argmax(discriminator(sol), axis = 1) == y_train)
  print(part123)
  print(part123/600)
  
  # on mesure la distorsion moyenne des adversaires générés
  distorsionPart = np.mean(np.sqrt(np.mean(np.square(x_train - sol), axis = 1)))
  liste_dist.append(distorsionPart)
  
  # on augmente les BDDs
  x_t = np.concatenate((x_t, sol))
  y_t = np.concatenate((y_t, y_train))
  soltest = generator(x_test)
  bdd_test = np.concatenate((bdd_test, soltest))
  
  distorsionPart = np.mean(np.sqrt(np.mean(np.square(x_test - soltest), axis = 1)))
  liste_dist_test.append(distorsionPart)
  
#################
# V - Résultats # 
#################

# 1 - À la recherche d'un bon c
# #############################

# c = 0 => erreur_train 0%, erreur_test 0.04% (4/10000)             # a des adversaires affreux !!!!!
# c = 0 => erreur_train 0%, erreur_test 0% (0/10000)

# c = 10 => erreur_train 0.57%, erreur_test 0.81% (81/10000)        # a des adversaires acceptables
# c = 10 => erreur-train 0.15%, erreur_test 0.54% (54/10000)

# c = 100 => erreur_train 1.48%, erreur_test 3.13% (313/10000)
# c = 100 => erreur_train 0.46%, erreur_test 1.73% (173/10000)
# c = 100 => erreur_train 3.83%, erreur_test 5.89% (589/10000)
# c = 100 => erreur_train 3.22%, erreur_test 5.18% (518/10000)
# c = 100 => erreur_train 1.33%, erreur_test 2.83% (283/10000)
# c = 100 => erreur_train 1.78%, erreur_test 3.52% (352/10000)
# c = 100 => erreur_train 0.97%, erreur_test 2.26% (226/10000)

# c = 200 => erreur_train 1.98%, erreur_test 4.57% (457/10000)
# c = 200 => erreur_train 4.72%, erreur_test 7.48% (748/10000)
# c = 200 => erreur_train 7.86%, erreur_test 11.19% (11.19/10000)

# c = 500 => erreur_train 6.34%, erreur_test 9.7% (970/10000) 
# c = 1000 => erreur_train 14.18%, erreur_test 17.89% (/10000)

# On a donc choisi c = 100


# 2 - Code pour afficher des adversaires à la fin
#################################################

import matplotlib.pyplot as plt

plt.close()
# Sélection de l'image à perturber
#n = np.random.randint(0,10000,1)[0] # on va perturber la 1000 ième image de la base de test qui correspond au chiffre 9
n = 1000
X = x_test
Y = to_categorical(y_test)
label_oiginal  = Y[n]  # un "9" si on utilise n=1000
image_original = X[n]

'''
Calcul du gradient du log de la vraisemblance du chiffre cible (proposé par l'adversaire)
par rapport aux entrées (image) >  vecteur de 784 composantes
'''
  

imReelle = image_original.reshape(1,784)

# Le + simle des algo : "iterative gradient" (on se limite à 100 itérations)
imAdversaire = generator(imReelle)

imAdversaire = imAdversaire.numpy() # On transforme le tf de TensorFlow en np.array de NumPy
Delta = imReelle - imAdversaire

# On edite le carré de la norme de Frobenius de la perturbation (moyenné par le nombre de pixels)
print("EQM = ",np.mean(Delta * Delta)) # pour ce cas EQM = 0.0015074897
print("2nd terme", loss_object(y_test[n], 1-discriminator(imAdversaire)))

plt.figure(figsize=(9, 3))
# Image réelle
plt.subplot(1,3,1)
plt.imshow(imReelle.reshape([28, 28]),cmap = "gray")
plt.title("Chiffre = " + np.str(np.argmax(discriminator(imReelle))))
print("Réelle : ", np.max(discriminator(imReelle)))
# Perturbation adverse
plt.subplot(1,3,2)
plt.imshow(Delta.reshape([28, 28]),cmap = "gray")
plt.title("Perturbation adversaire")
# Image modifiée par l'adversaire
plt.subplot(1,3,3)
plt.imshow(imAdversaire.reshape([28, 28]),cmap = "gray")
plt.title("Chiffre = " + np.str(np.argmax(discriminator(imAdversaire))))
print("Adversaire : ", np.max(discriminator(imAdversaire)))
plt.show()


# 2 - Code pour afficher l'évolution des adversaires 
####################################################

# on en profite alors pour regarder si les anciens adversaires sont désormais reconnus

import matplotlib.pyplot as plt


plt.close()
# Sélection de l'image à perturber
#n = np.random.randint(0,60000,1)[0] # on va perturber la 1000 ième image de la base de test qui correspond au chiffre 9
n = 1000
X = bdd_test
Y = to_categorical(y_test)
label_oiginal  = Y[n]  # un "9" si on utilise n=1000
image_original = X[n]
adversaire = [np.array([X[(i+1)*10000+n]]) for i in range(nbre_etapes_gene_disc)]
#adversaire1 = X[n+60000]

'''
Calcul du gradient du log de la vraisemblance du chiffre cible (proposé par l'adversaire)
par rapport aux entrées (image) >  vecteur de 784 composantes
'''
  

imReelle = image_original.reshape(1,784)
imAdversaire = [i.reshape(1,784) for i in adversaire]
Delta = [imReelle - i for i in imAdversaire]


plt.figure(figsize=(9, 9))
# Image réelle
plt.subplot(nbre_etapes_gene_disc,3,1)
plt.imshow(imReelle.reshape([28, 28]),cmap = "gray")
plt.title("Chiffre = " + np.str(np.argmax(discriminator(imReelle))))
print("Réelle : ", np.max(discriminator(imReelle)))

#nbre_etapes_gene_disc = 1

for i in range(nbre_etapes_gene_disc):
  # Perturbation adverse
  plt.subplot(nbre_etapes_gene_disc,3,2+i*3)
  plt.imshow(Delta[i].reshape([28, 28]),cmap = "gray")
  plt.title("Perturbation adversaire")
  # Image modifiée par l'adversaire
  plt.subplot(nbre_etapes_gene_disc,3,3+i*3)
  plt.imshow(imAdversaire[i].reshape([28, 28]),cmap = "gray")
  plt.title("Chiffre = " + np.str(np.argmax(discriminator(imAdversaire[i]))))
  print("Adversaire : ", np.max(discriminator(imAdversaire[i])))
plt.show()


# 3 - Code pour visualiser l'évolution de la distorsion en fonction du nombre d'étapes
######################################################################################

import numpy as np
import matplotlib.pyplot as plt

# nbre_etapes_gene_disc = 1

# La distorsion moyenne après 20 étapes
# a = np.array([0.073579594, 0.077717565, 0.08322092, 0.07832371, 0.087938294, 0.087290645, 0.098814584, 0.08776691, 0.100518204, 0.097730175, 0.09347145, 0.10769891, 0.11493808, 0.09993061, 0.11145804, 0.10446661, 0.10817391, 0.106853835, 0.1163971, 0.11360584])

a = np.array(liste_dist_test)
b = np.array(liste_dist)
c = np.array(liste_precision)

fig, ax = plt.subplots()

ax.plot(np.linspace(0,nbre_etapes_gene_disc-1, nbre_etapes_gene_disc),a, color = "blue", label = "distorsion test")
ax.plot(np.linspace(0,nbre_etapes_gene_disc-1, nbre_etapes_gene_disc), b, color = "green", label = "distorsion train")
ax.set_xlabel("nombre d'étapes")
ax.set_ylabel("distorsion moyenne")

ax2 = ax.twinx()
ax2.plot(np.linspace(0,nbre_etapes_gene_disc-1, nbre_etapes_gene_disc),c, color = "red", label = "précision test")
ax2.set_ylabel("précision")

plt.title("Évolution de la distorsion des adversaires générés")
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
plt.show()
