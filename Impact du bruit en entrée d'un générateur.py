# ##############################################################################
#       Programme de génération d'adveraire avec un réseau de neurones         #
#             et étude de l'impact du bruit en entrée                          #
# ##############################################################################

# Le but de cette partie est de vérifier une intuition :
# si on met une image et du bruit en entrée d'un réseau de neurones
# et qu'on lui demande de générer des adveraires, va-t-il vraiment ignorer
# le bruit ?
# Pour le tester, il suffit d'entraîner le générateur pour un classifieur donné
# et de le tester à image fixée, mais avec différents bruits.

# Le code s'articule en 5 parties :
#   I - la préparation des données
#   II - la construction du classifieur
#   III - la construction du générateur
#   IV - l' apprentissage
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
epochs = 10
steps_per_epoch = 500

(x_train, y_train), (x_test, y_test)=tf.keras.datasets.mnist.load_data()
x_train=(x_train.reshape(-1, 784)/255).astype(np.float32)
x_test= (x_test.reshape(-1,  784)/255).astype(np.float32)
train_ds=tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)


# II - Construction du classifieur
# ################################

def FC100_100_10(couches = (100,100)):
    Nh1, Nh2 = couches
    model = models.Sequential([
        layers.Dense(units=Nh1, activation='sigmoid',input_shape=(784,),
        layers.Dense(units=Nh2, activation='sigmoid',
        layers.Dense(units=10, activation='softmax')
    ])
    
    print("Structure du modele")
    model.summary()
    
    return(model)
    
# On fait appel aux fonctions "keras" très pratique
optimizer=tf.keras.optimizers.Adam()
loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
train_loss=tf.keras.metrics.Mean()
train_accuracy=tf.keras.metrics.SparseCategoricalAccuracy()

# ici "@tf.function" perm d'exécute + rapidement les instructions
@tf.function
def train_step(images, labels, model):
  with tf.GradientTape() as tape:
    predictions=model(images)
    loss=loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_loss(loss)
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


# III - Construction du générateur
# ################################

def FC500_500_10_ReLU(couches = (500,500)):
    Nh1, Nh2 = couches
    model = models.Sequential([
        layers.Dense(units=Nh1, activation='relu',input_shape=(884,)),
        layers.Dense(units=Nh2, activation='relu'),
        layers.Dense(units=784, activation='sigmoid')
    ])
    
    print("Structure du modele")
    model.summary()
    return(model)
    
def loss_object_gene(images_originales, images_generees, labels):
  loss = 100*tf.reduce_mean(tf.square(images_originales - images_generees))
  label_genere = discriminator(images_generees)
  a = loss_object(labels, 1-label_genere)
  loss += a        
  return(loss)
  

train_loss_gene=tf.keras.metrics.Mean()


@tf.function
def train_step_gene(images_et_bruit, images_originales, labels, model):
  with tf.GradientTape() as tape:
    images_generees=model(images_et_bruit)
    loss=loss_object_gene(images_originales, images_generees, labels)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_loss_gene(loss)

def train_gene(train_ds, nbr_entrainement, model):
  for entrainement in range(nbr_entrainement):
    start=time.time()
    for images_originales, labels in train_ds:
      a,d = images_originales.shape
      noise = tf.random.normal(shape = [a,100], mean = 0.0, stddev = 0.1, dtype = tf.float32)
      images_et_bruit = tf.concat([images_originales, noise], 1)
      train_step_gene(images_et_bruit, images_originales, labels, model)
    message='Entrainement {:04d}, loss: {:6.4f}, temps: {:7.4f}'
    
    print(message.format(entrainement+1,
                        train_loss_gene.result(),
                        time.time()-start))
    train_loss_gene.reset_states()

    
# IV - Apprentissage
# ##################

nbr_entrainement= 20
discriminator = FC100_100_10(couches = (100,100))
print("Entrainement")
train(train_ds, nbr_entrainement, discriminator)
discriminator.trainable = False

    
nbr_entrainement= 50
generator = FC500_500_10_ReLU()
print("Entrainement")
train_gene(train_ds, nbr_entrainement, generator)


# V - Visualisation des résultats
#################################

import matplotlib.pyplot as plt

plt.close()
#n = np.random.randint(0,10000,1)[0] # on va perturber la 1000 ième image de la base de test qui correspond au chiffre 9
n = 1000
X = x_test
Y = to_categorical(y_test)
# Sélection de l'image à perturber
label_oiginal  = Y[n]
image_original = X[n]

'''
Calcul du gradient du log de la vraisemblance du chiffre cible (proposé par l'adversaire)
par rapport aux entrées (image) >  vecteur de 784 composantes
'''
  

imReelle = image_original.reshape(1,784)
print(imReelle.shape)

noise = np.random.normal( loc = 0.0, scale = 0.1, size = (1,100))
imReelle_et_bruit = np.concatenate((imReelle, noise), axis = 1)

print(imReelle_et_bruit.shape)

# Le + simle des algo : "iterative gradient" (on se limite à 100 itérations)
imAdversaire = generator(imReelle_et_bruit)

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
