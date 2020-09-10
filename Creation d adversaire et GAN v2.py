import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
# from keras import layers, models
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

# Restructuration des donnée d'entrée pour les réseaux FC (Full connected)
x_train=(x_train.reshape(-1, 784)/255).astype(np.float32)
x_test= (x_test.reshape(-1,  784)/255).astype(np.float32)

# Nombre de classes
num_classes = len(np.unique(y_train))

# Services fournis par l'API tf.data (information sur le Web)
train_ds=tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)


# II - Construction du dicriminant
# ################################

def FC100_100_10(lambda0 = (10**(-5), 10**(-5), 10**(-6)), couches = (100,100)):
    Nh1, Nh2 = couches
    lambda1, lambda2, lambda3 = lambda0
    model = models.Sequential([
        layers.Dense(units=Nh1, activation='sigmoid',input_shape=(784,), kernel_regularizer=tf.keras.regularizers.l2(lambda1/100)),
        layers.Dense(units=Nh2, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(lambda2/100)),
        layers.Dense(units=10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(lambda3/10))
    ])
    
    print("Structure du modele")
    model.summary()
    
    return(model)


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

    
# IV - Apprentissage
# ##################


from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

#discriminator = FC100_100_10()

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
    
nbr_entrainement= 20
discriminator = FC100_100_10(couches = (100,100))
print("Entrainement")
train(train_ds, nbr_entrainement, discriminator)


#model.compile(loss='sparse_categorical_crossentropy', optimizer="Adam")

discriminator.trainable = False                         #!!! JE NE SAIS PAS CE QUE C'EST ???? 


def loss_object_gene(images_originales, images_generees, labels):
  loss = 100*tf.reduce_mean(tf.square(images_originales - images_generees))
  label_genere = discriminator(images_generees)
  a = loss_object(labels, 1-label_genere)
  #a = tf.equal(tf.argmax(discriminator(images_originales), axis = 1),tf.argmax(discriminator(images_generees), axis = 1))         #dans l'idéal il faudrait remplacer predict_images originales par le vrai label
  #a = tf.cast(a, tf.float32)
  loss += a        
  return(loss)
  
  

#loss_object_gene=tf.keras.losses.SparseCategoricalCrossentropy()
train_loss_gene=tf.keras.metrics.Mean()
#train_accuracy_gene=tf.keras.metrics.SparseCategoricalAccuracy()

@tf.function
def train_step_gene(images_originales, labels, model):
  with tf.GradientTape() as tape:
    images_generees=model(images_originales)
    loss=loss_object_gene(images_originales, images_generees, labels)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_loss_gene(loss) # + bas dans "def train" on comprend l'intérêt de cette instruction (édition des résultats

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
    
    
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    
# nbr_entrainement= 50
# generator = FC500_500_10_ReLU()
# print("Entrainement")
# train_gene(train_ds, nbr_entrainement, generator)

##
    
# optimizer=tf.keras.optimizers.Adam()
# loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
# train_loss=tf.keras.metrics.Mean()
# train_accuracy=tf.keras.metrics.SparseCategoricalAccuracy()
# test_loss=tf.keras.metrics.Mean()
# test_accuracy=tf.keras.metrics.SparseCategoricalAccuracy()
#     
# def test(xt,yt,model):
#   start=time.time()
#   predictions=model(xt)
#   t_loss=loss_object(yt, predictions)
#   test_loss(t_loss)
#   test_accuracy(yt, predictions)
#   message='Loss: {:6.4f}, accuracy: {:7.4f}%, temps: {:7.4f}'
#   print(message.format(test_loss.result(),
#                        test_accuracy.result()*100,
#                       time.time()-start))
    
a = discriminator(x_test)
b = np.max(a,  axis = 1)
c = np.mean(b)
print(len(b), c)
d = np.argmax(a, axis = 1)
e = y_test
f = (d==e)
print(len(d==e))
print(np.sum(f)/10000)
##

# for epoch in range(epochs):
#     for batch in range(steps_per_epoch):
#         image_selec = np.random.randint(0, x_test.shape[0], size = batch_size)
#         noise = np.array([x_test[i] for i in image_selec])
# 
#         #real_x = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]           #a priori inutile dans notre cas, donc je l'ai remplacé par noise dans la suite (nom à changer ^^)
#         
#         generator.train_on_batch(noise, noise)
#     
#     image_selec = np.random.randint(0, x_test.shape[0], size = batch_size)
#     noise = np.array([x_test[i] for i in image_selec])
#     
#     fake_x = generator.predict(noise)
#     
#     x = np.concatenate((noise, fake_x))
# 
#     y_gen = np.array([y_test[i] for i in image_selec]*2)
#     
#     train_ds=tf.data.Dataset.from_tensor_slices((x, y_gen)).batch(batch_size*2)
#     
#     nbr_entrainement= 1
#     train(train_ds, nbr_entrainement, discriminator)
# 
#     d_loss = discriminator.train_on_batch(x, y_gen)
#     # d_loss = 0
# 
#     # y_gen = np.array([y_test[i] for i in image_selec])                         #!!! ICI ON PEUT FAIRE INTERVENIR LES Y !!!!
#     #g_loss = gan.train_on_batch(noise, y_gen[:batch_size])
# 
# 
#     print(f'Epoch: {epoch} \t Discriminator Loss: {d_loss}')# \t\t Generator Loss: {g_loss}')

##

# # On fait appls aux fonctions "keras" très pratique
# optimizer=tf.keras.optimizers.Adam()
# loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
# train_loss=tf.keras.metrics.Mean()
# train_accuracy=tf.keras.metrics.SparseCategoricalAccuracy()
# test_loss=tf.keras.metrics.Mean()
# test_accuracy=tf.keras.metrics.SparseCategoricalAccuracy()
# 
# # ici "@tf.function" perm d'exécute + rapidement les instructions
# @tf.function
# def train_step(images, labels, model):
#   with tf.GradientTape() as tape:
#     predictions=model(images)
#     loss=loss_object(labels, predictions)
#   gradients = tape.gradient(loss, model.trainable_variables)
#   optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#   train_loss(loss) # + bas dans "def train" on comprend l'intérêt de cette instruction (édition des résultats
#   train_accuracy(labels, predictions) # idem
# 
# def train(train_ds, nbr_entrainement, model):
#   for entrainement in range(nbr_entrainement):
#     start=time.time()
#     for images, labels in train_ds:
#       train_step(images, labels, model)
#     message='Entrainement {:04d}, loss: {:6.4f}, accuracy: {:7.4f}%, temps: {:7.4f}'
#     print(message.format(entrainement+1,
#                          train_loss.result(),
#                          train_accuracy.result()*100,
#                          time.time()-start))
#     train_loss.reset_states()
#     train_accuracy.reset_states()

##################

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

