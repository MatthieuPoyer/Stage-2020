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
    
def loss_object_gene(images_originales, images_generees, labels):
  loss = c*tf.reduce_mean(tf.square(images_originales - images_generees))
  label_genere = discriminator(images_generees)
  a = loss_object(labels, 1-label_genere)
  #a = tf.equal(tf.argmax(discriminator(images_originales), axis = 1),tf.argmax(discriminator(images_generees), axis = 1))         #dans l'idéal il faudrait remplacer predict_images originales par le vrai label
  #a = tf.cast(a, tf.float32)
  loss += a        
  return(loss)
  
#on peut virer partie d'après si besoin !!! Si on veut changer paramètres

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
    

optimizer=tf.keras.optimizers.Adam()
loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
train_loss=tf.keras.metrics.Mean()
train_accuracy=tf.keras.metrics.SparseCategoricalAccuracy()
test_loss=tf.keras.metrics.Mean()
test_accuracy=tf.keras.metrics.SparseCategoricalAccuracy()

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

#######

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
  c = 2
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
  
  
  if not(np.argmax(predict) == l):
    return(0)
  
  elif not(possible):
    return(-1)
  
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
  return(np.mean(Delta * Delta))

#######

# L'algorithme :
# ##############


# I - Les paramètres :

c = 100

batch_size= 64

nbre_etapes_gene_disc = 7

# II - Préparation de la BDD et des RNs

(x_train, y_train), (x_test, y_test)=tf.keras.datasets.mnist.load_data()

# Restructuration des donnée d'entrée pour les réseaux FC (Full connected)
x_train=(x_train.reshape(-1, 784)/255).astype(np.float32)
x_test= (x_test.reshape(-1,  784)/255).astype(np.float32)

train_ds=tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

x_t = x_train
y_t = y_train
bdd_test = x_test

# Nombre de classes
num_classes = len(np.unique(y_train))

lambda0 = (0,0,0)
#lambda0 = (10**(-3), 10**(-3), 10**(-4))
discriminator = FC100_100_10(lambda0 = lambda0, couches = (100,100))
discriminator.compile(optimizer = optimizer, loss = loss_object, metrics=['accuracy'])
discriminator.summary()

#model.compile(loss='sparse_categorical_crossentropy', optimizer="Adam")

#loss_object_gene=tf.keras.losses.SparseCategoricalCrossentropy()
train_loss_gene=tf.keras.metrics.Mean()
#train_accuracy_gene=tf.keras.metrics.SparseCategoricalAccuracy()

# test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# generator = FC500_500_10_ReLU()
# generator.compile(optimizer = optimizer, loss = train_loss, metrics=['accuracy'])
# generator.summary()

generator = FC500_500_10_ReLU()

# III - La boucle

liste_dist_test = []
liste_dist = []
liste_dist_moy1 = []
liste_dist_moy2 = []

for i in range(nbre_etapes_gene_disc):

  # Services fournis par l'API tf.data (information sur le Web)
  t_ds=tf.data.Dataset.from_tensor_slices((x_t, y_t)).batch(batch_size)
  
  nbr_entrainement= 20
  start = time.time()
  print("Entrainement Disc", i)
  discriminator.fit(x_t, y_t,
          batch_size=batch_size,
          epochs=nbr_entrainement,
  )
  print(time.time()-start, (time.time()-start)/nbr_entrainement)
  print("On est à l'étape : ", i)
  print(" ")
  
  minimisons = [x for x in range(len(y_test))]
  shuffle(minimisons)
  liste1 = []
  liste2 = []
  for i in minimisons[:1000]:
      retour_adversaire1 = creation_Adversaire2(i, discriminator)
      retour_adversaire2 = IntriguingAdversersarialSansButPrecis(i,discriminator)
      liste1.append(retour_adversaire1[-1])
      liste2.append(retour_adversaire2)
  
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
  
  distorsion_moyenne1 = sum([np.sqrt(i) for i in liste1])/len(liste1)
  distorsion_moyenne2 = sum([np.sqrt(i) for i in liste2])/len(liste2)
  
  print("La distorsion moyenne dans le cas de l'iterative gradient est : ", distorsion_moyenne1)
  print("La distorsion moyenne dans le cas de Intriguing properties est : ", distorsion_moyenne2)
  
  liste_dist_moy1.append(distorsion_moyenne1)
  liste_dist_moy2.append(distorsion_moyenne2)
  
  #discriminator.trainable = False                         #!!! JE NE SAIS PAS CE QUE C'EST ????
  
  print("Jeu de test")
  test(x_test,y_test, discriminator)
  
  # nbr_entrainement= 50
  # print("Entrainement Géné", i)
  # generator.fit(x_train, y_train,
  #         batch_size=batch_size,
  #         epochs=nbr_entrainement,
  # )
  
  nbr_entrainement= 50
  print("Entrainement Géné", i)
  train_gene(train_ds, nbr_entrainement, generator)
  
  print("TRAIN, ERROR RATE")
  sol = generator(x_train)
  part123 = sum(np.argmax(discriminator(sol), axis = 1) == y_train)
  print(part123)
  print(part123/600)
  
  distorsionPart = np.mean(np.sqrt(np.mean((x_train - sol)*(x_train - sol), axis = 1)))
  
  liste_dist.append(distorsionPart)
  
  x_t = np.concatenate((x_t, sol))
  y_t = np.concatenate((y_t, y_train))
  soltest = generator(x_test)
  bdd_test = np.concatenate((x_t, soltest))
  
  distorsionPart = np.mean(np.sqrt(np.mean((x_test - soltest)*(x_test - soltest), axis = 1)))
  liste_dist_test.append(distorsionPart)

  
###

# a la recherche d'un bon c

print("TRAIN, ERROR RATE")
sol = generator(x_train)
part123 = sum(np.argmax(discriminator(sol), axis = 1) == y_train)
print(part123)
print(part123/600)

print("TEST, ERROR RATE")
sol = generator(x_test)
part123 = sum(np.argmax(discriminator(sol), axis = 1) == y_test)
print(part123)
print(part123/100)

# Résultats :
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



#################

import matplotlib.pyplot as plt

plt.close()
n = np.random.randint(0,60000,1)[0] # on va perturber la 1000 ième image de la base de test qui correspond au chiffre 9
#n = 1000
X = x_t       # !!!!! ATTENTION ICI C'est TRAIN !!!!!
Y = to_categorical(y_t)
# Sélection de l'image à perturber
label_oiginal  = Y[n]  # un "9"
image_original = X[n]
adversaire = [np.array([X[(i+1)*60000+n]]) for i in range(nbre_etapes_gene_disc)]
#adversaire1 = X[n+60000]

'''
Calcul du gradient du log de la vraisemblance du chiffre cible (proposé par l'adversaire)
par rapport aux entrées (image) >  vecteur de 784 composantes
'''
  

imReelle = image_original.reshape(1,784)
imAdversaire = [i.reshape(1,784) for i in adversaire]
Delta = [imReelle - i for i in imAdversaire]

# Le + simle des algo : "iterative gradient" (on se limite à 100 itérations)
imAdversaireTest = generator(imReelle)

imAdversaireTest = imAdversaireTest.numpy() # On transforme le tf de TensorFlow en np.array de NumPy
DeltaTest = imReelle - imAdversaireTest

# On edite le carré de la norme de Frobenius de la perturbation (moyenné par le nombre de pixels)
print("EQM1 = ",np.mean(Delta[0] * Delta[0])) # pour ce cas EQM = 0.0015074897
print("2nd terme", loss_object(y_train[n], 1-discriminator(imAdversaire[0])))

# On edite le carré de la norme de Frobenius de la perturbation (moyenné par le nombre de pixels)
print("EQMTest = ",np.mean(DeltaTest * DeltaTest)) # pour ce cas EQM = 0.0015074897
print("2nd terme", loss_object(y_train[n], 1-discriminator(imAdversaireTest)))

#plt.figure(figsize=(9, 6))
# Image réelle
plt.subplot(nbre_etapes_gene_disc,3,1)
plt.imshow(imReelle.reshape([28, 28]),cmap = "gray")
plt.title("Chiffre = " + np.str(np.argmax(discriminator(imReelle))))
print("Réelle : ", np.max(discriminator(imReelle)))

nbre_etapes_gene_disc = 20

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
# # Perturbation adverse
# plt.subplot(nbre_etapes_gene_disc+1,3,(nbre_etapes_gene_disc+1)*3-1)
# plt.imshow(DeltaTest.reshape([28, 28]),cmap = "gray")
# plt.title("Perturbation adversaire")
# # Image modifiée par l'adversaire
# plt.subplot(nbre_etapes_gene_disc+1,3,(nbre_etapes_gene_disc+1)*3)
# plt.imshow(imAdversaireTest.reshape([28, 28]),cmap = "gray")
# plt.title("Chiffre = " + np.str(np.argmax(discriminator(imAdversaireTest))))
# print("Adversaire : ", np.max(discriminator(imAdversaireTest)))
plt.show()

###


# La distorsion moyenne après 20 étapes

import numpy as np
import matplotlib.pyplot as plt

a = np.array([0.073579594, 0.077717565, 0.08322092, 0.07832371, 0.087938294, 0.087290645, 0.098814584, 0.08776691, 0.100518204, 0.097730175, 0.09347145, 0.10769891, 0.11493808, 0.09993061, 0.11145804, 0.10446661, 0.10817391, 0.106853835, 0.1163971, 0.11360584])

plt.plot(np.linspace(0,20,20),a)
plt.show()


####

# a = 50
# sol = 0
# for i in range(a):
#     sol += 50*7.4 + 20*(a+1)*2.6
# print("en secondes", sol)
# print("en minutes", sol/60)
# print("en heures", sol/3600)
# 
# print(" ")
# 
# print("Le temps mis sera de", sol//3600, "heures, de", (sol%3600)/60,"minutes, et de ", (sol%60), "secondes" )
# 
# print(" ")
# 
# a2 = 150
# sol2 = 0
# for i in range(a2):
#     sol2 += 5*7.4 + 1*(a2+1)*2.6
# print("en secondes", sol2)
# print("en minutes", sol2/60)
# print("en heures", sol2/3600)
# 
# print(" ")
# 
# print("Le temps mis sera de", sol2//3600, "heures, de", (sol2%3600)/60,"minutes, et de ", (sol2%60), "secondes" )