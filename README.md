# Robustesse des Réseaux de neurones par apprentissage antagoniste

>Vous trouverez dans ce répertoire les codes que j'ai utilisé durant mon stage.
D'ailleurs, tous les codes sont indépendants (donc certaines fonctions apparaissent plusieurs fois)

>Je vous invite à lire le rapport de stage, qui présente la démarche et les idées développés dans certains codes.

* Rapport de stage Matthieu Poyer
* Création d'Adversaire
  >C'est le premier code. Il crée le réseau de neurones, ainsi que des adveraires avec deux stratégies différentes : celle proposée par Szegedy et une banale descente en gradient. On y évalue l'impact du nombre d'étapes d'entraînement, ainsi que celui des couches internes.
* Étude de la régularisation et du dropout
  > Ce code permet d'évaluer les effets de la régularisation et du dropout sur la robustesse.
* Impact du bruit en entrée d'un générateur
  >Le but de cette partie est de répondre à la question : si on met une image et du bruit en entrée d'un réseau de neurones et qu'on lui demande de générer des adveraires, va-t-il vraiment ignorerle bruit ? 
  
* Algorithme de Madry accéléré
  >Nous avons proposé une version accélérée de la méthode de Madry. L'astuce consiste à générer des adveraires avec un réseau de neurones, afin de gagner du temps sur cette étape qui est longue.

PS : Les codes utilisent tensorflow 2
