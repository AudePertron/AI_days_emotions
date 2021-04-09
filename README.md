# Atelier IA détection des émotions sur un flux Webcam.

Dans cet atelier, nous allons essayer de détecter les émotions de votre visage via l'image de la webcam.
Nous détecterons les émotions suivantes:
- colere
- joie
- tristesse
- surprise

L'opération se réalisera en deux temps:
- Premièrement une détection des visages, pour cela nous utiliserons un modèle déjà existant (MTCNN)
- Ensuite un modèle de reconnaissance des émotions que nous réaliserons ensemble.

Pour entraîner un algorithme d'IA, il est nécessaire d'avoir à disposition une base de données labelisée. A savoir dans notre cas des photos de visage, avec une étiquette qui leur est associée (par exemple: photo1: colère, photo2: tristesse,.....)
L'entraînement d'un modèle prend du temps, même un modèle simple comme le notre. Pour gagner du temps lors de cette journée, nous vous proposons d'utiliser le modèle que nous avons entraîné au préalable, même si nous vous guideront à travers toutes les étapes du code dans ce Notebook. Vous serez en mesure de l'entraîner de A à Z après les IA days, si vous le désirez.

![exemple photo](/images/demo.png)

Il vous suffit de télécharger le contenu de ce dépôt GitHub, qui contient la base de données et le notebook Jupyter pour effectuer le travail. Ce notebook n'est pas rempli à 100%, mais des indications vous sont forunies pour compléter le code.

La base de données contient 19211 images de 48*48 pixels en niveaux de gris.

A la fin du code, nous ouvrirons la webcam et vous pourrez essayer votre modèle.
![test webcam](/images/test.png)

## Prérequis:
Nous vous recommandons de créer un nouvel environnement de travail avant de débuter.
Nous fournissons un fichier requirements.txt qui executera les installations nécessaires. Nous installerons Jupyter Notebook, mais si vous êtes déjà familiers avec Google Colab, vous pouvez faire tourner le code dessus.

## Installations
__N'oubliez pas de créer et d'activer un nouvel environnement de travail avant de démarrer__

Ouvrez votre terminal, et clonez ce dépôt GitHub:

`git clone https://github.com/AudePertron/AI_days_emotions`

installez les requirements:

`cd AI_days_emotions`

`pip install -r requirements.txt`

Ouvrez le notebook: 
`jupyter notebook`
