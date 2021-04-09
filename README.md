# Atelier IA détection des émotions sur un flux Webcam.

Dans cet atelier, nous allons essayer de détecter les émotions de votre visage via l'image de la webcam.
Nous détecterons les émotions suivantes:
- colere
- joie
- tristesse
- surprise

L'opération se réalisera en deux temps:
- Premièrement une détection des visages, pour cela nous utiliserons un modèle déjà existant (MTCNN)
- Ensuite un modèle de détectionndes émotions que nous réaliserons ensemble.

Pour entraîner un algorithme d'IA, il est nécessaire d'avoir à disposition une base de données labelisée. A savoir dans notre cas des photos de visage, avec une étiquette qui leur est associée (par exemple: photo1: colère, photo2: tristesse,.....)

Il vous suffit de télécharger le contenu de ce dépôt GitHub, qui contient la base de données et le notebook Jupyter pour effectuer le travail. Ce notebook n'est pas rempli à 100%, mais des indications vous sont forunies pour compléter le code.

La base de données contient 19211 images de 48*48 pixels en niveaux de gris.
