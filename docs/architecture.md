# Architecture du projet

Cette architecture constitue une première proposition et pourra être ajustée au fil du développement et des échanges de l’équipe.

MyTorch utilise **C++17 pour les calculs du réseau et les exécutables**, et
**Python pour préparer les données, piloter les expériences et analyser les
résultats**. Le Makefile est le point d'entrée commun pour compiler le projet
et lancer les tests ou les scripts.

## Les différentes parties

| Partie | Langage | Rôle |
| --- | --- | --- |
| `src/neural/` | C++ | Réseau générique : configuration, initialisation, sauvegarde/chargement, puis propagation et apprentissage. |
| `src/chess/` | C++ | Lecture des positions FEN, encodage numérique et lecture des labels du dataset. |
| `src/generator/` | C++ | Exécutable `my_torch_generator` : création de modèles à partir de configurations. |
| `src/analyzer/` | C++ | Exécutable `my_torch_analyzer` : orchestration des modes entraînement et prédiction. |
| `scripts/` (prévu) | Python | Préparation des datasets, lancement des expériences, métriques et graphiques. |
| `tests/` | C++ et Python | Tests des fonctions et des exécutables. |

Le module réseau manipule des vecteurs numériques sans connaître les règles
d'échecs ou la notation FEN. La partie échecs transforme les données selon les
contrats d'[entrée](input-encoding.md) et de [sortie](output-encoding.md).
L'analyseur relie ces deux parties.

## Circulation des données prévue

Pour l'entraînement :

```text
Datasets sources → préparation Python → fichiers train / validation / test
Fichier de configuration → générateur C++ → modèle initial
Modèle initial + données → analyseur C++ --train → modèle entraîné
Résultats des expériences → analyse Python → métriques et graphiques
```

Pour la prédiction :

```text
Fichier FEN → lecture et encodage C++ → réseau C++ → classes affichées
```

Les scripts Python appellent les exécutables avec `subprocess` et échangent
avec eux par fichiers et sorties de commandes. Aucun binding entre les deux
langages n'est nécessaire. Les boucles de calcul et les mises à jour des poids
restent dans le module C++ ; Python organise les expériences autour.

## Tests

- **C++** : tester directement les fonctions, notamment l'encodage, les calculs
  des couches, les gradients et la sauvegarde/chargement. Utiliser des exemples
  dont les résultats attendus sont connus.
- **Python** : lancer les exécutables et vérifier leurs arguments, stdout,
  stderr, codes de retour et fichiers produits. Couvrir aussi le parcours
  génération, entraînement, sauvegarde puis prédiction lorsqu'il sera disponible.

Les deux niveaux se complètent : un test C++ vérifie un calcul précis, un test
Python vérifie le comportement du programme complet. Il n'est pas nécessaire
de reproduire chaque test dans les deux langages.

## Makefile et scripts

Commandes disponibles :

- `make` : compiler les deux exécutables C++.
- `make test` : lancer les tests C++ et les tests des exécutables en Python.
- `make clean`, `make fclean`, `make re` : nettoyer et reconstruire.

Les futurs scripts seront également lancés par le Makefile, avec des cibles
comme `make dataset` et `make benchmark`. **Ces deux cibles n'existent pas encore.**
Les commandes Python utiliseront la variable `PYTHON` du Makefile, déjà définie
à `python3`. Les dépendances tierces éventuelles seront déclarées dans un fichier
dédié et utilisées dans un environnement virtuel.

Les outils seront des scripts `.py` versionnés. Les expériences devront conserver
les configurations, seeds, commandes et références des données utilisées pour
permettre leur reproduction. Les datasets d'entraînement devront rester
disponibles pour la soutenance, sans obligation de les versionner.

## État actuel

La génération, l'initialisation et la sauvegarde/chargement des réseaux sont
implémentées. Les tests C++ et les tests CLI Python sont lancés par `make test`.
Les contrats d'encodage et certaines interfaces sont déclarés ; l'encodage FEN,
la propagation, l'entraînement et la prédiction restent à implémenter, ainsi que
les scripts Python de préparation et d'évaluation.
