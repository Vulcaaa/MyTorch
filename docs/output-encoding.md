# Lire et encoder les sorties du réseau

Pour une position, le réseau doit choisir entre trois classes : `Nothing`,
`Check` et `Checkmate`. On ignore la couleur indiquée après le label du dataset.

Ce document propose l’ordre des sorties pour l’issue #1. Cet ordre doit rester
identique dans le lecteur de données, l’entraînement et la prédiction.

## 1. Trois classes, trois sorties

| Indice | Classe | Signification | Cible d’entraînement |
| --- | --- | --- | --- |
| 0 | `Nothing` | Ni échec ni mat | `[1, 0, 0]` |
| 1 | `Check` | Échec sans mat | `[0, 1, 0]` |
| 2 | `Checkmate` | Échec et mat | `[0, 0, 1]` |

Une position reçoit une seule classe. Si elle est mat, la cible est `Checkmate`,
pas `Check` en plus. Le pat n’est pas une classe distincte dans ce contrat.

L’ordre est donc toujours :

```text
[Nothing, Check, Checkmate]
```

## 2. Transformer un label du dataset en cible

On conserve les fichiers du dataset tels quels. À la lecture, on applique cette
correspondance :

| Label dans le fichier | Classe retenue | Cible |
| --- | --- | --- |
| `Nothing` | `Nothing` | `[1, 0, 0]` |
| `Check White` ou `Check Black` | `Check` | `[0, 1, 0]` |
| `Checkmate White` ou `Checkmate Black` | `Checkmate` | `[0, 0, 1]` |

Les labels simples `Check` et `Checkmate` sont également acceptés. Un label
inconnu est une erreur : il ne doit pas être transformé en `Nothing` par défaut.

Exemple tiré de `dataset/check/10_pieces.txt` :

```text
8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59 Check White
└────────────── FEN : entrées ─────────────┘ └─ cible ─┘

Check White → Check → [0, 1, 0]
```

La cible vient du dataset. Elle est fournie à l’entraînement pour comparer le
résultat calculé au résultat attendu ; elle ne fait jamais partie des entrées.

## 3. Lire une prédiction

Pour cette classification, on retient `softmax` en sortie, comme dans la
configuration d’exemple. Le réseau renvoie trois scores entre 0 et 1, dont la
somme vaut 1, à l’arrondi numérique près.

Exemple illustratif, sans calculer réellement la position précédente :

```text
Sorties : [0.05, 0.90, 0.05]
           ↑     ↑     ↑
        Nothing Check Checkmate

Le plus grand score est à l’indice 1 → prédiction : Check
```

On choisit simplement l’indice du plus grand score (`argmax`), sans seuil à 0.5.
En cas d’égalité exacte, on retient le premier indice pour garder un résultat
déterministe. Un score de `0.90` ne garantit pas que la prédiction est correcte
dans 90 % des cas : la qualité du modèle devra être mesurée.

En mode prédiction, aucune cible n’est nécessaire :

```text
FEN → encodeur → réseau → 3 scores → classe affichée
```

## 4. Garder le même contrat dans tout le projet

- La dernière couche doit contenir exactement **3 neurones**.
- La configuration d’exemple possède déjà ces 3 sorties et utilise `softmax`.
- Les cibles et les prédictions utilisent toujours le même ordre des classes.
- Le programme affiche `Nothing`, `Check` ou `Checkmate`, sans suffixe de couleur.
- Un modèle sauvegardé doit être interprété avec ce même contrat à son chargement.

Le module réseau reste générique : il manipule des vecteurs numériques. La
correspondance entre indices et noms des classes appartient à la partie échecs.
La règle de compatibilité avec les configurations et modèles sauvegardés est
définie dans [input-encoding.md](input-encoding.md#compatibilité-avec-le-code-actuel).
