# Encoder les entrées du réseau

Le réseau reçoit une liste de nombres. L’encodeur transforme une position FEN
en cette liste, toujours dans le même ordre.

Le contrat retenu pour l’issue #1 contient **777 entrées**, dans cet ordre :

| Indices du tableau | Information | Nombre de valeurs |
| --- | --- | --- |
| 0 à 767 | Position des pièces | 768 |
| 768 | Joueur au trait | 1 |
| 769 à 776 | Colonne de la cible de prise en passant | 8 |

Tous les indices commencent à zéro. Les droits de roque et les compteurs de
coups ne sont pas encodés dans ce contrat.

## 1. Une case = 12 valeurs

Chaque case possède 12 emplacements : un par type de pièce et par couleur.
L’ordre est le suivant :

| Indice | Pièce | Symbole FEN |
| --- | --- | --- |
| 0 | Pion blanc | `P` |
| 1 | Cavalier blanc | `N` |
| 2 | Fou blanc | `B` |
| 3 | Tour blanche | `R` |
| 4 | Dame blanche | `Q` |
| 5 | Roi blanc | `K` |
| 6 | Pion noir | `p` |
| 7 | Cavalier noir | `n` |
| 8 | Fou noir | `b` |
| 9 | Tour noire | `r` |
| 10 | Dame noire | `q` |
| 11 | Roi noir | `k` |

Une seule valeur vaut `1` si la case contient une pièce. Une case vide contient
uniquement des `0`. C’est un encodage dit « one-hot » pour les pièces.

```text
                  Pièces blanches   Pièces noires
                  P N B R Q K       p n b r q k
Tour blanche :   [0 0 0 1 0 0        0 0 0 0 0 0]
Roi noir :       [0 0 0 0 0 0        0 0 0 0 0 1]
Case vide :      [0 0 0 0 0 0        0 0 0 0 0 0]
```

## 2. Lire les 64 cases dans un ordre fixe

On suit l’ordre de la FEN, rangée par rangée, de gauche à droite :

```text
a8 b8 c8 d8 e8 f8 g8 h8    ← cases 0 à 7
a7 b7 c7 d7 e7 f7 g7 h7    ← cases 8 à 15
…
a1 b1 c1 d1 e1 f1 g1 h1    ← cases 56 à 63
```

On met les groupes de 12 valeurs bout à bout dans un seul tableau :

```text
[12 valeurs de a8, 12 valeurs de b8, …, 12 valeurs de h1]
```

Cela donne **64 × 12 = 768 entrées pour les pièces**. Il n’y a pas d’emplacement
supplémentaire pour une case vide : ses 12 zéros suffisent.

Pour retrouver une valeur dans ce tableau :

```text
indice dans le tableau = numéro de la case × 12 + indice de la pièce
```

## 3. Ajouter le joueur au trait

Le deuxième champ de la FEN indique qui doit jouer. On le place juste après
les 768 valeurs des pièces :

| Champ FEN | Joueur au trait | `inputs[768]` |
| --- | --- | --- |
| `w` | Blancs | `0` |
| `b` | Noirs | `1` |

Ce nombre représente une catégorie, pas la force d’un joueur.

## 4. Ajouter la cible de prise en passant

Le quatrième champ de la FEN contient `-` ou une case cible, comme `d6`.
On réserve **8 valeurs**, une par colonne de `a` à `h` :

```text
Colonne :   a   b   c   d   e   f   g   h
Indice :   769 770 771 772 773 774 775 776

Aucune :   [0   0   0   0   0   0   0   0]   ← champ « - »
Cible d6 : [0   0   0   1   0   0   0   0]   ← inputs[772] = 1
```

La rangée cible se déduit du joueur au trait : **6 si les blancs jouent**, **3 si
les noirs jouent**. Ainsi, `d6` avec `w` et `d3` avec `b` activent le même indice,
mais `inputs[768]` les distingue. Une cible sur une rangée incompatible est invalide.

On encode la cible indiquée dans la FEN. Sa présence ne garantit pas qu’une prise
en passant est effectivement légale : l’encodeur n’a pas à calculer ce coup.

## 5. Exemple du dataset

Première ligne de `dataset/check/10_pieces.txt` :

```text
8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59 Check White
```

Les `/` séparent les rangées. Un chiffre indique combien de cases vides se
suivent. Les deux premiers `8` représentent donc les rangées 8 et 7 entièrement
vides. Leurs 16 cases produisent les 192 premiers zéros du tableau.

La troisième rangée, `R2k4`, se lit ainsi :

```text
Case :   a6  b6  c6  d6  e6  f6  g6  h6
Pièce :  R   ·   ·   k   ·   ·   ·   ·
```

- `a6` est la case 16 : sa tour blanche active `inputs[16 × 12 + 3]`, soit `inputs[195]`.
- `b6` et `c6` sont vides : leurs groupes de 12 restent à zéro.
- `d6` est la case 19 : son roi noir active `inputs[19 × 12 + 11]`, soit `inputs[239]`.

On continue de la même façon jusqu’à `h1`.

Dans cette ligne, le deuxième champ est `b` et le quatrième est `-` :

```text
inputs[0..767]   = encodage des pièces décrit ci-dessus
inputs[768]      = 1                  ← noirs au trait
inputs[769..776] = [0 0 0 0 0 0 0 0]  ← aucune cible de prise en passant
```

Le tableau complet contient donc **768 + 1 + 8 = 777 valeurs**.

## 6. Donner les valeurs au réseau

```text
Ligne du dataset → extraire la FEN → encoder pièces + trait + en passant
                                 → tableau de 777 valeurs → réseau
```

Le tableau est construit en mémoire, puis transmis au réseau. Il n’est pas
nécessaire de le sauvegarder dans un autre fichier. Le fichier `.nn` contient
la configuration et les paramètres du modèle, pas les positions du dataset.

`Check White` est le résultat attendu pour l’entraînement : il ne doit jamais
être ajouté aux entrées. Pour nos trois classes, on retient ici `Check` et on
ignore le suffixe de couleur.

## Compatibilité avec le code actuel

Ce document définit le contrat ; l’encodeur n’est pas encore implémenté.
L’analyseur accepte uniquement les modèles à **777 entrées et 3 sorties**, avec
l’ordre de sortie **`[Nothing, Check, Checkmate]`** défini dans
[output-encoding.md](output-encoding.md). Un modèle dont les dimensions ne
correspondent pas doit être refusé explicitement.

La configuration d’exemple contient encore 768 entrées : sa première taille devra
passer à **777** pour utiliser cet encodage. Les anciens modèles à 768 entrées
doivent être régénérés avec une configuration compatible, puis entraînés pour
ce contrat.

Le générateur reste générique : il utilise les dimensions de la configuration
fournie et n’impose pas le contrat échecs. Aucun identifiant d’encodage
supplémentaire n’est ajouté au format des modèles pour cette issue ; les
dimensions seules ne permettent pas de vérifier l’ordre des classes, qui doit
être respecté lors de l’entraînement et de la prédiction.
