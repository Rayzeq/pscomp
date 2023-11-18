`pscomp` est un compilateur de pseudo-code pour l'IUT de Limoges. Il compile le pseudo-code en Python et fait un peu de type-checking.

## Installation

Vous aurez besoins de Python 3.10 (minimum) et pip ou pipx.
Installez le compilateur avec `pipx install git+https://github.com/Rayzeq/pscomp.git` (remplacez `pipx` par `pip` ou `pip3` si nécéssaire).


## Utilisation

Pour compiler un fichier, utilisez `pscomp <fichier>`.
Les fichiers `.txt`, `.pseudo` sont directement lus comme du pseudo code.
Vous pouvez aussi passer des fichiers markdown, auquel cas le compilateur traitera tous les blocs de code sans langage (ou avec `pseudo` comme langage) comme du pseudo-code.
