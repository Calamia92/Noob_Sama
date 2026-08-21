# Carnet d'essais

Ce carnet retrace les iterations mesurees pendant le projet. Le score compare
toujours la meme metrique :

```text
score = floors * 100 + rooms * 10 + kills + time * 0.1
```

La baseline de reference reste l'agent aleatoire sur 20 episodes de 100 steps :
score moyen 1.836, 0 kill et 0 salle terminee.

## Synthese

| Version | Changement teste | Signal observe | Decision |
| --- | --- | --- | --- |
| Baseline | Agent aleatoire | Score moyen 1.836, aucune progression de salle | Reference de comparaison |
| V1 | Q-learning sans direction de sortie | Environ 50 episodes, 0 salle terminee, plafond d'eval autour de 2.29 | Ajouter la direction de porte dans l'etat |
| V2 | Porte la plus proche | L'agent fait des allers-retours vers la porte par laquelle il vient d'entrer | Cibler la porte menant a la salle non visitee la plus proche |
| V3 | Direction de porte par mur | L'agent se colle aux murs sans s'aligner avec l'ouverture | Encoder la direction relative en 8 secteurs |
| V4 | Greedy deterministe | Blocage contre des obstacles non observes | Ajouter un detecteur de blocage en demo/eval |
| V5 | Alpha constant | Evals instables apres les pics | Decroissance alpha 0.25 -> 0.05 |
| V6 | Delegation heuristique | Plancher de perf plus stable, record eval 15.757 a ep. 975 | Conserver `heuristic` comme meta-action |
| V7 | Actions bas niveau enrichies | Plus de controle, mais espace d'action trop large si tout est appris directement | Garder les actions riches pour le controle heuristique |
| V8 | Intentions tactiques | Tests plus lisibles, mais les evals intent restent instables | Garder la version pratique Q-learning + garde-fous |

## Details des essais

### Baseline aleatoire

Commande :

```bash
python scripts/baseline_random.py --episodes 20 --max-steps 100 --seed 42
```

Resultat : score moyen 1.836, minimum 1.742, maximum 2.252. L'agent aleatoire
survit quelques secondes mais ne tue aucun ennemi et ne termine aucune salle.

### V1 - Etat aveugle aux portes

Premier probleme : apres un combat, toutes les positions d'une salle vide se
ressemblaient pour la Q-table. L'agent pouvait recevoir le bonus de salle, mais
il n'avait aucun signal d'etat pour apprendre ou se trouve la sortie.

Resultat observe : zero salle terminee sur les premiers essais courts, avec un
plafond d'evaluation autour de 2.29.

Correction : ajouter la direction de la porte cible dans l'etat discretise.

### V2 - Ping-pong entre salles

Apres ajout des portes, l'agent a commence a sortir des salles, mais il prenait
souvent la porte la plus proche. En entrant dans une nouvelle salle, cette porte
est souvent celle qui revient en arriere.

Resultat observe : allers-retours entre salles deja nettoyees.

Correction : calculer une porte cible par BFS dans le graphe du donjon pour
favoriser la salle non visitee la plus proche.

### V3 - Agent colle aux murs

Encoder seulement le mur de la porte (`up`, `down`, `left`, `right`) ne suffisait
pas. Si l'agent n'etait pas aligne avec l'ouverture, il pressait une direction et
restait bloque contre le mur.

Correction : encoder la direction relative vers la porte en 8 secteurs. L'agent
peut alors apprendre a s'aligner avant de traverser.

### V4 - Greedy deterministe bloque

En evaluation greedy, certaines positions non encodees dans l'etat provoquaient
des boucles : meme etat discret, meme action, meme blocage.

Correction : detecter 8 steps sans mouvement significatif hors combat et jouer
une action de sortie. Dans la version finale, cette sortie est deleguee a
l'heuristique plutot qu'a un coup aleatoire.

### V5 - Instabilite avec alpha constant

Les premiers entrainements montaient puis retombaient fortement, par exemple un
pic autour de 11 suivi d'une evaluation proche de 2. Le taux d'apprentissage
constant donnait trop de poids aux dernieres experiences.

Correction : faire decroitre alpha de 0.25 vers 0.05 pour stabiliser les acquis
en fin de run.

### V6 - Meta-action `heuristic`

L'exploration guidee seule aidait, mais l'agent ne savait pas toujours quand
laisser agir la politique de reference.

Correction : ajouter une action `heuristic` dans la Q-table. Le Q-learning reste
off-policy : il apprend a partir des coups joues, y compris ceux proposes par la
politique heuristique.

Resultat principal : meilleur agent a eval@975 avec score moyen 15.757, soit
environ 8.6 fois la baseline aleatoire.

### V7 - Controle bas niveau enrichi

Les observations visuelles ont montre que le controle simple etait limite en
combat. Des tirs diagonaux, tirs en mouvement et dashs directionnels ont ete
ajoutes.

Resultat : le controle heuristique devient meilleur, mais apprendre directement
toutes ces actions dans la Q-table agrandit trop l'espace d'action pour le budget
de deux jours.

Decision : garder ces actions dans le wrapper et dans les heuristiques, pas comme
espace principal d'apprentissage.

### V8 - Intentions tactiques

Derniere iteration experimentee : l'espace d'apprentissage est passe a des intentions
compactes (`fight`, `kite`, `dash_away`, `exit`, `loot`, `interact`, `wait`,
`heuristic`) resolues ensuite en actions clavier.

Resultat observe : les episodes d'entrainement peuvent etre bons, mais les
evaluations greedy restent instables apres migration de l'ancien modele. Cette
piste n'est donc pas retenue comme modele final.

Le record ponctuel `eval@1305` a 18.663 vient de la version pratique precedente :
modele Q-learning 16 actions, garde-fous heuristiques en evaluation/demo, et
controle bas niveau enrichi. L'evaluation finale annoncee dans le README reste
plus basse mais plus representative : moyenne 8.973 sur 20 episodes, avec des
runs faibles surtout dus aux combats ou l'agent perd trop de vie.

Decision : presenter la version finale comme un agent Q-learning tabulaire 16
actions avec controle bas niveau heuristique et garde-fous de demo, et garder
les intentions tactiques comme piste non retenue faute de stabilite.

## Limites connues

- Les donjons ne sont pas seedables, donc les evaluations courtes restent
  bruitees.
- Le modele sauvegarde final utilise encore l'ancien espace de 16 actions bas
  niveau ; le code sait le rejouer, mais la narration doit rester claire sur ce
  point.
- Les combats restent la principale source de runs faibles.
- L'evaluation finale est conservee dans `reports/final_eval.csv` pour relier
  directement les chiffres finaux au fichier source.

## Pistes avec plus de temps

- Evaluer chaque checkpoint sur plus d'episodes.
- Relancer un run complet avec une autre seed.
- Comparer l'agent final avec l'heuristique seule sur le meme protocole.
- Tester un petit DQN sur les memes features pour generaliser entre etats
  voisins.
