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

## Phase 2 - environnement turbo, DQN et clonage de comportement

Objectif de la phase : depasser le plafond de la Q-table (granularite de la
table) et viser un agent capable de finir un etage. Budget re-echantillonne :
episodes de 900 steps, baseline aleatoire re-mesuree sur ce budget.

| Version | Changement teste | Signal observe | Decision |
| --- | --- | --- | --- |
| V9 | Environnement turbo (boucle du jeu pilotee en synchrone) | ~540 steps/s au lieu de 4-5, temps de jeu exact par step | Adopte ; baselines re-mesurees (600 steps : 8.45, 900 steps : 10.54) |
| V10 | DQN 59 actions sur 70 features continues | Le greedy converge vers "camper dans la salle de depart" (evals figees a 13.54) | Retirer le revenu de survie de la recompense d'entrainement |
| V11 | Temps retire + cout par step + epsilon d'eval | L'agent traverse les portes puis se fige/meurt en combat ; evals instables | Ajouter une meta-action de delegation comme en V6 |
| V12 | Delegation heuristique + penalite HP 1.0 | Record eval 25.4 (2.4x la baseline) mais le greedy n'utilise jamais la delegation (0/900 mesure) | Delegation apprise par sequences, pas par steps isoles |
| V13 | Delegation collante (sequences de 12 steps) + reprise du meilleur | 7 etages termines a l'ENTRAINEMENT (scores 147-181), mais ecart train ~64 / eval ~6 : la perf appartient aux sequences heuristiques | Mesurer honnetement chaque composante |
| V14 | Eval comparative 20 episodes x 900 steps | DQN pur 6.66 (sous l'aleatoire 10.54) ; heuristique corrigee 63.66 avec 2 etages | Resultat negatif documente : le RL pur n'apprend pas le combat fin sur ce budget |
| V15 | Clonage de comportement (94 074 paires heuristiques, 87.8 % de precision val.) | **56.89 de moyenne sur 20 episodes, 5.4x la baseline**, 66 salles et 277 kills, reseau seul sans garde-fou | Modele retenu : `models/bc_agent.json` |

### V10-V13 - trois pathologies du DQN, mesurees et corrigees

1. **Camping** : le terme de temps du score (+0.1/s) paye l'immobilite sans
   risque ; le greedy convergeait vers 13.54 exactement (survie totale, zero
   contact) a chaque eval. Correctif : temps retire de la recompense
   d'ENTRAINEMENT (le score de comparaison ne change pas).
2. **Gel devant la porte** : trace pas-a-pas : l'agent marche droit vers la
   porte puis se fige a 164 px, mal aligne, sur une action statique (etat
   inchange -> meme argmax -> boucle infinie). Correctifs : cout de -0.02 par
   step et 2 % d'exploration en eval (protocole DQN standard).
3. **Delegation morte** : la meta-action "heuristic" tiree step par step
   pendant l'exploration ne montre jamais la valeur de l'heuristique (un step
   isole dans du bruit). Correctif : sequences collantes de 12 steps. Resultat :
   des etages complets a l'entrainement, mais le reseau seul ne les a pas
   internalises dans le budget disponible.

### V15 - apprentissage par demonstrations

150 episodes heuristiques enregistres en turbo (94 074 paires
features -> action, 11 etages dans les demonstrations), reseau entraine par
entropie croisee (30 epoques, ~2 min), precision de validation 87.8 %.

Le reseau clone atteint ~89 % du score de son professeur et nettoie 2 a 5
salles a chaque episode du protocole. Sa limite est celle du professeur : le
combat de boss (l'etage se finit "parfois", pas "a chaque run"). Piste
suivante notee : ameliorer le combat de boss de l'heuristique (les progres du
professeur se transferent au clone par re-collecte + re-clonage), puis
affinage RL type DQfD (perte supervisee + perte TD melangees).

Commandes de reproduction :

```bash
python scripts/collect_demos.py --episodes 150 --max-steps 900 --seed 42
python scripts/train_bc.py
python scripts/watch_dqn.py --agent models/bc_agent.json --episodes 3
```

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
