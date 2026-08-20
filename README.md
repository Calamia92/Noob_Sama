# Noob Sama - Gaming Agent

## Jeu choisi

Jeu : Eclipse Of Souls

Lien : https://kraich.itch.io/eclipse-of-souls

Type : roguelite action twin-stick jouable dans le navigateur via itch.io.

## Validation initiale

Le jeu se lance correctement depuis la page itch.io avec le bouton `Run game`.
Une fois le canvas charge, le menu principal s'ouvre et une run peut etre lancee
avec `Espace`, puis validation du premier heros disponible avec `Espace`.

Etats reperes dans le code du jeu :

- `title` : menu principal.
- `heroSelect` : choix du heros.
- `play` : partie en cours.
- `gameover` : mort.
- `victory` : victoire.

Le jeu maintient aussi des statistiques internes de run :

- `kills` : ennemis tues.
- `rooms` : salles terminees.
- `floors` : etages termines.
- `time` : temps de survie / temps de run.

## Controles confirmes

Les entrees clavier sont basees sur les touches physiques.

- Deplacement : `WASD` cote code, affiche `ZQSD` sur clavier AZERTY.
- Tir : fleches directionnelles, ou `IJKL`.
- Dash : `Espace`.
- Competence du heros : `Shift`.
- Interaction : `E`.
- Inventaire : `Tab`.
- Pause : `Escape`.
- Relancer : `R`.

## Observations de l'agent

Pour la premiere version, l'agent n'utilise pas les pixels du canvas. Le wrapper
lit l'etat interne expose par le jeu :

- `state` : etat courant du jeu.
- `hp` et `max_hp` : vie du joueur.
- `x`, `y` : position du joueur dans la salle.
- `floor` : profondeur actuelle.
- `kills` : ennemis tues pendant la run.
- `rooms` : salles terminees pendant la run.
- `floors` : etages termines pendant la run.
- `time` : temps ecoule dans la run.
- `gold` : or disponible.
- `item_count` : nombre d'objets deja obtenus.
- `room_type` : type de salle (`start`, `combat`, `shop`, `treasure`, etc.).
- `room_cleared` : indique si la salle courante est consideree terminee.
- `abyss_gate` : indique la presence d'une porte abyss optionnelle.
- `doors_open` : indique si les portes de la salle sont ouvertes.
- `portal_active` : indique si le portail d'etage est actif.
- `enemy_count` : nombre d'ennemis actifs.
- `nearest_enemy` : ennemi actif le plus proche, avec position, HP, type et distance.
- `nearest_pickup` : pickup le plus proche, avec type, rarete d'objet et distance.
- `nearest_shop_item` : objet ou soin achetable le plus proche, avec prix.
- `nearest_door` : porte ouverte la plus proche.
- `near_choice` : objet de salle item actuellement selectionnable avec `E`.
- `portal` : position et distance du portail actif apres boss.
- `available_doors` : portes ouvertes disponibles, triees par distance.
- `pickups` : liste courte des pickups visibles.

Ce choix evite un apprentissage depuis pixels bruts, trop couteux pour deux
jours, et permet de mesurer rapidement si l'agent fait mieux que le hasard.

## Actions de l'agent

Les actions disponibles dans le wrapper sont volontairement simples :

- `noop`
- `up`
- `down`
- `left`
- `right`
- `shoot_up`
- `shoot_down`
- `shoot_left`
- `shoot_right`
- `dash`
- `up_shoot_up`
- `down_shoot_down`
- `left_shoot_left`
- `right_shoot_right`
- `interact`
- deplacements diagonaux
- combinaisons deplacement + tir independant, par exemple `left_shoot_right`

La baseline aleatoire conserve l'ancien espace d'actions simple pour rester
reproductible. Les actions enrichies servent surtout au futur agent entraine et
a la politique heuristique.

Une politique heuristique de reference est disponible :

```bash
.venv/bin/python scripts/play_heuristic.py --watch --episodes 1 --max-steps 120
```

Elle n'est pas l'agent entraine. Elle sert de guide de comportement : viser
l'ennemi proche, garder ses distances, ramasser les pickups utiles, interagir
avec les objets/shops accessibles, puis aller vers les portes ou portails.

Cas speciaux deja pris en compte par cette heuristique :

- item room / treasure room : aller vers l'objet propose et valider avec `E`
  quand le choix est a portee.
- shop : acheter seulement si l'or suffit et si l'achat est utile, par exemple
  soin uniquement quand il manque de la vie.
- boss room : combattre comme une salle normale, puis aller au portail central
  et valider avec `E` quand le boss est vaincu.
- salles risquees (`altar`, `defi`, `scelle`, `gambler`) : ne pas declencher
  volontairement d'interaction optionnelle pour eviter de perdre une run de demo.
- abyss gate : detectee, mais non priorisee pour la premiere version car c'est
  un risque optionnel.

## Recompense et score proposes

Pour commencer simple, le score principal sera base sur la progression mesurable
de la run :

```text
score = floors * 100 + rooms * 10 + kills + time * 0.1
```

Ce score garde un signal meme si l'agent meurt vite : survivre un peu rapporte,
nettoyer une salle rapporte davantage, et changer d'etage rapporte beaucoup.

La recompense utilisee par `step(action)` mesure la difference de score entre
deux observations et ajoute une penalite si le joueur perd de la vie :

```text
reward = delta_score - hp_lost * 2
```

La baseline demandee par la consigne sera mesuree avec l'agent aleatoire sur le
meme score et le meme nombre de parties que l'agent entraine.

## Baseline aleatoire

Le wrapper a ete accelere (une seule lecture d'etat par step au lieu de trois,
soit environ 4 steps par seconde au lieu de 2). Comme le jeu tourne en temps
reel, ce changement modifie le temps de jeu couvert par un meme nombre de
steps : la baseline a donc ete re-mesuree sur l'environnement accelere pour
garder une comparaison honnete. Autre effet du changement : la somme des
rewards d'un episode correspond maintenant exactement au score final moins les
HP perdus (avant, ce qui se passait entre deux steps n'etait compte nulle part).

Baseline re-mesuree le 20 aout 2026 avec :

```bash
.venv/bin/python scripts/baseline_random.py --episodes 20 --max-steps 100 --seed 42
```

Resultats :

- Episodes : 20.
- Budget : 100 steps maximum par episode.
- Seed : 42.
- Score moyen : 1.836.
- Score minimum : 1.742.
- Score maximum : 2.252.
- Ecart-type : 0.132.
- Reward moyenne : 1.799.
- Kills : 0 sur tous les episodes.
- Salles terminees : 0 sur tous les episodes.

Conclusion : l'agent aleatoire survit quelques secondes mais ne progresse pas
dans le donjon avec ce budget. Le premier agent entraine devra etre compare sur
le meme nombre d'episodes et le meme budget de steps.

Donnees : `reports/random_baseline.csv`.

## Methode d'apprentissage

Algo retenu : **Q-learning tabulaire sur etat discretise**. Pourquoi celui-la
pour ce jeu-la : l'environnement tourne en temps reel dans un navigateur
(~4-5 steps par seconde, pas de GPU), donc le budget total d'apprentissage est
d'environ 30 000 steps — le deep RL est hors budget d'echantillons. Les
observations structurees se discretisent naturellement, et une Q-table se
sauvegarde en JSON de quelques dizaines de Ko, rechargeable trivialement et
inspectable a l'oeil nu.

L'etat compresse chaque observation en une cle compacte (~420 combinaisons
possibles, ~96 reellement rencontrees) : direction de l'ennemi le plus proche
(8 secteurs), sa distance (proche/moyen/loin, seuils 170/430 repris de
l'heuristique), vie basse ou non, direction de la porte la plus proche quand la
salle est vide, pickup utile a portee, interaction possible.

Trois mecanismes completent le Q-learning de base :

- **Exploration guidee** : pendant l'exploration, 60 % des coups sont joues par
  l'heuristique (`src/policies.py`) au lieu du pur hasard. Le Q-learning etant
  off-policy, il apprend de ces demonstrations ; en evaluation l'heuristique est
  totalement debranchee, seule la Q-table joue.
- **Shaping de porte** (entrainement seulement) : se rapprocher de la sortie
  d'une salle vide donne un petit signal continu (~+0.15/step) qui mene au
  +10/salle, trop rare pour etre decouvert seul. Le score de comparaison ne
  change pas.
- **Sauvegarde du meilleur** : eval greedy (epsilon=0) de 3 episodes tous les
  25 episodes ; si la moyenne bat le record, l'agent est sauvegarde dans
  `models/best_agent.json`.

Hyperparametres calcules depuis le budget d'echantillons (~9 visites par paire
etat-action) : alpha 0.2, gamma 0.97 (horizon ~33 steps, un trajet vers une
porte), epsilon 1.0 -> 0.10 atteint a 70 % du run, penalite HP 2.0.

Commande du run complet :

```bash
python scripts/train.py --episodes 300 --max-steps 100 --seed 42 --gamma 0.97 --epsilon-min 0.1 --guide-ratio 0.6 --door-shaping 0.005
```

## Resultats de l'entrainement

Run de 300 episodes x 100 steps (~1 h 45), scores dans
`reports/training_scores.csv` (une ligne par episode, train et eval).

Evals greedy (moyenne de 3 episodes, meme score que la baseline) :

| Episode | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 | 225 | 250 | 275 | 300 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Score | 6.46 | 1.73 | 5.99 | **11.39** | 7.62 | 2.19 | 2.66 | 2.16 | 2.56 | 2.29 | 5.69 | 2.34 |

- Meilleur agent (eval@100) : score moyen **11.39**, soit **6x la baseline
  aleatoire (1.836)**. Deux parties sur trois terminent une salle et font
  3 a 5 kills — ce que l'aleatoire n'a jamais fait une seule fois.
- Pendant l'entrainement : 615 kills et 101 salles terminees en 300 episodes.
- Le meilleur agent est conserve dans `models/best_agent.json` (rechargement
  verifie depuis un script neuf).

## Demo de l'agent entraine

Le script recharge `models/best_agent.json` depuis un script neuf, sans
relancer l'entrainement, et ouvre un navigateur visible :

```bash
python scripts/watch_agent.py --episodes 3
```

C'est la commande a utiliser pour la video de demonstration.

## Essais et difficultes (pour le carnet d'essais)

1. **V1 aveugle aux portes** : la premiere version de l'etat n'encodait pas la
   direction de la sortie. En 50 episodes, zero salle terminee et un plafond
   d'eval a 2.29 : dans une salle vide, tous les points donnaient le meme etat,
   l'agent ne pouvait structurellement pas apprendre a sortir. Correctif mesure :
   direction de porte dans l'etat + exploration guidee -> premiere salle
   terminee en eval des l'episode 25 du run suivant.
2. **Politique froussarde temporaire** : a l'eval@50, l'agent survivait 100
   steps sans un seul kill (score ~1.7) — les grosses penalites HP des morts
   recentes avaient temporairement ecrase les valeurs de combat apprises des
   demonstrations. Resolu de lui-meme avec plus d'episodes (pic a 11.39 a
   l'episode 100).
3. **Oscillation avec alpha constant** : apres le pic de l'episode 100, les
   evals oscillent (7.62 -> 2.19 -> ... -> 5.69 -> 2.34) sans converger : avec
   alpha fixe a 0.2, la table court toujours apres ses dernieres experiences.
   Le mecanisme "on garde le meilleur" protege le livrable. Avec plus de temps :
   alpha decroissant, davantage d'episodes d'eval par checkpoint (3 est bruite
   dans des donjons aleatoires non seedables), et un petit DQN sur les memes
   features pour generaliser entre etats voisins.
