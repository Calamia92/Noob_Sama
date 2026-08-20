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

On commence avec peu d'actions pour limiter l'espace de decision. D'autres
combinaisons pourront etre ajoutees seulement si la baseline et le premier
entrainement tournent correctement.

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

## Prochaine etape technique

Le wrapper d'environnement doit exposer une interface minimale :

```python
reset()
step(action)
observe()
get_score()
is_done()
```

Pour eviter l'OCR sur le canvas, la piste privilegiee est d'utiliser Playwright
et d'injecter une petite modification au chargement du jeu pour rendre l'instance
`Game` accessible depuis le navigateur, par exemple via `window.__eosGame`.
