from __future__ import annotations

from src.eos_env import Observation


RARITY_RANK = {
    None: 0,
    "common": 1,
    "rare": 2,
    "epic": 3,
    "legendary": 4,
}

RISKY_ROOM_TYPES = {"altar", "defi", "scelle", "gambler"}
COMBAT_PICKUP_RANGE = 250.0
ROOM_WIDTH = 1280.0
ROOM_HEIGHT = 720.0
WALL_MARGIN = 120.0

# Dodging thresholds. Enemy shots are slow orbs meant to be dodged by
# movement; hazards are telegraphed blast circles that resolve when their
# fuse runs out.
SHOT_THREAT_HORIZON = 0.8
SHOT_THREAT_RADIUS = 30.0
HAZARD_MARGIN = 28.0
HAZARD_DASH_WINDOW = 0.35
TELEGRAPH_STATES = {"aim", "windup", "prime"}


def heuristic_action(obs: Observation) -> str:
    """Small rule-based policy used as a smarter target than pure random."""
    if obs.state != "play":
        return "noop"

    # Survival first: leave telegraphed blast zones and sidestep incoming
    # projectiles before any other objective.
    escape = _hazard_escape(obs)
    if escape:
        return escape
    dodge = _shot_dodge(obs)
    if dodge:
        return dodge

    if (
        obs.near_choice
        and obs.near_choice["distance"] <= 58
        and obs.room_type not in RISKY_ROOM_TYPES
    ):
        return "interact"

    shop = obs.nearest_shop_item
    if shop and obs.room_type not in RISKY_ROOM_TYPES and _shop_item_is_useful(obs, shop):
        if shop["distance"] <= 58:
            return "interact"
        return _move_towards(obs, shop)

    enemy = obs.nearest_enemy
    if enemy:
        pickup = _best_pickup(obs)
        if pickup:
            return _move_towards(obs, pickup)
        return _combat_action(obs, enemy)

    # Loot the cleared room before leaving: gold, hearts and items stay on
    # the floor otherwise (the doors open as soon as the fight ends).
    pickup = _best_pickup(obs)
    if pickup:
        if pickup["type"] == "item" and pickup.get("choice") and pickup["distance"] <= 55:
            return "interact"
        if pickup["distance"] > 20:
            return _move_towards(obs, pickup)

    if obs.portal_active and obs.portal:
        if obs.portal["distance"] <= 65:
            return "interact"
        return _move_towards(obs, obs.portal)

    exit_door = obs.target_door or obs.nearest_door
    if obs.doors_open and exit_door:
        return _move_towards(obs, exit_door)

    return "noop"


def should_delegate_action(obs: Observation, action: str) -> bool:
    if obs.state != "play":
        return False

    if obs.enemy_count == 0 and (obs.doors_open or obs.portal_active):
        return True

    if obs.nearest_enemy and (obs.enemy_count >= 3 or obs.hp * 2 <= obs.max_hp):
        return True

    if obs.nearest_enemy and obs.enemy_count >= 2 and _is_static_combat_action(action):
        return True

    return False


def resolve_intent(obs: Observation, intent: str) -> str:
    if intent in {
        "noop",
        "up",
        "down",
        "left",
        "right",
        "up_left",
        "up_right",
        "down_left",
        "down_right",
        "shoot_up",
        "shoot_down",
        "shoot_left",
        "shoot_right",
        "shoot_up_left",
        "shoot_up_right",
        "shoot_down_left",
        "shoot_down_right",
        "dash",
        "dash_up",
        "dash_down",
        "dash_left",
        "dash_right",
        "dash_up_left",
        "dash_up_right",
        "dash_down_left",
        "dash_down_right",
    } or "_shoot_" in intent:
        return intent

    if intent == "heuristic":
        return heuristic_action(obs)
    if intent == "fight":
        return _fight_intent(obs)
    if intent == "kite":
        return _kite_intent(obs)
    if intent == "dash_away":
        return _dash_away_intent(obs)
    if intent == "exit":
        return _exit_intent(obs)
    if intent == "loot":
        return _loot_intent(obs)
    if intent == "interact":
        return "interact"
    if intent == "wait":
        return "noop"
    return heuristic_action(obs)


def _is_static_combat_action(action: str) -> bool:
    return action == "noop" or action == "dash" or action.startswith("shoot_")


def _fight_intent(obs: Observation) -> str:
    enemy = obs.nearest_enemy
    if enemy:
        return _combine(_defensive_move(obs, enemy), _shoot_towards(obs, enemy))
    return _exit_intent(obs)


def _kite_intent(obs: Observation) -> str:
    enemy = obs.nearest_enemy
    if enemy:
        return _combine(_move_away(obs, enemy), _shoot_towards(obs, enemy))
    return _exit_intent(obs)


def _dash_away_intent(obs: Observation) -> str:
    enemy = obs.nearest_enemy
    if enemy:
        return _dash_away(obs, enemy)
    return _exit_intent(obs)


def _exit_intent(obs: Observation) -> str:
    if obs.portal_active and obs.portal:
        if obs.portal["distance"] <= 65:
            return "interact"
        return _move_towards(obs, obs.portal)

    door = obs.target_door or obs.nearest_door
    if obs.doors_open and door:
        return _move_towards(obs, door)
    return heuristic_action(obs)


def _loot_intent(obs: Observation) -> str:
    if obs.near_choice and obs.near_choice["distance"] <= 58 and obs.room_type not in RISKY_ROOM_TYPES:
        return "interact"

    shop = obs.nearest_shop_item
    if shop and obs.room_type not in RISKY_ROOM_TYPES and _shop_item_is_useful(obs, shop):
        if shop["distance"] <= 58:
            return "interact"
        return _move_towards(obs, shop)

    pickup = _best_pickup(obs)
    if pickup:
        if pickup["type"] == "item" and pickup.get("choice") and pickup["distance"] <= 55:
            return "interact"
        return _move_towards(obs, pickup)
    return _exit_intent(obs)


def _best_pickup(obs: Observation) -> dict | None:
    if not obs.pickups:
        return None

    candidates = []
    for pickup in obs.pickups:
        kind = pickup.get("type")
        if obs.enemy_count > 0:
            if kind not in {"heart", "heartHalf"} or obs.hp_missing <= 0:
                continue
            if pickup.get("distance", 9999) > COMBAT_PICKUP_RANGE:
                continue
        if kind in {"heart", "heartHalf"} and obs.hp_missing <= 0:
            continue
        priority = {
            "heart": 100 if obs.hp_missing > 0 else 0,
            "heartHalf": 90 if obs.hp_missing > 0 else 0,
            "item": 70 + RARITY_RANK.get(pickup.get("item_rarity"), 0) * 5,
            "gold": 30,
        }.get(kind, 0)
        if priority:
            candidates.append((priority - pickup.get("distance", 9999) * 0.01, pickup))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def _shop_item_is_useful(obs: Observation, shop: dict) -> bool:
    if obs.gold < shop.get("price", 0):
        return False
    if shop.get("kind") in {"heart", "heartHalf"}:
        return obs.hp_missing > 0
    if shop.get("kind") == "item":
        return True
    return False


def _move_towards(obs: Observation, target: dict) -> str:
    return _move_from_delta(target["x"] - obs.x, target["y"] - obs.y)


def _move_away(obs: Observation, target: dict) -> str:
    return _move_from_delta(obs.x - target["x"], obs.y - target["y"])


def _shoot_towards(obs: Observation, target: dict) -> str:
    dx = target["x"] - obs.x
    dy = target["y"] - obs.y
    if abs(dx) > 45 and abs(dy) > 45:
        vertical = "down" if dy > 0 else "up"
        horizontal = "right" if dx > 0 else "left"
        return f"shoot_{vertical}_{horizontal}"
    if abs(dx) > abs(dy):
        return "shoot_right" if dx > 0 else "shoot_left"
    return "shoot_down" if dy > 0 else "shoot_up"


def _wall_room(x: float, y: float) -> float:
    return min(x, ROOM_WIDTH - x, y, ROOM_HEIGHT - y)


_DIRS = {
    "up": (0.0, -1.0),
    "down": (0.0, 1.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "up_left": (-0.707, -0.707),
    "up_right": (0.707, -0.707),
    "down_left": (-0.707, 0.707),
    "down_right": (0.707, 0.707),
}


def _best_direction(obs: Observation, preferred: list[tuple[float, float]]) -> str:
    """Pick the escape direction that also stays clear of every known
    threat (hazard circles, nearby enemies, walls), instead of blindly
    following the geometric ideal into another danger."""
    units = []
    for pdx, pdy in preferred:
        norm = (pdx * pdx + pdy * pdy) ** 0.5
        if norm > 1e-6:
            units.append((pdx / norm, pdy / norm))
    step = 90.0

    def score(ux: float, uy: float) -> float:
        nx, ny = obs.x + ux * step, obs.y + uy * step
        value = 2.0 * max((ux * px + uy * py for px, py in units), default=0.0)
        wall = _wall_room(nx, ny)
        if wall < 40:
            value -= 3.0
        elif wall < WALL_MARGIN:
            value -= 1.0
        for hazard in obs.hazards:
            clear = ((nx - hazard["x"]) ** 2 + (ny - hazard["y"]) ** 2) ** 0.5 - hazard["r"]
            if clear < HAZARD_MARGIN:
                value -= 4.0
            elif clear < 90:
                value -= 1.0
        for enemy in obs.enemies:
            d = ((nx - enemy["x"]) ** 2 + (ny - enemy["y"]) ** 2) ** 0.5
            if d < 70:
                value -= 3.0
            elif d < 130:
                value -= 1.0
        return value

    return max(_DIRS, key=lambda name: score(*_DIRS[name]))


def _hazard_escape(obs: Observation) -> str | None:
    for hazard in obs.hazards:
        if hazard["distance"] > hazard["r"] + HAZARD_MARGIN:
            continue
        dx = obs.x - hazard["x"]
        dy = obs.y - hazard["y"]
        if abs(dx) < 1 and abs(dy) < 1:
            dx = ROOM_WIDTH / 2 - obs.x
            dy = ROOM_HEIGHT / 2 - obs.y
        move = _best_direction(obs, [(dx, dy)])
        if hazard["remaining"] < HAZARD_DASH_WINDOW:
            return "dash_" + move
        enemy = obs.nearest_enemy
        if enemy and "_" not in move:
            return f"{move}_{_shoot_towards(obs, enemy)}"
        return move
    return None


def _shot_dodge(obs: Observation) -> str | None:
    for shot in obs.shots:
        rx = obs.x - shot["x"]
        ry = obs.y - shot["y"]
        vx = shot.get("vx", 0.0)
        vy = shot.get("vy", 0.0)
        speed_sq = vx * vx + vy * vy
        if speed_sq <= 1e-6:
            continue
        t_close = (rx * vx + ry * vy) / speed_sq
        if t_close < 0 or t_close > SHOT_THREAT_HORIZON:
            continue
        cx = rx - vx * t_close
        cy = ry - vy * t_close
        if cx * cx + cy * cy > SHOT_THREAT_RADIUS * SHOT_THREAT_RADIUS:
            continue
        # Sidestep perpendicular to the shot's path, whichever side is
        # clear of walls, hazards and other enemies.
        move = _best_direction(obs, [(-vy, vx), (vy, -vx)])
        enemy = obs.nearest_enemy
        if enemy and "_" not in move:
            return f"{move}_{_shoot_towards(obs, enemy)}"
        return move
    return None


def _strafe_move(obs: Observation, enemy: dict) -> str:
    ex = enemy["x"] - obs.x
    ey = enemy["y"] - obs.y
    return _best_direction(obs, [(-ey, ex), (ey, -ex)])


def _boss_attack_response(obs: Observation, enemy: dict, shot: str) -> str | None:
    """Pattern-specific answers to the guardians' signature attacks."""
    if enemy.get("state") != "attack":
        return None
    attack = enemy.get("attack")
    distance = enemy.get("distance", 9999)
    step = enemy.get("attack_step", 0)

    if attack in {"tripleLunge", "heavyCharge"}:
        if attack == "heavyCharge" and step >= 2:
            # Stunned against a wall: the free-damage window.
            if distance > 320:
                return _combine(_move_towards(obs, enemy), shot)
            return shot
        # Aiming or mid-lunge: the charge follows the boss-player line, so
        # keep clearing it sideways, with a dash when the maw is close
        # (each lunge covers ~160 px, walking alone barely clears it).
        move = _strafe_move(obs, enemy)
        if distance < 260:
            return "dash_" + move if move != "noop" else _dash_away(obs, enemy)
        if "_" not in move:
            return f"{move}_{shot}"
        return move

    if attack == "slamQuake":
        # Bodyslam blast (radius 140) resolves the instant it lands: the
        # only reliable dodge is distance during the wind-up.
        if distance < 220:
            move = _move_away(obs, enemy)
            if "_" not in move and move != "noop":
                return f"{move}_{shot}"
            return move if move != "noop" else _dash_away(obs, enemy)
        return shot

    return None


def _combat_action(obs: Observation, enemy: dict) -> str:
    shot = _shoot_towards(obs, enemy)
    distance = enemy.get("distance", 9999)
    boss = bool(enemy.get("boss"))

    if boss:
        response = _boss_attack_response(obs, enemy, shot)
        if response:
            return response

    # A telegraphed attack aims at the player's current position: strafing
    # sideways breaks the line before the enemy commits.
    if enemy.get("state") in TELEGRAPH_STATES and distance < 420:
        move = _strafe_move(obs, enemy)
        if "_" not in move:
            return f"{move}_{shot}"
        return move
    if enemy.get("state") == "dash" and distance < 260:
        move = _strafe_move(obs, enemy)
        return "dash_" + move if move != "noop" else _dash_away(obs, enemy)

    # Bosses hit harder on contact and lunge further: keep more distance.
    near, retreat, defend, engage = (150, 260, 420, 560) if boss else (105, 190, 330, 470)
    if distance < retreat:
        if distance < near:
            return _dash_away(obs, enemy)
        return _combine(_move_away(obs, enemy), shot)

    hp_low = obs.hp * 2 <= obs.max_hp
    crowded = obs.enemy_count >= 2
    if hp_low or crowded or distance < defend:
        return _combine(_defensive_move(obs, enemy), shot)

    if distance > engage:
        return _combine(_move_towards(obs, enemy), shot)

    return shot


def _dash_away(obs: Observation, target: dict) -> str:
    move = _move_away(obs, target)
    if move == "noop":
        # Overlapping the enemy: the away-vector collapses to zero, so pick
        # any escape direction instead of dashing in place.
        move = _move_towards_center_if_near_wall(obs)
        if move == "noop":
            move = "up" if obs.y > ROOM_HEIGHT / 2 else "down"
    return "dash_" + move


def _defensive_move(obs: Observation, enemy: dict) -> str:
    wall_escape = _move_towards_center_if_near_wall(obs)
    if wall_escape != "noop":
        return wall_escape

    dx = enemy["x"] - obs.x
    dy = enemy["y"] - obs.y
    if abs(dx) > abs(dy):
        return "up" if obs.y > ROOM_HEIGHT / 2 else "down"
    return "left" if obs.x > ROOM_WIDTH / 2 else "right"


def _move_towards_center_if_near_wall(obs: Observation) -> str:
    if obs.x < WALL_MARGIN:
        return "right"
    if obs.x > ROOM_WIDTH - WALL_MARGIN:
        return "left"
    if obs.y < WALL_MARGIN:
        return "down"
    if obs.y > ROOM_HEIGHT - WALL_MARGIN:
        return "up"
    return "noop"


def _combine(move: str, shoot: str) -> str:
    if move == "noop":
        return shoot
    if "_" in move:
        # Keep the action space compact: diagonal movement does not get a
        # shooting combo in the default action set.
        return shoot
    return f"{move}_{shoot}"


def _move_from_delta(dx: float, dy: float) -> str:
    deadzone = 24
    horizontal = "right" if dx > deadzone else "left" if dx < -deadzone else ""
    vertical = "down" if dy > deadzone else "up" if dy < -deadzone else ""
    if vertical and horizontal:
        return f"{vertical}_{horizontal}"
    return vertical or horizontal or "noop"
