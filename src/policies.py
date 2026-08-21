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


def heuristic_action(obs: Observation) -> str:
    """Small rule-based policy used as a smarter target than pure random."""
    if obs.state != "play":
        return "noop"

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

    if obs.portal_active and obs.portal:
        if obs.portal["distance"] <= 65:
            return "interact"
        return _move_towards(obs, obs.portal)

    exit_door = obs.target_door or obs.nearest_door
    if obs.doors_open and exit_door:
        return _move_towards(obs, exit_door)

    pickup = _best_pickup(obs)
    if pickup:
        if pickup["type"] == "item" and pickup.get("choice") and pickup["distance"] <= 55:
            return "interact"
        if pickup["distance"] > 20:
            return _move_towards(obs, pickup)

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


def _combat_action(obs: Observation, enemy: dict) -> str:
    shot = _shoot_towards(obs, enemy)
    distance = enemy.get("distance", 9999)

    if distance < 190:
        if distance < 105:
            return _dash_away(obs, enemy)
        return _combine(_move_away(obs, enemy), shot)

    hp_low = obs.hp * 2 <= obs.max_hp
    crowded = obs.enemy_count >= 2
    if hp_low or crowded or distance < 330:
        return _combine(_defensive_move(obs, enemy), shot)

    if distance > 470:
        return _combine(_move_towards(obs, enemy), shot)

    return shot


def _dash_away(obs: Observation, target: dict) -> str:
    move = _move_away(obs, target)
    return "dash" if move == "noop" else "dash_" + move


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
