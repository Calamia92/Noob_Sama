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


def heuristic_action(obs: Observation) -> str:
    """Small rule-based policy used as a smarter target than pure random."""
    if obs.state != "play":
        return "noop"

    if obs.near_choice and obs.near_choice["distance"] <= 58:
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
        if pickup["distance"] > 20:
            return _move_towards(obs, pickup)

    if obs.portal_active and obs.portal:
        if obs.portal["distance"] <= 65:
            return "interact"
        return _move_towards(obs, obs.portal)

    exit_door = obs.target_door or obs.nearest_door
    if obs.doors_open and exit_door:
        return _move_towards(obs, exit_door)

    enemy = obs.nearest_enemy
    if enemy:
        shot = _shoot_towards(obs, enemy)
        if enemy["distance"] < 170:
            return _combine(_move_away(obs, enemy), shot)
        if enemy["distance"] > 430:
            return _combine(_move_towards(obs, enemy), shot)
        return shot

    return "noop"


def _best_pickup(obs: Observation) -> dict | None:
    if not obs.pickups:
        return None

    candidates = []
    for pickup in obs.pickups:
        kind = pickup.get("type")
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
    if abs(dx) > abs(dy):
        return "shoot_right" if dx > 0 else "shoot_left"
    return "shoot_down" if dy > 0 else "shoot_up"


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
