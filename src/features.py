from __future__ import annotations

from src.eos_env import Observation


ROOM_W = 1280.0
ROOM_H = 720.0
DIAG = (ROOM_W**2 + ROOM_H**2) ** 0.5

ROOM_TYPES = [
    "start",
    "combat",
    "shop",
    "treasure",
    "boss",
    "altar",
    "defi",
    "scelle",
    "gambler",
]

# Distance covered in one step at full speed, used to normalise the motion
# feature (a value near zero while a move key is held means "stuck").
STEP_TRAVEL = 60.0


def _entity(obs: Observation, entity: dict | None, extra: list[float] | None = None) -> list[float]:
    """present flag + normalised offset and distance toward an entity."""
    if not entity:
        return [0.0, 0.0, 0.0, 0.0] + [0.0] * len(extra or [])
    values = [
        1.0,
        (entity["x"] - obs.x) / ROOM_W,
        (entity["y"] - obs.y) / ROOM_H,
        entity.get("distance", DIAG) / DIAG,
    ]
    return values + (extra or [])


def featurize(obs: Observation, previous: Observation | None = None) -> list[float]:
    """Flatten an observation into a normalised feature vector."""
    if obs.state != "play":
        return [0.0] * N_FEATURES

    features: list[float] = [
        obs.x / ROOM_W,
        obs.y / ROOM_H,
        obs.hp / obs.max_hp if obs.max_hp else 0.0,
        obs.hp_missing / obs.max_hp if obs.max_hp else 0.0,
        min(obs.gold, 200) / 200.0,
        min(obs.item_count, 12) / 12.0,
        float(obs.doors_open),
        float(obs.portal_active),
        float(obs.room_cleared),
        float(obs.abyss_gate),
        min(obs.enemy_count, 8) / 8.0,
        min(obs.pickup_count, 8) / 8.0,
    ]

    features.extend(1.0 if obs.room_type == kind else 0.0 for kind in ROOM_TYPES)

    for i in range(3):
        enemy = obs.enemies[i] if i < len(obs.enemies) else None
        extra = None
        if enemy:
            max_hp = enemy.get("max_hp") or 1
            extra = [
                enemy.get("hp", 0) / max_hp,
                1.0 if enemy.get("elite") or enemy.get("boss") else 0.0,
            ]
        else:
            extra = [0.0, 0.0]
        features.extend(_entity(obs, enemy, extra))

    for i in range(2):
        pickup = obs.pickups[i] if i < len(obs.pickups) else None
        extra = None
        if pickup:
            kind = pickup.get("type")
            extra = [
                1.0 if kind in {"heart", "heartHalf"} else 0.0,
                1.0 if kind == "item" else 0.0,
                1.0 if kind == "gold" else 0.0,
            ]
        else:
            extra = [0.0, 0.0, 0.0]
        features.extend(_entity(obs, pickup, extra))

    features.extend(_entity(obs, obs.target_door))
    features.extend(_entity(obs, obs.portal))

    choice = obs.near_choice
    features.extend([1.0 if choice else 0.0, choice["distance"] / DIAG if choice else 0.0])

    shop = obs.nearest_shop_item
    affordable = 1.0 if shop and obs.gold >= shop.get("price", 0) else 0.0
    features.extend(_entity(obs, shop, [affordable]))

    if previous is not None and previous.state == "play":
        travel = ((obs.x - previous.x) ** 2 + (obs.y - previous.y) ** 2) ** 0.5
        speed = min(travel / STEP_TRAVEL, 1.0)
        features.extend([speed, 1.0 if speed < 0.05 else 0.0])
    else:
        features.extend([0.0, 0.0])

    return features


def _count_features() -> int:
    from dataclasses import fields

    dummy_values = {
        "state": "play",
        "hp": 6,
        "max_hp": 6,
        "x": 0.0,
        "y": 0.0,
        "floor": 1,
        "kills": 0,
        "rooms": 0,
        "floors": 0,
        "time": 0.0,
        "hp_missing": 0,
        "gold": 0,
        "item_count": 0,
        "room_type": "start",
        "room_cleared": False,
        "abyss_gate": False,
        "doors_open": False,
        "portal_active": False,
        "enemy_count": 0,
        "pickup_count": 0,
        "nearest_enemy": None,
        "nearest_pickup": None,
        "nearest_shop_item": None,
        "nearest_door": None,
        "target_door": None,
        "near_choice": None,
        "portal": None,
        "available_doors": [],
        "pickups": [],
        "enemies": [],
    }
    assert {f.name for f in fields(Observation)} == set(dummy_values)
    return len(featurize(Observation(**dummy_values)))


N_FEATURES = 0
N_FEATURES = _count_features()
