from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright


GAME_URL = "https://html-classic.itch.zone/html/18868512/index.html?v=1787145254"


ACTIONS: dict[str, list[str]] = {
    "noop": [],
    "up": ["KeyW"],
    "down": ["KeyS"],
    "left": ["KeyA"],
    "right": ["KeyD"],
    "up_left": ["KeyW", "KeyA"],
    "up_right": ["KeyW", "KeyD"],
    "down_left": ["KeyS", "KeyA"],
    "down_right": ["KeyS", "KeyD"],
    "shoot_up": ["ArrowUp"],
    "shoot_down": ["ArrowDown"],
    "shoot_left": ["ArrowLeft"],
    "shoot_right": ["ArrowRight"],
    "dash": ["Space"],
    "interact": ["KeyE"],
    "up_shoot_up": ["KeyW", "ArrowUp"],
    "up_shoot_down": ["KeyW", "ArrowDown"],
    "up_shoot_left": ["KeyW", "ArrowLeft"],
    "up_shoot_right": ["KeyW", "ArrowRight"],
    "down_shoot_up": ["KeyS", "ArrowUp"],
    "down_shoot_down": ["KeyS", "ArrowDown"],
    "down_shoot_left": ["KeyS", "ArrowLeft"],
    "down_shoot_right": ["KeyS", "ArrowRight"],
    "left_shoot_up": ["KeyA", "ArrowUp"],
    "left_shoot_down": ["KeyA", "ArrowDown"],
    "left_shoot_left": ["KeyA", "ArrowLeft"],
    "left_shoot_right": ["KeyA", "ArrowRight"],
    "right_shoot_up": ["KeyD", "ArrowUp"],
    "right_shoot_down": ["KeyD", "ArrowDown"],
    "right_shoot_left": ["KeyD", "ArrowLeft"],
    "right_shoot_right": ["KeyD", "ArrowRight"],
}


TERMINAL_STATES = {"gameover", "victory"}


RANDOM_BASELINE_ACTIONS = [
    "noop",
    "up",
    "down",
    "left",
    "right",
    "shoot_up",
    "shoot_down",
    "shoot_left",
    "shoot_right",
    "dash",
    "up_shoot_up",
    "down_shoot_down",
    "left_shoot_left",
    "right_shoot_right",
]


@dataclass(frozen=True)
class Observation:
    state: str
    hp: float
    max_hp: float
    x: float
    y: float
    floor: int
    kills: int
    rooms: int
    floors: int
    time: float
    hp_missing: float
    gold: int
    item_count: int
    room_type: str | None
    room_cleared: bool
    abyss_gate: bool
    doors_open: bool
    portal_active: bool
    enemy_count: int
    pickup_count: int
    nearest_enemy: dict[str, Any] | None
    nearest_pickup: dict[str, Any] | None
    nearest_shop_item: dict[str, Any] | None
    nearest_door: dict[str, Any] | None
    target_door: dict[str, Any] | None
    near_choice: dict[str, Any] | None
    portal: dict[str, Any] | None
    available_doors: list[dict[str, Any]]
    pickups: list[dict[str, Any]]


class EclipseEnv:
    def __init__(
        self,
        *,
        headless: bool = True,
        slow_mo: int = 0,
        step_seconds: float = 0.15,
    ) -> None:
        self.headless = headless
        self.slow_mo = slow_mo
        self.step_seconds = step_seconds
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self.page: Page | None = None
        self._last_obs: Observation | None = None

    def __enter__(self) -> "EclipseEnv":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def start(self) -> None:
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo,
        )
        self._context = self._browser.new_context(viewport={"width": 1280, "height": 720})
        self.page = self._context.new_page()
        self.page.route("**/js/main.js", self._patch_main_js)
        self._last_obs = None

    def close(self) -> None:
        # Tolerant teardown so a crashed browser can be restarted cleanly.
        for closer in (
            lambda: self._context.close() if self._context else None,
            lambda: self._browser.close() if self._browser else None,
            lambda: self._playwright.stop() if self._playwright else None,
        ):
            try:
                closer()
            except Exception:
                pass
        self._context = None
        self._browser = None
        self._playwright = None
        self.page = None
        self._last_obs = None

    def reset(self) -> Observation:
        page = self._require_page()
        page.goto(GAME_URL, wait_until="domcontentloaded")
        self._wait_for_game()
        page.keyboard.press("Space")
        self._wait_for_state("heroSelect")
        page.keyboard.press("Space")
        self._wait_for_state("play")
        time.sleep(0.3)
        return self.observe()

    def step(self, action: str) -> tuple[Observation, float, bool, dict[str, Any]]:
        if action not in ACTIONS:
            raise ValueError(f"Unknown action {action!r}. Available actions: {sorted(ACTIONS)}")

        # Reuse the previous snapshot as "before": a single evaluate per step
        # roughly doubles the real-time step rate.
        before = self._last_obs if self._last_obs is not None else self.observe()
        self._hold_keys(ACTIONS[action], self.step_seconds)
        after = self.observe()
        reward = self._reward(before, after)
        done = after.state in TERMINAL_STATES
        return after, reward, done, {"action": action}

    def observe(self) -> Observation:
        data = self._eval_game(
            """() => {
                const g = window.__eosGame;
                const dbg = window.__eosDebug || {};
                const p = g.player || {};
                const s = g.stats || {};
                const room = g.room || null;
                const node = g.node || null;

                const dist = (x, y) => Math.hypot((p.x ?? 0) - x, (p.y ?? 0) - y);
                const activeEnemies = [];
                const enemies = dbg.enemies;
                if (enemies) {
                    for (let i = 0; i < enemies.count; i++) {
                        const e = enemies.items[i];
                        if (!e || e.hp <= 0 || e.spawnTimer > 0) continue;
                        activeEnemies.push({
                            x: e.x ?? 0,
                            y: e.y ?? 0,
                            hp: e.hp ?? 0,
                            max_hp: e.maxHp ?? 0,
                            type: e.type?.id || e.type?.name || null,
                            state: e.state || null,
                            elite: !!e.elite,
                            boss: !!e.type?.boss,
                            distance: dist(e.x ?? 0, e.y ?? 0),
                        });
                    }
                }
                activeEnemies.sort((a, b) => a.distance - b.distance);

                const activePickups = [];
                const pickups = dbg.pickups;
                if (pickups) {
                    for (let i = 0; i < pickups.count; i++) {
                        const item = pickups.items[i];
                        if (!item) continue;
                        activePickups.push({
                            x: item.x ?? 0,
                            y: item.y ?? 0,
                            type: item.type || null,
                            value: item.value ?? 0,
                            choice: !!item.choice,
                            item_name: item.item?.name || null,
                            item_rarity: item.item?.rarity || null,
                            item_desc: item.item?.desc || null,
                            distance: dist(item.x ?? 0, item.y ?? 0),
                        });
                    }
                }
                activePickups.sort((a, b) => a.distance - b.distance);

                let availableDoors = [];
                let nearestDoor = null;
                if (room?.node?.doors && room.doorsOpen) {
                    const centers = {
                        up: { x: room.w / 2, y: 0 },
                        down: { x: room.w / 2, y: room.h },
                        left: { x: 0, y: room.h / 2 },
                        right: { x: room.w, y: room.h / 2 },
                    };
                    availableDoors = Object.entries(room.node.doors)
                        .filter(([, open]) => !!open)
                        .map(([dir]) => ({
                            dir,
                            x: centers[dir].x,
                            y: centers[dir].y,
                            distance: dist(centers[dir].x, centers[dir].y),
                        }))
                        .sort((a, b) => a.distance - b.distance);
                    nearestDoor = availableDoors[0] || null;
                }

                // Door leading toward the closest unvisited room (BFS over the
                // dungeon graph). Prevents ping-ponging between cleared rooms.
                let targetDoor = nearestDoor;
                if (availableDoors.length && g.dungeon?.nodes && node) {
                    const dirs = { up: [0, -1], right: [1, 0], down: [0, 1], left: [-1, 0] };
                    const key = (x, y) => x + ',' + y;
                    const seen = new Set([key(node.gx, node.gy)]);
                    const queue = [[node.gx, node.gy, null]];
                    let bestDir = null;
                    while (queue.length) {
                        const [cx, cy, firstDir] = queue.shift();
                        const cur = g.dungeon.nodes.get(key(cx, cy));
                        if (!cur) continue;
                        if (!cur.visited && firstDir) { bestDir = firstDir; break; }
                        for (const [dir, [dx, dy]] of Object.entries(dirs)) {
                            if (!cur.doors || !cur.doors[dir]) continue;
                            const nk = key(cx + dx, cy + dy);
                            if (seen.has(nk)) continue;
                            seen.add(nk);
                            queue.push([cx + dx, cy + dy, firstDir || dir]);
                        }
                    }
                    if (bestDir) {
                        targetDoor = availableDoors.find((d) => d.dir === bestDir) || nearestDoor;
                    }
                }

                let nearestShopItem = null;
                if (node?.type === 'shop' && node.shopStock && room?.shopSlots) {
                    const slots = room.shopSlots();
                    const shopItems = node.shopStock
                        .map((slot, i) => {
                            if (!slot || slot.sold || !slots[i]) return null;
                            return {
                                x: slots[i].x,
                                y: slots[i].y,
                                kind: slot.kind || null,
                                price: slot.price ?? 0,
                                item_name: slot.item?.name || null,
                                item_rarity: slot.item?.rarity || null,
                                item_desc: slot.item?.desc || null,
                                distance: dist(slots[i].x, slots[i].y),
                            };
                        })
                        .filter(Boolean)
                        .sort((a, b) => a.distance - b.distance);
                    nearestShopItem = shopItems[0] || null;
                }

                const nearChoice = g.nearChoice ? {
                    x: g.nearChoice.x ?? 0,
                    y: g.nearChoice.y ?? 0,
                    type: g.nearChoice.type || null,
                    value: g.nearChoice.value ?? 0,
                    choice: !!g.nearChoice.choice,
                    item_name: g.nearChoice.item?.name || null,
                    item_rarity: g.nearChoice.item?.rarity || null,
                    item_desc: g.nearChoice.item?.desc || null,
                    distance: dist(g.nearChoice.x ?? 0, g.nearChoice.y ?? 0),
                } : null;

                const portal = room?.portalActive ? {
                    x: room.w / 2,
                    y: room.h / 2,
                    distance: dist(room.w / 2, room.h / 2),
                } : null;

                return {
                    state: g.state,
                    hp: p.hp ?? 0,
                    max_hp: p.maxHp ?? 0,
                    x: p.x ?? 0,
                    y: p.y ?? 0,
                    floor: g.floorNum ?? 0,
                    kills: s.kills ?? 0,
                    rooms: s.rooms ?? 0,
                    floors: s.floors ?? 0,
                    time: s.time ?? 0,
                    hp_missing: Math.max(0, (p.maxHp ?? 0) - (p.hp ?? 0)),
                    gold: p.gold ?? 0,
                    item_count: p.items?.length ?? 0,
                    room_type: node?.type || null,
                    room_cleared: !!node?.cleared,
                    abyss_gate: !!node?.abyssGate,
                    doors_open: !!room?.doorsOpen,
                    portal_active: !!room?.portalActive,
                    enemy_count: activeEnemies.length,
                    pickup_count: activePickups.length,
                    nearest_enemy: activeEnemies[0] || null,
                    nearest_pickup: activePickups[0] || null,
                    nearest_shop_item: nearestShopItem,
                    nearest_door: nearestDoor,
                    target_door: targetDoor,
                    near_choice: nearChoice,
                    portal,
                    available_doors: availableDoors,
                    pickups: activePickups.slice(0, 8),
                };
            }"""
        )
        obs = Observation(**data)
        self._last_obs = obs
        return obs

    @staticmethod
    def compute_score(obs: Observation) -> float:
        return obs.floors * 100 + obs.rooms * 10 + obs.kills + obs.time * 0.1

    def get_score(self) -> float:
        obs = self._last_obs if self._last_obs is not None else self.observe()
        return self.compute_score(obs)

    def is_done(self) -> bool:
        obs = self._last_obs if self._last_obs is not None else self.observe()
        return obs.state in TERMINAL_STATES

    def _hold_keys(self, keys: list[str], seconds: float) -> None:
        page = self._require_page()
        for key in keys:
            page.keyboard.down(key)
        time.sleep(seconds)
        for key in reversed(keys):
            page.keyboard.up(key)

    def _reward(self, before: Observation, after: Observation) -> float:
        return (
            (after.floors - before.floors) * 100
            + (after.rooms - before.rooms) * 10
            + (after.kills - before.kills)
            + max(0.0, after.time - before.time) * 0.1
            - max(0.0, before.hp - after.hp) * 2
        )

    def _eval_game(self, expression: str) -> Any:
        page = self._require_page()
        return page.evaluate(expression)

    def _wait_for_game(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._require_page().evaluate("() => !!window.__eosGame"):
                return
            time.sleep(0.05)
        raise TimeoutError("Game instance was not exposed as window.__eosGame.")

    def _wait_for_state(self, state: str, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            current = self._require_page().evaluate("() => window.__eosGame && window.__eosGame.state")
            if current == state:
                return
            time.sleep(0.05)
        current = self._require_page().evaluate("() => window.__eosGame && window.__eosGame.state")
        raise TimeoutError(f"Expected game state {state!r}, got {current!r}.")

    def _require_page(self) -> Page:
        if self.page is None:
            raise RuntimeError("Environment is not started. Use EclipseEnv as a context manager.")
        return self.page

    @staticmethod
    def _patch_main_js(route: Any) -> None:
        response = route.fetch()
        source = response.text()
        patched = source.replace(
            "import { Game } from './game/game.js';",
            "import { Game } from './game/game.js';\n"
            "import { enemies } from './game/enemy.js';\n"
            "import { pickups } from './game/pickup.js';",
            1,
        )
        patched = patched.replace(
            "const game = new Game(view);",
            "const game = new Game(view); "
            "window.__eosGame = game; "
            "window.__eosDebug = { enemies, pickups };",
            1,
        )
        route.fulfill(
            response=response,
            body=patched,
            headers={**response.headers, "content-type": "application/javascript"},
        )
