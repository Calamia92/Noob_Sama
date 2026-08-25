from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright


GAME_URL = "https://html-classic.itch.zone/html/18868512/index.html?v=1787145254"

# The game simulates with a fixed 120 Hz timestep (see its core/loop.js).
SIM_RATE = 120


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
    "shoot_up_left": ["ArrowUp", "ArrowLeft"],
    "shoot_up_right": ["ArrowUp", "ArrowRight"],
    "shoot_down_left": ["ArrowDown", "ArrowLeft"],
    "shoot_down_right": ["ArrowDown", "ArrowRight"],
    "dash": ["Space"],
    "dash_up": ["KeyW", "Space"],
    "dash_down": ["KeyS", "Space"],
    "dash_left": ["KeyA", "Space"],
    "dash_right": ["KeyD", "Space"],
    "dash_up_left": ["KeyW", "KeyA", "Space"],
    "dash_up_right": ["KeyW", "KeyD", "Space"],
    "dash_down_left": ["KeyS", "KeyA", "Space"],
    "dash_down_right": ["KeyS", "KeyD", "Space"],
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
    "up_shoot_up_left": ["KeyW", "ArrowUp", "ArrowLeft"],
    "up_shoot_up_right": ["KeyW", "ArrowUp", "ArrowRight"],
    "up_shoot_down_left": ["KeyW", "ArrowDown", "ArrowLeft"],
    "up_shoot_down_right": ["KeyW", "ArrowDown", "ArrowRight"],
    "down_shoot_up_left": ["KeyS", "ArrowUp", "ArrowLeft"],
    "down_shoot_up_right": ["KeyS", "ArrowUp", "ArrowRight"],
    "down_shoot_down_left": ["KeyS", "ArrowDown", "ArrowLeft"],
    "down_shoot_down_right": ["KeyS", "ArrowDown", "ArrowRight"],
    "left_shoot_up_left": ["KeyA", "ArrowUp", "ArrowLeft"],
    "left_shoot_up_right": ["KeyA", "ArrowUp", "ArrowRight"],
    "left_shoot_down_left": ["KeyA", "ArrowDown", "ArrowLeft"],
    "left_shoot_down_right": ["KeyA", "ArrowDown", "ArrowRight"],
    "right_shoot_up_left": ["KeyD", "ArrowUp", "ArrowLeft"],
    "right_shoot_up_right": ["KeyD", "ArrowUp", "ArrowRight"],
    "right_shoot_down_left": ["KeyD", "ArrowDown", "ArrowLeft"],
    "right_shoot_down_right": ["KeyD", "ArrowDown", "ArrowRight"],
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


# Installed at document start so every page load exposes the same one-call
# state snapshot, shared by the real-time and turbo stepping modes.
OBSERVE_SETUP_JS = """
window.__eosObserve = () => {
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
                attack: e.attackName || null,
                attack_step: e.atkStep ?? 0,
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

    // Door leading toward the best unvisited room (BFS over the dungeon
    // graph, i.e. the minimap). Rooms are ranked by type so the run gears
    // up (treasure, shop) before facing the guardian, which is always
    // routed last.
    let targetDoor = nearestDoor;
    let targetRoomType = null;
    let targetRoomHops = 0;
    if (availableDoors.length && g.dungeon?.nodes && node) {
        const dirs = { up: [0, -1], right: [1, 0], down: [0, 1], left: [-1, 0] };
        const key = (x, y) => x + ',' + y;
        const seen = new Set([key(node.gx, node.gy)]);
        const queue = [[node.gx, node.gy, null, 0]];
        const PRIORITY = { treasure: 5, shop: 4, miniboss: 2, boss: 1 };
        let best = null;
        while (queue.length) {
            const [cx, cy, firstDir, hops] = queue.shift();
            const cur = g.dungeon.nodes.get(key(cx, cy));
            if (!cur) continue;
            if (!cur.visited && firstDir) {
                const prio = PRIORITY[cur.type] ?? 3;
                const score = prio * 100 - hops;
                if (!best || score > best.score) {
                    best = { score, firstDir, type: cur.type || null, hops };
                }
                continue;
            }
            for (const [dir, [dx, dy]] of Object.entries(dirs)) {
                if (!cur.doors || !cur.doors[dir]) continue;
                const nk = key(cx + dx, cy + dy);
                if (seen.has(nk)) continue;
                seen.add(nk);
                queue.push([cx + dx, cy + dy, firstDir || dir, hops + 1]);
            }
        }
        if (best) {
            targetDoor = availableDoors.find((d) => d.dir === best.firstDir) || nearestDoor;
            targetRoomType = best.type;
            targetRoomHops = best.hops;
        }
    }

    // Enemies still materialising (spawn window): invisible to the active
    // list, yet the right moment to pre-position instead of freezing.
    const spawningEnemies = [];
    if (enemies) {
        for (let i = 0; i < enemies.count; i++) {
            const e = enemies.items[i];
            if (!e || e.hp <= 0 || !(e.spawnTimer > 0)) continue;
            spawningEnemies.push({
                x: e.x ?? 0,
                y: e.y ?? 0,
                distance: dist(e.x ?? 0, e.y ?? 0),
            });
        }
    }
    spawningEnemies.sort((a, b) => a.distance - b.distance);

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

    // Enemy projectiles and telegraphed blast zones: both are dodgeable
    // by movement, provided the agent can actually see them.
    const activeShots = [];
    const shots = dbg.enemyShots;
    if (shots) {
        for (let i = 0; i < shots.count; i++) {
            const sh = shots.items[i];
            if (!sh) continue;
            activeShots.push({
                x: sh.x ?? 0,
                y: sh.y ?? 0,
                vx: sh.vx ?? 0,
                vy: sh.vy ?? 0,
                radius: sh.radius ?? 6,
                distance: dist(sh.x ?? 0, sh.y ?? 0),
            });
        }
    }
    activeShots.sort((a, b) => a.distance - b.distance);

    const activeHazards = [];
    if (dbg.hazards) {
        for (const hz of dbg.hazards) {
            activeHazards.push({
                x: hz.x,
                y: hz.y,
                r: hz.r,
                remaining: Math.max(0, (hz.fuse ?? 0) - (hz.t ?? 0)),
                distance: dist(hz.x, hz.y),
            });
        }
    }
    activeHazards.sort((a, b) => a.distance - b.distance);

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
        enemies: activeEnemies.slice(0, 3),
        shots: activeShots.slice(0, 6),
        hazards: activeHazards.slice(0, 6),
        spawning: spawningEnemies.slice(0, 3),
        target_room_type: targetRoomType,
        target_room_hops: targetRoomHops,
    };
};
"""


# One synchronous env step: press the keys, advance the fixed-step
# simulation, release, optionally draw, and read the new state — a single
# browser round-trip instead of key events plus a real-time sleep.
STEP_TURBO_JS = """({ codes, frames, render }) => {
    for (const code of codes) window.dispatchEvent(new KeyboardEvent('keydown', { code }));
    window.__eosAdvance(frames, 1 / 120);
    for (const code of codes) window.dispatchEvent(new KeyboardEvent('keyup', { code }));
    if (render) window.__eosRender();
    return window.__eosObserve();
}"""


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
    enemies: list[dict[str, Any]]
    shots: list[dict[str, Any]]
    hazards: list[dict[str, Any]]
    spawning: list[dict[str, Any]]
    target_room_type: str | None
    target_room_hops: int


class EclipseEnv:
    def __init__(
        self,
        *,
        headless: bool = True,
        slow_mo: int = 0,
        step_seconds: float = 0.15,
        turbo: bool = False,
    ) -> None:
        self.headless = headless
        self.slow_mo = slow_mo
        self.step_seconds = step_seconds
        # Turbo drives the game's fixed-step loop synchronously instead of
        # holding keys in real time: same simulated duration per step, but
        # the wall-clock cost drops to the simulation itself.
        self.turbo = turbo
        self._frames_per_step = max(1, round(step_seconds * SIM_RATE))
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
        # ignore_https_errors: tolerate TLS-inspecting proxies whose root
        # certificate Chromium does not know (blocks goto and route.fetch).
        self._context = self._browser.new_context(
            viewport={"width": 1280, "height": 720},
            ignore_https_errors=True,
        )
        if self.turbo:
            self._context.add_init_script("window.__eosTurbo = true;")
        self._context.add_init_script(OBSERVE_SETUP_JS)
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
        if self.turbo:
            self._advance_to_state(("heroSelect", "play"))
            self._advance_to_state(("play",))
            self._advance(int(0.3 * SIM_RATE))
        else:
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
        if self.turbo:
            after = self._step_turbo(ACTIONS[action])
        else:
            self._hold_keys(ACTIONS[action], self.step_seconds)
            after = self.observe()
        reward = self._reward(before, after)
        done = after.state in TERMINAL_STATES
        return after, reward, done, {"action": action}

    def observe(self) -> Observation:
        data = self._eval_game("() => window.__eosObserve()")
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

    def _step_turbo(self, codes: list[str]) -> Observation:
        data = self._require_page().evaluate(
            STEP_TURBO_JS,
            {
                "codes": codes,
                "frames": self._frames_per_step,
                "render": not self.headless,
            },
        )
        obs = Observation(**data)
        self._last_obs = obs
        return obs

    def _advance(self, frames: int) -> None:
        self._require_page().evaluate(
            "(frames) => window.__eosAdvance(frames, 1 / 120)", frames
        )

    def _advance_to_state(self, states: tuple[str, ...], *, max_frames: int = 2400) -> None:
        # Menus only react while the simulation runs, so tap the key and
        # advance in small chunks until one of the requested states shows up.
        page = self._require_page()
        advanced = 0
        while True:
            current = page.evaluate("() => window.__eosGame.state")
            if current in states:
                return
            if advanced >= max_frames:
                raise TimeoutError(f"Expected game state {states!r}, got {current!r}.")
            page.evaluate(
                """(code) => {
                    window.dispatchEvent(new KeyboardEvent('keydown', { code }));
                    window.__eosAdvance(2, 1 / 120);
                    window.dispatchEvent(new KeyboardEvent('keyup', { code }));
                    window.__eosAdvance(10, 1 / 120);
                }""",
                "Space",
            )
            advanced += 12

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
            if self._require_page().evaluate(
                "() => !!window.__eosGame && (!window.__eosTurbo || !!window.__eosAdvance)"
            ):
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
            "import { pickups } from './game/pickup.js';\n"
            "import { enemyShots } from './game/enemyshot.js';\n"
            "import { hazards } from './game/boss.js';\n"
            "import { consommer } from './core/hitstop.js';",
            1,
        )
        patched = patched.replace(
            "const game = new Game(view);",
            "const game = new Game(view); "
            "window.__eosGame = game; "
            "window.__eosDebug = { enemies, pickups, enemyShots, hazards };",
            1,
        )
        # Audio adds startup cost and has no effect on the simulation.
        patched = patched.replace(
            "initInput(() => { initAudio(); startMusic(); });",
            "initInput(window.__eosTurbo ? null : () => { initAudio(); startMusic(); });",
            1,
        )
        # Expose the fixed-step loop so turbo mode can drive the simulation
        # synchronously (same hitstop behaviour as the real loop). Real-time
        # mode keeps the original requestAnimationFrame loop.
        patched = patched.replace(
            "  startLoop({\n"
            "    update: (dt) => { pollGamepad(); game.update(dt); },\n"
            "    frameEnd: () => endFrame(),\n"
            "    render: () => game.render(ctx),\n"
            "  });",
            "  const loopHooks = {\n"
            "    update: (dt) => { pollGamepad(); game.update(dt); },\n"
            "    frameEnd: () => endFrame(),\n"
            "    render: () => game.render(ctx),\n"
            "  };\n"
            "  window.__eosAdvance = (frames, dt) => {\n"
            "    for (let i = 0; i < frames; i += 1) {\n"
            "      if (!consommer()) loopHooks.update(dt);\n"
            "    }\n"
            "    loopHooks.frameEnd();\n"
            "  };\n"
            "  window.__eosRender = () => loopHooks.render(0);\n"
            "  if (!window.__eosTurbo) startLoop(loopHooks);",
            1,
        )
        route.fulfill(
            response=response,
            body=patched,
            headers={**response.headers, "content-type": "application/javascript"},
        )
