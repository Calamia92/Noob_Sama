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
    "shoot_up": ["ArrowUp"],
    "shoot_down": ["ArrowDown"],
    "shoot_left": ["ArrowLeft"],
    "shoot_right": ["ArrowRight"],
    "dash": ["Space"],
    "up_shoot_up": ["KeyW", "ArrowUp"],
    "down_shoot_down": ["KeyS", "ArrowDown"],
    "left_shoot_left": ["KeyA", "ArrowLeft"],
    "right_shoot_right": ["KeyD", "ArrowRight"],
}


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

    def close(self) -> None:
        if self._context:
            self._context.close()
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()
        self._context = None
        self._browser = None
        self._playwright = None
        self.page = None

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

        before = self.observe()
        self._hold_keys(ACTIONS[action], self.step_seconds)
        after = self.observe()
        reward = self._reward(before, after)
        return after, reward, self.is_done(), {"action": action}

    def observe(self) -> Observation:
        data = self._eval_game(
            """() => {
                const g = window.__eosGame;
                const p = g.player || {};
                const s = g.stats || {};
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
                };
            }"""
        )
        return Observation(**data)

    def get_score(self) -> float:
        obs = self.observe()
        return obs.floors * 100 + obs.rooms * 10 + obs.kills + obs.time * 0.1

    def is_done(self) -> bool:
        return self.observe().state in {"gameover", "victory"}

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
            "const game = new Game(view);",
            "const game = new Game(view); window.__eosGame = game;",
            1,
        )
        route.fulfill(
            response=response,
            body=patched,
            headers={**response.headers, "content-type": "application/javascript"},
        )
