"""Terraforming Mars - multiplayer runner for the heuristic bot.

Drives one or more bot processes against each other or against a human on the
open-source Terraforming Mars server. The bot's judgement lives in tm_bot.py;
this module only talks to the server, decides whose turn it is and posts the
answers that tm_bot.decide() produces.

Main modes:
    --vs-human      create a two-player game and play against a human
    --join          join an existing game via its player id
    --ab-crn        paired A/B match against a frozen champion module
    --auto-games    unattended self-play series

Requires: requests, tm_bot.py and card_db.json next to this file.
"""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

import requests

# from tm_bot: the actual judgement
from tm_bot import (
    CARD_DB, MC_RESERVE,
    load_card_db,
    get_state, post_input,
    handle_or, decide,
    score_card, score_action,
    turns_left, can_convert_plants, can_convert_heat,
    choose_best_space, build_payment,
    param_progress_from_state,
    _score_milestone, _score_award,
)
try:
    from tm_model import game_state_features, card_features
except ImportError:
    def game_state_features(state): return []
    def card_features(info, **kw): return []

# search helpers from tm_mcts
try:
    import shadow_analyze as _shadow
except Exception:
    _shadow = None

from tm_mcts import (
    rollout_score, decide_rollout,
    take_snapshot, restore_snapshot,
    get_last_save_id, rollback_to_save,
    wait_for_db, resolve_game_id,
    ROLLOUT_MOVES, MAX_CANDIDATES, MCTS_MIN_DELTA,
    POLL_INTERVAL, POST_WAIT, ERROR_WAIT, MAX_ERRORS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mcts_mp")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_URL      = "http://localhost:9000"
LOCK_FILE        = "tm_mcts_lock.json"   # in the working directory
LOCK_TIMEOUT     = 90.0   # seconds until a crashed bot loses the lock
LOCK_POLL        = 0.5   # seconds between lock polls
MCTS_DATA_FILE   = "mcts_data_mp.jsonl"

# colours the server knows
VALID_COLORS = ["red", "blue", "yellow", "green", "black", "purple"]


# ---------------------------------------------------------------------------
# MCTSLock - atomic process coordination through a file
# ---------------------------------------------------------------------------

class MCTSLock:
    """Coordinates several bot processes through an atomic lock file.

    Only one process at a time may run a search with rollbacks, otherwise the
    processes would see each other's temporary states.
    """

    def __init__(self, my_color: str, game_id: str, lock_file: str = LOCK_FILE):
        self.my_color  = my_color
        self.game_id   = game_id
        self.lock_file = lock_file

    def acquire(self, save_id: int) -> bool:
        """Wait until the lock is free, then take it. Blocks until it is available
        or the timeout expires.
        """
        waited = 0.0
        while True:
            state = self._read()
            locked_by = state.get("locked_by")

            if locked_by is None:
                # free - try to take it
                self._write(save_id)
                # wait briefly and read again (race condition check)
                time.sleep(0.15)
                state2 = self._read()
                if state2.get("locked_by") == self.my_color:
                    log.debug("🔒 Lock erworben (%s, save=%d)", self.my_color, save_id)
                    return True
                # another bot was faster - keep waiting
                time.sleep(LOCK_POLL)
                waited += LOCK_POLL
                continue

            elif locked_by == self.my_color:
                # our own lock (e.g. after restarting from a crash)
                since = state.get("since", 0)
                if time.time() - since > LOCK_TIMEOUT:
                    log.warning("🔒 Eigener Timeout-Lock gefunden – neu erwerben")
                    self._write(save_id)
                return True

            else:
                # another bot holds the lock
                since = state.get("since", 0)
                age   = time.time() - since
                if age > LOCK_TIMEOUT:
                    log.warning("🔒 Lock-Timeout von '%s' (%.0fs) – übernehme",
                                locked_by, age)
                    self._write(save_id)
                    return True

                # wait
                if waited == 0:
                    log.info("⏳ Warte auf MCTS-Lock von '%s'...", locked_by)
                time.sleep(LOCK_POLL)
                waited += LOCK_POLL
                if waited > 0 and int(waited) % 10 == 0:
                    log.info("⏳ Warte seit %.0fs auf Lock von '%s'", waited, locked_by)

    def release(self):
        """Release the lock."""
        self._write_free()
        log.debug("🔓 Lock freigegeben (%s)", self.my_color)

    def is_locked_by_other(self) -> bool:
        """Is another bot inside the search right now?"""
        state = self._read()
        lb = state.get("locked_by")
        return lb is not None and lb != self.my_color

    def _write(self, save_id: int):
        data = {
            "locked_by": self.my_color,
            "game_id":   self.game_id,
            "save_id":   save_id,
            "since":     time.time(),
        }
        self._atomic_write(data)

    def _write_free(self):
        data = {
            "locked_by": None,
            "game_id":   self.game_id,
            "save_id":   -1,
            "since":     time.time(),
        }
        self._atomic_write(data)

    def _atomic_write(self, data: dict):
        """Write atomically via a temporary file plus os.replace()."""
        tmp = self.lock_file + f".tmp.{self.my_color}"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, self.lock_file)
        except Exception as e:
            log.warning("Lock-Schreib-Fehler: %s", e)
            try:
                os.unlink(tmp)
            except Exception:
                pass

    def _read(self) -> dict:
        try:
            with open(self.lock_file, encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"locked_by": None}


# ---------------------------------------------------------------------------
# Creating a multiplayer game
# ---------------------------------------------------------------------------

# The three OFFICIAL boards. Deliberately no community or Pathfinders boards: the bot
# knows the milestones and awards of these three completely (_milestone_gap covers
# Tharsis, Elysium and Hellas, and so does _AWARD_KEYS). On the others it would be
# blind and give away 5-15 VP in milestones and awards.
OFFICIAL_BOARDS = ("tharsis", "hellas", "elysium")


COLORS_TO_NAMES = {
    "red":    "Bot-Rot",
    "blue":   "Bot-Blau",
    "yellow": "Bot-Gelb",
    "green":  "Bot-Grün",
    "black":  "Bot-Schwarz",
    "purple": "Bot-Lila",
}


def create_mp_game_with_undo(
    base_url: str,
    n_players: int,
    colors: list[str] | None = None,
    draft: bool = False,
    seed: float | None = None,
    random_first: bool = True,
    cloned_game_id: str | None = None,
    board: str = "tharsis",
    human_color: str | None = None,
    human_name: str = "apehead",
    expansions: set | None = None,
    settings: dict | None = None,
) -> dict[str, str]:
    """Create a multiplayer game with undoOption enabled for n_players players.

    Undo is what makes the rollback-based search possible at all: a probe move
    can be played and taken back again.
    """
    if colors is None:
        colors = VALID_COLORS[:n_players]

    # The server reads ONLY `players[i].first` to decide who starts; the payload field
    # `randomFirstPlayer` exists in the CLIENT only (the web UI rolls the dice itself and
    # then sets `first`), so sending it has no effect. Hardcoding `first: (i == 0)` made
    # the first colour in the list the starting player every time.
    # The starting player rotates each generation afterwards, but generation 1 - and with
    # an odd final generation one more - would systematically go to the same seat.
    # IMPORTANT for the paired A/B: with a seed set, the starting player is DERIVED from
    # it, so both sides of a pair get the same seat. Only without a seed is it really
    # random. random_first=False keeps fixed seats.
    if random_first:
        _first_idx = (random.Random(f"first:{seed}").randrange(len(colors))
                      if seed is not None else random.randrange(len(colors)))
    else:
        _first_idx = 0

    players = [
        {
            "name":      (human_name if c == human_color
                          else COLORS_TO_NAMES.get(c, f"Bot-{c}")),
            "color":     c,
            "beginner":  False,
            "handicap":  0,
            "first":     (i == _first_idx),
        }
        for i, c in enumerate(colors)
    ]

    _exp = expansions or set()
    payload = {
        "players": players,
        "expansions": {
            "corpera": True,
            "promo": "promo" in _exp, "venus": "venus" in _exp, "colonies": "colonies" in _exp,
            "prelude": "prelude" in _exp, "prelude2": "prelude2" in _exp,
            "turmoil": "turmoil" in _exp, "community": "community" in _exp,
            "ares": "ares" in _exp, "moon": "moon" in _exp,
            "pathfinders": "pathfinders" in _exp, "ceo": "ceo" in _exp,
            "starwars": "starwars" in _exp, "underworld": "underworld" in _exp,
            "deltaProject": "deltaProject" in _exp,
        },
        "board": board,
        "seed": (seed if seed is not None else random.random()),
        "randomFirstPlayer": random_first,
        "undoOption": True,
        "showTimers": False,
        "fastModeOption": False,
        "showOtherPlayersVP": True,   # visible in multiplayer for better rollouts
        "aresExtremeVariant": False,
        "politicalAgendasExtension": "Standard",
        "solarPhaseOption": False,
        "removeNegativeGlobalEventsOption": False,
        "modularMA": False,
        "draftVariant": draft,
        "initialDraft": False,
        "preludeDraftVariant": False,
        "ceosDraftVariant": False,
        "startingCorporations": 2,
        "shuffleMapOption": False,
        "randomMA": "No randomization",
        "includeFanMA": False,
        "soloTR": False,
        "customCorporationsList": [], "bannedCards": [], "includedCards": [],
        "customColoniesList": [], "customPreludes": [],
        "requiresMoonTrackCompletion": False,
        "requiresVenusTrackCompletion": False,
        "moonStandardProjectVariant": False,
        "moonStandardProjectVariant1": False,
        "altVenusBoard": False,
        "twoCorpsVariant": False,
        "customCeos": [], "startingCeos": 3, "startingPreludes": 4,
    }

    # ── Take over the real round settings (tm_settings.json). Without this the runner
    # creates games with DIFFERENT options than the real round (board, fast mode, fan
    # milestones, number of corporations ...) and the bot is tested under the wrong
    # conditions. 'players' always stays the runner's list (bot plus human); everything
    # else (expansions, bannedCards, customCorporationsList, board, randomMA, includeFanMA,
    # fastModeOption, shuffleMapOption, startingCorporations/Ceos) comes from the file.
    # An explicit --expansions overrides the expansions from the file.
    if settings:
        _skip = {"players", "seed"}
        for k, v in settings.items():
            if k in _skip:
                continue
            payload[k] = v
        if expansions:   # --expansions takes precedence over the file
            payload["expansions"] = {
                "corpera": True,
                **{m: (m in _exp) for m in (
                    "promo", "venus", "colonies", "prelude", "prelude2", "turmoil",
                    "community", "ares", "moon", "pathfinders", "ceo", "starwars",
                    "underworld", "deltaProject")},
            }
        # numbers are sometimes strings in the file ("4") - the server expects int
        for _k in ("startingCorporations", "startingCeos", "startingPreludes"):
            if isinstance(payload.get(_k), str) and payload[_k].isdigit():
                payload[_k] = int(payload[_k])
        payload["seed"] = seed if seed is not None else random.random()

    # Deterministic deck: clone an existing game (exact shuffle order).
    if cloned_game_id:
        payload["clonedGamedId"] = cloned_game_id

    # The updated server needs noticeably longer for the draft setup; several concurrent
    # creategame requests (--parallel) exceeded the old timeout and the whole pair was
    # lost. Hence a longer timeout plus retry with backoff. Creation is idempotent enough
    # (a new game id per attempt); only with cloned_game_id is care needed - there the
    # same clone is requested again rather than rolled anew.
    data = None
    last_exc = None
    for attempt in range(4):
        try:
            r = requests.post(f"{base_url}/api/creategame", json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            break
        except requests.RequestException as _e:
            last_exc = _e
            time.sleep(2.0 * (attempt + 1))   # 2s, 4s, 6s
    if data is None:
        raise last_exc

    game_id    = data["id"]
    color_to_id = {p["color"]: p["id"] for p in data["players"]}

    log.info("🎮 Spiel erstellt: %s | %d Spieler: %s",
             game_id, n_players, list(color_to_id.keys()))

    # wait until the game is in the database
    for _ in range(20):
        time.sleep(1.0)
        try:
            check = requests.put(
                f"{base_url}/load_game",
                json={"gameId": game_id, "rollbackCount": 0},
                timeout=5,
            )
            if check.status_code == 200:
                log.info("   Spiel in DB verfügbar")
                break
        except Exception:
            pass

    return game_id, color_to_id


# ---------------------------------------------------------------------------
# Coordination file for unattended game creation
# ---------------------------------------------------------------------------

GAME_COORD_FILE = "tm_mp_game.json"

READY_FILE    = "tm_mp_ready.json"
COLOR_REG_FILE = "tm_mp_colors.json"   # colour registration


def find_card_db() -> str:
    """Find card_db.json in the working directory or a parent directory."""
    import os
    candidates = [
        "card_db.json",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "card_db.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "card_db.json nicht gefunden. Bitte mit build_card_db.py erzeugen."
    )


def register_color(allowed_colors: list[str] | None = None, timeout: float = 10.0) -> str:
    """Register this bot process and return a free colour.

    allowed restricts the choice; the registration file coordinates several
    processes that start at the same time.
    """
    import time as _time
    candidates = allowed_colors if allowed_colors else VALID_COLORS
    start = _time.time()
    while _time.time() - start < timeout:
        try:
            # read the current state
            try:
                with open(COLOR_REG_FILE, encoding="utf-8") as f:
                    reg = json.load(f)
                # clean up stale registrations (> 5 min)
                reg = {c: ts for c, ts in reg.items()
                       if _time.time() - ts < 300}
            except (FileNotFoundError, json.JSONDecodeError):
                reg = {}

            # find the next free colour among those allowed
            for color in candidates:
                if color not in reg:
                    # try to reserve this colour
                    reg[color] = _time.time()
                    tmp = COLOR_REG_FILE + f".{color}.tmp"
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(reg, f)
                    os.replace(tmp, COLOR_REG_FILE)
                    # wait briefly and check that we really hold it
                    _time.sleep(0.2)
                    with open(COLOR_REG_FILE, encoding="utf-8") as f:
                        reg2 = json.load(f)
                    if reg2.get(color) == reg[color]:
                        log.info("🎨 Farbe automatisch zugeteilt: %s", color)
                        return color
                    # another bot was faster - try again
                    break
        except Exception as e:
            log.warning("Farb-Registrierung Fehler: %s", e)
        _time.sleep(0.3)

    raise RuntimeError("Konnte keine freie Farbe registrieren")


def release_color(color: str):
    """Release the registered colour after the game."""
    try:
        with open(COLOR_REG_FILE, encoding="utf-8") as f:
            reg = json.load(f)
        reg.pop(color, None)
        tmp = COLOR_REG_FILE + f".{color}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(reg, f)
        os.replace(tmp, COLOR_REG_FILE)
    except Exception:
        pass


def signal_ready(my_color: str, game_num: int):
    """Signal that this bot is ready for the next game."""
    try:
        # read the current state
        try:
            with open(READY_FILE, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("game_num") != game_num:
                data = {"game_num": game_num, "ready": []}
        except (FileNotFoundError, json.JSONDecodeError):
            data = {"game_num": game_num, "ready": []}

        if my_color not in data["ready"]:
            data["ready"].append(my_color)

        tmp = READY_FILE + f".{my_color}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, READY_FILE)
        log.info("   ✅ Bereit für Partie %d signalisiert", game_num)
    except Exception as e:
        log.warning("   signal_ready Fehler: %s", e)


def wait_for_all_ready(all_colors: list, game_num: int, timeout: float = 60.0) -> bool:
    """Wait until all colours are ready. Returns True on success."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with open(READY_FILE, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("game_num") == game_num:
                ready = set(data.get("ready", []))
                missing = [c for c in all_colors if c not in ready]
                if not missing:
                    return True
                log.info("   ⏳ Warte auf: %s (Partie %d)", missing, game_num)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(1.0)
    log.warning("   Timeout beim Warten auf alle Bots")
    return False




def write_game_coord(game_id: str, color_to_id: dict[str, str], game_num: int = 0):
    """Write the game ids into the coordination file for the other bot processes."""
    data = {
        "game_id":     game_id,
        "color_to_id": color_to_id,
        "created_at":  time.time(),
        "game_num":    game_num,
    }
    tmp = GAME_COORD_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, GAME_COORD_FILE)
    log.info("📋 Spiel-IDs geschrieben: %s", GAME_COORD_FILE)


def _try_read_coord(my_color: str, game_num: int = 0, max_age: float = 30.0) -> tuple[str, str] | None:
    """Read the coordination file if it exists, is fresh and belongs to the right
    game number.
    """
    try:
        with open(GAME_COORD_FILE, encoding="utf-8") as f:
            data = json.load(f)
        age = time.time() - data.get("created_at", 0)
        # the game number has to match (0 = any)
        file_num = data.get("game_num", 0)
        if game_num > 0 and file_num != game_num:
            return None
        if age < max_age:
            player_id = data["color_to_id"].get(my_color)
            if player_id:
                return data["game_id"], player_id
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    return None


def read_game_coord(my_color: str, timeout: float = 60.0, game_num: int = 0) -> tuple[str, str] | None:
    """Read the game ids from the coordination file.
    Waits up to timeout seconds for it to appear.
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            with open(GAME_COORD_FILE, encoding="utf-8") as f:
                data = json.load(f)
            player_id = data["color_to_id"].get(my_color)
            if player_id:
                age      = time.time() - data.get("created_at", 0)
                file_num = data.get("game_num", 0)
                # check the game number (0 = any)
                num_ok = (game_num == 0 or file_num == game_num)
                if age < 300 and num_ok:
                    return data["game_id"], player_id
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass
        time.sleep(1.0)
        log.info("⏳ Warte auf Spiel-Koordinierungsdatei (%s)...", GAME_COORD_FILE)

    log.error("Koordinierungsdatei nicht gefunden nach %.0fs", timeout)
    return None


# ---------------------------------------------------------------------------
# Rollout in multiplayer: simulates opponent moves
# ---------------------------------------------------------------------------

def do_rollout_mp(
    base_url:    str,
    player_id:   str,
    all_player_ids: list[str],
    n_moves:     int,
    game_id:     str,
    simple:      bool = False,
) -> float:
    """Multiplayer rollout: play n_moves moves and return the rollout score.

    Opponent moves are simulated as well, so the resulting position is realistic
    rather than one where only this bot has acted.
    """
    moves_made = 0

    try:
        init_state = get_state(base_url, player_id)
        start_gen  = init_state.get("game", {}).get("generation", 1)
        start_save = get_last_save_id(base_url, game_id)
    except Exception:
        return 0.0

    for _ in range(n_moves):
        # are we to move?
        try:
            state = get_state(base_url, player_id)
        except Exception:
            break

        phase   = state.get("game", {}).get("phase", "")
        cur_gen = state.get("game", {}).get("generation", 1)

        if phase == "end":
            break
        if cur_gen != start_gen:
            break   # no moves across a generation boundary (undo is not possible there)

        is_active = state.get("thisPlayer", {}).get("isActive", False)
        waiting   = state.get("waitingFor")

        if is_active and waiting:
            wtype = waiting.get("type", "")
            if wtype == "player":
                # player selection - choose our own colour
                result = decide_rollout(state)
            else:
                result = decide_rollout(state)

            if result is None:
                break
            try:
                post_input(base_url, player_id, result)
                moves_made += 1
                time.sleep(POST_WAIT * 0.5)   # faster in a rollout
            except Exception:
                break

        elif not simple and not is_active and waiting is None:
            # an opponent is to move - simulate one move for each of them
            acted = False
            for opp_id in all_player_ids:
                if opp_id == player_id:
                    continue
                try:
                    opp_state = get_state(base_url, opp_id)
                    if not opp_state.get("thisPlayer", {}).get("isActive", False):
                        continue
                    opp_waiting = opp_state.get("waitingFor")
                    if not opp_waiting:
                        continue
                    opp_result = decide_rollout(opp_state)
                    if opp_result is None:
                        continue
                    post_input(base_url, opp_id, opp_result)
                    moves_made += 1
                    time.sleep(POST_WAIT * 0.5)
                    acted = True
                    break   # only one opponent per iteration
                except Exception:
                    continue
            if not acted:
                time.sleep(POLL_INTERVAL)
        else:
            time.sleep(POLL_INTERVAL)

    # measure the final score
    try:
        final_state = get_state(base_url, player_id)
        score = rollout_score(final_state)
    except Exception:
        score = 0.0

    # roll back to the starting position
    if moves_made > 0:
        time.sleep(0.3 + moves_made * 0.05)
        ok = rollback_to_save(base_url, game_id, start_save)
        if not ok:
            log.warning("  MP-Rollback fehlgeschlagen (%d Züge)", moves_made)

    return score


# ---------------------------------------------------------------------------
# Multiplayer search: evaluate candidates while holding the lock
# ---------------------------------------------------------------------------

def handle_or_mcts_mp(
    state:          dict,
    base_url:       str,
    player_id:      str,
    all_player_ids: list[str],
    lock:           MCTSLock,
    game_id:        str,
    n_rollouts:     int = ROLLOUT_MOVES,
    max_candidates: int = MAX_CANDIDATES,
    simple_rollout: bool = False,
) -> tuple[dict, float]:
    """Multiplayer version of the search over the top-level options, with locking.

    Sequence:
      1. take the lock
      2. remember the current save state
      3. play a probe move, roll out, undo
      4. repeat for every candidate, then decide
    """
    # base decision (fallback)
    raw_result = handle_or(state)
    if raw_result is None:
        return raw_result, 0.0

    # collect candidates (identical to the single-process version in tm_mcts.py)
    waiting  = state["waitingFor"]
    options  = waiting.get("options", [])
    player   = state["thisPlayer"]
    mc       = player.get("megacredits", 0)
    candidates = []

    for i, opt in enumerate(options):
        otype = opt.get("type", "")
        title = str(opt.get("title", "")).lower()

        if opt.get("buttonLabel") == "Undo":
            continue

        if otype == "option" and "pass" not in title and "undo" not in title:
            sc = score_action("heat", state) if "heat" in title else 5
            candidates.append((sc, {
                "type": "or", "runId": state["runId"], "index": i,
                "response": {"type": "option"},
            }))
        elif otype == "option":
            candidates.append((5, {
                "type": "or", "runId": state["runId"], "index": i,
                "response": {"type": "option"},
            }))
        elif otype == "space" and can_convert_plants(state):
            spaces = opt.get("spaces", [])
            if spaces:
                space_map = {s["id"]: s for s in state["game"]["spaces"]}
                best = choose_best_space(
                    spaces, space_map,
                    tile_type="greenery",
                    player_id=player.get("color"),
                )
                candidates.append((score_action("greenery", state), {
                    "type": "or", "runId": state["runId"], "index": i,
                    "response": {"type": "space", "spaceId": best},
                }))
        elif otype == "or" and "milestone" in str(opt.get("title", "")).lower():
            # nested milestone: {type:"or", title:"Claim a milestone", options:[...]}
            sc = _score_milestone(str(opt.get("title", "")), state)
            if sc > 0:
                sub_options = opt.get("options", [])
                if sub_options:
                    ms_name = sub_options[0].get("title", "?")
                    candidates.append((sc, {
                        "type": "or", "runId": state["runId"], "index": i,
                        "response": {
                            "type": "or", "index": 0,
                            "response": {"type": "option"},
                        },
                    }))
        elif otype == "or" and ("award" in str(opt.get("title", "")).lower() or "fund" in str(opt.get("title", "")).lower()):
            # nested award: {type:"or", title:"Fund an award", options:[...]}
            sc = _score_award(str(opt.get("title", "")), state)
            if sc > 0:
                sub_options = opt.get("options", [])
                if sub_options:
                    candidates.append((sc, {
                        "type": "or", "runId": state["runId"], "index": i,
                        "response": {
                            "type": "or", "index": 0,
                            "response": {"type": "option"},
                        },
                    }))
        elif otype == "projectCard":
            all_cards = opt.get("cards", [])
            SP_NAMES  = {"Aquifer", "Greenery", "City", "Power Plant:SP", "Asteroid:SP"}

            hand = [c for c in all_cards
                    if not c["name"].endswith(":SP") and c["name"] not in SP_NAMES]
            for c in hand:
                cost = c.get("calculatedCost", 0)
                if cost <= mc - MC_RESERVE:
                    sc = score_card(c, state)
                    candidates.append((sc, {
                        "type": "or", "runId": state["runId"], "index": i,
                        "response": {"type": "projectCard",
                                     "card": c["name"],
                                     "payment": build_payment(c, player)},
                    }))

            for sp_name, action in [("Aquifer", "ocean_sp"), ("Greenery", "greenery_sp"),
                                     ("Asteroid:SP", "temp_sp"), ("City", "city_sp")]:
                for c in all_cards:
                    if c["name"] == sp_name:
                        cost = c.get("calculatedCost", 999)
                        if cost <= mc - MC_RESERVE:
                            sc = score_action(action, state)
                            if sc > 0:
                                candidates.append((sc, {
                                    "type": "or", "runId": state["runId"], "index": i,
                                    "response": {"type": "projectCard",
                                                 "card": sp_name,
                                                 "payment": build_payment(c, player)},
                                }))

    if not candidates:
        return raw_result, 0.0

    candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = candidates[:max_candidates]

    # --- Fast, invisible leaf (one-ply feature prediction in Python).
    #     No server rollout or snapshot -> no browser false alarms, much faster.
    #     Switch: TM_FAST_LEAF=1 (uses value_model.joblib). ---
    import os as _os_fl
    if _os_fl.environ.get("TM_FAST_LEAF", "") not in ("", "0", "false", "False"):
        try:
            from learned_value import fast_leaf_decision
            _fres = fast_leaf_decision(state, top_candidates, raw_result, CARD_DB)
        except Exception as _e:
            _fres = None
            log.warning("  Fast-Leaf Fehler: %s", _e)
        if _fres is not None:
            best_payload, best_v, _scored = _fres
            log.info("  ⚡ Fast-Leaf (Python, kein Server-Rollout): %d Kandidaten", len(_scored))
            _wf = (state.get("waitingFor") or {}).get("options", [])
            for _v, _heur, _pl in _scored[:8]:
                _r = _pl.get("response", {})
                _nm = _r.get("card")
                if not _nm:
                    _ix = _pl.get("index")
                    _nm = (str(_wf[_ix].get("title", "?"))[:28]
                           if isinstance(_ix, int) and 0 <= _ix < len(_wf) else _r.get("type", "?"))
                log.info("      %+7.1f  %s", _v, _nm)
            try:
                best_payload["runId"] = state["runId"]
            except Exception:
                pass
            _chosen = (best_payload.get("response", {}).get("card")
                       or best_payload.get("response", {}).get("type", "?"))
            if _chosen == "option":
                _chosen = "Pass"
            log.info("  ✓ Fast-Leaf wählt: %s (v=%.1f)", _chosen, best_v)
            return best_payload, best_v
        elif not getattr(handle_or_mcts_mp, "_fastleaf_warned", False):
            handle_or_mcts_mp._fastleaf_warned = True
            log.warning("  ⚠ TM_FAST_LEAF gesetzt, aber gelerntes Modell NICHT geladen "
                        "→ falle auf Server-Rollout zurück (Störung!). "
                        "Prüfe value_model.joblib bzw. TM_VALUE_MODEL.")

    # --- take the lock ---
    current_save = get_last_save_id(base_url, game_id)
    lock.acquire(current_save)

    try:
        # sanity check: save_id has not changed while we were waiting
        save_after_lock = get_last_save_id(base_url, game_id)
        if save_after_lock != current_save:
            log.warning("  Save-ID verändert während Lock-Wartezeit (%d → %d) "
                        "– aktualisiere Basis",
                        current_save, save_after_lock)
            current_save = save_after_lock

        log.info("  🎲 MP-MCTS: %d Kandidaten × %d Rollout-Züge",
                 len(top_candidates), n_rollouts)

        base_score = rollout_score(state)
        results: list[tuple[float, float, dict]] = []
        best_payload = raw_result
        best_rollout = -999.0

        for heuristic_score, payload in top_candidates:
            resp = payload.get("response", {}) if isinstance(payload, dict) else {}
            display_name = resp.get("card", resp.get("type", "?"))
            if display_name == "option":
                display_name = "Pass"
            log.info("    → Teste: %s (h=%.1f)", display_name, heuristic_score)

            # remember the current save state (standard undo instead of a custom snapshot).
            # AFTER refreshing runId; save_id may have changed since taking the lock.
            snapshot_save = get_last_save_id(base_url, game_id)
            if snapshot_save < 0:
                log.warning("    Save-ID nicht abrufbar – MCTS übersprungen")
                break
            # verify we really are to move after the snapshot
            try:
                snap_check = get_state(base_url, player_id)
                if not snap_check.get("thisPlayer", {}).get("isActive", False):
                    log.warning("    Nach Snapshot nicht mehr aktiv – MCTS abbrechen")
                    break
            except Exception:
                pass

            # play the probe move - runId MUST be fresh
            try:
                try:
                    cur_state = get_state(base_url, player_id)
                    payload["runId"] = cur_state["runId"]
                except Exception as e:
                    log.warning("    runId-Refresh fehlgeschlagen: %s", e)

                post_input(base_url, player_id, payload)
                time.sleep(POST_WAIT)
            except Exception as e:
                log.warning("    Probe-Zug fehlgeschlagen: %s", e)
                rollback_to_save(base_url, game_id, snapshot_save)
                continue

            # rollout
            rollout_sc = do_rollout_mp(
                base_url, player_id, all_player_ids,
                n_rollouts, game_id, simple=simple_rollout,
            )
            delta = rollout_sc - base_score
            log.info("    Rollout: %.1f (Δ%+.1f)", rollout_sc, delta)

            results.append((rollout_sc, heuristic_score, payload))
            if rollout_sc > best_rollout:
                best_rollout = rollout_sc

            # back to the state before this probe move (standard undo)
            ok = rollback_to_save(base_url, game_id, snapshot_save)
            if not ok:
                log.warning("    Rollback fehlgeschlagen – breche MCTS ab")
                break

        # hybrid decision
        if results:
            max_r  = max(r[0] for r in results)
            min_r  = min(r[0] for r in results)
            spread = max_r - min_r
            if spread >= MCTS_MIN_DELTA:
                best_rollout, _, best_payload = max(results, key=lambda x: x[0])
                log.info("  📊 MCTS entscheidet (spread=%.1f)", spread)
            else:
                _, _, best_payload = max(results, key=lambda x: x[1])
                best_rollout = max_r
                log.info("  📊 Heuristik entscheidet (spread=%.1f < %.1f)",
                         spread, MCTS_MIN_DELTA)

        # refresh runId after all restores
        try:
            cur_state = get_state(base_url, player_id)
            best_payload["runId"] = cur_state["runId"]
        except Exception:
            pass

        # NO post_input here - run_mcts_bot_mp sends the move
        # (the lock is released there after the post)

    finally:
        # ALWAYS release the lock, including on errors
        lock.release()

    chosen = (best_payload.get("response", {}).get("card")
              or best_payload.get("response", {}).get("type", "?"))
    if chosen == "option":
        chosen = "Pass"
    log.info("  ✓ MP-MCTS wählt: %s (rollout=%.1f)", chosen, best_rollout)

    return best_payload, best_rollout


# ---------------------------------------------------------------------------
# Final label for multiplayer data
# ---------------------------------------------------------------------------

def compute_final_label_mp(
    my_state:  dict,
    all_final: dict[str, dict],   # {player_id: final_state}
    my_id:     str,
) -> tuple[float, int, int]:
    """Final label for multiplayer training data.
    Returns (label, my_vp, won).
    """
    my_vp = my_state["thisPlayer"]["victoryPointsBreakdown"]["total"]

    opp_vps = [
        s["thisPlayer"]["victoryPointsBreakdown"]["total"]
        for pid, s in all_final.items()
        if pid != my_id
    ]

    if not opp_vps:
        return my_vp / 60.0, my_vp, 1

    best_opp_vp = max(opp_vps)
    vp_diff     = my_vp - best_opp_vp
    won         = my_vp >= max(opp_vps + [my_vp])

    base      = 1.0 if won else -1.0
    vp_bonus  = max(-1.0, min(1.0, vp_diff / 20.0))
    label     = base + vp_bonus

    # compute the rank
    all_vps = sorted([my_vp] + opp_vps, reverse=True)
    rank    = all_vps.index(my_vp) + 1

    return label, my_vp, rank


# ---------------------------------------------------------------------------
# Main bot loop (multiplayer)
# ---------------------------------------------------------------------------

def run_mcts_bot_mp(
    base_url:       str,
    player_id:      str,
    game_id:        str,
    my_color:       str,
    all_player_ids: list[str],
    n_rollouts:     int   = ROLLOUT_MOVES,
    max_candidates: int   = MAX_CANDIDATES,
    data_file:      str   = MCTS_DATA_FILE,
    simple_rollout: bool  = False,
    server_id:      str   = "EIERWIRBRAUCHENEIER",
    enable_mcts:    bool  = True,
    human_opponent: bool  = False,
):
    log.info("🤖 MP-MCTS-Bot | Farbe: %s | %s", my_color, player_id)
    log.info("   Rollouts: %d | Kandidaten: %d | Einfacher Rollout: %s",
             n_rollouts, max_candidates, simple_rollout)
    log.info("   MCTS aktiv: %s%s", enable_mcts,
             "" if enable_mcts else " (reiner Heuristik-Sparringspartner)")
    log.info("   Mitspieler IDs: %s", all_player_ids)
    log.info("   🌐 %s/player?id=%s", base_url, player_id)

    # Are foreign player ids known? In a real game (joined through our own id only)
    # they are not - then the "everyone inactive" end-of-game detection must not fire,
    # because the opponents cannot be polled, and the end of the game is recognised
    # through phase == "end" alone.
    _know_opponents = any(pid != player_id for pid in all_player_ids)

    lock             = MCTSLock(my_color, game_id)
    db_ready         = False
    errors           = 0
    last_key         = None
    # Draft repick pools this bot has already answered in this game.
    # Without this guard the bot answers the same draft round MORE THAN ONCE: the normal
    # dedup below hangs on gameAge, and gameAge RISES as soon as the human drafts, so the
    # dedup stops matching, a second answer goes out for the same pool, the server has
    # already processed the card -> "Card <name> not found" -> HTTP 400 -> abort after
    # four attempts.
    repick_done: set = set()
    last_action_time = time.time()   # timestamp of the last successful move
    mcts_transitions: list[dict] = []
    post_error_counts: dict = {}   # failures per input (guards against endless loops)

    while True:
        try:
            state  = get_state(base_url, player_id)
            errors = 0
        except Exception as e:
            errors += 1
            if errors >= MAX_ERRORS:
                log.error("Zu viele Fehler: %s", e)
                break
            time.sleep(ERROR_WAIT)
            continue

        game    = state.get("game", {})
        phase   = game.get("phase", "")
        waiting = state.get("waitingFor")
        my_plants = state.get("thisPlayer", {}).get("plants", 0)

        # End of game. IMPORTANT: while 8 or more plants are pending, do NOT end through the
        # phase heuristic - after the last generation comes the production phase (whose phase
        # is not in the list) and only THEN the final greenery prompt. Leaving here would hang
        # the server waiting for a placement that never comes.
        # With 8 or more plants only an explicit phase == "end" ends the loop.
        # The prelude phase (and future expansion phases) are NOT the end of the game.
        # Two safeguards: 'preludes' explicitly whitelisted, plus a generation > 1 guard
        # (no game ends in generation 1), which also catches unknown early phases.
        _gen = state.get("game", {}).get("generation", 0) or 0
        game_over = (phase == "end") or (
            not waiting and not _can_place_final_greenery(state) and _gen > 1 and phase not in
            ("research", "action", "drafting", "initialdrafting", "solar",
             "preludes", "prelude", "initialcards")
        )

        # Real end-of-game detection without a timeout:
        # 1. phase == "end" -> explicit from the server
        # 2. all parameters maxed AND no player has moves left
        #    (recognisable: every player has isActive=False and waiting=None)
        if not game_over and not waiting and phase == "action":
            try:
                game_data = state.get("game", {})
                oxygen = game_data.get("oxygenLevel", 0)
                temp   = game_data.get("temperature", -30)
                oceans = game_data.get("oceans", 0)
                params_full = (oxygen >= 14 and temp >= 8 and oceans >= 9)

                # IMPORTANT: do not declare the game over while 8 or more plants are pending.
                # At the end of the game the final greenery phase runs and the bot still has to
                # convert those plants. Aborting here hangs the server on a greenery placement
                # that never arrives.
                #
                # Disabled against HUMANS: "everyone inactive" is a RACE. Between two of its own
                # actions (endgame liquidation, say) the bot is briefly inactive and an opponent who
                # has already passed is too, so a single snapshot reports the end of the game while
                # the server is about to give the bot the move again. For human games therefore -
                # consistently with the idle backstop below, which is disabled as well - ONLY
                # phase == "end" counts; the recovery loop picks things up again by itself.
                if (_know_opponents and not human_opponent and params_full
                        and not is_active and not waiting and not _can_place_final_greenery(state)):
                    # all parameters maxed and this bot inactive.
                    # check whether all other players are inactive too
                    # (waiting for the opponents' bonus greenery actions).
                    if not hasattr(run_mcts_bot_mp, "_params_full_since"):
                        run_mcts_bot_mp._params_full_since = None
                    if run_mcts_bot_mp._params_full_since is None:
                        run_mcts_bot_mp._params_full_since = time.time()

                    all_inactive = True
                    for opp_id in all_player_ids:
                        try:
                            opp_state = get_state(base_url, opp_id)
                            if (opp_state.get("thisPlayer", {}).get("isActive", False)
                                    or opp_state.get("waitingFor")):
                                all_inactive = False
                                break
                        except Exception:
                            pass

                    waited = time.time() - run_mcts_bot_mp._params_full_since
                    if all_inactive or (not human_opponent and waited > 30):
                        reason = "alle inaktiv" if all_inactive else f"Timeout {waited:.0f}s"
                        log.info("   O2=%d Temp=%d Ozeane=%d maximal → Spielende (%s)",
                                 oxygen, temp, oceans, reason)
                        game_over = True
                else:
                    if hasattr(run_mcts_bot_mp, "_params_full_since"):
                        run_mcts_bot_mp._params_full_since = None

                # fallback: the server has changed phase
                fresh = get_state(base_url, player_id)
                if fresh.get("game", {}).get("phase") == "end":
                    state = fresh
                    game_over = True
            except Exception:
                pass

        if game_over:
            vp_breakdown = state["thisPlayer"].get("victoryPointsBreakdown", {})
            my_vp = vp_breakdown.get("total", state["thisPlayer"]["terraformRating"])
            my_tr = state["thisPlayer"]["terraformRating"]

            # final label from the VP comparison with all opponents
            all_final = {player_id: state}
            for opp_id in all_player_ids:
                if opp_id != player_id:
                    try:
                        all_final[opp_id] = get_state(base_url, opp_id)
                    except Exception:
                        pass

            final_label, _, rank = compute_final_label_mp(state, all_final, player_id)

            all_vps = {pid: s["thisPlayer"]["victoryPointsBreakdown"]["total"]
                       for pid, s in all_final.items()}
            won = (my_vp >= max(all_vps.values()))

            log.info("🏁 VP: %d | TR: %d | Rang: %d | Label: %+.2f | Gewonnen: %s",
                     my_vp, my_tr, rank, final_label, won)
            log.info("   Alle VPs: %s", all_vps)

            # save the training data
            if mcts_transitions and data_file:
                with open(data_file, "a", encoding="utf-8") as f:
                    for t in mcts_transitions:
                        t["label"]    = final_label
                        t["final_vp"] = my_vp
                        t["final_tr"] = my_tr
                        t["won"]      = won
                        t["rank"]     = rank
                        f.write(json.dumps(t) + "\n")
                log.info("💾 %d MCTS-Transitions gespeichert (label=%+.2f) → %s",
                         len(mcts_transitions), final_label, data_file)

            # make sure the lock is released at the end of the game
            lock.release()
            break

        player    = state.get("thisPlayer", {})
        is_active = player.get("isActive", False)
        wtype     = waiting.get("type", "") if waiting else ""

        # initialCards: isActive is often False, but the bot still has to answer
        # research phase 'card': buying cards is needed even with isActive=False
        # research phase waiting=None: the server waits for another player -> wait
        # If something sits in OUR waitingFor, it is our input - regardless of isActive.
        # isActive is unreliable in simultaneous and cleanup phases: the FINAL greenery
        # conversion at the end of the game arrives as 'space' with isActive=False, and the
        # old condition (without 'space') swallowed it -> hang on "place final greenery".
        # In simultaneous phases the server sets waitingFor only for the player to move.
        need_action = bool(waiting)

        # Special case: phase=action, isActive=False, waiting=None
        # May mean the server is still processing, that someone else really is to move,
        # or that a rollback bent the state and we wrongly believe we are inactive.
        # Fetch a fresh state and recover.
        if not need_action and phase == "action" and not waiting and not is_active:
            if not hasattr(run_mcts_bot_mp, "_idle_action_count"):
                run_mcts_bot_mp._idle_action_count = 0
            run_mcts_bot_mp._idle_action_count += 1
            # fetch a fresh state on EVERY idle poll (not only every fifth)
            try:
                fresh = get_state(base_url, player_id)
                fresh_waiting = fresh.get("waitingFor")
                fresh_active  = fresh.get("thisPlayer", {}).get("isActive", False)
                fresh_phase   = fresh.get("game", {}).get("phase", "")
                if fresh_waiting or fresh_active:
                    state = fresh
                    need_action = True
                    last_key = None   # force re-evaluation
                    run_mcts_bot_mp._idle_action_count = 0
                elif fresh_phase == "end":
                    state = fresh
                    break   # end of game
            except Exception:
                pass
            # Recovery nudge: if the bot hangs in this state for a while, reset last_key
            # periodically so that a possibly swallowed action is retried (defuses a
            # rollback-induced deadlock).
            if not need_action and run_mcts_bot_mp._idle_action_count % 50 == 0:
                last_key = None
            # Diagnostic safeguard: if the bot idles for a long time with 8 or more plants and
            # no waitingFor, the final greenery prompt is apparently not showing up in the
            # polled state. Dump the full state (including opponents) once.
            if (_can_place_final_greenery(state) and run_mcts_bot_mp._idle_action_count == 30
                    and not getattr(run_mcts_bot_mp, "_hang_dumped", False)):
                try:
                    opp_states = {oid: get_state(base_url, oid)
                                  for oid in all_player_ids if oid != player_id}
                    # Is an opponent waiting? Then it is simply their turn -> not a hang.
                    if any(s.get("waitingFor") for s in opp_states.values()):
                        run_mcts_bot_mp._idle_action_count = 0   # check again later
                    else:
                        dump = {"self": get_state(base_url, player_id),
                                "opponents": opp_states}
                        fn = f"hang_debug_{int(time.time())}.json"
                        json.dump(dump, open(fn, "w"), indent=2, default=str)
                        log.warning("  ⚠ %d Pflanzen + idle, niemand am Zug → %s (BITTE HOCHLADEN)",
                                    my_plants, fn)
                        run_mcts_bot_mp._hang_dumped = True
                except Exception as e:
                    log.error("  Hang-Dump fehlgeschlagen: %s", e)
            # after 90 s of real idling, assume the game is over
            # Disabled against human opponents: a good move often takes longer.
            if not human_opponent and run_mcts_bot_mp._idle_action_count * POLL_INTERVAL > 90:
                log.warning("  90s idle → Spielende angenommen")
                break
        else:
            if hasattr(run_mcts_bot_mp, "_idle_action_count"):
                run_mcts_bot_mp._idle_action_count = 0

        if not need_action:
            if not hasattr(run_mcts_bot_mp, "_poll_count"):
                run_mcts_bot_mp._poll_count = 0
            run_mcts_bot_mp._poll_count += 1
            if run_mcts_bot_mp._poll_count % 33 == 0:
                log.info("   ⏳ Warte | phase=%s isActive=%s waiting=%s",
                         phase, is_active, wtype or "None")
            # SHADOW ANALYSIS (--vs-human only, pure observation): while the HUMAN is to move,
            # the bot logs what IT would do in that position, plus the scores. The difference
            # between human and bot is exactly what a bot-versus-bot match cannot measure.
            if _shadow is not None and human_opponent:
                for _hid in all_player_ids:
                    if _hid != player_id:
                        try:
                            _shadow.shadow_step(base_url, _hid, get_state, decide, my_color)
                        except Exception as _se:
                            # Do NOT swallow this silently: a NameError in the shadow module once left only
                            # game-start markers in the log for seven games - the error was invisible and the
                            # playing time was lost. The FIRST error is logged loudly (with a traceback),
                            # after that only every 50th occurrence, so the console is not spammed. The game
                            # continues either way - the analysis must never disturb play.
                            import traceback as _tb2
                            run_mcts_bot_mp._shadow_errs = getattr(
                                run_mcts_bot_mp, "_shadow_errs", 0) + 1
                            _n = run_mcts_bot_mp._shadow_errs
                            if _n == 1 or _n % 50 == 0:
                                log.error("  ❌ SCHATTEN-ANALYSE FEHLGESCHLAGEN (%d×): %s: %s",
                                          _n, type(_se).__name__, _se)
                                log.error("     -> im Log fehlen die Entscheidungspunkte! "
                                          "%s", _tb2.format_exc().strip().splitlines()[-1])
            time.sleep(POLL_INTERVAL)
            continue

        if wtype == "player":
            # The runner used to answer the player selection ITSELF by choosing its own colour,
            # which meant the bot always attacked itself. With Fish ("Select player to decrease
            # plants production") it lowered its OWN plant production although the opponent had
            # some. tm_bot.handle_player distinguishes correctly between an attack
            # (decrease/remove/steal/lose -> opponent) and a bonus (-> self), it was simply
            # never called from here.
            players = waiting.get("players", [])
            if players:
                colors  = [p if isinstance(p, str) else p.get("color", "") for p in players]
                payload = None
                try:
                    _res = decide(state)
                    payload = _res[0] if isinstance(_res, tuple) else _res
                except Exception as _pe:
                    log.warning("  player-Handler Fehler: %s – Fallback", _pe)
                if not payload:
                    chosen  = my_color if my_color in colors else colors[0]
                    payload = {"type": "player", "runId": state["runId"], "player": chosen}
                log.info("  👤 Spielerauswahl -> %s", payload.get("player"))
                try:
                    post_input(base_url, player_id, payload)
                    time.sleep(POST_WAIT)
                except Exception:
                    pass
            continue

        # Deduplication - never deduplicate initialCards (the server only changes the state
        # once both players have answered, so the key stays the same)
        if not waiting:
            time.sleep(POLL_INTERVAL)
            continue
        # gameAge distinguishes genuinely new decisions that carry an identical signature
        # (e.g. several consecutive "Place any final greenery from plants").
        # Within the same server state gameAge stays put -> the dedup holds and nothing is
        # sent twice; after a placement gameAge rises -> the next one is handled.
        key = (wtype, str(waiting.get("title", "")),
               tuple(c.get("name", "") for c in waiting.get("cards", [])),
               (state.get("game") or {}).get("gameAge"))
        if key == last_key and wtype != "initialCards":
            time.sleep(POLL_INTERVAL)
            continue

        # DRAFT REPICK: already answered this card pool? Then do NOT send again - the chosen
        # card has already been processed server-side and a second attempt fails with
        # "Card <name> not found" (HTTP 400). Wait until the server offers a new pack (then
        # the pool key changes).
        if _is_draft_repick(state):
            _pk = _repick_pool_key(state)
            if _pk in repick_done:
                time.sleep(POLL_INTERVAL)
                continue
            repick_done.add(_pk)

        gen = game.get("generation", 1)
        mc  = player.get("megacredits", 0)
        tr  = player.get("terraformRating", 14)
        log.info("[Gen %d] MC:%d TR:%d | %s", gen, mc, tr,
                 str(waiting.get("title", ""))[:40])

        # check that the database is ready (needed for the search)
        if enable_mcts and not db_ready and phase == "action":
            db_ready = wait_for_db(base_url, game_id, max_wait=5)
            if db_ready:
                log.info("   Spiel in DB – MCTS aktiviert")
            else:
                log.info("   Spiel nicht in DB – MCTS deaktiviert")

        _wt = waiting.get("title", "")
        _wtitle = _wt.lower() if isinstance(_wt, str) else ""
        _is_action_menu = "take your" in _wtitle   # main action menu only, no inline or-resolution of cards
        use_mcts = (enable_mcts and phase == "action" and wtype == "or"
                    and db_ready and _is_action_menu)
        payload    = None
        rollout_sc = 0.0
        payload    = None   # always initialise
        rollout_sc = 0.0

        if use_mcts:
            try:
                # fetch the state again - it may be stale after a long wait for the lock
                try:
                    state = get_state(base_url, player_id)
                    waiting = state.get("waitingFor")
                    if not waiting or not state.get("thisPlayer", {}).get("isActive", False):
                        last_key = None
                        time.sleep(POLL_INTERVAL)
                        continue
                    wtype = waiting.get("type", "")
                    if wtype != "or":
                        use_mcts = False
                except Exception:
                    pass

                if use_mcts:
                    payload, rollout_sc = handle_or_mcts_mp(
                        state, base_url, player_id, all_player_ids,
                        lock, game_id,
                        n_rollouts=n_rollouts,
                        max_candidates=max_candidates,
                        simple_rollout=simple_rollout,
                    )
                else:
                    result = decide(state)
                    payload = result[0] if isinstance(result, tuple) else result
                    rollout_sc = 0.0
            except Exception as e:
                log.warning("  MP-MCTS Fehler: %s – fallback zu decide()", e)
                lock.release()   # make sure
                result = decide(state)
                payload = result[0] if isinstance(result, tuple) else result
                rollout_sc = 0.0

            # collect training data
            card_name = payload.get("response", {}).get("card", "") if payload else ""
            if card_name and card_name not in ("Pass", ""):
                info     = CARD_DB.get(card_name, {})
                s_feats  = game_state_features(state)
                c_feats  = card_features(info)
                progress = param_progress_from_state(state)
                mcts_transitions.append({
                    "card":           card_name,
                    "rollout_sc":     rollout_sc,
                    "generation":     gen,
                    "param_progress": progress,
                    "state_feats":    s_feats,
                    "card_feats":     c_feats,
                })

        else:
            result = decide(state)
            if result is None:
                last_key = key
                time.sleep(POLL_INTERVAL)
                continue
            payload = result[0] if isinstance(result, tuple) else result
            if payload is None:
                last_key = key
                time.sleep(POLL_INTERVAL)
                continue

        if payload is None:
            last_key = key
            time.sleep(POLL_INTERVAL)
            continue

        if payload is None:
            last_key = key
            time.sleep(POLL_INTERVAL)
            continue
        resp = payload.get("response", {}) if isinstance(payload, dict) else {}
        log.info("  → %s", resp.get("card", resp.get("type", "?")))

        try:
            save_before = get_last_save_id(base_url, game_id) if enable_mcts else -1
            post_input(base_url, player_id, payload)
            last_action_time = time.time()   # successful move
            # --- Transition logging (bot self-play, full state per move).
            #     Optional, off by default - enable with TM_LOG_TRANSITIONS=1. ---
            import os as _os_tl
            if _os_tl.environ.get("TM_LOG_TRANSITIONS", "") not in ("", "0", "false", "False"):
                try:
                    import json as _json
                    _p = _os_tl.path.join(_os_tl.path.dirname(_os_tl.path.abspath(__file__)), "transitions.jsonl")
                    with open(_p, "a", encoding="utf-8") as _tlog:
                        _tlog.write(_json.dumps({
                            "state": state,
                            "move": {k: v for k, v in payload.items() if k != "_label"},
                        }, default=lambda o: list(o) if isinstance(o, set) else str(o)) + "\n")
                    globals()["_TLOG_N"] = globals().get("_TLOG_N", 0) + 1
                    log.info("  📝 %d Zeilen -> %s", globals()["_TLOG_N"], _p)
                except Exception as _e:
                    log.error("  ⚠ Logging-Fehler: %s", _e)
            # --- end ---
            last_key = key
            time.sleep(POST_WAIT)
            if enable_mcts:
                # Search mode: wait actively until the server has processed the move (the save id
                # changes). Until then keep last_key, so no double move fires. A fallback limit
                # guards against hanging.
                waited_post = POST_WAIT
                while waited_post < 3.0:
                    try:
                        if get_last_save_id(base_url, game_id) != save_before:
                            last_key = None   # the state has changed -> read it again
                            break
                    except Exception:
                        last_key = None
                        break
                    time.sleep(POLL_INTERVAL)
                    waited_post += POLL_INTERVAL
            else:
                # Heuristic mode (--join): post_input is synchronous, so the server state is already
                # updated on return. The save_id check is not reliable here (no separate game id;
                # game_id == player_id makes get_last_save_id return a constant), so last_key would
                # → get_last_save_id liefert konstant -1), wodurch last_key nie
                # never be reset. An action without a follow-up input (Asteroid, say) then leads back
                # to "Take your next action" with an identical key, the double-move guard blocks the
                # next action and the bot polls forever without logging. Hence read again directly.
                # neu lesen.
                last_key = None
            # Successful post -> clear the rejection list (it only applies to the current error
            # cycle, not to the whole game).
            try:
                import tm_bot as _tb
                if _tb._draft_rejected:
                    _tb._draft_rejected.clear()
            except Exception:
                pass
        except requests.HTTPError as e:
            # Count failures per input (robust against oscillating states where successful posts
            # in between would keep resetting a plain consecutive counter).
            # The card names belong in the key: during a draft the TITLE is identical across ALL
            # rounds, so failures from different draft rounds would add up into one abort.
            # rounds, so failures from different draft rounds would add up into one abort. With
            # the cards in the key every round counts on its own.
            err_key = (wtype, str(waiting.get("title", ""))[:80],
                       tuple(c.get("name", "") for c in (waiting.get("cards") or [])))
            post_error_counts[err_key] = post_error_counts.get(err_key, 0) + 1
            n_err = post_error_counts[err_key]
            log.warning("  HTTP Fehler (%d× für diesen Input): %s",
                        n_err, e.response.text[:200] if e.response else e)
            if n_err == 1:
                # First failure -> full diagnosis: what did the server want, what did the bot send?
                # (Helps when a handler is missing.)
                log.warning("  ⚠️ Unbeantworteter Input – waitingFor: %r", waiting)
                log.warning("  ⚠️ Gesendete Antwort: %r", payload)
            if n_err >= 4:
                log.error("  ❌ Input %d× nicht beantwortbar – Abbruch, um eine "
                          "Endlosschleife zu vermeiden. Bitte obige waitingFor-"
                          "Struktur melden, dann lässt sich der Handler ergänzen.",
                          n_err)
                break
            # DISCARD THE DRAFT CACHE: _draft_choice_cache pins the choice per card pool for
            # repick stability. If the server has just REJECTED that choice ("Card <name> not
            # found"), it is exactly the wrong one - otherwise the bot would resend it on every
            # retry and abort after four attempts. Clearing the cache makes the next attempt
            # retry and abort after four attempts. Clearing the cache makes the next attempt
            # decide afresh against the current pool.
            try:
                import re as _re
                import tm_bot as _tb
                _tb._draft_choice_cache.clear()
                # The server names the rejected card in plain text ("Error: Card <name> not found").
                # Block that card specifically, so the next attempt picks the NEXT BEST one instead
                # of the same one again.
                _txt = e.response.text if e.response is not None else ""
                for _m in _re.finditer(r"Card (.+?) not found", _txt):
                    _tb._draft_rejected.add(_m.group(1).strip())
                if _tb._draft_rejected:
                    log.warning("  ⚠️ Vom Server abgelehnte Draft-Karten: %s",
                                sorted(_tb._draft_rejected))
            except Exception:
                pass
            # last_key=None: the next poll re-reads the state
            last_key = None
            time.sleep(ERROR_WAIT)
        except Exception as e:
            log.warning("  Fehler: %s", e)
            last_key = None
            time.sleep(ERROR_WAIT)


# ---------------------------------------------------------------------------
# Unattended game mode: one bot creates the game, the others read the ids
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sequenzieller Ein-Prozess-Self-Play-Treiber
#
# A single process drives ALL bots strictly one after another. Because Terraforming
# Mars is turn-based, only one bot is to move at any time (exception: draft and
# research, where several choose in parallel - there they are simply served in turn,
# without rollback). While one bot runs its rollouts including rollbacks, NO other
# process polls the game, so there are no ghost states, no double moves and no
# rollback deadlock.
# Lock, ready files and coordination files are unnecessary here.
# ---------------------------------------------------------------------------

def _can_place_final_greenery(state: dict) -> bool:
    """True when the bot can still place at least one greenery from plants at the
    end of the game (8 plants, or 7 with Ecoline).
    """
    try:
        return can_convert_plants(state)
    except Exception:
        return state.get("thisPlayer", {}).get("plants", 0) >= 8


def _needs_action(state: dict) -> bool:
    """True when the player has to make an input in the given state."""
    waiting = state.get("waitingFor")
    if not waiting:
        return False
    phase     = state.get("game", {}).get("phase", "")
    is_active = state.get("thisPlayer", {}).get("isActive", False)
    wtype     = waiting.get("type", "")
    return bool(
        is_active
        or wtype in ("initialCards", "card", "payment", "amount")
        or phase == "drafting"
    )


def _step_player(
    state, base_url, player_id, all_player_ids, game_id, my_color,
    enable_mcts, lock, db_ready, game_id_for_db,
    n_rollouts, max_candidates, simple_rollout, transitions,
    mcts_allowed_now: bool = True,
    decide_fn=None,
) -> bool:
    """Make ONE decision for player_id and post it.
    Returns True when a move was actually sent.
    """
    decide_fn = decide_fn or decide   # switchable heuristic variant per colour
    waiting = state.get("waitingFor")
    phase   = state.get("game", {}).get("phase", "")
    player  = state.get("thisPlayer", {})
    wtype   = waiting.get("type", "")
    gen     = state.get("game", {}).get("generation", 1)
    mc      = player.get("megacredits", 0)
    tr      = player.get("terraformRating", 14)

    log.info("[%s|Gen %d] MC:%d TR:%d | %s", my_color, gen, mc, tr,
             str(waiting.get("title", ""))[:40])

    # Player selection: delegate to the heuristic module. Modules without a 'player'
    # handler (a frozen champion) return None -> fall back to the previous behaviour
    # (own colour).
    if wtype == "player":
        players = waiting.get("players", [])
        if players:
            payload = None
            try:
                result  = decide_fn(state)
                payload = result[0] if isinstance(result, tuple) else result
            except Exception as e:
                log.warning("  player-Handler Fehler: %s – Fallback", e)
            if not payload:
                colors  = [p if isinstance(p, str) else p.get("color", "") for p in players]
                chosen  = my_color if my_color in colors else colors[0]
                payload = {"type": "player", "runId": state["runId"], "player": chosen}
            try:
                post_input(base_url, player_id, payload)
                time.sleep(POST_WAIT)
                return True
            except Exception:
                return False
        return False

    # check the database is ready (needed for the search); db_ready is a 1-element list
    if enable_mcts and not db_ready[0] and phase == "action":
        db_ready[0] = wait_for_db(base_url, game_id_for_db, max_wait=5)
        if db_ready[0]:
            log.info("   Spiel in DB – MCTS aktiviert")

    _wt = waiting.get("title", "")
    _wtitle = _wt.lower() if isinstance(_wt, str) else ""
    _is_action_menu = "take your" in _wtitle   # main action menu only, no inline or-resolution of cards
    use_mcts   = (enable_mcts and mcts_allowed_now and phase == "action"
                  and wtype == "or" and db_ready[0] and _is_action_menu)
    payload    = None
    rollout_sc = 0.0

    if use_mcts:
        try:
            payload, rollout_sc = handle_or_mcts_mp(
                state, base_url, player_id, all_player_ids,
                lock, game_id,
                n_rollouts=n_rollouts,
                max_candidates=max_candidates,
                simple_rollout=simple_rollout,
            )
        except Exception as e:
            log.warning("  MCTS Fehler: %s – fallback decide()", e)
            try:
                lock.release()
            except Exception:
                pass
            result  = decide_fn(state)
            payload = result[0] if isinstance(result, tuple) else result
        # collect training data (search bot only)
        card_name = payload.get("response", {}).get("card", "") if payload else ""
        if card_name and card_name not in ("Pass", ""):
            info = CARD_DB.get(card_name, {})
            transitions.append({
                "card":           card_name,
                "rollout_sc":     rollout_sc,
                "generation":     gen,
                "param_progress": param_progress_from_state(state),
                "state_feats":    game_state_features(state),
                "card_feats":     card_features(info),
            })
    else:
        result  = decide_fn(state)
        payload = result[0] if isinstance(result, tuple) else result

    if payload is None:
        return False

    resp = payload.get("response", {}) if isinstance(payload, dict) else {}
    log.info("  → %s", resp.get("card", resp.get("type", "?")))

    try:
        save_before = get_last_save_id(base_url, game_id)
        post_input(base_url, player_id, payload)
        # Wait briefly for the server to process (save id change), so the next poll sees the
        # following state instead of the same move again.
        time.sleep(POST_WAIT)
        waited = POST_WAIT
        while waited < 2.0:
            try:
                if get_last_save_id(base_url, game_id) != save_before:
                    break
            except Exception:
                break
            time.sleep(POLL_INTERVAL)
            waited += POLL_INTERVAL
        # NOTE: a "move without state change" warning based on save_id was built and removed
        # again: with a small POST_WAIT the save id is not a reliable per-input signal (it
        # produced false alarms on demonstrably effective moves). Real no-op detection would
        # alarms on demonstrably effective moves). Real no-op detection would
        # need a state comparison (a fresh GET), not the save id.
        return True
    except requests.HTTPError as e:
        body = ""
        try:
            body = e.response.text[:300] if e.response is not None else ""
        except Exception:
            body = ""
        sc = e.response.status_code if e.response is not None else "?"
        log.warning("  HTTP %s Fehler. Server-Body: %r", sc, body)
        # The server body of this 400 is often EMPTY, so the actual diagnosis is WHAT the bot
        # sent and in response to WHAT. Both are printed once.
        log.warning("  ⚠️ Gesendete Payload: %r", payload)
        log.warning("  ⚠️ waitingFor-Titel: %r | type=%r",
                    (state.get("waitingFor") or {}).get("title"),
                    (state.get("waitingFor") or {}).get("type"))
        time.sleep(ERROR_WAIT)
        return False
    except Exception as e:
        log.warning("  Fehler: %s", e)
        time.sleep(ERROR_WAIT)
        return False


def _finalize_selfplay(base_url, color_to_id, mcts_color, transitions, data_file,
                       prod_snap=None):
    """Final scoring: final VP, label, and saving the training data."""
    mcts_id   = color_to_id[mcts_color]
    all_final = {}
    for pid in color_to_id.values():
        try:
            all_final[pid] = get_state(base_url, pid)
        except Exception:
            pass
    if mcts_id not in all_final:
        log.warning("   Finalen State des MCTS-Bots nicht erreichbar – kein Label")
        return None

    state = all_final[mcts_id]
    vp_bd = state["thisPlayer"].get("victoryPointsBreakdown", {})
    my_vp = vp_bd.get("total", state["thisPlayer"]["terraformRating"])
    my_tr = state["thisPlayer"]["terraformRating"]
    final_label, _, rank = compute_final_label_mp(state, all_final, mcts_id)

    all_vps = {pid: s["thisPlayer"].get("victoryPointsBreakdown", {}).get("total", 0)
               for pid, s in all_final.items()}
    won = (my_vp >= max(all_vps.values())) if all_vps else False

    log.info("🏁 %s | VP:%d TR:%d | Rang:%d | Label:%+.2f | Gewonnen:%s",
             mcts_color, my_vp, my_tr, rank, final_label, won)
    log.info("   Alle VPs: %s", all_vps)
    # Engine diagnostics: production per colour at the end of the game (plants are the key
    # indicator). Lets the plant lever be measured straight from the A/B log.
    for _pid, _s in all_final.items():
        _tp  = _s.get("thisPlayer", {})
        _col = next((c for c, p in color_to_id.items() if p == _pid), _pid)
        _vpb = _tp.get("victoryPointsBreakdown", {})
        _sum = (_tp.get("megacreditProduction", 0) + _tp.get("steelProduction", 0)
                + _tp.get("titaniumProduction", 0) + _tp.get("plantProduction", 0)
                + _tp.get("energyProduction", 0) + _tp.get("heatProduction", 0))
        log.info("   🏭 %-5s | PFLANZEN:%d | Prod-Summe:%d | KARTEN-VP:%d | Greenery-VP:%d | Meilenstein-VP:%d",
                 _col, _tp.get("plantProduction", 0), _sum,
                 _vpb.get("victoryPoints", 0), _vpb.get("greenery", 0), _vpb.get("milestones", 0))

    if transitions and data_file:
        with open(data_file, "a", encoding="utf-8") as f:
            for t in transitions:
                t["label"], t["final_vp"], t["final_tr"] = final_label, my_vp, my_tr
                t["won"], t["rank"] = won, rank
                f.write(json.dumps(t) + "\n")
        log.info("💾 %d MCTS-Transitions gespeichert (label=%+.2f) → %s",
                 len(transitions), final_label, data_file)

    # Result by COLOUR (the player id changes per game, the colour is stable) for
    # aggregation across several games.
    # BEHAVIOUR DATA per colour: final production, final tiles and the generation 3/5/8
    # snapshots. Lets the strategy chains be measured in the A/B without human games.
    _behav = {}
    for _c, _pid in color_to_id.items():
        _tp = all_final.get(_pid, {}).get("thisPlayer", {})
        _behav[_c] = {
            "end_prod": {
                "mc":    _tp.get("megacreditProduction", 0),
                "steel": _tp.get("steelProduction", 0),
                "titan": _tp.get("titaniumProduction", 0),
                "plant": _tp.get("plantProduction", 0),
                "energy":_tp.get("energyProduction", 0),
                "heat":  _tp.get("heatProduction", 0),
            },
            "cities":    _tp.get("citiesCount", 0),
            "vp_parts":  _tp.get("victoryPointsBreakdown", {}),
            "snap":      (prod_snap or {}).get(_c, {}),
        }
    return {
        "vps_by_color": {c: all_vps.get(pid, 0) for c, pid in color_to_id.items()},
        # MC at the end of the game: the official tiebreaker on equal VP
        "mc_by_color": {c: all_final.get(pid, {}).get("thisPlayer", {}).get("megacredits", 0)
                        for c, pid in color_to_id.items()},
        "behav_by_color": _behav,
        "mcts_color":   mcts_color,
        "mcts_vp":      my_vp,
        "mcts_rank":    rank,
        "mcts_won":     won,
    }


def _repick_pool_key(state):
    """Identifier of the current repick card set (colour plus sorted names).
    Changes as soon as the server offers a new pack.
    """
    w = state.get("waitingFor") or {}
    color = (state.get("thisPlayer") or {}).get("color")
    names = tuple(sorted(c.get("name", "") for c in (w.get("cards") or [])))
    return (color, names)


def _is_draft_repick(state):
    """True during the DRAFT REPICK phase: the player has already drafted and may
    change the choice until everyone has picked.
    """
    w = state.get("waitingFor") or {}
    if w.get("type") != "card":
        return False
    tp = state.get("thisPlayer") or {}
    if tp.get("needsToDraft"):   # a real first draft -> normal progress
        return False
    title = w.get("title")
    msg = title.get("message", "") if isinstance(title, dict) else str(title)
    return "change your selection" in msg


def _wf_signature(color, waiting):
    """Progress signature of a waitingFor state. The title alone is not enough,
    because consecutive decisions can carry an identical title.
    """
    w = waiting or {}
    cards = tuple(sorted(c.get("name", "") for c in (w.get("cards") or [])))
    return (color, w.get("type", ""), str(w.get("title", ""))[:60],
            cards, w.get("min"), w.get("max"))


def load_decide_variant(module_name: str, db_path: str | None = None):
    """Load decide() from an alternative heuristic module (e.g. a frozen champion),
    so that two variants can play against each other in one process.
    """
    import importlib
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        log.error("Champion-Modul '%s' nicht ladbar: %s", module_name, e)
        raise SystemExit(
            f"ABBRUCH: Champion-Modul '{module_name}' nicht ladbar ({e}). "
            f"Ohne echten Champion waere der A/B-Lauf ein wertloser A/A-Test."
        )
    if db_path and hasattr(mod, "load_card_db"):
        try:
            mod.load_card_db(db_path)
        except Exception as e:
            log.warning("load_card_db fuer '%s' fehlgeschlagen: %s", module_name, e)
    return mod.decide


def _summarize_crn(crn_games: list[dict], n_games: int):
    """A/B evaluation: paired VP margin challenger minus champion with a 95 %
    confidence interval. Each pair plays the same deck twice with swapped seats.
    """
    import math
    import statistics
    from collections import defaultdict

    log.info("══════════════════════════════════════════════════")
    if not crn_games:
        log.warning("📊 CRN: keine gewertete Partie (alle abgebrochen?)")
        log.info("══════════════════════════════════════════════════")
        return

    # Write the raw per-game data. Aggregating and discarding it loses the information
    # HOW the games came about - the VP margin alone does not show whether one side had
    # collapse games (no income over many generations), which is invisible in a paired
    # margin because it cancels out.
    try:
        import json as _json, time as _time
        _fn = f"ab_games_{int(_time.time())}.json"
        with open(_fn, "w", encoding="utf-8") as _f:
            _json.dump(crn_games, _f, indent=1, default=str)
        log.info(f"   Rohdaten der Einzelpartien: {_fn}")
    except Exception as _e:
        log.warning(f"   Rohdaten konnten nicht geschrieben werden: {_e}")

    by_pair: dict[int, list] = defaultdict(list)
    for g in crn_games:
        by_pair[g["pair"]].append(g)

    pair_margins = []
    all_d        = []
    chall_wins = champ_wins = ties = 0
    for idx in sorted(by_pair):
        ds = []
        for g in by_pair[idx]:
            cv = g["vps"].get(g["chall"], 0)
            mv = g["vps"].get(g["champ"], 0)
            ds.append(cv - mv)
            all_d.append(cv - mv)
            if   cv > mv: chall_wins += 1
            elif mv > cv: champ_wins += 1
            else:
                # TM rule: on equal VP the megacredits at the end of the game decide
                cm = g.get("mcs", {}).get(g["chall"], 0)
                mm = g.get("mcs", {}).get(g["champ"], 0)
                if   cm > mm: chall_wins += 1
                elif mm > cm: champ_wins += 1
                else:         ties += 1
        if len(ds) == 2:   # only complete pairs count
            pair_margins.append(sum(ds) / 2.0)

    # Distribution of the individual results - shows downward outliers (collapse games)
    # that stay invisible in the paired margin because they cancel out there.
    _alle_vps = []
    for g in crn_games:
        _alle_vps.append((g["vps"].get(g["chall"], 0), "C"))
        _alle_vps.append((g["vps"].get(g["champ"], 0), "M"))
    if _alle_vps:
        _c = sorted(v for v, s in _alle_vps if s == "C")
        _m = sorted(v for v, s in _alle_vps if s == "M")
        def _q(xs, p):
            return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else 0
        log.info("   VP-Verteilung (Ausreisser nach unten = Kollapspartien):")
        log.info(f"      Challenger: min {min(_c)} | 10%% {_q(_c,.1)} | Median {_q(_c,.5)} | max {max(_c)}")
        log.info(f"      Champion  : min {min(_m)} | 10%% {_q(_m,.1)} | Median {_q(_m,.5)} | max {max(_m)}")
        _schwelle = 60
        log.info(f"      Partien unter {_schwelle} VP: "
                 f"Challenger {sum(1 for v in _c if v < _schwelle)} | "
                 f"Champion {sum(1 for v in _m if v < _schwelle)}")

    n_pairs = len(pair_margins)
    log.info("📊 CRN-Auswertung: %d vollständige Paare | %d/%d Partien gewertet",
             n_pairs, len(all_d), n_games)

    if n_pairs >= 2:
        mean_d = statistics.mean(pair_margins)
        sd     = statistics.stdev(pair_margins)
        se     = sd / math.sqrt(n_pairs)
        half   = 1.96 * se
        log.info("   VP-Marge Challenger−Champion (gepaart): %+.2f  "
                 "[95%%-CI %+.2f … %+.2f | SD %.2f | n=%d Paare]",
                 mean_d, mean_d - half, mean_d + half, sd, n_pairs)
        if   mean_d - half > 0: verdict = "Challenger signifikant besser"
        elif mean_d + half < 0: verdict = "Champion signifikant besser"
        else:                   verdict = "kein signifikanter Unterschied (CI enthält 0)"
        log.info("   → %s", verdict)
    elif n_pairs == 1:
        log.info("   VP-Marge (1 Paar, kein CI): %+.2f", pair_margins[0])

    if all_d:
        log.info("   Siege: Challenger %d | Champion %d | Gleichstand %d  (von %d Partien)",
                 chall_wins, champ_wins, ties, len(all_d))
    log.info("══════════════════════════════════════════════════")



# ---------------------------------------------------------------------------
# Paralleler A/B-CRN-Modus (--parallel N)
# ---------------------------------------------------------------------------

import threading as _threading
from concurrent.futures import ThreadPoolExecutor as _TPE


class _PairLogRouter(logging.Handler):
    """Buffers log records per worker thread; the main thread writes them out, so
    the output of parallel games does not interleave.
    """
    def __init__(self, passthrough: list[logging.Handler]):
        super().__init__()
        self.passthrough = passthrough
        self.local = _threading.local()
    def emit(self, record: logging.LogRecord) -> None:
        buf = getattr(self.local, "buf", None)
        if buf is not None:
            buf.append(record)
        else:
            for h in self.passthrough:
                if record.levelno >= h.level:
                    h.handle(record)


def _play_crn_game(base_url: str, game_id: str, color_to_id: dict,
                   decide_by_color: dict, game_num: int, n_games: int) -> dict | None:
    """Play one game to the end heuristically (search off), with the same guards as
    the sequential self-play driver.
    """
    all_player_ids = list(color_to_id.values())
    prev_sig, prev_save, stuck, idle = None, -1, 0, 0
    repick_done = set()   # repick pools already answered in this game
    STUCK_ABORT   = 8
    GAME_TIME_CAP = 1200   # the updated server is noticeably slower under --parallel load;
                          # real games reached generation 16 and ran into the old cap
    game_start    = time.time()

    # ── BEHAVIOUR LOGGING (for the strategy work) ──
    # Snapshot of each player's production at generation 3/5/8. The loop polls every
    # player anyway, so this costs one dict. Purpose: make the STRATEGY CHAINS measurable
    # that the VP margin does not show - "does the bot build the physical terraforming
    # engine (plants/energy/heat), or does it sink into steel and titanium, which only
    # make cards cheaper?". Passed into crn_games by _finalize_selfplay at the end.
    _prod_snap: dict = {}   # {colour: {generation: {production type: value}}}
    _SNAP_GENS = (3, 5, 8)

    while True:
        acted = False
        if time.time() - game_start > GAME_TIME_CAP:
            log.error("   ⏱ Partie %d läuft >%ds ohne Spielende – Abbruch", game_num, GAME_TIME_CAP)
            return None
        for color, pid in color_to_id.items():
            try:
                state = get_state(base_url, pid)
            except Exception:
                continue
            _g = state.get("game", {}).get("generation", 0) or 0
            if _g in _SNAP_GENS and _g not in _prod_snap.get(color, {}):
                _tp = state.get("thisPlayer", {})
                _prod_snap.setdefault(color, {})[_g] = {
                    "mc":    _tp.get("megacreditProduction", 0),
                    "steel": _tp.get("steelProduction", 0),
                    "titan": _tp.get("titaniumProduction", 0),
                    "plant": _tp.get("plantProduction", 0),
                    "energy":_tp.get("energyProduction", 0),
                    "heat":  _tp.get("heatProduction", 0),
                }
            if state.get("game", {}).get("phase", "") == "end":
                return _finalize_selfplay(base_url, color_to_id,
                                          next(iter(color_to_id)), [], None,
                                          prod_snap=_prod_snap)
            if _needs_action(state):
                waiting = state.get("waitingFor") or {}
                sig = _wf_signature(color, waiting)
                try:
                    cur_save = get_last_save_id(base_url, game_id)
                except Exception:
                    cur_save = prev_save
                if _is_draft_repick(state):
                    stuck = 0   # repick: not a hang
                    _pk = _repick_pool_key(state)
                    if _pk in repick_done:
                        # Already drafted this pool -> do NOT answer again, otherwise the server keeps the
                        # focus on this player and the other one never gets a turn
                        # dran (Live-Lock unter --parallel, 15.07.). Ueberspringen = warten.
                        continue
                    repick_done.add(_pk)
                elif sig == prev_sig and cur_save == prev_save:
                    stuck += 1
                else:
                    stuck = 0
                prev_sig, prev_save = sig, cur_save
                if stuck >= STUCK_ABORT:
                    log.error("   ❌ Entscheidung %d× ohne Fortschritt (%s) – Partie %d abgebrochen",
                              stuck, sig[2], game_num)
                    return None
                ok = _step_player(
                    state, base_url, pid, all_player_ids, game_id, color,
                    enable_mcts=False, lock=None, db_ready=[True],
                    game_id_for_db=game_id, n_rollouts=0, max_candidates=0,
                    simple_rollout=True, transitions=[],
                    mcts_allowed_now=False,
                    decide_fn=decide_by_color.get(color),
                )
                if ok:
                    acted, idle = True, 0
                break
        if not acted:
            idle += 1
            if idle * POLL_INTERVAL > 5:
                try:
                    any_active = any(
                        get_state(base_url, p).get("thisPlayer", {}).get("isActive", False)
                        or get_state(base_url, p).get("waitingFor")
                        for p in color_to_id.values())
                    if not any_active:
                        g = get_state(base_url, all_player_ids[0]).get("game", {})
                        if (g.get("oxygenLevel", 0) >= 14 and g.get("temperature", -30) >= 8
                                and g.get("oceans", 0) >= 9):
                            log.info("   Niemand aktiv + Parameter maximal → Spielende")
                            return _finalize_selfplay(base_url, color_to_id,
                                                      next(iter(color_to_id)), [], None,
                                                      prod_snap=_prod_snap)
                except Exception:
                    pass
            if idle * POLL_INTERVAL > 120:
                log.warning("   120s ohne Zug → Abbruch (Partie %d)", game_num)
                return None
            time.sleep(POLL_INTERVAL)


def _run_crn_pair(pair_no: int, n_pairs: int, base_url: str, n_players: int,
                  all_colors: list[str], draft: bool, champion_decide,
                  flush=None, master_seed=None):
    """One complete pair of games: game A with a fresh deck, game B as a clone with
    swapped seats - common random numbers, so the deck cancels out of the margin.
    """
    out_results, out_crn = [], []
    prev_id = None
    # Derive this pair's seed deterministically from (master_seed, pair_no): a separate
    # RNG instance, so it is parallel-safe (no shared global state).
    # The same master_seed reproduces pair pair_no with an identical deck.
    pair_seed = (random.Random(f"{master_seed}:{pair_no}").random()
                 if master_seed is not None else None)
    seat1, seat2 = all_colors[0], all_colors[1]
    for half in (0, 1):
        game_num = pair_no * 2 + 1 + half
        is_o2    = (half == 1)
        champ    = seat1 if is_o2 else seat2
        chall    = seat2 if is_o2 else seat1
        decide_by_color = {champ: champion_decide}
        log.info("── Partie %d/%d ──", game_num, n_pairs * 2)
        try:
            game_id, color_to_id = create_mp_game_with_undo(
                base_url, n_players, all_colors, draft=draft,
                random_first=False, seed=(pair_seed if not is_o2 else None),
                cloned_game_id=(prev_id if is_o2 else None))
        except Exception as e:
            log.error("Spiel-Erstellung fehlgeschlagen: %s", e)
            return out_results, out_crn
        if not is_o2:
            prev_id = game_id
            log.info("   🎲 Paar %d Seed: %r", pair_no, pair_seed)
        log.info("   Spiel: %s | IDs: %s", game_id, color_to_id)
        for color, pid in color_to_id.items():
            rolle = "Champion" if color == champ else "Challenger"
            log.info("   🌐 %-9s %s/player?id=%s", f"{color}/{rolle}:", base_url, pid)
        res = _play_crn_game(base_url, game_id, color_to_id, decide_by_color,
                             game_num, n_pairs * 2)
        if res:
            out_results.append(res)
            out_crn.append({"pair": pair_no, "seed": pair_seed,
                            "champ": champ, "chall": chall,
                            "vps": res["vps_by_color"],
                            "mcs": res.get("mc_by_color", {}),
                            "behav": res.get("behav_by_color", {})})
        if flush:
            flush()   # print the game block immediately instead of only at the end of the pair
    return out_results, out_crn


def run_ab_crn_parallel(base_url: str, n_players: int, all_colors: list[str],
                        draft: bool, n_pairs: int, champion_decide, workers: int,
                        master_seed=None):
    """Run N pairs on `workers` threads. Pairs are independent (their own games and
    decks), so they can run in parallel.
    """
    root = logging.getLogger()
    orig_handlers = root.handlers[:]
    router = _PairLogRouter(orig_handlers)
    root.handlers = [router]
    flush_lock = _threading.Lock()
    all_results, all_crn = [], []

    def worker(pair_no: int):
        # buf is still None here -> this line goes straight to console and file
        log.info("▶ Paar %d/%d gestartet", pair_no + 1, n_pairs)
        router.local.buf = []

        def flush():
            buf = router.local.buf
            router.local.buf = []
            with flush_lock:
                for rec in buf:
                    for h in orig_handlers:
                        if rec.levelno >= h.level:
                            h.handle(rec)

        try:
            return _run_crn_pair(pair_no, n_pairs, base_url, n_players,
                                 all_colors, draft, champion_decide,
                                 flush=flush, master_seed=master_seed)
        finally:
            flush()
            router.local.buf = None

    try:
        log.info("🧪 A/B-CRN parallel | %d Paare auf %d Workern | "
                 "Ausgabe gepuffert je Partie, Fortschritt alle 30s", n_pairs, workers)
        from concurrent.futures import as_completed as _as_completed
        with _TPE(max_workers=workers) as pool:
            futures = {pool.submit(worker, p): p for p in range(n_pairs)}
            pending = set(futures)
            last_beat = time.time()
            while pending:
                done = set()
                for fut in list(pending):
                    if fut.done():
                        results, crn = fut.result()
                        all_results.extend(results)
                        all_crn.extend(crn)
                        done.add(fut)
                pending -= done
                if pending and time.time() - last_beat >= 30:
                    log.info("⏳ Fortschritt: %d/%d Paare fertig, %d laufen",
                             n_pairs - len(pending), n_pairs, min(workers, len(pending)))
                    last_beat = time.time()
                if pending:
                    time.sleep(1.0)
    finally:
        root.handlers = orig_handlers
    _summarize_crn(all_crn, n_pairs * 2)
    log.info("✅ %d Partien abgeschlossen", n_pairs * 2)


def _print_replay_command(master_seed):
    """Print a ready-to-copy command that repeats this run with identical decks."""
    import sys
    argv = sys.argv
    parts, i = [], 1
    while i < len(argv):
        if argv[i] == "--seed":
            i += 2;  continue   # skip the flag and its value
        if argv[i].startswith("--seed="):
            i += 1;  continue
        parts.append(argv[i]);  i += 1
    cmd = f"py -3.12 {argv[0]} " + " ".join(parts) + f" --seed {master_seed}"
    log.info("══════════════════════════════════════════════════")
    log.info("🎲 REPLAY (identische Decks) — diesen Befehl fuer den zweiten Lauf kopieren:")
    log.info("   %s", cmd)
    log.info("══════════════════════════════════════════════════")


def run_sequential_selfplay(
    base_url:       str,
    n_players:      int,
    all_colors:     list[str],
    mcts_color:     str,
    n_rollouts:     int,
    max_candidates: int,
    data_file:      str,
    simple_rollout: bool,
    draft:          bool,
    n_games:        int = 1,
    decide_by_color:    dict | None = None,   # colour -> decide() variant
    enable_mcts_global: bool = True,          # False => reiner Heuristik-A/B
    random_first:       bool = True,          # False => feste Sitze (reproduzierbar)
    roles_by_color:     dict | None = None,   # colour -> label for the evaluation
    crn:                bool = False,          # gepaarter CRN-Modus via clonedGamedId
    champion_decide=None,                      # Champion-Heuristik (CRN)
    master_seed=None,   # paired A/B: decks deterministic per pair
):
    """One process, all bots sequentially. The root fix against the rollback race."""
    log.info("🎮 Sequenzieller Self-Play-Treiber | %d Spieler | %d Spiele",
             n_players, n_games)
    log.info("   MCTS-Bot: %s | übrige Farben: reine Heuristik", mcts_color)

    results: list[dict] = []   # one entry per scored game
    crn_games: list[dict] = []   # paired margin raw data per game
    prev_clone_id = None   # game id of the odd game, for cloning

    for game_num in range(1, n_games + 1):
        log.info("── Partie %d/%d ──", game_num, n_games)

        # Odd game = fresh deck (champion in seat 2), even game = clone of the previous deck
        # with swapped roles (champion in seat 1). Both orientations therefore see the same
        # deck, and seat and starting player are balanced across the pair.
        cur_cloned_id   = None
        cur_random_first = random_first
        if crn:
            is_o2 = (game_num % 2 == 0)
            cur_random_first = False
            seat1, seat2 = all_colors[0], all_colors[1]
            champ_color  = seat1 if is_o2 else seat2
            chall_color  = seat2 if is_o2 else seat1
            decide_by_color = {champ_color: champion_decide}
            roles_by_color  = {champ_color: "Champion", chall_color: "Challenger"}
            cur_cloned_id   = prev_clone_id if is_o2 else None

        cur_seed = None
        if crn and master_seed is not None and (game_num % 2 == 1):
            # fresh (odd) game: deterministic deck seed per pair
            cur_seed = random.Random(f"{master_seed}:{(game_num - 1) // 2}").random()

        try:
            game_id, color_to_id = create_mp_game_with_undo(
                base_url, n_players, all_colors, draft=draft,
                random_first=cur_random_first, seed=cur_seed,
                cloned_game_id=cur_cloned_id)
        except Exception as e:
            log.error("Spiel-Erstellung fehlgeschlagen: %s", e)
            continue

        if crn and game_num % 2 == 1:
            prev_clone_id = game_id   # this source clones the next game

        all_player_ids = list(color_to_id.values())
        write_game_coord(game_id, color_to_id, game_num=game_num)  # fuer externen Korpus-Sammler
        log.info("   Spiel: %s | IDs: %s", game_id, color_to_id)
        for color, pid in color_to_id.items():
            rolle = (roles_by_color or {}).get(
                color, "MCTS" if (enable_mcts_global and color == mcts_color)
                                 else "Heuristik")
            log.info("   🌐 %-9s %s/player?id=%s", f"{color}/{rolle}:", base_url, pid)

        locks       = {c: MCTSLock(c, game_id) for c in color_to_id}
        db_ready    = {c: [False] for c in color_to_id}
        transitions: list[dict] = []   # search bot only
        idle        = 0

        # --- Loop and progress guard ---
        # Allow the search only on the FIRST or-action of a turn: only there does the
        # decision sit on a save boundary that rollback_to_save returns to cleanly. On
        # follow-up actions the rollback would undo what has already happened.
        # gespielte erste Aktion loeschen (-> HTTP 400 -> Endlosschleife).
        mcts_first_action_pending = True
        prev_sig   = None   # (colour, waitingFor type, title) of the last action
        prev_save  = -1   # global save id at the last action
        stuck      = 0   # how often the same decision came round without progress
        repick_done = set()   # repick pools this player has already answered
        STUCK_FORCE_HEURISTIC = 3   # from here on, disable the search for this decision
        STUCK_ABORT           = 8   # from here on, abort the game instead of hanging
        # Catch-all: aborts a game that runs unrealistically long. Also catches hangs where
        # the save id does advance but no real end of the game is reached (it fires where the
        # signature and idle guards do not). Normal two-player games take about two minutes.
        GAME_TIME_CAP = 360  # Sekunden
        game_start    = time.time()
        res           = None   # finalize result of this game (None = aborted)

        while True:
            acted      = False
            game_over  = False
            abort_game = False

            if time.time() - game_start > GAME_TIME_CAP:
                log.error("   ⏱ Partie %d läuft >%ds ohne Spielende – Abbruch "
                          "(Hänger-Schutz). Letzte Entscheidung: %r",
                          game_num, GAME_TIME_CAP, prev_sig)
                break

            for color, pid in color_to_id.items():
                try:
                    state = get_state(base_url, pid)
                except Exception:
                    continue

                if state.get("game", {}).get("phase", "") == "end":
                    game_over = True
                    break

                if _needs_action(state):
                    is_mcts = enable_mcts_global and (color == mcts_color)
                    waiting = state.get("waitingFor") or {}

                    # progress since the last action? signature plus global save id
                    sig = _wf_signature(color, waiting)
                    try:
                        cur_save = get_last_save_id(base_url, game_id)
                    except Exception:
                        cur_save = prev_save
                    if _is_draft_repick(state):
                        stuck = 0   # repick: not a hang
                        _pk = _repick_pool_key(state)
                        if _pk in repick_done:
                            # Already drafted this pool -> do NOT answer again, otherwise the server keeps the
                            # focus on this player and the other one never gets a turn (live lock). Skipping
                            # means waiting for the other player. A NEW pack (different key) is answered again.
                            continue
                        repick_done.add(_pk)
                    elif sig == prev_sig and cur_save == prev_save:
                        stuck += 1
                    else:
                        stuck = 0
                    prev_sig, prev_save = sig, cur_save

                    # Fix 1: search only on the first or-action of the turn.
                    mcts_allowed_now = is_mcts and mcts_first_action_pending

                    # Fix 2: if the same decision sticks, first force the heuristic, then (if that does
                    # not help either) abort the game, so a single input cannot block a whole series.
                    if stuck >= STUCK_ABORT:
                        log.error("   ❌ Entscheidung %d× ohne Fortschritt (%s) – "
                                  "Partie %d abgebrochen (Endlosschleifen-Schutz). "
                                  "waitingFor: %r",
                                  stuck, sig[2], game_num, waiting)
                        abort_game = True
                        break
                    if stuck >= STUCK_FORCE_HEURISTIC and mcts_allowed_now:
                        log.warning("   ⚠️ Entscheidung %d× ohne Fortschritt (%s) – "
                                    "erzwinge Heuristik statt MCTS", stuck, sig[2])
                        mcts_allowed_now = False

                    ok = _step_player(
                        state, base_url, pid, all_player_ids, game_id, color,
                        enable_mcts=is_mcts, lock=locks[color],
                        db_ready=db_ready[color], game_id_for_db=game_id,
                        n_rollouts=n_rollouts, max_candidates=max_candidates,
                        simple_rollout=simple_rollout,
                        transitions=transitions,
                        mcts_allowed_now=mcts_allowed_now,
                        decide_fn=(decide_by_color or {}).get(color),
                    )
                    if ok:
                        acted = True
                        idle  = 0
                        # The turn's search budget is spent with the first or-action;
                        # a move by the opponent opens a new turn.
                        if is_mcts and mcts_allowed_now and waiting.get("type") == "or":
                            mcts_first_action_pending = False
                        elif not is_mcts:
                            mcts_first_action_pending = True
                    break   # poll again after a move (whose turn is it now?)

            if abort_game:
                break

            if game_over:
                res = _finalize_selfplay(base_url, color_to_id, mcts_color,
                                         transitions, data_file)
                break

            if not acted:
                idle += 1
                # backup end-of-game: nobody active and all global parameters maxed
                if idle * POLL_INTERVAL > 5:
                    try:
                        any_active = False
                        for pid in color_to_id.values():
                            s = get_state(base_url, pid)
                            if (s.get("thisPlayer", {}).get("isActive", False)
                                    or s.get("waitingFor")):
                                any_active = True
                                break
                        if not any_active:
                            ms = get_state(base_url, color_to_id[mcts_color]).get("game", {})
                            if (ms.get("oxygenLevel", 0) >= 14
                                    and ms.get("temperature", -30) >= 8
                                    and ms.get("oceans", 0) >= 9):
                                log.info("   Niemand aktiv + Parameter maximal → Spielende")
                                res = _finalize_selfplay(base_url, color_to_id, mcts_color,
                                                         transitions, data_file)
                                break
                    except Exception:
                        pass
                # Diagnostics: every ~10 s print the full server state of both players, so it is
                # visible what the server is waiting for.
                if idle % 20 == 0:
                    parts = []
                    for c, p in color_to_id.items():
                        try:
                            s  = get_state(base_url, p)
                            g  = s.get("game", {})
                            w  = s.get("waitingFor")
                            parts.append(
                                "%s: phase=%s gen=%s aktiv=%s waiting=%s "
                                "O2=%s Temp=%s Ozeane=%s" % (
                                    c, g.get("phase", "?"), g.get("generation", "?"),
                                    s.get("thisPlayer", {}).get("isActive", False),
                                    (w.get("type", "?") if w else "None"),
                                    g.get("oxygenLevel", "?"), g.get("temperature", "?"),
                                    g.get("oceans", "?"),
                                )
                            )
                        except Exception as e:
                            parts.append("%s: <State-Fehler %s>" % (c, e))
                    log.info("   ⏳ Niemand am Zug | %s", " || ".join(parts))
                if idle * POLL_INTERVAL > 120:
                    log.warning("   120s ohne Zug → Abbruch (Spiel evtl. hängengeblieben)")
                    break
                time.sleep(POLL_INTERVAL)

        # Record the result of this game (None = aborted, discarded).
        if res:
            results.append(res)
            if crn:
                crn_games.append({
                    "pair":  (game_num - 1) // 2,
                    "champ": all_colors[0] if game_num % 2 == 0 else all_colors[1],
                    "chall": all_colors[1] if game_num % 2 == 0 else all_colors[0],
                    "vps":   res["vps_by_color"],
                    "mcs":   res.get("mc_by_color", {}),
                    "behav": res.get("behav_by_color", {}),
                })

        # clean up (lock file, if the search wrote one)
        try:
            os.unlink(LOCK_FILE)
        except Exception:
            pass
        time.sleep(2.0)

    # ── summary ──
    if crn:
        _summarize_crn(crn_games, n_games)
    elif results:
        n_scored = len(results)
        log.info("══════════════════════════════════════════════════")
        log.info("📊 Auswertung: %d von %d Partien gewertet", n_scored, n_games)
        for color in all_colors:
            # "win" = leading in this game (a tie counts for every leading colour, which with
            # two players is practically always unambiguous)
            wins = sum(1 for r in results
                       if r["vps_by_color"].get(color, -1) >= max(r["vps_by_color"].values()))
            avg_vp = sum(r["vps_by_color"].get(color, 0) for r in results) / n_scored
            rolle  = (roles_by_color or {}).get(
                color, "MCTS" if (enable_mcts_global and color == mcts_color)
                                 else "Heuristik")
            log.info("   %-7s (%-9s): %2d/%d Siege (%3.0f%%) | ⌀VP %.1f",
                     color, rolle, wins, n_scored, 100 * wins / n_scored, avg_vp)
        avg_rank  = sum(r["mcts_rank"] for r in results) / n_scored
        mcts_wins = sum(1 for r in results if r["mcts_won"])
        log.info("   → MCTS-Referenz (%s): %d/%d Siege, ⌀Rang %.2f",
                 mcts_color, mcts_wins, n_scored, avg_rank)
        log.info("   (Gleichstände zählen für jede führende Farbe als Sieg)")
        log.info("══════════════════════════════════════════════════")
    else:
        log.warning("📊 Keine Partie lieferte ein verwertbares Ergebnis "
                    "(alle abgebrochen?)")

    log.info("✅ %d Partien abgeschlossen", n_games)


def run_auto_games(
    base_url:       str,
    my_color:       str,
    n_games:        int,
    n_players:      int,
    all_colors:     list[str],
    n_rollouts:     int,
    max_candidates: int,
    data_file:      str,
    simple_rollout: bool,
    server_id:      str,
    draft:          bool = False,
    mcts_role:      str  = "creator",
):
    """Unattended multi-game mode.

    The bot with the first colour (alphabetically) creates the game and writes the
    ids into the coordination file; the others read them from there.
    """
    log.info("🎮 Auto-Modus | Farbe: %s | %d Spieler | %d Spiele",
             my_color, n_players, n_games)

    # the creator is always the first colour in all_colors (deterministic)
    creator_color = all_colors[0]
    is_creator    = (my_color == creator_color)
    log.info("   Ersteller: %s | Ich bin Ersteller: %s", creator_color, is_creator)

    # Resolve the search role: only ONE bot may run the search, otherwise rollbacks race.
    enable_mcts = (mcts_role == "all") or (mcts_role == "creator" and is_creator)
    log.info("   MCTS-Rolle: %s | Dieser Bot fährt MCTS: %s", mcts_role, enable_mcts)
    if mcts_role == "all" and n_players > 1:
        log.warning("   ⚠ mcts-role=all: mehrere Bots rollen zurück – Rollback-Race möglich!")

    for game_num in range(1, n_games + 1):
        log.info("── Partie %d/%d ──", game_num, n_games)

        is_creator_this_round = is_creator
        my_player_id  = None
        all_player_ids = []

        # delete the old coordination file so no game from a previous round is read
        if is_creator:
            try:
                os.unlink(GAME_COORD_FILE)
            except FileNotFoundError:
                pass

        # all bots signal readiness
        signal_ready(my_color, game_num)

        if is_creator:
            # the creator waits until all others are ready, then creates the game
            log.info("   Warte auf alle Bots...")
            wait_for_all_ready(all_colors, game_num, timeout=300.0)
            log.info("   Erstelle neues Spiel als %s...", my_color)
            try:
                game_id, color_to_id = create_mp_game_with_undo(
                    base_url, n_players, all_colors, draft=draft
                )
                write_game_coord(game_id, color_to_id, game_num=game_num)
                my_player_id   = color_to_id[my_color]
                all_player_ids = list(color_to_id.values())
            except Exception as e:
                log.error("Spiel-Erstellung fehlgeschlagen: %s", e)
                time.sleep(5)
                continue
        else:
            # non-creators: wait until the coordination file is written
            result = read_game_coord(my_color, timeout=300.0, game_num=game_num)
            if result is None:
                log.error("Koordinierungsdatei nicht erhalten – überspringe Partie %d",
                          game_num)
                continue
            game_id, my_player_id = result
            try:
                with open(GAME_COORD_FILE) as f:
                    coord = json.load(f)
                all_player_ids = list(coord["color_to_id"].values())
            except Exception:
                all_player_ids = [my_player_id]

        log.info("   Spiel: %s | Meine ID: %s", game_id, my_player_id)

        try:
            run_mcts_bot_mp(
                base_url       = base_url,
                player_id      = my_player_id,
                game_id        = game_id,
                my_color       = my_color,
                all_player_ids = all_player_ids,
                n_rollouts     = n_rollouts,
                max_candidates = max_candidates,
                data_file      = data_file,
                simple_rollout = simple_rollout,
                server_id      = server_id,
                enable_mcts    = enable_mcts,
            )
        except Exception as e:
            import traceback
            log.error("Partie %d fehlgeschlagen: %s", game_num, e)
            log.error("Traceback:\n%s", traceback.format_exc())

        # clean up after the game
        if is_creator:
            try:
                os.unlink(GAME_COORD_FILE)
            except Exception:
                pass
        # clean up lock and ready files
        for f in [LOCK_FILE, READY_FILE]:
            try:
                os.unlink(f)
            except Exception:
                pass

        # wait so the other bots can notice the end of the game
        # before the next coordination file is written
        time.sleep(100.0)   # wait until all bots have noticed the end of the game (timeout 90 s)

    release_color(my_color)
    log.info("✅ %d Partien abgeschlossen", n_games)


def run_vs_human(
    base_url:       str,
    bot_color:      str,
    draft:          bool,
    n_rollouts:     int   = ROLLOUT_MOVES,
    max_candidates: int   = MAX_CANDIDATES,
    data_file:      str   = MCTS_DATA_FILE,
    simple_rollout: bool  = False,
    enable_mcts:    bool  = True,
    expansions:     set   = None,
    settings:       dict  = None,
    board:          str   = "random",
):
    """One versus one: bot against a human player.

    Creates a two-player game and prints the join link for the human.
    """
    colors = VALID_COLORS[:2]
    if bot_color not in colors:
        bot_color = colors[0]
    human_color = next(c for c in colors if c != bot_color)

    # Menschenpartien: zufaelliges offizielles Board (Tharsis/Hellas/Elysium ->
    # variance in milestones and awards) and the draft variant, as in a real match
    # and in the later deployment environment.
    # "random" picks one of the three official boards, which varies the milestones and
    # awards between games. A --settings file still takes precedence over this.
    if board == "random":
        board = random.choice(OFFICIAL_BOARDS)
    game_id, color_to_id = create_mp_game_with_undo(
        base_url, 2, colors, draft=draft, board=board,
        human_color=human_color, expansions=expansions, settings=settings)
    bot_id   = color_to_id[bot_color]
    human_id = color_to_id[human_color]

    # Confirmation: what was sent, and which board the server REALLY created (the board
    # name is not in the state -> recognised through the board-specific milestones)
    print(f"[Spielerstellung] gesendet: board='{board}', draftVariant={draft}, "
          f"expansions=corpera+{sorted(expansions) if expansions else '(nur Base)'}")
    _BOARDS = {
        "THARSIS":  {"Terraformer", "Mayor", "Gardener", "Builder", "Planner"},
        "HELLAS":   {"Diversifier", "Tactician", "Polar Explorer", "Energizer", "Rim Settler"},
        "ELYSIUM":  {"Generalist", "Specialist", "Ecologist", "Tycoon", "Legend"},
    }
    try:
        _st = requests.get(f"{base_url}/api/player", params={"id": bot_id}, timeout=10).json()
        _ms = {m.get("name") for m in _st.get("game", {}).get("milestones", [])}
        _board = next((b for b, s in _BOARDS.items() if s & _ms), "UNBEKANNT")
        print(f"[Spielerstellung] Board laut Meilensteinen: {_board}   {sorted(_ms)}")
    except Exception as _e:
        print(f"[Spielerstellung] Board-Check uebersprungen: {_e}")

    log.info("=" * 64)
    log.info("👤 DU spielst als %s – öffne diesen Link im Browser:", human_color.upper())
    log.info("      %s/player?id=%s", base_url, human_id)
    if enable_mcts:
        log.info("🤖 Bot spielt als %s (MCTS) und zieht automatisch.", bot_color.upper())
        log.info("   Der Bot wartet geduldig auf deine Züge – kein 90s-Abbruch.")
    else:
        log.info("🤖 Bot spielt als %s (REINE HEURISTIK, keine Rollouts).", bot_color.upper())
        log.info("   Kein Server-Snapshot/Restore -> deine Züge werden NICHT zurückgesetzt.")
    log.info("=" * 64)

    run_mcts_bot_mp(
        base_url       = base_url,
        player_id      = bot_id,
        game_id        = game_id,
        my_color       = bot_color,
        all_player_ids = [bot_id, human_id],
        n_rollouts     = n_rollouts,
        max_candidates = max_candidates,
        data_file      = data_file,
        simple_rollout = simple_rollout,
        enable_mcts    = enable_mcts,
        human_opponent = True,
    )


def run_join_game(
    base_url:       str,
    player_id:      str,
    n_rollouts:     int   = ROLLOUT_MOVES,
    max_candidates: int   = MAX_CANDIDATES,
    data_file:      str   = MCTS_DATA_FILE,
    simple_rollout: bool  = False,
):
    """Join an existing game using only the bot's own player id.

    Nothing is created here - the game already exists, the bot only answers.
    """
    state    = get_state(base_url, player_id)
    my_color = state.get("thisPlayer", {}).get("color", "?")
    game_id  = state.get("game", {}).get("id") or state.get("id") or f"join-{player_id}"

    log.info("=" * 64)
    log.info("🤝 Beitritt zu laufender Partie als %s", my_color.upper())
    log.info("   player_id=%s | game=%s", player_id, game_id)
    log.info("   Reine Heuristik – keine Rollouts, kein Undo, fair gegenüber Mitspielern.")
    log.info("=" * 64)

    run_mcts_bot_mp(
        base_url       = base_url,
        player_id      = player_id,
        game_id        = game_id,
        my_color       = my_color,
        all_player_ids = [player_id],   # own id only -> no access to other players
        n_rollouts     = n_rollouts,
        max_candidates = max_candidates,
        data_file      = data_file,
        simple_rollout = simple_rollout,
        enable_mcts    = False,   # fair: expert system only
        human_opponent = True,   # real opponents -> no idle abort
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TM Multiplayer MCTS Bot (2–6 Spieler)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # 2 Bots, 50 Spiele automatisch
  Terminal 1: py -3.12 tm_mcts_mp.py --color red  --auto-games 50
  Terminal 2: py -3.12 tm_mcts_mp.py --color blue --auto-games 50

  # 4 Bots, 20 Spiele
  Terminal 1: py -3.12 tm_mcts_mp.py --color red    --auto-games 20 --players 4
  Terminal 2: py -3.12 tm_mcts_mp.py --color blue   --auto-games 20 --players 4
  Terminal 3: py -3.12 tm_mcts_mp.py --color yellow --auto-games 20 --players 4
  Terminal 4: py -3.12 tm_mcts_mp.py --color green  --auto-games 20 --players 4

  # Manuell (bestehendes Spiel)
  py -3.12 tm_mcts_mp.py --player-id pXXX --game-id gYYY --color red
        """
    )

    parser.add_argument("--color",         default=None,
                        choices=VALID_COLORS + [None],
                        help="Eigene Farbe (optional – wird automatisch zugeteilt)")
    parser.add_argument("--auto-games",    type=int, default=0,
                        help="Automatisch N Spiele spielen")
    parser.add_argument("--players",       type=int, default=2,
                        choices=range(2, 7),
                        help="Anzahl Spieler (2-6, default: 2)")
    parser.add_argument("--colors",        nargs="+", default=None,
                        help="Farben aller Spieler (default: erste N Farben)")
    parser.add_argument("--player-id",     default=None,
                        help="Player-ID (für manuellen Modus)")
    parser.add_argument("--game-id",       default=None,
                        help="Game-ID (für manuellen Modus)")
    parser.add_argument("--all-player-ids", nargs="+", default=None,
                        help="Alle Player-IDs (für Rollout-Gegner im manuellen Modus)")
    parser.add_argument("--url",           default=DEFAULT_URL)
    parser.add_argument("--rollouts",      default=ROLLOUT_MOVES, type=int,
                        help=f"Rollout-Züge pro Kandidat (default: {ROLLOUT_MOVES})")
    parser.add_argument("--candidates",    default=MAX_CANDIDATES, type=int,
                        help=f"Max. Kandidaten (default: {MAX_CANDIDATES})")
    parser.add_argument("--db",            default=None,
                        help="Kartendatenbank (default: card_db.json im Arbeitsverzeichnis)")
    parser.add_argument("--settings", default=None,
                        help="tm_settings.json der echten Runde: uebernimmt board, randomMA, "
                             "includeFanMA, fastModeOption, shuffleMapOption, bannedCards, "
                             "customCorporationsList, startingCorporations/Ceos usw. "
                             "(--expansions ueberschreibt die Erweiterungen der Datei)")
    parser.add_argument("--expansions", default="",
                        help="Komma-Liste der Erweiterungen fuer --vs-human, z.B. "
                             "'venus,prelude,prelude2'. corpera ist immer an. "
                             "(default: nur Base+Corporate Era)")
    parser.add_argument("--champion-db",   default=None,
                        help="Eingefrorene Kartendatenbank NUR fuer den Champion "
                             "(default: gleiche wie --db). Erlaubt sauberes Gaten von card_db-Aenderungen.")
    parser.add_argument("--data",          default=MCTS_DATA_FILE,
                        help=f"Ausgabedatei für Trainingsdaten (default: {MCTS_DATA_FILE})")
    parser.add_argument("--server-id",     default="EIERWIRBRAUCHENEIER")
    parser.add_argument("--board", default="random",
                        choices=[*OFFICIAL_BOARDS, "random"],
                        help="board for --vs-human (default: random = one of "
                             "Tharsis/Hellas/Elysium). A --settings file takes precedence.")
    parser.add_argument("--draft", action="store_true",
                        help="Draft-Variante aktivieren (jeder Spieler wählt Karten aus Paket)")
    parser.add_argument("--simple-rollout", action="store_true",
                        help="Gegner während Rollout ignorieren (Option A, schneller)")
    parser.add_argument("--mcts-role", choices=["creator", "all", "none"],
                        default="creator",
                        help="Wer fährt MCTS: 'creator' = nur Spielersteller (Default, "
                             "verhindert Rollback-Race), 'all' = alle Bots (racy!), "
                             "'none' = reine Heuristik")
    parser.add_argument("--lock-timeout",  default=LOCK_TIMEOUT, type=float,
                        help=f"Sekunden bis abgestürzter Bot Lock verliert (default: {LOCK_TIMEOUT})")
    parser.add_argument("--sequential", action="store_true",
                        help="Sequenzieller Ein-Prozess-Self-Play: EIN Prozess steuert "
                             "alle Bots streng nacheinander (kein Rollback-Race). "
                             "Nur dieses eine Terminal starten. Erste Farbe fährt MCTS.")

    parser.add_argument("--ab", action="store_true",
                        help="A/B-Heuristikvergleich (Ein-Prozess, MCTS AUS, feste Sitze): "
                             "Challenger = Arbeitsversion tm_bot.decide, Champion = "
                             "eingefrorenes Modul. Nutzt den --sequential-Treiber.")
    parser.add_argument("--champion-module", default="tm_bot_champion",
                        help="Modulname der eingefrorenen Champion-Heuristik "
                             "(default: tm_bot_champion)")
    parser.add_argument("--champion-color", default=None,
                        help="Farbe, die der Champion spielt (default: zweite Farbe). "
                             "Für ein CRN-Paar denselben --seed mit getauschter Farbe laufen lassen.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Master-Seed für --ab-crn. Pro Paar wird daraus "
                             "deterministisch ein Deck-Seed abgeleitet (parallel-fest). "
                             "Ohne Angabe wird einer erzeugt; der Lauf gibt am Ende den "
                             "fertigen Replay-Befehl aus. Gleicher --seed = identische "
                             "Decks → deckgenauer A/B-Vergleich über zwei Läufe.")
    parser.add_argument("--ab-crn", action="store_true",
                        help="Gepaartes A/B mit echtem CRN: jede ungerade Partie ist ein "
                             "frisches Deck (Champion Sitz 2), jede gerade Partie klont es "
                             "(clonedGamedId) mit getauschten Rollen (Champion Sitz 1). "
                             "--auto-games zählt hier PAARE (1 Paar = 2 Partien). MCTS AUS.")

    parser.add_argument("--vs-human", action="store_true",
                        help="1v1 gegen Menschen: erstellt Spiel mit Undo, gibt dir den "
                             "Browser-Link, Bot spielt automatisch (kein Idle-Abbruch)")

    parser.add_argument("--no-mcts", action="store_true",
                        help="Nur mit --vs-human: Bot spielt REIN HEURISTISCH (keine "
                             "Server-Rollouts/Undo) -> deine Zuege werden nicht zurueckgesetzt.")

    parser.add_argument("--join", action="store_true",
                        help="Tritt einer bestehenden Partie nur über --player-id bei und "
                             "spielt rein heuristisch (fair: keine Rollouts/kein Undo). "
                             "Zum Ersetzen eines Spielers in einer echten Runde.")

    parser.add_argument("--post-wait", type=float, default=None,
                        help="Wartezeit (s) nach jedem POST (Default 2.0). "
                             "0.3-0.5 beschleunigt Partien stark; vorher per "
                             "A/A-Lauf verifizieren (Null-Paare muessen 0 bleiben)")
    parser.add_argument("--poll", type=float, default=None,
                        help="Poll-Intervall (s) wenn niemand am Zug ist (Default 2.0)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Anzahl parallel laufender CRN-Paare (nur --ab-crn). "
                             "Jedes Paar laeuft intern sequenziell (A, Klon, B); "
                             "Logs werden pro Paar gepuffert ausgegeben")

    parser.add_argument("--log", action="store_true",
                        help="Schreibt das Log zusätzlich in logs/tm_<modus>_<timestamp>.log "
                             "(UTF-8, zeilenweise geflusht – läuft auch weiter, wenn die "
                             "Konsole z.B. durch QuickEdit-Markierung blockiert ist)")

    args = parser.parse_args()

    # speed: override POST_WAIT / POLL_INTERVAL in BOTH modules
    # (this module imports the names by value from tm_bot)
    import tm_bot as _tb
    if args.post_wait is not None:
        _tb.POST_WAIT = args.post_wait
        globals()["POST_WAIT"] = args.post_wait
    if args.poll is not None:
        _tb.POLL_INTERVAL = args.poll
        globals()["POLL_INTERVAL"] = args.poll

    # File logging: handler on the root logger, so tm_bot and tm_mcts land there too.
    if args.log:
        if args.ab_crn:      mode_tag = "ab-crn"
        elif args.ab:        mode_tag = "ab"
        elif args.vs_human:  mode_tag = "vs-human"
        elif args.join:      mode_tag = "join"
        elif args.sequential: mode_tag = "sequential"
        elif args.auto_games > 0: mode_tag = "auto"
        else:                mode_tag = "manuell"
        os.makedirs("logs", exist_ok=True)
        ts       = time.strftime("%Y-%m-%d_%H-%M-%S")
        log_path = os.path.join("logs", f"tm_{mode_tag}_{ts}.log")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-5s %(message)s", datefmt="%H:%M:%S"))
        logging.getLogger().addHandler(fh)
        log.info("📝 Log-Datei: %s", log_path)

    # Kartendatenbank finden
    db_path = args.db or find_card_db()
    load_card_db(db_path)
    # The champion runs on a frozen card database when given, so card database changes
    # become asymmetric (challenger only) and therefore measurable.
    champ_db_path = args.champion_db or db_path
    if args.champion_db:
        log.info("Champion nutzt eingefrorene card_db: %s", champ_db_path)

    # Gepaartes CRN-A/B: Decks via clonedGamedId reproduziert, Rollen/Sitze
    # balanced across the pair. --auto-games counts pairs.
    if args.ab_crn:
        all_colors = args.colors or VALID_COLORS[:args.players]
        champ_decide = load_decide_variant(args.champion_module, champ_db_path)
        pairs   = args.auto_games if args.auto_games > 0 else 30
        n_games = pairs * 2
        # Master seed: taken from --seed or generated. Per pair a
        # a deck seed is derived from it deterministically (parallel-safe).
        master_seed = args.seed if args.seed is not None else random.randrange(2**31)
        log.info("🧪 A/B-CRN | %d Paare (=%d Partien) | Champion-Modul=%s | "
                 "MCTS AUS | Decks gepaart via clonedGamedId | master_seed=%d",
                 pairs, n_games, args.champion_module, master_seed)
        if args.parallel > 1:
            run_ab_crn_parallel(
                base_url=args.url, n_players=args.players,
                all_colors=all_colors, draft=args.draft, n_pairs=pairs,
                champion_decide=champ_decide, workers=args.parallel,
                master_seed=master_seed)
            _print_replay_command(master_seed)
            return
        run_sequential_selfplay(
            base_url       = args.url,
            n_players      = args.players,
            all_colors     = all_colors,
            mcts_color     = all_colors[0],   # finalize label only; the search is off
            n_rollouts     = args.rollouts,
            max_candidates = args.candidates,
            data_file      = args.data,
            simple_rollout = args.simple_rollout,
            draft          = args.draft,
            n_games        = n_games,
            enable_mcts_global = False,
            random_first       = False,
            crn                = True,
            champion_decide    = champ_decide,
            master_seed        = master_seed,
        )
        _print_replay_command(master_seed)
        return

    # A/B-Heuristikvergleich: Challenger (Live) vs. Champion (eingefroren),
    # Search off globally, fixed seats. Reproducible through --seed.
    if args.ab:
        all_colors = args.colors or VALID_COLORS[:args.players]
        if args.seed is not None:
            random.seed(args.seed)   # reproducible sequence of game seeds
        champ_color = args.champion_color or (
            all_colors[1] if len(all_colors) > 1 else all_colors[0])
        champ_decide = load_decide_variant(args.champion_module, champ_db_path)
        # The champion colour uses the frozen variant; all other colours fall back to the
        # live heuristic (challenger).
        decide_by_color = {champ_color: champ_decide}
        roles_by_color  = {c: ("Champion" if c == champ_color else "Challenger")
                           for c in all_colors}
        n_games = args.auto_games if args.auto_games > 0 else 1
        log.info("🧪 A/B | Champion=%s (%s) vs. Challenger=übrige | MCTS AUS | "
                 "feste Sitze | %d Spiele%s", champ_color, args.champion_module,
                 n_games, f" | seed={args.seed}" if args.seed is not None else "")
        run_sequential_selfplay(
            base_url       = args.url,
            n_players      = args.players,
            all_colors     = all_colors,
            mcts_color     = champ_color,   # for the finalize label only; the search is off globally
            n_rollouts     = args.rollouts,
            max_candidates = args.candidates,
            data_file      = args.data,
            simple_rollout = args.simple_rollout,
            draft          = args.draft,
            n_games        = n_games,
            decide_by_color    = decide_by_color,
            enable_mcts_global = False,
            random_first       = False,
            roles_by_color     = roles_by_color,
        )
        return

    # Sequential single-process mode: one process drives all bots,
    # no colour registration or coordination files needed.
    if args.sequential:
        all_colors = args.colors or VALID_COLORS[:args.players]
        n_games    = args.auto_games if args.auto_games > 0 else 1
        run_sequential_selfplay(
            base_url       = args.url,
            n_players      = args.players,
            all_colors     = all_colors,
            mcts_color     = all_colors[0],
            n_rollouts     = args.rollouts,
            max_candidates = args.candidates,
            data_file      = args.data,
            simple_rollout = args.simple_rollout,
            draft          = args.draft,
            n_games        = n_games,
        )
        return

    # Echte Runden-Einstellungen laden (--settings tm_settings.json)
    _settings = None
    if getattr(args, "settings", None):
        with open(args.settings, encoding="utf-8-sig") as _f:
            _settings = json.load(_f)
        _exp_on = [k for k, v in (_settings.get("expansions") or {}).items() if v]
        print(f"[Settings] {args.settings}: board={_settings.get('board')} "
              f"randomMA={_settings.get('randomMA')} fanMA={_settings.get('includeFanMA')} "
              f"fastMode={_settings.get('fastModeOption')} shuffleMap={_settings.get('shuffleMapOption')} "
              f"corps={_settings.get('startingCorporations')} ceos={_settings.get('startingCeos')} "
              f"| Erweiterungen: {_exp_on} | {len(_settings.get('bannedCards') or [])} gebannte Karten")

    # One versus one against a human: create the game, print the link, bot plays on.
    if args.vs_human:
        run_vs_human(
            base_url       = args.url,
            bot_color      = args.color or VALID_COLORS[0],
            draft          = args.draft or bool(_settings and _settings.get("draftVariant")),
            expansions     = {e.strip() for e in args.expansions.split(",") if e.strip()},
            settings       = _settings,
            n_rollouts     = args.rollouts,
            max_candidates = args.candidates,
            data_file      = args.data,
            simple_rollout = args.simple_rollout,
            enable_mcts    = not args.no_mcts,
            board          = args.board,
        )
        return

    # Join a real game: own player id only, pure heuristic, fair.
    if args.join:
        if not args.player_id:
            parser.error("--join benötigt --player-id <eigene Spieler-ID>")
        run_join_game(
            base_url       = args.url,
            player_id      = args.player_id,
            n_rollouts     = args.rollouts,
            max_candidates = args.candidates,
            data_file      = args.data,
            simple_rollout = args.simple_rollout,
        )
        return

    # determine the colour: automatically or manually
    if args.color:
        my_color = args.color
    else:
        my_color = register_color(allowed_colors=VALID_COLORS[:args.players])

    # determine the colours of all players
    if args.colors:
        all_colors = args.colors
    else:
        all_colors = VALID_COLORS[:args.players]

    # make sure my_color is in all_colors
    if my_color not in all_colors:
        all_colors = VALID_COLORS[:args.players]

    # Lock-Timeout global setzen
    import tm_mcts_mp as _self
    _self.LOCK_TIMEOUT = args.lock_timeout

    if args.auto_games > 0:
        run_auto_games(
            base_url       = args.url,
            my_color       = my_color,
            n_games        = args.auto_games,
            n_players      = args.players,
            all_colors     = all_colors,
            n_rollouts     = args.rollouts,
            max_candidates = args.candidates,
            data_file      = args.data,
            simple_rollout = args.simple_rollout,
            server_id      = args.server_id,
            draft          = args.draft,
            mcts_role      = args.mcts_role,
        )

    elif args.player_id and args.game_id:
        # Manueller Modus
        all_player_ids = args.all_player_ids or [args.player_id]
        run_mcts_bot_mp(
            base_url       = args.url,
            player_id      = args.player_id,
            game_id        = args.game_id,
            my_color       = my_color,
            all_player_ids = all_player_ids,
            n_rollouts     = args.rollouts,
            max_candidates = args.candidates,
            data_file      = args.data,
            simple_rollout = args.simple_rollout,
            server_id      = args.server_id,
            enable_mcts    = (args.mcts_role != "none"),
        )

    else:
        parser.error("Entweder --auto-games oder (--player-id + --game-id) angeben")


if __name__ == "__main__":
    main()
