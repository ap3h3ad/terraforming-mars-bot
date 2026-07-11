"""
Terraforming Mars – MCTS Bot (Stufe 1)

Erweitert tm_bot.py um Monte Carlo Tree Search:
- Für jeden Kandidaten-Zug: POST, Rollout, Undo
- Wählt Zug mit bestem Rollout-Score
- Speichert Trainingsdaten mit echten Rollout-Scores

Voraussetzung: Spiel muss mit undoOption: true erstellt werden!

Ausführen:
  py -3.12 tm_mcts.py --player-id pXXX --url http://localhost:9000
  py -3.12 tm_mcts.py --player-id pXXX --rollouts 8 --candidates 4
"""

import argparse
import json
import logging
import os
import random
import time

import requests

# Importiere alles aus tm_bot
from tm_bot import (
    CARD_DB, MC_RESERVE,
    load_card_db,
    get_state, post_input,
    handle_or, decide,
    score_card, score_action,
    turns_left, can_convert_plants, can_convert_heat,
    choose_best_space, build_payment,
)
try:
    from tm_model import game_state_features, card_features
except ImportError:
    def game_state_features(state): return []
    def card_features(info, **kw): return []

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mcts")

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
DEFAULT_URL       = "http://localhost:9000"
POLL_INTERVAL     = 0.3
POST_WAIT         = 0.4
ERROR_WAIT        = 2.0
MAX_ERRORS        = 8
ROLLOUT_MOVES     = 8    # Zufällige Züge pro Rollout
MCTS_MIN_DELTA    = 3.0  # Minimaler Rollout-Spread damit MCTS die Heuristik überstimmt
MAX_CANDIDATES    = 5    # Beste Kandidaten die evaluiert werden
MCTS_DATA_FILE    = "mcts_data.jsonl"

# ---------------------------------------------------------------------------
# Rollout-Score: misst Spielzustand nach Rollout
# ---------------------------------------------------------------------------

def rollout_score(state: dict) -> float:
    """
    Bewertet den Spielzustand nach einem Rollout.
    Kombiniert TR, Produktion und Parameter-Fortschritt.
    Mit TM_LEARNED_LEAF=1: stattdessen der GELERNTE End-Margin (Weg A).
    """
    import os as _os
    if _os.environ.get("TM_LEARNED_LEAF", "") not in ("", "0", "false", "False"):
        try:
            from learned_value import learned_value_from_server
            _lm = learned_value_from_server(state)
            if _lm is not None:
                return _lm
        except Exception:
            pass

    p = state.get("thisPlayer", {})
    g = state.get("game", {})

    tr      = p.get("terraformRating", 14)
    mc_prod = p.get("megacreditProduction", 0)
    st_prod = p.get("steelProduction", 0)
    ti_prod = p.get("titaniumProduction", 0)
    pl_prod = p.get("plantProduction", 0)
    en_prod = p.get("energyProduction", 0)
    ht_prod = p.get("heatProduction", 0)

    oxygen  = g.get("oxygenLevel", 0)
    temp    = g.get("temperature", -30)
    oceans  = g.get("oceans", 0)

    vp_estimate = state.get("thisPlayer", {}).get(
        "victoryPointsBreakdown", {}).get("total", tr)

    prod_value = (
        mc_prod * 5 + st_prod * 8 + ti_prod * 10 +
        pl_prod * 10 + en_prod * 7 + ht_prod * 6
    )
    param_progress = (
        oxygen / 14.0 * 20 + (temp + 30) / 38.0 * 15 + oceans / 9.0 * 15
    )
    return vp_estimate * 2 + prod_value * 0.5 + param_progress


def find_undo_index(state: dict) -> int | None:
    """Findet den Index der Undo-Option im waitingFor."""
    waiting = state.get("waitingFor", {})
    options = waiting.get("options", [])
    for i, opt in enumerate(options):
        label = opt.get("buttonLabel", "")
        title = str(opt.get("title", "")).lower()
        if label == "Undo" or "undo" in title:
            return i
    return None


def do_undo(base_url: str, game_id: str, state: dict) -> bool:
    """Macht den letzten Zug rückgängig via load_game (1 Zug = 2 Saves)."""
    try:
        r = requests.put(
            f"{base_url}/load_game",
            json={"gameId": game_id, "rollbackCount": 2},
            timeout=10,
        )
        if r.status_code == 200:
            time.sleep(POST_WAIT)
            return True
        log.warning("  load_game Undo fehlgeschlagen: %d %s", r.status_code, r.text[:80])
        return False
    except Exception as e:
        log.warning("  Undo fehlgeschlagen: %s", e)
        return False


def create_game_with_undo(base_url: str) -> tuple[str, str]:
    """Erstellt ein neues Solo-Spiel mit undoOption:True und wartet bis es in der DB ist."""
    import random
    payload = {
        "players": [{"name": "mcts", "color": "red", "beginner": False, "handicap": 0, "first": True}],
        "expansions": {
            "corpera": True, "promo": False, "venus": False, "colonies": False,
            "prelude": False, "prelude2": False, "turmoil": False, "community": False,
            "ares": False, "moon": False, "pathfinders": False, "ceo": False,
            "starwars": False, "underworld": False, "deltaProject": False,
        },
        "board": "tharsis", "seed": random.random(), "randomFirstPlayer": False,
        "undoOption": True, "showTimers": False, "fastModeOption": False,
        "showOtherPlayersVP": False, "aresExtremeVariant": False,
        "politicalAgendasExtension": "Standard", "solarPhaseOption": False,
        "removeNegativeGlobalEventsOption": False, "modularMA": False,
        "draftVariant": False, "initialDraft": False, "preludeDraftVariant": False,
        "ceosDraftVariant": False, "startingCorporations": 2, "shuffleMapOption": False,
        "randomMA": "No randomization", "includeFanMA": False, "soloTR": False,
        "customCorporationsList": [], "bannedCards": [], "includedCards": [],
        "customColoniesList": [], "customPreludes": [],
        "requiresMoonTrackCompletion": False, "requiresVenusTrackCompletion": False,
        "moonStandardProjectVariant": False, "moonStandardProjectVariant1": False,
        "altVenusBoard": False, "twoCorpsVariant": False,
        "customCeos": [], "startingCeos": 3, "startingPreludes": 4,
    }
    r = requests.post(f"{base_url}/api/creategame", json=payload, timeout=15)
    r.raise_for_status()
    data = r.json()
    game_id  = data["id"]
    player_id = data["players"][0]["id"]

    # Warte bis save_id=0 in der DB ist (max 15 Sekunden)
    for _ in range(15):
        time.sleep(1.0)
        check = requests.put(
            f"{base_url}/load_game",
            json={"gameId": game_id, "rollbackCount": 0},
            timeout=5,
        )
        if check.status_code == 200:
            log.info("   Spiel in DB verfügbar nach Erstellung")
            return game_id, player_id
    log.warning("   Spiel nicht in DB nach 15s – trotzdem fortfahren")
    return game_id, player_id


def wait_for_db(base_url: str, game_id: str, max_wait: int = 15) -> bool:
    """Wartet bis das Spiel in der DB ist (load_game verfügbar)."""
    for _ in range(max_wait):
        try:
            r = requests.put(
                f"{base_url}/load_game",
                json={"gameId": game_id, "rollbackCount": 0},
                timeout=5,
            )
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def get_save_count(base_url: str, game_id: str) -> int:
    """Gibt die Anzahl der Saves in der DB zurück."""
    try:
        r = requests.get(f"{base_url}/api/game/history",
                         params={"id": game_id, "serverId": "EIERWIRBRAUCHENEIER"},
                         timeout=5)
        if r.status_code == 200:
            return len(r.json())
    except Exception:
        pass
    return -1


def wait_for_new_save(base_url: str, game_id: str, current_max: int, timeout: int = 8) -> int:
    """Wartet bis ein neuer Save in der DB ist und gibt die neue max save_id zurück."""
    for _ in range(timeout * 2):
        time.sleep(0.5)
        r = requests.get(f"{base_url}/api/game/history",
                         params={"id": game_id, "serverId": "EIERWIRBRAUCHENEIER"},
                         timeout=5)
        if r.status_code == 200:
            saves = r.json()
            if saves:
                new_max = max(saves)
                if new_max > current_max:
                    return new_max
    return current_max


def rollback_to_save(base_url: str, game_id: str, target_save_id: int) -> bool:
    """Rollback bis zu einer bestimmten save_id (löscht alle neueren Saves)."""
    try:
        # Hole aktuelle Save-IDs
        r = requests.get(f"{base_url}/api/game/history",
                         params={"id": game_id, "serverId": "EIERWIRBRAUCHENEIER"},
                         timeout=5)
        if r.status_code != 200:
            return False
        save_ids = r.json()
        if not save_ids:
            return False
        
        # Zähle Saves die gelöscht werden sollen (alle > target_save_id)
        to_delete = sum(1 for s in save_ids if s > target_save_id)
        if to_delete == 0:
            return True  # Schon beim Ziel
        
        r2 = requests.put(
            f"{base_url}/load_game",
            json={"gameId": game_id, "rollbackCount": to_delete},
            timeout=10,
        )
        if r2.status_code == 200:
            time.sleep(POST_WAIT)
            return True
        log.warning("  load_game Rollback fehlgeschlagen: %d %s", r2.status_code, r2.text[:80])
        return False
    except Exception as e:
        log.warning("  Rollback fehlgeschlagen: %s", e)
        return False


def do_rollback(base_url: str, game_id: str, n_moves: int) -> bool:
    """Macht n_moves Züge rückgängig (Kompatibilität)."""
    return rollback_to_save(base_url, game_id, 0)  # Fallback


# ---------------------------------------------------------------------------
# Rollout: spiele N zufällige Züge und messe den Score
# ---------------------------------------------------------------------------

def take_snapshot(base_url: str, game_id: str) -> str | None:
    """Erstellt einen In-Memory Snapshot des aktuellen Game-States."""
    try:
        r = requests.post(f"{base_url}/api/mcts/snapshot",
                         json={"gameId": game_id}, timeout=10)
        if r.status_code == 200:
            return r.json()["snapshotId"]
        log.warning("Snapshot fehlgeschlagen: %d %s", r.status_code, r.text[:80])
    except Exception as e:
        log.warning("Snapshot Fehler: %s", e)
    return None


def restore_snapshot(base_url: str, snapshot_id: str, game_id: str) -> bool:
    """Stellt einen In-Memory Snapshot wieder her."""
    try:
        r = requests.post(f"{base_url}/api/mcts/restore",
                         json={"snapshotId": snapshot_id, "gameId": game_id},
                         timeout=10)
        if r.status_code == 200:
            time.sleep(0.3)
            return True
        log.warning("Restore fehlgeschlagen: %d %s", r.status_code, r.text[:80])
    except Exception as e:
        log.warning("Restore Fehler: %s", e)
    return False


def get_last_save_id(base_url: str, game_id: str) -> int:
    """Gibt die höchste save_id in der DB zurück via game history API."""
    try:
        r = requests.get(f"{base_url}/api/game/history",
                         params={"id": game_id, "serverId": "EIERWIRBRAUCHENEIER"},
                         timeout=5)
        if r.status_code == 200:
            saves = r.json()
            if saves:
                return max(saves)
    except Exception:
        pass
    return -1


def do_rollout(base_url: str, player_id: str, n_moves: int, game_id: str | None = None) -> float:
    if game_id is None:
        game_id = player_id
    # Merke Startgeneration für Generationsgrenze
    
    """
    Spielt n_moves zufällige Züge und gibt den Rollout-Score zurück.
    Macht danach alle Züge rückgängig.
    """
    moves_made = 0
    states_for_undo = []

    # Merke Startgeneration – kein Rollout über Generationsgrenzen
    try:
        init_state = get_state(base_url, player_id)
        start_gen = init_state.get("game", {}).get("generation", 1)
    except Exception:
        start_gen = 99

    for _ in range(n_moves):
        # State holen
        try:
            state = get_state(base_url, player_id)
        except Exception:
            break

        phase = state.get("game", {}).get("phase", "")
        cur_gen = state.get("game", {}).get("generation", 1)

        if phase == "end":
            break

        # Stoppe Rollout wenn Generation wechselt (Undo nicht mehr möglich)
        if cur_gen != start_gen:
            break

        waiting = state.get("waitingFor")
        if not waiting:
            time.sleep(POLL_INTERVAL)
            continue

        wtype = waiting.get("type", "")
        if wtype == "player":
            break  # Multiplayer: anderer Spieler dran

        # Entscheide zufällig (leicht gewichtet nach Score)
        result = decide_rollout(state)
        if result is None:
            break

        payload = result
        try:
            post_input(base_url, player_id, payload)
            states_for_undo.append(state)
            moves_made += 1
            time.sleep(POST_WAIT)
        except Exception:
            break

    # Finalen Score messen
    try:
        final_state = get_state(base_url, player_id)
        score = rollout_score(final_state)
    except Exception:
        score = 0.0

    # Alle Rollout-Züge rückgängig machen
    if moves_made > 0:
        # Warte damit alle async Saves in SQLite ankommen
        time.sleep(0.5 + moves_made * 0.1)
        ok = do_rollback(base_url, game_id, moves_made)
        if not ok:
            log.warning("  Rollback von %d Zügen fehlgeschlagen", moves_made)

    return score


def decide_rollout(state: dict) -> dict | None:
    """
    Trifft eine Entscheidung für den Rollout.

    Spielstärke-Hebel: zuerst die vollwertige Heuristik decide() versuchen,
    damit der Rollout unter vernünftigem Folgeverhalten ausgewertet wird
    (statt unter Zufallszügen, die Engine-/Produktionskarten systematisch
    unterbewerten). Nur wenn decide() den Fall nicht abdeckt (None), fällt
    die bisherige Zufallslogik als robuster Rückfall ein. Kostet keine
    zusätzlichen HTTP-Calls – dieselbe Rollout-Länge, klügere Zugwahl.
    """
    try:
        heuristic = decide(state)
        if heuristic is not None:
            return heuristic
    except Exception:
        pass   # Fällt auf Zufallslogik zurück

    waiting = state.get("waitingFor", {})
    wtype   = waiting.get("type", "")
    player  = state.get("thisPlayer", {})
    mc      = player.get("megacredits", 0)
    runId   = state["runId"]

    if wtype == "initialCards":
        # Einfache Auswahl
        responses = []
        for opt in waiting.get("options", []):
            cards = opt.get("cards", [])
            min_c = opt.get("min", 0)
            title = str(opt.get("title", "")).lower()
            if "corporation" in title:
                chosen = [cards[0]["name"]] if cards else []
            elif "prelude" in title:
                chosen = [c["name"] for c in cards[:min_c]]
            else:
                chosen = []
            responses.append({"type": "card", "cards": chosen})
        return {"type": "initialCards", "runId": runId, "responses": responses}

    if wtype == "card":
        return {"type": "card", "runId": runId, "cards": []}

    if wtype == "or":
        options = waiting.get("options", [])
        # Filtere Undo-Option raus
        valid = [(i, o) for i, o in enumerate(options)
                 if o.get("buttonLabel") != "Undo"]
        if not valid:
            return None

        # Wähle zufällig aus den Top-Optionen
        random.shuffle(valid)
        for i, opt in valid[:3]:
            otype = opt.get("type", "")
            if otype == "option":
                return {"type": "or", "runId": runId, "index": i,
                        "response": {"type": "option"}}
            elif otype == "projectCard":
                cards = opt.get("cards", [])
                # Wähle zufällig eine bezahlbare Karte
                affordable = [c for c in cards if c.get("calculatedCost", 0) <= mc - 5]
                if affordable:
                    c = random.choice(affordable)
                    return {"type": "or", "runId": runId, "index": i,
                            "response": {"type": "projectCard",
                                        "card": c["name"],
                                        "payment": build_payment(c)}}
                return {"type": "or", "runId": runId, "index": i,
                        "response": {"type": "option"}}
            elif otype == "space":
                spaces = opt.get("spaces", [])
                if spaces:
                    return {"type": "or", "runId": runId, "index": i,
                            "response": {"type": "space",
                                        "spaceId": random.choice(spaces)}}

        # Fallback: erste Option
        i, opt = valid[0]
        return {"type": "or", "runId": runId, "index": i,
                "response": {"type": "option"}}

    if wtype == "space":
        spaces = waiting.get("spaces", [])
        if spaces:
            return {"type": "space", "runId": runId,
                    "spaceId": random.choice(spaces)}

    if wtype == "payment":
        amount = waiting.get("amount", 3) or 3
        return {"type": "payment", "runId": runId, "payment": build_payment(amount)}

    if wtype == "amount":
        return {"type": "amount", "runId": runId, "amount": waiting.get("min", 0)}

    if wtype == "player":
        players = waiting.get("players", [])
        if players:
            return {"type": "player", "runId": runId,
                    "player": players[0] if isinstance(players[0], str)
                              else players[0].get("color", "red")}

    return None


# ---------------------------------------------------------------------------
# MCTS Hauptlogik: evaluiere Top-Kandidaten mit Rollouts
# ---------------------------------------------------------------------------

def handle_or_mcts(
    state: dict,
    base_url: str,
    player_id: str,
    n_rollouts: int = ROLLOUT_MOVES,
    max_candidates: int = MAX_CANDIDATES,
    game_id: str | None = None,
) -> tuple[dict, float]:
    if game_id is None:
        game_id = player_id
    """
    MCTS-Version von handle_or:
    1. Bewerte Kandidaten mit aktuellem Modell/Heuristik
    2. Evaluiere Top-K mit Rollouts
    3. Gib besten Zug + Rollout-Score zurück
    """
    # Hole Kandidaten vom normalen handle_or
    raw_result = handle_or(state)
    if raw_result is None or not isinstance(raw_result, dict):
        return raw_result, 0.0

    # Hole alle Kandidaten (nicht nur den besten)
    from tm_bot import handle_or as _handle_or
    waiting  = state["waitingFor"]
    options  = waiting.get("options", [])
    player   = state["thisPlayer"]
    mc       = player.get("megacredits", 0)

    # Sammle Kandidaten manuell
    candidates = []
    for i, opt in enumerate(options):
        otype = opt.get("type", "")
        title = str(opt.get("title", "")).lower()

        if opt.get("buttonLabel") == "Undo":
            continue  # Undo nie im MCTS evaluieren

        if otype == "option" and "pass" not in title and "undo" not in title:
            # Nicht-Pass Aktionen (Heat→Temp etc.)
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
                from tm_bot import choose_best_space
                space_map = {s["id"]: s for s in state["game"]["spaces"]}
                best = choose_best_space(spaces, space_map)
                candidates.append((score_action("greenery", state), {
                    "type": "or", "runId": state["runId"], "index": i,
                    "response": {"type": "space", "spaceId": best},
                }))
        elif otype == "projectCard":
            all_cards = opt.get("cards", [])
            SP_NAMES = {"Aquifer", "Greenery", "City", "Power Plant:SP", "Asteroid:SP"}

            # Handkarten
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
                                    "payment": build_payment(c)},
                    }))

            # Standard-Projekte
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
                                                "payment": build_payment(c)},
                                }))

    if not candidates:
        return raw_result, 0.0

    # Sortiere und nehme Top-K
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = candidates[:max_candidates]

    log.info("  🎲 MCTS: evaluiere %d Kandidaten mit je %d Rollout-Zügen",
             len(top_candidates), n_rollouts)

    best_payload  = raw_result
    best_rollout  = -999.0
    base_score    = rollout_score(state)
    results: list[tuple[float, float, dict]] = []  # (rollout_sc, heuristic_sc, payload)

    for heuristic_score, payload in top_candidates:
        card_name = payload.get("response", {}).get("card", "")
        display_name = card_name or payload["response"].get("type", "?")
        if display_name == "option": display_name = "Pass"
        log.info("    → Teste: %s (h=%.1f)", display_name, heuristic_score)

        # 1. Snapshot vor Kandidaten-Zug
        snapshot_id = take_snapshot(base_url, game_id)
        if snapshot_id is None:
            log.warning("    Snapshot fehlgeschlagen – überspringe MCTS")
            return best_payload, best_rollout

        try:
            post_input(base_url, player_id, payload)
            time.sleep(POST_WAIT)
        except Exception as e:
            log.warning("    POST fehlgeschlagen: %s", e)
            restore_snapshot(base_url, snapshot_id, game_id)
            continue

        # 2. Rollout
        rollout_sc = do_rollout(base_url, player_id, n_rollouts, game_id=game_id)
        delta = rollout_sc - base_score
        log.info("    Rollout-Score: %.1f (delta: %+.1f)", rollout_sc, delta)

        results.append((rollout_sc, heuristic_score, payload))
        if rollout_sc > best_rollout:
            best_rollout = rollout_sc

        # 3. Restore zum Snapshot vor diesem Kandidaten
        ok = restore_snapshot(base_url, snapshot_id, game_id)
        if not ok:
            log.warning("    Restore fehlgeschlagen – überspringe restliche Kandidaten")
            break

    # Hybrid-Entscheidung: MCTS nur wenn Spread gross genug
    if results:
        max_r = max(r[0] for r in results)
        min_r = min(r[0] for r in results)
        spread = max_r - min_r
        if spread >= MCTS_MIN_DELTA:
            # MCTS hat klare Präferenz
            best_rollout, _, best_payload = max(results, key=lambda x: x[0])
            log.info("  📊 MCTS entscheidet (spread=%.1f)", spread)
        else:
            # Zu ähnlich → Heuristik entscheidet
            _, _, best_payload = max(results, key=lambda x: x[1])
            best_rollout = max_r
            log.info("  📊 Heuristik entscheidet (spread=%.1f < %.1f)", spread, MCTS_MIN_DELTA)

    # Aktualisiere runId nach allen Restores
    try:
        current_state = get_state(base_url, player_id)
        best_payload["runId"] = current_state["runId"]
    except Exception:
        pass

    _chosen = best_payload.get("response", {}).get("card") or best_payload.get("response", {}).get("type", "?")
    if _chosen == "option": _chosen = "Pass"
    log.info("  ✓ MCTS wählt: %s (rollout=%.1f)", _chosen, best_rollout)

    return best_payload, best_rollout


# ---------------------------------------------------------------------------
# Hauptbot-Loop mit MCTS
# ---------------------------------------------------------------------------

def resolve_game_id(base_url: str, player_id: str, server_id: str = "EIERWIRBRAUCHENEIER") -> str | None:
    """Findet die Game-ID für eine Player-ID über die Games-Liste."""
    try:
        r = requests.get(f"{base_url}/api/games", params={"serverId": server_id}, timeout=10)
        if r.status_code != 200:
            return None
        games = r.json()
        for g in games:
            if player_id in g.get("participantIds", []):
                return g["gameId"]
    except Exception as e:
        log.warning("resolve_game_id fehlgeschlagen: %s", e)
    return None


def run_mcts_bot(
    base_url: str,
    player_id: str,
    n_rollouts: int = ROLLOUT_MOVES,
    max_candidates: int = MAX_CANDIDATES,
    data_file: str = MCTS_DATA_FILE,
    server_id: str = "EIERWIRBRAUCHENEIER",
):
    log.info("🤖 MCTS-Bot | %s | %s", base_url, player_id)
    log.info("   Rollouts: %d | Kandidaten: %d", n_rollouts, max_candidates)
    log.info("   🌐 Spiel: %s/player?id=%s", base_url, player_id)

    # Game-ID für load_game Rollback ermitteln
    game_id = resolve_game_id(base_url, player_id, server_id)
    if game_id:
        log.info("   Game-ID: %s", game_id)
    else:
        log.warning("   Game-ID nicht gefunden – Rollback nur via Player-ID")
        game_id = player_id  # Fallback

    db_ready = False  # Wird True sobald Spiel in DB persistiert ist

    errors = 0
    last_key = None
    mcts_transitions = []  # (state_feats, card_feats, card_name, rollout_score)

    while True:
        try:
            state = get_state(base_url, player_id)
            errors = 0
        except Exception as e:
            errors += 1
            if errors >= MAX_ERRORS:
                log.error("Zu viele Fehler: %s", e)
                break
            time.sleep(ERROR_WAIT)
            continue

        game  = state.get("game", {})
        phase = game.get("phase", "")
        waiting = state.get("waitingFor")

        # Spielende erkennen: phase=end ODER (kein waitingFor und phase nicht research/action/drafting)
        game_over = (phase == "end") or (not waiting and phase not in ("research", "action", "drafting", "initialdrafting", "solar"))
        if game_over:
            vp_breakdown = state["thisPlayer"].get("victoryPointsBreakdown", {})
            vp  = vp_breakdown.get("total", state["thisPlayer"]["terraformRating"])
            tr  = state["thisPlayer"]["terraformRating"]
            won = game.get("isSoloModeWin", False)
            log.info("🏁 VP: %d | TR: %d | Gewonnen: %s", vp, tr, won)

            # Weise finales VP-Label zu (normalisiert: VP / 60)
            # 60 VP = gutes Solo-Ergebnis; skaliert auf ~0.5-1.0 Bereich
            final_label = vp / 60.0

            # Speichere Trainingsdaten mit finalem Label
            if mcts_transitions and data_file:
                with open(data_file, "a", encoding="utf-8") as f:
                    for t in mcts_transitions:
                        t["label"] = final_label
                        t["final_vp"] = vp
                        t["final_tr"] = tr
                        f.write(json.dumps(t) + "\n")
                log.info("💾 %d MCTS-Transitions gespeichert (VP=%d, label=%.3f) → %s",
                         len(mcts_transitions), vp, final_label, data_file)
            break

        player  = state.get("thisPlayer", {})

        if not player.get("isActive", False) or not waiting:
            time.sleep(POLL_INTERVAL)
            continue

        wtype = waiting.get("type", "")
        if wtype == "player":
            time.sleep(POLL_INTERVAL)
            continue

        # Deduplizierung
        key = (wtype, str(waiting.get("title", "")),
               tuple(c.get("name", "") for c in waiting.get("cards", [])))
        if key == last_key:
            time.sleep(POLL_INTERVAL)
            continue

        gen = game.get("generation", 1)
        mc  = player.get("megacredits", 0)
        tr  = player.get("terraformRating", 14)
        log.info("[Gen %d] MC:%d TR:%d | %s", gen, mc, tr, str(waiting.get("title", ""))[:40])

        # Prüfe ob Spiel in DB ist (nötig für load_game Rollback)
        if not db_ready and phase == "action":
            db_ready = wait_for_db(base_url, game_id, max_wait=5)
            if db_ready:
                log.info("   Spiel in DB verfügbar – MCTS aktiviert")
            else:
                log.info("   Spiel noch nicht in DB – MCTS deaktiviert")

        # MCTS in der Action-Phase für or-Typ (load_game erlaubt immer Rollback)
        use_mcts = (phase == "action" and wtype == "or" and db_ready)

        if use_mcts:
            payload, rollout_sc = handle_or_mcts(
                state, base_url, player_id, n_rollouts, max_candidates, game_id=game_id)

            # Trainingsdaten speichern (Label wird am Spielende gesetzt)
            card_name = payload.get("response", {}).get("card", "")
            if card_name and card_name not in ("Pass", ""):
                info    = CARD_DB.get(card_name, {})
                s_feats = game_state_features(state)
                c_feats = card_features(info)
                gen     = state.get("game", {}).get("generation", 1)
                mcts_transitions.append({
                    "card":         card_name,
                    "rollout_sc":   rollout_sc,   # Rollout-Score (Hilfsinfo)
                    "generation":   gen,
                    "state_feats":  s_feats,
                    "card_feats":   c_feats,
                })
        else:
            result = decide(state)
            if result is None:
                last_key = key
                time.sleep(POLL_INTERVAL)
                continue
            # decide() gibt (payload, card_name, card_cost) zurück
            if isinstance(result, tuple):
                payload = result[0]
            else:
                payload = result
            if payload is None:
                last_key = key
                time.sleep(POLL_INTERVAL)
                continue

        log.info("  → %s", payload.get("response", {}).get("card",
                  payload.get("response", {}).get("type", "?")))

        try:
            log.debug("  Sende Payload: %s", str(payload)[:200])
            post_input(base_url, player_id, payload)
            last_key = None
            time.sleep(POST_WAIT)
        except requests.HTTPError as e:
            log.warning("  HTTP Fehler: %s", e.response.text[:200] if e.response else e)
            log.warning("  Payload war: %s", json.dumps(payload))
            last_key = key
            time.sleep(ERROR_WAIT)
        except Exception as e:
            log.warning("  Fehler: %s", e)
            last_key = key
            time.sleep(ERROR_WAIT)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TM MCTS Bot")
    parser.add_argument("--player-id",   default=None,
                        help="Player-ID (ohne --auto-games)")
    parser.add_argument("--auto-games",  type=int, default=0,
                        help="Automatisch N Partien spielen (erstellt Spiele selbst)")
    parser.add_argument("--url",         default=DEFAULT_URL)
    parser.add_argument("--rollouts",    default=ROLLOUT_MOVES, type=int,
                        help="Rollout-Züge pro Kandidat (default: 8)")
    parser.add_argument("--candidates",  default=MAX_CANDIDATES, type=int,
                        help="Max. Kandidaten die evaluiert werden (default: 5)")
    parser.add_argument("--db",          default="card_db.json")
    parser.add_argument("--data",        default=MCTS_DATA_FILE,
                        help="Ausgabedatei für MCTS-Trainingsdaten")
    parser.add_argument("--server-id",   default="EIERWIRBRAUCHENEIER",
                        help="Server-ID für die Games-Liste")
    parser.add_argument("--no-mcts",     action="store_true",
                        help="MCTS deaktivieren (normaler Bot)")
    args = parser.parse_args()

    load_card_db(args.db)

    if args.auto_games > 0:
        # Automatischer Multi-Spiel-Modus
        log.info("🎮 Auto-Modus: %d Partien", args.auto_games)
        for i in range(args.auto_games):
            log.info("── Partie %d/%d ──", i + 1, args.auto_games)
            try:
                game_id, player_id = create_game_with_undo(args.url)
                log.info("   Spiel: %s | Spieler: %s", game_id, player_id)
                run_mcts_bot(
                    args.url, player_id,
                    n_rollouts=args.rollouts,
                    max_candidates=args.candidates,
                    data_file=args.data,
                    server_id=args.server_id,
                )
            except Exception as e:
                log.error("Partie %d fehlgeschlagen: %s", i + 1, e)
            time.sleep(2)
        log.info("✅ Alle %d Partien abgeschlossen", args.auto_games)
    elif args.no_mcts:
        # Normaler Bot-Modus
        from tm_bot import run_bot
        run_bot(args.url, args.player_id, poll=0.3)
    elif args.player_id:
        run_mcts_bot(
            args.url, args.player_id,
            n_rollouts=args.rollouts,
            max_candidates=args.candidates,
            data_file=args.data,
            server_id=args.server_id,
        )
    else:
        parser.error("Entweder --player-id oder --auto-games angeben")


if __name__ == "__main__":
    main()
