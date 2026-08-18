"""
Terraforming Mars – Multiplayer MCTS Bot

Ermöglicht 2–6 MCTS-Bots gegeneinander. Jeder Bot läuft als eigener
Prozess und koordiniert sich über eine atomare Lock-Datei damit immer
nur ein Bot gleichzeitig im MCTS-Modus (Probe + Rollout + Undo) ist.

Warum Lock nötig:
  load_game setzt den KOMPLETTEN Spielzustand zurück – also auch Züge
  anderer Spieler die seit dem Snapshot gemacht wurden. Ohne Lock würde
  Bot B seinen Fortschritt verlieren wenn Bot A gerade im Rollout ist.

Ablauf pro Zug:
  1. Lock erwerben (warte wenn anderer Bot im MCTS ist)
  2. MCTS: Top-K Kandidaten via Snapshot/Rollout/Restore evaluieren
  3. Besten Zug spielen
  4. Lock freigeben

Rollout-Strategie im Multiplayer:
  Nach jedem eigenen Rollout-Zug: wenn Gegner dran ist, spiele
  einen schnellen heuristischen Zug für ihn (Option B aus Konzept).
  Für den ersten Start ist Option A (Gegner ignorieren) als Fallback
  implementiert (--simple-rollout Flag).

Ausführen (2 Bots):
  Terminal 1: py -3.12 tm_mcts_mp.py --color red   --auto-games 50
  Terminal 2: py -3.12 tm_mcts_mp.py --color blue  --auto-games 50

Ausführen (manuell, bereits laufendes Spiel):
  py -3.12 tm_mcts_mp.py --player-id pXXX --color red

Ausführen (4 Bots):
  Terminal 1: py -3.12 tm_mcts_mp.py --color red    --auto-games 20
  Terminal 2: py -3.12 tm_mcts_mp.py --color blue   --auto-games 20
  Terminal 3: py -3.12 tm_mcts_mp.py --color yellow --auto-games 20
  Terminal 4: py -3.12 tm_mcts_mp.py --color green  --auto-games 20
  (alle Terminals im selben Verzeichnis starten)

Hinweis: Vom Bot erstellte Spiele laufen OHNE undoOption - der Server wuerde sonst
bei jeder Aktion einen vollstaendigen Snapshot schreiben (Datenbank waechst enorm).
Die MCTS-Suche braucht Undo und ist deshalb nur in Partien nutzbar, die anderswo
mit Undo erstellt wurden (Oberflaeche + --join).
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import requests

# Importiere aus tm_bot (Basis-Logik)
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

# Importiere MCTS-Hilfsfunktionen aus tm_mcts
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
# Konfiguration
# ---------------------------------------------------------------------------

DEFAULT_URL      = "http://localhost:9000"
LOCK_FILE        = "tm_mcts_lock.json"   # Im Arbeitsverzeichnis
LOCK_TIMEOUT     = 90.0    # Sekunden bis abgestürzter Bot die Sperre verliert
LOCK_POLL        = 0.5     # Sekunden zwischen Lock-Polls
MCTS_DATA_FILE   = "mcts_data_mp.jsonl"

# Farben die der TM-Server kennt
VALID_COLORS = ["red", "blue", "yellow", "green", "black", "purple"]

# Zeitdeckel je Partie (Sekunden) - LETZTE Rueckfallebene, nicht der Haenger-Schutz.
# 31.07.: 900 s war viel zu eng. Ein PAAR sind zwei Partien nacheinander im selben
# Worker; bei gemessenen 25 min je Paar unter 24-Worker-Last dauert eine Partie im
# Mittel 12.5 min - die Haelfte liegt darueber. Ergebnis: 131 von 600 Partien
# abgebrochen (mit 1800 s waren es 23). Gegen echte Haenger wirkt der
# Fortschrittswaechter (8x dieselbe Entscheidung), der binnen Sekunden greift; der
# Zeitdeckel muss nur verhindern, dass ein Worker unbegrenzt festhaengt.
# Frueher: 1200 im
# parallelen CRN-Pfad (am 15.07. wegen des langsameren Servers erhoeht) und 360 im
# sequenziellen Treiber, der nie mitgezogen wurde. Mit reichhaltigen Settings
# (Venus/Ares/CEO/Solar-Phase) dauern Partien ein Vielfaches der frueheren zwei Minuten,
# deshalb ein gemeinsamer, grosszuegiger Wert - ueber --game-cap anpassbar.
GAME_TIME_CAP = 3600

# 28.07.: Der Server liefert bei POST /player/input das vollstaendige aktualisierte
# Spielermodell zurueck (PlayerInput.ts Z.82) - der Runner verwarf es und pruefte
# stattdessen mit bis zu vier zusaetzlichen save_id-Abfragen, ob der Zug angekommen ist.
# Die Antwort selbst ist der Beweis. Mit den reichhaltigen Settings hat sich die Zahl
# der Entscheidungen je Partie fast verdoppelt (39.8 statt 20.5 Aktionen), und der
# Server ist der Engpass - jede eingesparte Anfrage zaehlt.
# Abschaltbar ueber --no-post-reuse, um bei identischem Seed gegenpruefen zu koennen.
POST_REUSE = True
# 28.07. abends: POST_REUSE allein war zu scharf. Zusammen mit dem gelockerten
# _needs_action (bedient jede offene Eingabe) entstand ein Rennen - lieferte der Server
# nach dem POST noch kurz den alten Zustand, beantwortete der Bot dieselbe Aufforderung
# ein zweites Mal ("Not waiting for anything", 17x in 28 min). Eine kurze feste Pause
# schliesst das Fenster und kostet nur einen Bruchteil der alten Nachkontrolle
# (1 GET + 2 s Schlaf + bis zu 3 weitere GETs).
POST_REUSE_WAIT = 0.4

# 28.07. spaet abends: Die feste Pause allein reicht NICHT. Die Serverlogs zeigen
# "GameLoader loaded game ... from database" - der Server laedt Partien bei Bedarf aus
# der Datenbank nach, was Sekunden bis Minuten dauern kann. Gegen ein Rennen dieser
# Groessenordnung ist keine feste Wartezeit zu bemessen.
# Deshalb wird jetzt GEPRUEFT statt gewartet: Der Treiber merkt sich, welche Aufforderung
# er zuletzt beantwortet hat (Typ, Titel, Karten, gameAge). Kommt exakt dieselbe zurueck,
# ist der Zustand veraltet - dann wird NICHT geantwortet, sondern kurz gewartet und neu
# gelesen. gameAge steigt, sobald der Server eine Eingabe verarbeitet hat; innerhalb
# desselben Zustands bleibt es stehen. Denselben Mechanismus nutzt der --vs-human-Pfad
# seit dem 15.07. (dort als `last_key`), der A/B-Treiber hatte ihn nie.
# Nach STALE_MAX_SKIPS Versuchen wird trotzdem gesendet - lieber ein 400er als ein Haenger.
STALE_MAX_SKIPS   = 10
STALE_RECHECK_WAIT = 0.5
_LAST_ANSWERED: dict = {}


def _answer_key(state: dict) -> tuple:
    """Kennzeichnet eine Aufforderung eindeutig - inklusive gameAge als Fortschrittsmarker."""
    w = state.get("waitingFor") or {}
    return (w.get("type"),
            str(w.get("title", ""))[:80],
            tuple(c.get("name", "") for c in (w.get("cards") or [])),
            (state.get("game") or {}).get("gameAge"))


# ---------------------------------------------------------------------------
# MCTSLock – atomare Prozess-Koordination via Datei
# ---------------------------------------------------------------------------

class MCTSLock:
    """
    Koordiniert mehrere MCTS-Bot-Prozesse über eine atomare Lock-Datei.

    Nur ein Bot darf gleichzeitig im MCTS-Modus sein (Probe + Rollout + Undo),
    weil load_game den gesamten Spielzustand zurücksetzt.

    Atomares Schreiben via temporäre Datei + os.replace() verhindert
    halbgeschriebene Zustände bei gleichzeitigem Zugriff.
    """

    def __init__(self, my_color: str, game_id: str, lock_file: str = LOCK_FILE):
        self.my_color  = my_color
        self.game_id   = game_id
        self.lock_file = lock_file

    def acquire(self, save_id: int) -> bool:
        """
        Warte bis Lock frei, dann erwerbe ihn.
        Blockiert bis Lock verfügbar oder Timeout des anderen Bots.
        Gibt True zurück wenn Lock erworben.
        """
        waited = 0.0
        while True:
            state = self._read()
            locked_by = state.get("locked_by")

            if locked_by is None:
                # Frei – versuche zu erwerben
                self._write(save_id)
                # Kurz warten + nochmal lesen (Race Condition check)
                time.sleep(0.15)
                state2 = self._read()
                if state2.get("locked_by") == self.my_color:
                    log.debug("🔒 Lock erworben (%s, save=%d)", self.my_color, save_id)
                    return True
                # Anderer Bot war schneller – weiter warten
                time.sleep(LOCK_POLL)
                waited += LOCK_POLL
                continue

            elif locked_by == self.my_color:
                # Eigener Lock (z.B. nach Neustart nach Absturz)
                since = state.get("since", 0)
                if time.time() - since > LOCK_TIMEOUT:
                    log.warning("🔒 Eigener Timeout-Lock gefunden – neu erwerben")
                    self._write(save_id)
                return True

            else:
                # Anderer Bot hat Lock
                since = state.get("since", 0)
                age   = time.time() - since
                if age > LOCK_TIMEOUT:
                    log.warning("🔒 Lock-Timeout von '%s' (%.0fs) – übernehme",
                                locked_by, age)
                    self._write(save_id)
                    return True

                # Warte
                if waited == 0:
                    log.info("⏳ Warte auf MCTS-Lock von '%s'...", locked_by)
                time.sleep(LOCK_POLL)
                waited += LOCK_POLL
                if waited > 0 and int(waited) % 10 == 0:
                    log.info("⏳ Warte seit %.0fs auf Lock von '%s'", waited, locked_by)

    def release(self):
        """Lock freigeben."""
        self._write_free()
        log.debug("🔓 Lock freigegeben (%s)", self.my_color)

    def is_locked_by_other(self) -> bool:
        """Prüft ob ein anderer Bot gerade im MCTS ist."""
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
        """Schreibt atomisch über temporäre Datei + os.replace()."""
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
# Multiplayer-Spielerstellung
# ---------------------------------------------------------------------------

COLORS_TO_NAMES = {
    "red":    "Bot-Rot",
    "blue":   "Bot-Blau",
    "yellow": "Bot-Gelb",
    "green":  "Bot-Grün",
    "black":  "Bot-Schwarz",
    "purple": "Bot-Lila",
}


# Die drei OFFIZIELLEN Boards. Bewusst keine Community-/Pathfinders-Boards: der Bot
# kennt die Meilensteine und Awards dieser drei vollstaendig (_milestone_gap deckt
# Tharsis/Elysium/Hellas ab, _AWARD_KEYS ebenso), bei den uebrigen waere er blind und
# wuerde 5-15 VP an Meilensteinen/Awards verschenken.
OFFICIAL_BOARDS = ("tharsis", "hellas", "elysium")


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
    """
    Erstellt ein Multiplayer-Spiel fuer n_players Spieler (ohne undoOption, s.u.).
    Gibt {color: player_id} zurück.

    Bei --auto-games erstellt der erste Bot (alphabetisch sortierte Farbe)
    das Spiel und schreibt die IDs in eine geteilte Datei.
    Andere Bots lesen die IDs daraus.
    """
    if colors is None:
        colors = VALID_COLORS[:n_players]

    # ★ STARTSPIELER 20.07. (apeheads Beobachtung "der Bot ist IMMER Spieler 1"):
    # Hier stand fest `"first": (i == 0)` - der erste Spieler der Farbliste war IMMER
    # Startspieler, also immer der Bot. Der Payload schickt zwar `randomFirstPlayer`
    # mit, aber das Feld existiert NUR im Client (die Web-UI wuerfelt selbst und setzt
    # dann `first`); im gesamten src/server/ des Servers kommt es nicht vor - der Server
    # liest ausschliesslich `players[i].first`. Das Flag verpuffte also wirkungslos.
    # Der Startspieler rotiert danach zwar jede Generation
    # (Game.ts: firstIndex = (firstIndex + 1) % players.length), aber Generation 1 und
    # - bei ungerader Endgeneration - eine weitere gingen systematisch an den Bot.
    # WICHTIG fuer den CRN-A/B: bei gesetztem Seed wird der Startspieler DARAUS
    # abgeleitet, damit beide Seiten eines Paares denselben Sitz haben. Nur ohne Seed
    # (Live-Partien) wird echt gewuerfelt. random_first=False behaelt feste Sitze.
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
        # 31.07.: Der vom Bot erstellte Spiele NIE mit Undo. Der Server speichert damit
        # bei JEDER Aktion einen vollstaendigen Snapshot statt einmal je Runde
        # (Player.ts Z.1444) - nach zwei A/B-Laeufen war die Datenbank 24 GB gross, und
        # das Nachladen bremste den Server auf ein Drittel des Durchsatzes.
        # Wer Undo braucht, erstellt die Partie in der Oberflaeche und laesst den Bot
        # per --join beitreten; dann kommt die Einstellung von dort.
        "undoOption": False,
        "showTimers": False,
        "fastModeOption": False,
        "showOtherPlayersVP": True,   # Im MP sichtbar für bessere Rollouts
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

    # ── Echte Runden-Einstellungen uebernehmen (tm_settings.json). Ohne das erstellte der
    # Runner Spiele mit ANDEREN Optionen als die reale Runde (board=tharsis statt 'random all',
    # kein Fast Mode, kein Fan-MA, randomMA aus, 2 statt 4 Korporationen ...) -> der Bot wurde
    # unter falschen Bedingungen getestet. 'players' bleibt IMMER die Runner-Liste (Bot+Mensch);
    # alles andere (inkl. expansions, bannedCards, customCorporationsList, board, randomMA,
    # includeFanMA, fastModeOption, shuffleMapOption, startingCorporations/Ceos) kommt aus der
    # Datei. Ein explizites --expansions ueberschreibt die Erweiterungen der Datei.
    if settings:
        _skip = {"players", "seed"}
        for k, v in settings.items():
            if k in _skip:
                continue
            payload[k] = v
        if expansions:                       # --expansions hat Vorrang vor der Datei
            payload["expansions"] = {
                "corpera": True,
                **{m: (m in _exp) for m in (
                    "promo", "venus", "colonies", "prelude", "prelude2", "turmoil",
                    "community", "ares", "moon", "pathfinders", "ceo", "starwars",
                    "underworld", "deltaProject")},
            }
        # Zahlen kommen in der Datei teils als String ("4") - der Server erwartet int.
        for _k in ("startingCorporations", "startingCeos", "startingPreludes"):
            if isinstance(payload.get(_k), str) and payload[_k].isdigit():
                payload[_k] = int(payload[_k])
        payload["seed"] = seed if seed is not None else random.random()

    # Deterministisches Deck: vorhandene Partie klonen (exakte Mischreihenfolge).
    if cloned_game_id:
        payload["clonedGamedId"] = cloned_game_id

    # Der aktualisierte Server (~07/2026) braucht fuers Draft-Setup deutlich laenger;
    # mehrere gleichzeitige creategame-Requests (--parallel) ueberschritten das alte
    # timeout=15 -> "Read timed out", das ganze Paar brach ab (15.07.). Daher: laengeres
    # Timeout + Retry mit Backoff. Die Erstellung ist idempotent genug (neue game_id je
    # Versuch); nur bei cloned_game_id ist Vorsicht noetig -> dort NICHT neu wuerfeln,
    # sondern denselben Klon erneut anfragen.
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

    # Warte bis Spiel in DB ist
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
# Koordinierungsdatei für automatische Spielerstellung
# ---------------------------------------------------------------------------

GAME_COORD_FILE = "tm_mp_game.json"

READY_FILE    = "tm_mp_ready.json"
COLOR_REG_FILE = "tm_mp_colors.json"   # Farb-Registrierung


def find_card_db() -> str:
    """Sucht card_db.json im Arbeitsverzeichnis und Elternverzeichnissen."""
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
    """
    Registriert diesen Bot-Prozess und gibt eine freie Farbe zurück.
    allowed_colors schränkt die Auswahl auf gültige Spielerfarben ein.
    """
    import time as _time
    candidates = allowed_colors if allowed_colors else VALID_COLORS
    start = _time.time()
    while _time.time() - start < timeout:
        try:
            # Lese aktuellen Stand
            try:
                with open(COLOR_REG_FILE, encoding="utf-8") as f:
                    reg = json.load(f)
                # Bereinige alte Registrierungen (>5min)
                reg = {c: ts for c, ts in reg.items()
                       if _time.time() - ts < 300}
            except (FileNotFoundError, json.JSONDecodeError):
                reg = {}

            # Finde nächste freie Farbe aus den erlaubten
            for color in candidates:
                if color not in reg:
                    # Versuche diese Farbe zu reservieren
                    reg[color] = _time.time()
                    tmp = COLOR_REG_FILE + f".{color}.tmp"
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(reg, f)
                    os.replace(tmp, COLOR_REG_FILE)
                    # Kurz warten und prüfen ob wir wirklich diese Farbe haben
                    _time.sleep(0.2)
                    with open(COLOR_REG_FILE, encoding="utf-8") as f:
                        reg2 = json.load(f)
                    if reg2.get(color) == reg[color]:
                        log.info("🎨 Farbe automatisch zugeteilt: %s", color)
                        return color
                    # Anderer Bot war schneller – nochmal versuchen
                    break
        except Exception as e:
            log.warning("Farb-Registrierung Fehler: %s", e)
        _time.sleep(0.3)

    raise RuntimeError("Konnte keine freie Farbe registrieren")


def release_color(color: str):
    """Gibt die registrierte Farbe nach Spielende frei."""
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
    """Signalisiert dass dieser Bot für die nächste Partie bereit ist."""
    try:
        # Lese aktuellen Stand
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
    """Wartet bis alle Farben bereit sind. Gibt True zurück wenn erfolgreich."""
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
    """Schreibt Spiel-IDs in Koordinierungsdatei für andere Bot-Prozesse."""
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
    """
    Liest Koordinierungsdatei falls sie existiert, frisch ist und zur richtigen Partie gehört.
    """
    try:
        with open(GAME_COORD_FILE, encoding="utf-8") as f:
            data = json.load(f)
        age = time.time() - data.get("created_at", 0)
        # Partie-Nummer muss übereinstimmen (0 = beliebig)
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
    """
    Liest Spiel-IDs aus Koordinierungsdatei.
    Wartet bis zu timeout Sekunden. game_num=0 akzeptiert beliebige Partie.
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
                # Partie-Nummer prüfen (0 = beliebig)
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
# Rollout im Multiplayer: simuliert Gegner-Züge
# ---------------------------------------------------------------------------

def do_rollout_mp(
    base_url:    str,
    player_id:   str,
    all_player_ids: list[str],
    n_moves:     int,
    game_id:     str,
    simple:      bool = False,
) -> float:
    """
    Multiplayer-Rollout: spielt n_moves Züge und gibt Rollout-Score zurück.

    simple=False (Option B): Simuliert auch Gegner-Züge heuristisch.
    simple=True  (Option A): Ignoriert Gegner, spielt nur eigene Züge.

    Stellt danach den Zustand via rollback_to_save wieder her.
    WICHTIG: Lock muss bereits gehalten werden wenn diese Funktion aufgerufen wird.
    """
    moves_made = 0

    try:
        init_state = get_state(base_url, player_id)
        start_gen  = init_state.get("game", {}).get("generation", 1)
        start_save = get_last_save_id(base_url, game_id)
    except Exception:
        return 0.0

    for _ in range(n_moves):
        # Prüfe ob wir dran sind
        try:
            state = get_state(base_url, player_id)
        except Exception:
            break

        phase   = state.get("game", {}).get("phase", "")
        cur_gen = state.get("game", {}).get("generation", 1)

        if phase == "end":
            break
        if cur_gen != start_gen:
            break   # Keine Züge über Generationsgrenze (Undo nicht möglich)

        is_active = state.get("thisPlayer", {}).get("isActive", False)
        waiting   = state.get("waitingFor")

        if is_active and waiting:
            wtype = waiting.get("type", "")
            if wtype == "player":
                # Spielerauswahl – eigene Farbe wählen
                result = decide_rollout(state)
            else:
                result = decide_rollout(state)

            if result is None:
                break
            try:
                post_input(base_url, player_id, result)
                moves_made += 1
                time.sleep(POST_WAIT * 0.5)  # Schneller im Rollout
            except Exception:
                break

        elif not simple and not is_active and waiting is None:
            # Gegner ist dran – simuliere einen Zug für jeden Gegner
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
                    break  # Nur ein Gegner pro Iteration
                except Exception:
                    continue
            if not acted:
                time.sleep(POLL_INTERVAL)
        else:
            time.sleep(POLL_INTERVAL)

    # Finalen Score messen
    try:
        final_state = get_state(base_url, player_id)
        score = rollout_score(final_state)
    except Exception:
        score = 0.0

    # Zurückrollen zum Ausgangszustand
    if moves_made > 0:
        time.sleep(0.3 + moves_made * 0.05)
        ok = rollback_to_save(base_url, game_id, start_save)
        if not ok:
            log.warning("  MP-Rollback fehlgeschlagen (%d Züge)", moves_made)

    return score


# ---------------------------------------------------------------------------
# MP-MCTS: evaluiere Kandidaten mit Lock
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
    """
    MP-Version von handle_or_mcts mit Lock-Mechanismus.

    Ablauf:
      1. Lock erwerben
      2. Für jeden Top-K Kandidaten: Snapshot → Probe → Rollout → Restore
      3. Besten Zug wählen
      4. Zug spielen
      5. Lock freigeben
    """
    # 31.07.: Diese Suche setzt Rollbacks voraus (rollback_to_save). Vom Bot erstellte
    # Partien laufen seit dem 31.07. OHNE undoOption - dort ist sie nicht nutzbar.
    # In Partien aus der Oberflaeche (+ --join) funktioniert sie weiterhin.
    if not (state.get("game") or {}).get("undoOption", True):
        log.warning("  MCTS-Suche uebersprungen (Partie ohne undoOption) - reine Heuristik")
        return handle_or(state), 0.0

    # Hole Basis-Entscheidung (Fallback)
    raw_result = handle_or(state)
    if raw_result is None:
        return raw_result, 0.0

    # Kandidaten sammeln (identisch zu handle_or_mcts in tm_mcts.py)
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
            # Verschachtelter Meilenstein: {type:"or", title:"Claim a milestone", options:[...]}
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
            # Verschachtelter Award: {type:"or", title:"Fund an award", options:[...]}
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

    # --- Schnelles, unsichtbares Blatt (1-Halbzug-Feature-Vorhersage in Python).
    #     Kein Server-Rollout/Snapshot -> keine Browser-Fehlalarme, viel schneller.
    #     Schalter: TM_FAST_LEAF=1 (nutzt value_model.joblib). ---
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

    # --- Lock erwerben ---
    current_save = get_last_save_id(base_url, game_id)
    lock.acquire(current_save)

    try:
        # Sanity-Check: save_id hat sich nicht verändert während wir warteten
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

            # Aktuellen Save-Stand merken (Standard-Undo statt Custom-Snapshot).
            # NACH dem runId-Refresh; save_id kann sich seit Lock-Erwerb geändert haben.
            snapshot_save = get_last_save_id(base_url, game_id)
            if snapshot_save < 0:
                log.warning("    Save-ID nicht abrufbar – MCTS übersprungen")
                break
            # Validiere dass wir wirklich dran sind nach dem Snapshot
            try:
                snap_check = get_state(base_url, player_id)
                if not snap_check.get("thisPlayer", {}).get("isActive", False):
                    log.warning("    Nach Snapshot nicht mehr aktiv – MCTS abbrechen")
                    break
            except Exception:
                pass

            # Probe-Zug spielen – runId MUSS frisch sein
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

            # Rollout
            rollout_sc = do_rollout_mp(
                base_url, player_id, all_player_ids,
                n_rollouts, game_id, simple=simple_rollout,
            )
            delta = rollout_sc - base_score
            log.info("    Rollout: %.1f (Δ%+.1f)", rollout_sc, delta)

            results.append((rollout_sc, heuristic_score, payload))
            if rollout_sc > best_rollout:
                best_rollout = rollout_sc

            # Zurück zum Zustand vor diesem Probe-Zug (Standard-Undo)
            ok = rollback_to_save(base_url, game_id, snapshot_save)
            if not ok:
                log.warning("    Rollback fehlgeschlagen – breche MCTS ab")
                break

        # Hybrid-Entscheidung
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

        # runId nach allen Restores aktualisieren
        try:
            cur_state = get_state(base_url, player_id)
            best_payload["runId"] = cur_state["runId"]
        except Exception:
            pass

        # KEIN post_input hier – run_mcts_bot_mp sendet den Zug
        # (Lock wird dort nach dem Post freigegeben)

    finally:
        # Lock IMMER freigeben – auch bei Fehlern
        lock.release()

    chosen = (best_payload.get("response", {}).get("card")
              or best_payload.get("response", {}).get("type", "?"))
    if chosen == "option":
        chosen = "Pass"
    log.info("  ✓ MP-MCTS wählt: %s (rollout=%.1f)", chosen, best_rollout)

    return best_payload, best_rollout


# ---------------------------------------------------------------------------
# Finales Label für MP-Daten
# ---------------------------------------------------------------------------

def compute_final_label_mp(
    my_state:  dict,
    all_final: dict[str, dict],   # {player_id: final_state}
    my_id:     str,
) -> tuple[float, int, int]:
    """
    Berechnet finales Label für MP-Trainingsdaten.
    Gibt (label, my_vp, won_rank) zurück.

    Label: +1.0 bis +2.0 (Sieg) oder -2.0 bis -1.0 (Niederlage)
    Basiert auf VP-Differenz zum besten Gegner.
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

    # Rang berechnen
    all_vps = sorted([my_vp] + opp_vps, reverse=True)
    rank    = all_vps.index(my_vp) + 1

    return label, my_vp, rank


# ---------------------------------------------------------------------------
# Haupt-Bot-Loop (Multiplayer)
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

    # Sind fremde Spieler-IDs bekannt? In einer echten Partie (Beitritt nur über
    # die eigene ID) ist das nicht der Fall – dann darf die "alle inaktiv"-
    # Spielende-Erkennung nicht greifen (man kann die Gegner nicht abfragen),
    # und das Spielende wird ausschließlich über phase=="end" erkannt.
    _know_opponents = any(pid != player_id for pid in all_player_ids)

    lock             = MCTSLock(my_color, game_id)
    db_ready         = False
    errors           = 0
    last_key         = None
    # Draft-Repick-Pools, die dieser Bot in dieser Partie schon beantwortet hat.
    # Ohne diesen Schutz antwortet der Bot MEHRFACH auf dieselbe Draft-Runde: die
    # normale Dedup unten haengt an gameAge, und gameAge STEIGT sobald der Mensch
    # draftet -> Dedup greift nicht mehr -> zweite Antwort auf denselben Pool ->
    # der Server hat die Karte schon verarbeitet -> "Card <Name> not found" -> HTTP 400
    # -> nach 4 Versuchen Abbruch (apeheads Absturz 18.07., Gen 12). Die beiden A/B-
    # Schleifen hatten diesen Schutz laengst (15.07.), der --vs-human-Pfad nicht.
    repick_done: set = set()
    last_action_time = time.time()   # Zeitstempel letzter erfolgreicher Zug
    mcts_transitions: list[dict] = []
    post_error_counts: dict = {}     # Fehlschläge je Input (gegen Endlosschleifen)

    # SCHATTEN-POLL (13.08.): frueher stand dieser Aufruf NUR im Zweig
    # "if not need_action" - also nur, wenn der Bot selbst nichts zu tun hatte.
    # In der DRAFTPHASE sind aber BEIDE Spieler gleichzeitig gefragt: der Bot hat
    # immer etwas zu tun und pollte den Menschen deshalb nie. Folge: von 113
    # Entscheidungspunkten einer Partie stammte genau EINER aus dem Draft - fuer
    # die Frage "wer greift welche Karte ab" war der Schattenbot blind.
    # Jetzt wird bei JEDEM Schleifendurchlauf gepollt. Die Dedup ueber _sig im
    # Schattenmodul verhindert Doppeleintraege; Mehrkosten sind ein GET je
    # Bot-Entscheidung, gegen einen menschlichen Gegner irrelevant.
    def _shadow_poll():
        if _shadow is None or not human_opponent:
            return
        for _hid in all_player_ids:
            if _hid == player_id:
                continue
            try:
                _shadow.shadow_step(base_url, _hid, get_state, decide, my_color)
            except Exception as _se:
                import traceback as _tb2
                run_mcts_bot_mp._shadow_errs = getattr(
                    run_mcts_bot_mp, "_shadow_errs", 0) + 1
                _n = run_mcts_bot_mp._shadow_errs
                if _n == 1 or _n % 50 == 0:
                    log.error("  \u274c SCHATTEN-ANALYSE FEHLGESCHLAGEN (%d\u00d7): %s: %s",
                              _n, type(_se).__name__, _se)
                    log.error("     -> im Log fehlen die Entscheidungspunkte! %s",
                              _tb2.format_exc().strip().splitlines()[-1])

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

        # Spielende. WICHTIG: solange >=8 Pflanzen anstehen, NICHT ueber die
        # Phasen-Heuristik beenden - nach der letzten Gen kommt die Produktionsphase
        # (phase nicht in der Liste) und DANN die finale Greenery-Abfrage. Stieg der
        # Bot hier aus, haengt der Server auf die nie kommende Platzierung.
        # Bei >=8 Pflanzen beendet nur noch das explizite phase=="end".
        # Die Prelude-Phase (und kuenftige Erweiterungs-Phasen) sind KEIN Spielende.
        # Zwei Absicherungen: 'preludes' explizit in der Whitelist, UND eine Gen>1-Sperre
        # (kein Spiel endet in Generation 1) - faengt auch unbekannte fruehe Phasen ab.
        _gen = state.get("game", {}).get("generation", 0) or 0
        game_over = (phase == "end") or (
            not waiting and not _can_place_final_greenery(state) and _gen > 1 and phase not in
            ("research", "action", "drafting", "initialdrafting", "solar",
             "preludes", "prelude", "initialcards")
        )

        # Echte Spielende-Erkennung ohne Timeout:
        # 1. phase == "end" → explizit vom Server
        # 2. Alle Parameter maximal UND kein Spieler hat mehr Züge
        #    (erkennbar: alle Spieler haben isActive=False und waiting=None)
        if not game_over and not waiting and phase == "action":
            try:
                game_data = state.get("game", {})
                oxygen = game_data.get("oxygenLevel", 0)
                temp   = game_data.get("temperature", -30)
                oceans = game_data.get("oceans", 0)
                params_full = (oxygen >= 14 and temp >= 8 and oceans >= 9)

                # WICHTIG: nicht fuer beendet erklaeren, solange >=8 Pflanzen anstehen.
                # Am Spielende laeuft die finale Greenery-Phase - der Bot muss die
                # Pflanzen noch umwandeln. Bricht er hier ab, haengt der Server auf die
                # nie kommende Greenery-Platzierung ("place final greenery").
                #
                # Gegen MENSCHEN abgeschaltet: 'alle inaktiv' ist ein RACE. Zwischen zwei
                # eigenen Aktionen (z.B. Endgame-Liquidation: Karte verkaufen) ist der Bot
                # kurz inaktiv und der bereits gepasste Gegner ebenfalls -> ein einzelner
                # Snapshot meldet faelschlich Spielende, obwohl der Server den Bot gleich
                # wieder am Zug hat. Fuer Menschen-Spiele gilt darum - konsistent zum
                # ebenfalls deaktivierten 90s-Idle-Backstop (s.u.) - NUR phase=="end";
                # die Recovery-Schleife erholt sich bei Reaktivierung von selbst.
                if (_know_opponents and not human_opponent and params_full
                        and not is_active and not waiting and not _can_place_final_greenery(state)):
                    # Alle Parameter maximal + dieser Bot inaktiv.
                    # Prüfe ob auch alle anderen Spieler inaktiv sind
                    # (warten auf Bonusgreenery-Aktionen der Gegner).
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

                # Fallback: Server hat phase gewechselt
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

            # Finales Label aus VP-Vergleich mit allen Gegnern
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

            # Trainingsdaten speichern
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

            # Lock beim Spielende sicherstellen freigeben
            lock.release()
            break

        player    = state.get("thisPlayer", {})
        is_active = player.get("isActive", False)
        wtype     = waiting.get("type", "") if waiting else ""

        # initialCards: isActive ist oft False, Bot muss aber trotzdem antworten
        # Research-Phase 'card': Kartenkauf auch bei isActive=False nötig
        # Research-Phase waiting=None: Server wartet auf anderen Spieler → warten
        # Steht etwas in UNSEREM waitingFor, ist es unsere Eingabe - unabhaengig von
        # isActive. isActive ist in Simultan-/Aufraeumphasen unzuverlaessig: z.B. die
        # FINALE Greenery-Umwandlung am Spielende kommt als 'space' mit isActive=False;
        # die alte Bedingung (ohne 'space') verschluckte sie -> Haenger "place final greenery".
        # In Simultanphasen setzt der Server waitingFor nur fuer den Spieler, der dran ist.
        need_action = bool(waiting)

        _shadow_poll()   # auch waehrend der Draftphase, s. Kommentar oben

        # Sonderfall: phase=action, isActive=False, waiting=None
        # Kann bedeuten dass Server noch verarbeitet ODER wirklich andere dran ist
        # ODER dass ein Rollback (MCTS) den Zustand verbogen hat und wir uns
        # fälschlich für inaktiv halten. Aktiv frischen State holen und erholen.
        if not need_action and phase == "action" and not waiting and not is_active:
            if not hasattr(run_mcts_bot_mp, "_idle_action_count"):
                run_mcts_bot_mp._idle_action_count = 0
            run_mcts_bot_mp._idle_action_count += 1
            # Bei JEDEM Idle-Poll frischen State holen (nicht nur alle 5)
            try:
                fresh = get_state(base_url, player_id)
                fresh_waiting = fresh.get("waitingFor")
                fresh_active  = fresh.get("thisPlayer", {}).get("isActive", False)
                fresh_phase   = fresh.get("game", {}).get("phase", "")
                if fresh_waiting or fresh_active:
                    state = fresh
                    need_action = True
                    last_key = None          # erzwinge Neubewertung
                    run_mcts_bot_mp._idle_action_count = 0
                elif fresh_phase == "end":
                    state = fresh
                    break   # Spielende
            except Exception:
                pass
            # Recovery-Anstoß: hängt der Bot länger in diesem Zustand, periodisch
            # last_key zurücksetzen, damit eine evtl. verschluckte Aktion neu
            # versucht wird (entschärft Rollback-induzierten Deadlock).
            if not need_action and run_mcts_bot_mp._idle_action_count % 50 == 0:
                last_key = None
            # Diagnose-Sicherung: haengt der Bot mit >=8 Pflanzen lange idle OHNE
            # waitingFor, taucht die finale Greenery-Abfrage offenbar nicht im
            # gepollten Zustand auf. Einmalig vollen Zustand (inkl. Gegner) sichern.
            if (_can_place_final_greenery(state) and run_mcts_bot_mp._idle_action_count == 30
                    and not getattr(run_mcts_bot_mp, "_hang_dumped", False)):
                try:
                    opp_states = {oid: get_state(base_url, oid)
                                  for oid in all_player_ids if oid != player_id}
                    # Wartet ein Gegner? Dann ist es schlicht sein Zug -> kein Hang.
                    if any(s.get("waitingFor") for s in opp_states.values()):
                        run_mcts_bot_mp._idle_action_count = 0   # spaeter neu pruefen
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
            # Nach 90s echtem Idle → Spielende annehmen (vorher 5 Min).
            # Gegen menschliche Gegner deaktiviert: ein guter Zug dauert oft länger.
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
            time.sleep(POLL_INTERVAL)
            continue

        if wtype == "player":
            # BUGFIX 18.07. (apeheads Fish-Befund): Hier stand "eigene Farbe wählen" —
            # der Runner beantwortete die Spielerauswahl SELBST und griff damit immer
            # den Bot an. Bei Fish ("Select player to decrease plants production")
            # senkte der Bot so seine EIGENE Pflanzenproduktion, obwohl apehead welche
            # hatte. Ants funktionierte, weil es über SelectCard läuft und den normalen
            # Pfad nimmt. tm_bot.handle_player unterscheidet längst korrekt zwischen
            # Angriff (decrease/remove/steal/lose -> Gegner) und Bonus (-> selbst), wurde
            # hier aber nie aufgerufen. Die A/B-Schleife (s.u.) machte es schon richtig;
            # wie beim Draft-Absturz war NUR der --vs-human-Pfad übersehen worden.
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

        # Deduplizierung – initialCards nie deduplizieren (Server ändert State erst
        # wenn beide Spieler geantwortet haben, daher bleibt key gleich)
        if not waiting:
            time.sleep(POLL_INTERVAL)
            continue
        # gameAge unterscheidet echte neue Entscheidungen mit identischer Signatur
        # (z.B. mehrere aufeinanderfolgende "Place any final greenery from plants").
        # Innerhalb desselben Server-Zustands bleibt gameAge gleich -> Dedup haelt,
        # kein Doppel-Senden; nach einer Platzierung steigt gameAge -> naechste wird bearbeitet.
        key = (wtype, str(waiting.get("title", "")),
               tuple(c.get("name", "") for c in waiting.get("cards", [])),
               (state.get("game") or {}).get("gameAge"))
        if key == last_key and wtype != "initialCards":
            time.sleep(POLL_INTERVAL)
            continue

        # DRAFT-REPICK: diesen Kartenpool schon beantwortet? Dann NICHT erneut senden -
        # die gewaehlte Karte ist serverseitig bereits verarbeitet und ein zweiter Versuch
        # scheitert mit "Card <Name> not found" (HTTP 400). Warten, bis der Server ein
        # neues Paeckchen anbietet (dann aendert sich der Pool-Key).
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

        # DB-Bereitschaft prüfen (nötig für MCTS)
        if enable_mcts and not db_ready and phase == "action":
            db_ready = wait_for_db(base_url, game_id, max_wait=5)
            if db_ready:
                log.info("   Spiel in DB – MCTS aktiviert")
            else:
                log.info("   Spiel nicht in DB – MCTS deaktiviert")

        _wt = waiting.get("title", "")
        _wtitle = _wt.lower() if isinstance(_wt, str) else ""
        _is_action_menu = "take your" in _wtitle   # nur Hauptaktionsmenue, keine Inline-OR-Kartenaufloesung
        use_mcts = (enable_mcts and phase == "action" and wtype == "or"
                    and db_ready and _is_action_menu)
        payload    = None
        rollout_sc = 0.0
        payload    = None   # Immer initialisieren
        rollout_sc = 0.0

        if use_mcts:
            try:
                # State nochmal holen – könnte veraltet sein wenn lange auf Lock gewartet
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
                lock.release()   # Sicherstellen
                result = decide(state)
                payload = result[0] if isinstance(result, tuple) else result
                rollout_sc = 0.0

            # Trainingsdaten sammeln
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
            last_action_time = time.time()   # Erfolgreicher Zug
            # --- Transitions-Logging (Bot-Selbstspiel, Voll-Zustand pro Zug).
            #     Optional, Default AUS – mit TM_LOG_TRANSITIONS=1 einschalten. ---
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
            # --- Ende ---
            last_key = key
            time.sleep(POST_WAIT)
            if enable_mcts:
                # MCTS: aktiv warten bis der Server den Zug verarbeitet hat
                # (Save-ID ändert sich). Bis dahin last_key beibehalten, damit
                # kein Doppelzug feuert. Fallback-Grenze gegen Hängenbleiben.
                waited_post = POST_WAIT
                while waited_post < 3.0:
                    try:
                        if get_last_save_id(base_url, game_id) != save_before:
                            last_key = None   # State hat sich geändert → neu lesen
                            break
                    except Exception:
                        last_key = None
                        break
                    time.sleep(POLL_INTERVAL)
                    waited_post += POLL_INTERVAL
            else:
                # Heuristik-Modus (--join): post_input ist synchron, der Server-
                # State ist nach Rückkehr bereits aktualisiert. Die save_id-Prüfung
                # ist hier nicht verlässlich (keine MCTS-game_id; game_id == player_id
                # → get_last_save_id liefert konstant -1), wodurch last_key nie
                # zurückgesetzt würde. Folge: eine Aktion ohne eigenen Folge-Input
                # (z.B. Asteroid:SP) führt zurück zu "Take your next action" mit
                # identischem key → der Doppelzug-Schutz (key == last_key) blockiert
                # die Folgeaktion und der Bot pollt endlos ohne Log. Daher direkt
                # neu lesen.
                last_key = None
            # Erfolgreicher Post -> Sperrliste zuruecksetzen (sie gilt nur fuer den
            # aktuellen Fehlerzyklus, nicht fuer die ganze Partie).
            try:
                import tm_bot as _tb
                # 17.08.: Auch hier nur die TRANSIENTEN Sperren loesen. Die Persistenz-
                # regel (PERSIST_FAIL_LIMIT) wurde am 15.08. nur im A/B-Pfad
                # (_step_player) eingebaut; dieser --vs-human-Pfad leerte weiterhin nach
                # JEDEM erfolgreichen Post alles. Folge live gesehen: Self-replicating
                # Robots scheiterte, wurde gesperrt, ein anderer Zug gelang, die Sperre
                # fiel weg, der Bot griff erneut zu - vierter Fehlversuch, Partieabbruch.
                # Exakt derselbe Zyklus wie am 15.08., nur im anderen Treiber.
                if hasattr(_tb, "clear_transient_rejects"):
                    _tb.clear_transient_rejects(game_id)
                else:
                    if _tb._draft_rejected:
                        _tb._draft_rejected.clear()
                    if getattr(_tb, "_play_rejected", None):
                        _tb._play_rejected.clear()
            except Exception:
                pass
        except requests.HTTPError as e:
            # Fehlschläge je Input zählen (robust gegen oszillierende Zustände,
            # bei denen erfolgreiche Zwischenposts einen reinen "consecutive"-
            # Zähler immer wieder zurücksetzen würden).
            # Die Kartennamen gehoeren in den Schluessel: im Draft ist der TITEL ueber
            # ALLE Runden identisch ("{'data': [{'type': 2, 'value': 'blue'}]}"), sodass
            # sich Fehler aus verschiedenen Draft-Runden zu einem Abbruch aufsummierten
            # (apehead 18.07.: Abbruch bei "4x", obwohl in dieser Runde nur 1-2 Versuche
            # scheiterten). Mit den Karten im Schluessel zaehlt jede Runde fuer sich.
            err_key = (wtype, str(waiting.get("title", ""))[:80],
                       tuple(c.get("name", "") for c in (waiting.get("cards") or [])))
            post_error_counts[err_key] = post_error_counts.get(err_key, 0) + 1
            n_err = post_error_counts[err_key]
            log.warning("  HTTP Fehler (%d× für diesen Input): %s",
                        n_err, e.response.text[:200] if e.response else e)
            if n_err == 1:
                # Erster Fehlschlag → volle Diagnose: was wollte der Server,
                # was hat der Bot gesendet? (Hilft, fehlende Handler zu bauen.)
                log.warning("  ⚠️ Unbeantworteter Input – waitingFor: %r", waiting)
                log.warning("  ⚠️ Gesendete Antwort: %r", payload)
            if n_err >= 4:
                log.error("  ❌ Input %d× nicht beantwortbar – Abbruch, um eine "
                          "Endlosschleife zu vermeiden. Bitte obige waitingFor-"
                          "Struktur melden, dann lässt sich der Handler ergänzen.",
                          n_err)
                break
            # DRAFT-CACHE VERWERFEN (apeheads Absturz 18.07., die eigentliche Ursache):
            # _draft_choice_cache haelt die Wahl pro Kartenpool fest (Repick-Stabilitaet).
            # Hat der Server sie gerade ABGELEHNT ("Card <Name> not found"), ist genau diese
            # Wahl falsch - der Bot wuerde sie sonst bei jedem Retry erneut senden und nach
            # 4 Versuchen abbrechen. Cache leeren => naechster Versuch entscheidet frisch
            # gegen den aktuellen Pool.
            try:
                import re as _re
                import tm_bot as _tb
                _tb._draft_choice_cache.clear()
                # Der Server nennt die abgelehnte Karte im Klartext:
                # "Error: Card <Name> not found". Diese Karte gezielt sperren, damit der
                # Bot beim naechsten Versuch die NAECHSTBESTE waehlt statt dieselbe erneut.
                _txt = e.response.text if e.response is not None else ""
                for _m in _re.finditer(r"Card (.+?) not found", _txt):
                    _tb._draft_rejected.add(_m.group(1).strip())
                # 28.07.: Zahlungs-Ablehnung ("Did not spend enough to pay for card").
                # Der Server nennt die Karte NICHT - aber wir wissen, welche wir gerade
                # geschickt haben. Ohne diese Sperre wiederholt der Bot dieselbe Zahlung
                # bis zum Partie-Abbruch (gesehen am 28.07., Self-replicating Robots).
                if "pay for card" in _txt or "Did not spend enough" in _txt:
                    _card = None
                    try:
                        _card = (payload.get("response") or {}).get("card")
                    except Exception:
                        pass
                    if _card:
                        # 17.08.: ueber note_reject, damit der Fehlversuchszaehler
                        # laeuft und die Karte ab dem zweiten Mal fuer den REST der
                        # Partie gesperrt bleibt (siehe oben).
                        if hasattr(_tb, "note_reject"):
                            _tb.note_reject("play", _card)
                        else:
                            _tb._play_rejected.add(_card)
                        log.warning("  ⚠️ Zahlung abgelehnt, Karte gesperrt: %s (%d. Versuch)",
                                    _card,
                                    getattr(_tb, "_play_fail_count", {}).get(_card, 1))
                if _tb._draft_rejected:
                    log.warning("  ⚠️ Vom Server abgelehnte Draft-Karten: %s",
                                sorted(_tb._draft_rejected))
            except Exception:
                pass
            # last_key=None: nächster Poll versucht State neu zu lesen
            last_key = None
            time.sleep(ERROR_WAIT)
        except Exception as e:
            log.warning("  Fehler: %s", e)
            last_key = None
            time.sleep(ERROR_WAIT)


# ---------------------------------------------------------------------------
# Auto-Spiel-Modus: ein Bot erstellt Spiel, andere lesen IDs
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sequenzieller Ein-Prozess-Self-Play-Treiber
#
# Ein einziger Prozess steuert ALLE Bots streng nacheinander. Da Terraforming
# Mars rundenbasiert ist, ist zu jedem Zeitpunkt nur ein Bot am Zug (Ausnahme:
# Draft/Research, wo mehrere parallel wählen – dort wird einfach abwechselnd
# bedient, ohne Rollback). Während der MCTS-Bot seine Rollouts inkl. Rollback
# fährt, pollt KEIN anderer Prozess die Partie – damit gibt es keine
# Geisterzustände, keine Doppelzüge und keinen Rollback-Deadlock mehr.
# Lock, ready-Dateien und Koordinierungsdateien sind hier überflüssig.
# ---------------------------------------------------------------------------

def _can_place_final_greenery(state: dict) -> bool:
    """True, wenn der Bot am Spielende noch mind. eine Greenery aus Pflanzen legen kann.
    ERSETZT die frueher hart kodierte Schwelle `my_plants < 8`: die echten Greenery-Kosten
    variieren (Ecoline = 6 Pflanzen statt 8). Mit Ecoline und 6-7 Pflanzen dachte der alte
    Code faelschlich `< 8 -> keine Greenery mehr anstehend` und stieg aus der Warteschleife
    aus -> der Server haengt auf die nie kommende finale Greenery-Platzierung (17.07.)."""
    try:
        return can_convert_plants(state)
    except Exception:
        return state.get("thisPlayer", {}).get("plants", 0) >= 8


def _needs_action(state: dict) -> bool:
    """True, wenn der Spieler im gegebenen State eine Eingabe machen muss."""
    waiting = state.get("waitingFor")
    if not waiting:
        return False
    # 28.07.: Frueher wurde nur bedient, wenn isActive gesetzt war ODER der Typ in einer
    # kurzen Liste stand ODER die Draft-Phase lief. Das ist zu eng: Der Server setzt
    # waitingFor NUR fuer den Spieler, der am Zug ist - isActive ist dagegen in Simultan-
    # und Cleanup-Phasen unzuverlaessig. Gefunden an der SOLAR-Phase (solarPhaseOption,
    # World Government): dort kam ein 'or' mit isActive=False, fiel durch alle drei Raster
    # und die Partie stand bis zum Abbruch. Dieselbe Fehlerklasse war im --vs-human-Pfad
    # laengst behoben (finale Gruenflaechen-Umwandlung als 'space' mit isActive=False);
    # der A/B-Treiber hatte die alte, enge Fassung behalten.
    return True


def _acting_bot_module(decide_fn):
    """Das Bot-Modul, zu dem die handelnde decide-Funktion gehoert.

    14.08.: Die Ablehnungssperren (_play_rejected/_draft_rejected) wurden bisher HART
    in `tm_bot` eingetragen und geleert. Im A/B spielen aber ZWEI Module - Challenger
    `tm_bot` und Champion `tm_bot_champion` - und die Sitze tauschen je Paar. Filtert
    handle_or im Champion gegen dessen eigene (leere) Menge, wiederholt der Bot dieselbe
    abgelehnte Zahlung bis zum Partieabbruch (gesehen 14.08.: Self-replicating Robots,
    8x, Partie 50 abgebrochen; 3 von 50 Partien des Laufs verloren). Im --vs-human-Pfad
    gibt es nur ein Modul, deshalb fiel es beim Portieren am 28.07. nicht auf.
    ⚠ Nicht ersatzweise in ALLE Module eintragen - das wuerde den einen Arm wegen eines
      Fehlers des anderen sperren und die beiden Arme unvergleichbar machen.
    """
    try:
        return sys.modules.get(getattr(decide_fn, "__module__", "") or "tm_bot")
    except Exception:
        return None


def _step_player(
    state, base_url, player_id, all_player_ids, game_id, my_color,
    enable_mcts, lock, db_ready, game_id_for_db,
    n_rollouts, max_candidates, simple_rollout, transitions,
    mcts_allowed_now: bool = True,
    decide_fn=None,
) -> bool:
    """
    Trifft EINE Entscheidung für player_id und postet sie.
    Gibt True zurück, wenn ein Zug erfolgreich gepostet wurde.
    Wiederverwendung der erprobten Bausteine: handle_or_mcts_mp / decide().
    """
    decide_fn = decide_fn or decide   # umschaltbare Heuristik-Variante pro Farbe

    # Veralteten Zustand erkennen, statt blind erneut zu antworten (s. _LAST_ANSWERED).
    _key = _answer_key(state)
    if _key[0] != "initialCards":          # dort aendert sich der Zustand erst, wenn BEIDE geantwortet haben
        _prev, _skips = _LAST_ANSWERED.get(player_id, (None, 0))
        if _key == _prev:
            if _skips < STALE_MAX_SKIPS:
                _LAST_ANSWERED[player_id] = (_prev, _skips + 1)
                time.sleep(STALE_RECHECK_WAIT)
                return False               # Aufrufer liest neu
            log.debug("  Zustand nach %d Versuchen unveraendert - sende trotzdem", _skips)
    waiting = state.get("waitingFor")
    phase   = state.get("game", {}).get("phase", "")
    player  = state.get("thisPlayer", {})
    wtype   = waiting.get("type", "")
    gen     = state.get("game", {}).get("generation", 1)
    mc      = player.get("megacredits", 0)
    tr      = player.get("terraformRating", 14)

    log.info("[%s|Gen %d] MC:%d TR:%d | %s", my_color, gen, mc, tr,
             str(waiting.get("title", ""))[:40])

    # Spielerauswahl: an das Heuristik-Modul delegieren (z.B. Cloud-Seeding-Fix
    # im Challenger). Module ohne 'player'-Handler (eingefrorener Champion)
    # liefern None → Fallback auf das bisherige Verhalten (eigene Farbe).
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

    # DB-Bereitschaft prüfen (nötig für MCTS); db_ready ist 1-elementige Liste (by-ref)
    if enable_mcts and not db_ready[0] and phase == "action":
        db_ready[0] = wait_for_db(base_url, game_id_for_db, max_wait=5)
        if db_ready[0]:
            log.info("   Spiel in DB – MCTS aktiviert")

    _wt = waiting.get("title", "")
    _wtitle = _wt.lower() if isinstance(_wt, str) else ""
    _is_action_menu = "take your" in _wtitle   # nur Hauptaktionsmenue, keine Inline-OR-Kartenaufloesung
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
        # Trainingsdaten sammeln (nur MCTS-Bot)
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
        save_before = None if POST_REUSE else get_last_save_id(base_url, game_id)
        _new_state = post_input(base_url, player_id, payload)
        # Enthaelt die Antwort ein Spielermodell, hat der Server den Zug verarbeitet -
        # Warten und Nachfragen sind dann ueberfluessig (s. POST_REUSE oben).
        _LAST_ANSWERED[player_id] = (_key, 0)
        _confirmed = POST_REUSE and isinstance(_new_state, dict) and bool(_new_state.get("runId"))
        if _confirmed:
            # Kurze Schutzpause gegen veraltete Folgezustaende (s. POST_REUSE_WAIT).
            if POST_REUSE_WAIT > 0:
                time.sleep(POST_REUSE_WAIT)
        else:
            # Kurz auf Serververarbeitung warten (Save-ID-Wechsel), damit der
            # nächste Poll den Folgezustand sieht statt denselben Zug erneut.
            time.sleep(POST_WAIT)
            if save_before is not None:
                waited = POST_WAIT
                while waited < 2.0:
                    try:
                        if get_last_save_id(base_url, game_id) != save_before:
                            break
                    except Exception:
                        break
                    time.sleep(POLL_INTERVAL)
                    waited += POLL_INTERVAL
        # HINWEIS: Eine "Zug ohne Zustandsaenderung"-Warnung auf save_id-Basis
        # wurde am 07.06. eingebaut und wieder entfernt: die save_id ist bei
        # kleinem POST_WAIT kein verlaessliches Pro-Input-Signal (1953 Fehl-
        # alarme auf nachweislich wirksamen Zuegen, Lauf 21:41). Echte No-Op-
        # Erkennung braeuchte einen State-Vergleich (re-GET), nicht die save_id.
        # 14.08.: Sperrliste nach erfolgreichem Post leeren - sie gilt nur fuer den
        # aktuellen Fehlerzyklus. Im A/B-Pfad fehlte das ganz: eine einmal gesperrte
        # Karte blieb im Worker-Prozess ueber ALLE folgenden Partien gesperrt.
        try:
            _tb_ok = _acting_bot_module(decide_fn)
            if _tb_ok is not None:
                if hasattr(_tb_ok, "clear_transient_rejects"):
                    # 15.08.: gibt NUR frei, was seltener als PERSIST_FAIL_LIMIT mal
                    # gescheitert ist; bei neuer Partie wird alles zurueckgesetzt.
                    _tb_ok.clear_transient_rejects(game_id)
                else:
                    for _n in ("_draft_rejected", "_play_rejected", "_space_rejected"):
                        if getattr(_tb_ok, _n, None):
                            getattr(_tb_ok, _n).clear()
        except Exception:
            pass
        return True
    except requests.HTTPError as e:
        body = ""
        try:
            body = e.response.text[:300] if e.response is not None else ""
        except Exception:
            body = ""
        sc = e.response.status_code if e.response is not None else "?"
        log.warning("  HTTP %s Fehler. Server-Body: %r", sc, body)
        # Der Server-Body ist bei diesem 400 oft LEER -> die eigentliche Diagnose ist,
        # WAS der Bot gesendet hat und WORAUF. Beides einmalig ausgeben (15.07.).
        log.warning("  ⚠️ Gesendete Payload: %r", payload)
        log.warning("  ⚠️ waitingFor-Titel: %r | type=%r",
                    (state.get("waitingFor") or {}).get("title"),
                    (state.get("waitingFor") or {}).get("type"))
        # 28.07.: Ablehnungen gezielt sperren, sonst schickt der Bot dieselbe Antwort
        # bis zum Partie-Abbruch (8x ohne Fortschritt). Der A/B-Treiber hatte diese
        # Behandlung nie - sie stand nur im --vs-human-Pfad.
        try:
            import re as _re
            _tb = _acting_bot_module(decide_fn)      # 14.08.: NICHT mehr hart tm_bot
            if _tb is not None:
                for _m in _re.finditer(r"Card (.+?) not found", body or ""):
                    _tb._draft_choice_cache.clear()
                    _tb._draft_rejected.add(_m.group(1).strip())
                if "pay for card" in (body or "") or "Did not spend enough" in (body or ""):
                    _card = (payload.get("response") or {}).get("card")
                    if _card:
                        # 15.08.: ueber note_reject, damit eine dauerhaft unbezahlbare
                        # Karte nach PERSIST_FAIL_LIMIT Versuchen fuer den REST DER PARTIE
                        # gesperrt bleibt und nicht nach jedem erfolgreichen Post
                        # zurueckkehrt (Endlosschleife, live 15.08.).
                        if hasattr(_tb, "note_reject"):
                            _tb.note_reject("play", _card)
                        else:
                            _tb._play_rejected.add(_card)
                        log.warning("  ⚠️ Zahlung abgelehnt, Karte gesperrt (%s): %s (%d. Versuch)",
                                    getattr(_tb, "__name__", "?"), _card,
                                    getattr(_tb, "_play_fail_count", {}).get(_card, 1))
                # 15.08.: dasselbe fuer FELDER. "Selected space is occupied" /
                # "no spaces available" liess den Bot dasselbe Feld achtmal senden,
                # danach Partieabbruch (Gen 19, Feld 51, Partie 24).
                _body = (body or "")
                if "space is occupied" in _body or "no spaces available" in _body \
                        or "Selected space" in _body:
                    _sid = (payload.get("response") or payload or {}).get("spaceId")
                    if _sid and hasattr(_tb, "note_reject"):
                        _tb.note_reject("space", _sid)
                        log.warning("  ⚠️ Feld abgelehnt, gesperrt (%s): %s (%d. Versuch)",
                                    getattr(_tb, "__name__", "?"), _sid,
                                    getattr(_tb, "_space_fail_count", {}).get(_sid, 1))
        except Exception:
            pass
        time.sleep(ERROR_WAIT)
        return False
    except Exception as e:
        log.warning("  Fehler: %s", e)
        time.sleep(ERROR_WAIT)
        return False


def _finalize_selfplay(base_url, color_to_id, mcts_color, transitions, data_file,
                       prod_snap=None):
    """Endwertung: finale VPs, Label berechnen, Trainingsdaten speichern."""
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
    # Engine-Diagnose: Produktion pro Farbe am Spielende (PFLANZEN = Schluessel-
    # indikator). Erlaubt, den Pflanzen-Hebel direkt im A/B-Log zu messen.
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

    # Ergebnis nach FARBE (pid wechselt je Partie, Farbe ist stabil) für die
    # Aggregation über mehrere Partien in run_sequential_selfplay.
    # VERHALTENS-DATEN je Farbe (fuer den Strategie-Layer): Endproduktion, Endkacheln
    # und die Gen-3/5/8-Snapshots. Erlaubt, die Strategie-Ketten im A/B zu messen,
    # ohne Menschpartien.
    def _conv_counts(_p):
        _out = {"heat": 0, "plant": 0}
        for _m in list(sys.modules.values()):
            _c = getattr(_m, "CONV_COUNTS", None)
            if isinstance(_c, dict) and _p in _c:
                for _k in _out:
                    _out[_k] += (_c[_p] or {}).get(_k, 0)
        return _out

    def _dict_from_modules(_p, _attr):
        """Kartenbezogene Zaehler (ACT_COUNTS/PLAY_GEN) aus dem jeweiligen Bot-Modul.
        Challenger und Champion sind verschiedene Module, die pid ist eindeutig."""
        _out = {}
        for _m in list(sys.modules.values()):
            _c = getattr(_m, _attr, None)
            if isinstance(_c, dict) and _p in _c:
                for _k, _v in (_c[_p] or {}).items():
                    if _attr == "PLAY_GEN":
                        _out.setdefault(_k, _v)
                    else:
                        _out[_k] = _out.get(_k, 0) + _v
        return _out

    # 13.08.: GLOBALPARAMETER am Spielende. behav kannte bisher nur Spielerwerte -
    # ob Venus in einer Partie ueberhaupt gestiegen ist, war nicht nachvollziehbar.
    # Genau daran haengt aber die Frage, warum venusgesperrte Engines zu 64 % auf der
    # Hand liegenbleiben: der Bot prognostiziert den Venusanstieg mit demselben Prior
    # wie Sauerstoff (1.0 je Generation), obwohl niemand gezwungen ist, Venus zu heben.
    _gm = {}
    for _pid in color_to_id.values():
        _g = (all_final.get(_pid) or {}).get("game") or {}
        if _g:
            _gm = {"temperature": _g.get("temperature", -30),
                   "oxygen":      _g.get("oxygenLevel", 0),
                   "oceans":      _g.get("oceans", 0),
                   "venus":       _g.get("venusScaleLevel", 0),
                   "generation":  _g.get("generation", 0)}
            # 14.08.: VERLAUF der Globalparameter. Der Server fuehrt
            # `globalsPerGeneration` (der Bot liest es bereits fuer LEVER_ENDGAME_RATE).
            # Damit laesst sich pruefen, ob die Parameter BESCHLEUNIGT steigen -
            # `_param_rate` extrapoliert linear vom Spielstart, und wenn die zweite
            # Haelfte schneller ist, faellt `gtu` zu gross aus und torgesperrte Engines
            # werden mit activations = 0 abgelehnt (Birds/Penguins, live 13.08.).
            # Genau dieser Fehler wurde fuer r_eff mit LEVER_HORIZON_CURVE behoben;
            # _gens_to_global_req hat die Behandlung nie bekommen.
            try:
                _hist = _g.get("globalsPerGeneration") or []
                _gm["history"] = [{"temperature": _h.get("temperature"),
                                   "oxygen":      _h.get("oxygen", _h.get("oxygenLevel")),
                                   "oceans":      _h.get("oceans"),
                                   "venus":       _h.get("venus", _h.get("venusScaleLevel"))}
                                  for _h in _hist if isinstance(_h, dict)]
            except Exception:
                _gm["history"] = []
            break

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
            # 13.08.: Ressourcen-BESTAND am Spielende (nicht Produktion). Ohne ihn ist
            # "ungenutzte Waerme am Spielende" nicht messbar - live lagen 14-22 Waerme
            # brach, waehrend die Temperatur ihr Maximum erreichte.
            "end_res": {
                "mc":     _tp.get("megaCredits", 0),
                "steel":  _tp.get("steel", 0),
                "titan":  _tp.get("titanium", 0),
                "plant":  _tp.get("plants", 0),
                "energy": _tp.get("energy", 0),
                "heat":   _tp.get("heat", 0),
            },
            # 13.08.: Umwandlungen (Waerme->Temperatur, Pflanzen->Gruenflaeche). Die
            # Zaehler liegen im jeweiligen Bot-Modul (CONV_COUNTS, pid-basiert);
            # Challenger und Champion sind verschiedene Module, deshalb ueber alle
            # geladenen Module summiert - die pid ist eindeutig, Doppelzaehlung
            # ausgeschlossen.
            "conv":      _conv_counts(_pid),
            "globals":   dict(_gm),          # 13.08.: Endstand der Globalparameter
            # 13.08.: Aktivierungen und Ausspielgeneration je Karte - trennt
            # "zu spaet gespielt" von "nie priorisiert" bei den Engines mit 0 Markern.
            # 14.08.: Trichter seen -> drafted -> bought fuer Ressourcen-VP-Engines
            "funnel":    _dict_from_modules(_pid, "FUNNEL"),
            # 16.08.: wie oft/wann feuert das Leerlaufgeld-Flag
            "idle":      _dict_from_modules(_pid, "IDLE_STATS"),
            "card_act":  _dict_from_modules(_pid, "ACT_COUNTS"),
            "play_gen":  _dict_from_modules(_pid, "PLAY_GEN"),
            "cities":    _tp.get("citiesCount", 0),
            "vp_parts":  _tp.get("victoryPointsBreakdown", {}),
            "snap":      (prod_snap or {}).get(_c, {}),
            # 23.07.: Tableau-Namen mitschreiben. Befund aus den Live-Partien: apehead
            # hat 19 Ressourcen-VP-Karten (Ants/Birds/Livestock ...) im Tableau, der Bot
            # nur 5 - das erklaert den -23-Karten-VP-Rueckstand fast vollstaendig.
            # Bewertung und Kaufschwelle sind NICHT der Engpass (geprueft), und der
            # Schattenlog zeigt die EIGENEN Zuege des Bots nicht. Im Selbstspiel sind
            # beide Seiten sichtbar -> hier laesst sich messen, ob der Bot diese Engine
            # ueberhaupt baut, wenn kein Mensch ihm die Karten wegnimmt.
            "tableau":   [ (_x.get("name") if isinstance(_x, dict) else _x)
                           for _x in (_tp.get("tableau") or []) ],
            # 23.07.: Ressourcen, die am Spielende AUF Karten liegen. Karten im Tableau
            # sind nicht dasselbe wie Punkte - die Kette lautet kaufen -> spielen ->
            # SAMMELN -> Punkte. apehead endet mit 130 Ressourcen, der Bot mit 62.
            # Ohne diese Zahl sieht man nicht, ob eine gekaufte Engine ueberhaupt laeuft.
            # 25.07.: HANDKARTEN am Spielende. Ohne sie ist aus ab_games NICHT
            # entscheidbar, ob eine Zielkarte (a) nie gekauft, (b) gekauft und nie
            # gespielt oder (c) verkauft wurde - Tableau zeigt nur (b)/(c) als Fehlen.
            # ACHTUNG: Die Handkarten liegen auf der OBERSTEN Ebene des
            # PlayerViewModel (state["cardsInHand"]), NICHT unter thisPlayer - dort
            # steht nur cardsInHandNbr (Hand-Fix 18.07.). all_final[pid] IST der
            # volle State des jeweiligen Spielers, der Zugriff geht also direkt.
            "hand":      [ (_x.get("name") if isinstance(_x, dict) else _x)
                           for _x in (all_final.get(_pid, {}).get("cardsInHand") or []) ],
            "card_res":  { (_x.get("name") if isinstance(_x, dict) else _x):
                           (_x.get("resources") or 0)
                           for _x in (_tp.get("tableau") or [])
                           if isinstance(_x, dict) and (_x.get("resources") or 0) > 0 },
        }
    return {
        "vps_by_color": {c: all_vps.get(pid, 0) for c, pid in color_to_id.items()},
        # M€ am Spielende: offizieller TM-Tiebreaker bei VP-Gleichstand
        "mc_by_color": {c: all_final.get(pid, {}).get("thisPlayer", {}).get("megacredits", 0)
                        for c, pid in color_to_id.items()},
        "behav_by_color": _behav,
        "mcts_color":   mcts_color,
        "mcts_vp":      my_vp,
        "mcts_rank":    rank,
        "mcts_won":     won,
    }


def _repick_pool_key(state):
    """Kennung des aktuellen Repick-Kartensets (Farbe + sortierte Namen). Aendert sich, sobald
    der Server ein neues Paeckchen anbietet -> dann darf der Spieler wieder antworten."""
    w = state.get("waitingFor") or {}
    color = (state.get("thisPlayer") or {}).get("color")
    names = tuple(sorted(c.get("name", "") for c in (w.get("cards") or [])))
    return (color, names)


def _is_draft_repick(state):
    """True in der DRAFT-REPICK-Phase: der Spieler hat bereits gedraftet (needsToDraft=False),
    darf seine Karte aber aendern, bis alle anderen gewaehlt haben (Server-Aenderung ~07/2026,
    message 'You can change your selection until all players have selected'). Der Bot antwortet
    hier deterministisch immer dieselbe Karte -- das ist KEIN Haenger, sondern legitimes Warten
    auf die Mitspieler. Der stuck-Zaehler darf in diesem Zustand NICHT hochlaufen, sonst wird
    die Partie faelschlich abgebrochen."""
    w = state.get("waitingFor") or {}
    if w.get("type") != "card":
        return False
    tp = state.get("thisPlayer") or {}
    if tp.get("needsToDraft"):        # echter Erst-Draft -> normaler Fortschritt
        return False
    title = w.get("title")
    msg = title.get("message", "") if isinstance(title, dict) else str(title)
    return "change your selection" in msg


def _wf_signature(color, waiting):
    """Fortschritts-Signatur eines waitingFor-Zustands. WICHTIG: der Titel allein reicht
    NICHT -- im Draft-Repick ("You can change your selection...") ist der Titel ueber alle
    Anfragen identisch, waehrend sich der KARTENINHALT aendert (das ist der Fortschritt).
    Ein titel-basierter Haenger-Schutz haelt den Draft faelschlich fuer eine Endlosschleife
    und bricht die Partie ab (15.07.). Darum Karten-Namen + min/max mit aufnehmen."""
    w = waiting or {}
    cards = tuple(sorted(c.get("name", "") for c in (w.get("cards") or [])))
    return (color, w.get("type", ""), str(w.get("title", ""))[:60],
            cards, w.get("min"), w.get("max"))


def load_decide_variant(module_name: str, db_path: str | None = None):
    """Laedt decide() aus einem alternativen Heuristik-Modul (z. B. eingefrorener
    Champion 'tm_bot_champion'). Laedt dessen Karten-DB nach, damit die Variante
    in sich konsistent ist. Bricht bei Importfehler HART ab – ein stiller
    Fallback auf die Live-Heuristik wuerde unbemerkt A/A messen und
    faelschlich 'keine Verbesserung' ausweisen."""
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
        # 01.08.: HART abbrechen, wenn die Datei fehlt. load_card_db() wirft KEINE
        # Ausnahme - es schreibt nur eine Warnung und laesst CARD_DB leer ("Bot laeuft
        # ohne Kartenbewertung"). Ein Lauf mit --champion-db auf einen falschen Pfad lief
        # dadurch vier Stunden gegen einen blinden Champion und meldete +21.45 VP; der
        # Champion hatte 0 Karten-VP und praktisch keine Produktion. Dieselbe Begruendung
        # wie beim Importfehler oben: ein stiller Fallback misst Unsinn.
        if not os.path.exists(db_path):
            raise SystemExit(
                f"ABBRUCH: --champion-db '{db_path}' nicht gefunden. Der Champion liefe "
                f"ohne Kartenbewertung - das Ergebnis waere wertlos.")
        try:
            mod.load_card_db(db_path)
        except Exception as e:
            raise SystemExit(
                f"ABBRUCH: card_db '{db_path}' fuer '{module_name}' nicht ladbar ({e}).")
        n = len(getattr(mod, "CARD_DB", {}) or {})
        if n < 100:
            raise SystemExit(
                f"ABBRUCH: card_db '{db_path}' enthaelt nur {n} Karten - das kann nicht "
                f"stimmen. Der Champion waere praktisch blind.")
        log.info("Champion-Kartendatenbank: %d Karten aus %s", n, db_path)
    return mod.decide


def _summarize_crn(crn_games: list[dict], n_games: int):
    """CRN-Auswertung: gepaarte VP-Marge Challenger−Champion mit 95%-CI.
    Jedes Paar = zwei Partien auf identischem Deck (clonedGamedId) mit
    getauschten Rollen/Sitzen; deren Mittel kuerzt Deck- und Sitzeffekt heraus."""
    import math
    import statistics
    from collections import defaultdict

    log.info("══════════════════════════════════════════════════")
    if not crn_games:
        log.warning("📊 CRN: keine gewertete Partie (alle abgebrochen?)")
        log.info("══════════════════════════════════════════════════")
        return

    # ★ 20.07.: Rohdaten wegschreiben. Bisher wurden die Einzelpartien nur aggregiert
    # und dann verworfen - die VP-Marge allein sagt nichts darueber, WIE die Partien
    # zustande kamen. Konkreter Anlass: LEVER_RESOURCE_SYNERGY zielte auf die
    # KOLLAPSPARTIEN (Bot mit 0 M-Produktion ueber acht Generationen), und ob es die im
    # Bot-Duell ueberhaupt gibt, laesst sich ohne die Einzelwerte nicht pruefen.
    try:
        import json as _json, time as _time
        _ab = globals().get("_ABORT_COUNT", 0)
        if _ab:
            log.warning("   ⏱ %d Partien am Zeitdeckel (%ds) abgebrochen - "
                        "bei hoher Zahl --game-cap pruefen", _ab, GAME_TIME_CAP)
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
                # TM-Regel: VP-Gleichstand -> M€ am Spielende entscheidet
                cm = g.get("mcs", {}).get(g["chall"], 0)
                mm = g.get("mcs", {}).get(g["champ"], 0)
                if   cm > mm: chall_wins += 1
                elif mm > cm: champ_wins += 1
                else:         ties += 1
        if len(ds) == 2:                       # nur vollständige Paare werten
            pair_margins.append(sum(ds) / 2.0)

    # Verteilung der Einzelergebnisse - zeigt Ausreisser nach unten (Kollapspartien),
    # die in der gepaarten Marge unsichtbar bleiben, weil sie sich dort herauskuerzen.
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
    """Puffert Log-Records pro Worker-Thread; der Hauptthread schreibt durch.
    So bleiben die Zeilen eines Paares im Log zusammenhaengend (Auswerte-
    Skripte parsen weiterhin '── Partie N ──'-Bloecke)."""
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
    """Eine Partie heuristisch zu Ende spielen (MCTS aus). Gleiche Guards wie
    run_sequential_selfplay: Zeit-Cap, Stuck-Erkennung, Idle-Spielende."""
    all_player_ids = list(color_to_id.values())
    prev_sig, prev_save, stuck, idle = None, -1, 0, 0
    repick_done = set()   # Repick-Pools, die in dieser Partie schon beantwortet wurden
    STUCK_ABORT   = 8
    # Zeitdeckel: modulweite Konstante GAME_TIME_CAP (s.o.), per --game-cap setzbar
    game_start    = time.time()

    # ── VERHALTENS-LOGGING (20.07., fuer den Strategie-Layer) ──
    # Snapshot der Produktion je Spieler bei Gen 3/5/8. Der Loop pollt ohnehin jeden
    # Spieler, also kostet es nur ein Dict. Zweck: die STRATEGIE-KETTEN messbar machen,
    # die der VP-A/B nicht sieht - "baut der Bot die physische Terraform-Engine
    # (Pflanzen/Energie/Waerme) auf, oder versackt er in Stahl/Titan, das nur Karten
    # verbilligt?". Wird am Ende durch _finalize_selfplay in crn_games gereicht.
    _prod_snap: dict = {}          # {color: {gen: {prod-typ: wert}}}
    _SNAP_GENS = (3, 5, 8)

    while True:
        acted = False
        if time.time() - game_start > GAME_TIME_CAP:
            globals()["_ABORT_COUNT"] = globals().get("_ABORT_COUNT", 0) + 1
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
                # 31.07.: frueher get_last_save_id(). Die save_id steigt aber NUR mit
                # undoOption bei jeder Aktion - ohne Undo nur einmal je Runde, dann
                # haelt der Waechter zwei gleiche "Take your next action" faelschlich
                # fuer einen Haenger. gameAge steigt bei jedem Logeintrag (Game.ts
                # Z.1671), unabhaengig von Undo - und steht schon im Zustand, spart
                # also zusaetzlich eine Serveranfrage je Schleifendurchlauf.
                cur_save = (state.get("game") or {}).get("gameAge", prev_save)
                if _is_draft_repick(state):
                    stuck = 0                     # Repick: kein Haenger
                    _pk = _repick_pool_key(state)
                    if _pk in repick_done:
                        # Diesen Pool schon gedraftet -> NICHT erneut antworten, sonst behaelt
                        # der Server den Fokus auf diesem Spieler und der Mitspieler kommt nie
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
                  flush=None, master_seed=None, settings=None):
    """Ein vollstaendiges CRN-Paar: Partie A (frisches Deck), Partie B (Klon,
    getauschte Sitze). Liefert (results, crn_games). flush() gibt den bisher
    gepufferten Log-Block aus (wird nach jeder Partie gerufen)."""
    out_results, out_crn = [], []
    prev_id = None
    # Seed dieses Paars deterministisch aus (master_seed, pair_no) ableiten:
    # eigene RNG-Instanz -> parallel-fest (kein geteilter globaler Zustand).
    # Gleicher master_seed reproduziert Paar pair_no mit identischem Deck.
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
                cloned_game_id=(prev_id if is_o2 else None), settings=settings)
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
            flush()   # Partie-Block sofort ausgeben statt erst am Paar-Ende
    return out_results, out_crn


def run_ab_crn_parallel(base_url: str, n_players: int, all_colors: list[str],
                        draft: bool, n_pairs: int, champion_decide, workers: int,
                        master_seed=None, settings=None):
    """N CRN-Paare auf `workers` Threads. Paare sind unabhaengig (eigene
    Spiele/Decks); Logs werden pro Paar gepuffert und am Stueck ausgegeben."""
    root = logging.getLogger()
    orig_handlers = root.handlers[:]
    router = _PairLogRouter(orig_handlers)
    root.handlers = [router]
    flush_lock = _threading.Lock()
    all_results, all_crn = [], []

    def worker(pair_no: int):
        # buf ist hier noch None -> diese Zeile geht sofort an Konsole/Datei
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
                                 flush=flush, master_seed=master_seed, settings=settings)
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
    """Gibt einen kopierfertigen Befehl aus, der diesen Lauf mit identischen
    Decks wiederholt: alle Original-Argumente aus sys.argv, vorhandenes --seed
    entfernt, am Ende --seed <master_seed> angehaengt."""
    import sys
    argv = sys.argv
    parts, i = [], 1
    while i < len(argv):
        if argv[i] == "--seed":
            i += 2;  continue            # Flag + Wert ueberspringen
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
    decide_by_color:    dict | None = None,   # Farbe -> decide()-Variante
    enable_mcts_global: bool = True,          # False => reiner Heuristik-A/B
    random_first:       bool = True,          # False => feste Sitze (reproduzierbar)
    roles_by_color:     dict | None = None,   # Farbe -> Label fuer die Auswertung
    crn:                bool = False,          # gepaarter CRN-Modus via clonedGamedId
    champion_decide=None,                      # Champion-Heuristik (CRN)
    master_seed=None,                          # CRN: Decks deterministisch je Paar
    settings=None,                             # tm_settings.json der echten Runde
):
    """Ein Prozess, alle Bots sequenziell. Wurzel-Lösung gegen den Rollback-Race."""
    log.info("🎮 Sequenzieller Self-Play-Treiber | %d Spieler | %d Spiele",
             n_players, n_games)
    log.info("   MCTS-Bot: %s | übrige Farben: reine Heuristik", mcts_color)

    results: list[dict] = []   # ein Eintrag je gewerteter Partie
    crn_games: list[dict] = [] # CRN: pro Partie Marge-Rohdaten
    prev_clone_id = None       # CRN: Game-ID der ungeraden Partie zum Klonen

    for game_num in range(1, n_games + 1):
        log.info("── Partie %d/%d ──", game_num, n_games)

        # CRN: ungerade Partie = frisches Deck (Champion auf Sitz 2), gerade
        # Partie = Klon des vorigen Decks mit getauschten Rollen (Champion Sitz 1).
        # So sehen beide Orientierungen dasselbe Deck, Sitz/Startspieler ist
        # über das Paar ausbalanciert.
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
            # frische (ungerade) Partie: deterministischer Deck-Seed je Paar
            cur_seed = random.Random(f"{master_seed}:{(game_num - 1) // 2}").random()

        try:
            game_id, color_to_id = create_mp_game_with_undo(
                base_url, n_players, all_colors, draft=draft,
                random_first=cur_random_first, seed=cur_seed,
                cloned_game_id=cur_cloned_id, settings=settings)
        except Exception as e:
            log.error("Spiel-Erstellung fehlgeschlagen: %s", e)
            continue

        if crn and game_num % 2 == 1:
            prev_clone_id = game_id   # diese Quelle klont die nächste Partie

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
        transitions: list[dict] = []   # nur MCTS-Bot
        idle        = 0

        # --- Schleifen-/Fortschrittsschutz (Fix gegen Endlosschleife) ---
        # MCTS nur auf der ERSTEN or-Aktion eines Turns zulassen: nur dort sitzt
        # die Entscheidung auf einer Save-Grenze, zu der rollback_to_save sauber
        # zurueckkehrt. Auf Folge-Aktionen wuerde der Rollback die bereits
        # gespielte erste Aktion loeschen (-> HTTP 400 -> Endlosschleife).
        mcts_first_action_pending = True
        prev_sig   = None    # (Farbe, waitingFor-Typ, Titel) der letzten Aktion
        prev_save  = -1      # globale save_id bei der letzten Aktion
        stuck      = 0       # wie oft dieselbe Entscheidung ohne Fortschritt kam
        repick_done = set()  # Repick-Pools, die dieser Spieler schon beantwortet hat
        STUCK_FORCE_HEURISTIC = 3   # ab hier MCTS fuer diese Entscheidung abschalten
        STUCK_ABORT           = 8   # ab hier Partie abbrechen statt zu haengen
        # Catch-all: bricht eine Partie ab, die unrealistisch lange dauert. Faengt
        # auch Hänger, bei denen die save_id zwar fortschreitet, aber kein echtes
        # Spielende erreicht wird (greift dort, wo Signatur- und Idle-Guard nicht
        # anschlagen). Normale 2P-Partien dauern hier ~2 min.
        # Zeitdeckel: modulweite Konstante GAME_TIME_CAP (s.o.)
        game_start    = time.time()
        res           = None   # Finalize-Ergebnis dieser Partie (None = abgebrochen)

        while True:
            acted      = False
            game_over  = False
            abort_game = False

            if time.time() - game_start > GAME_TIME_CAP:
                globals()["_ABORT_COUNT"] = globals().get("_ABORT_COUNT", 0) + 1
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

                    # Fortschritt seit der letzten Aktion? Signatur + globale save_id.
                    sig = _wf_signature(color, waiting)
                    # 31.07.: gameAge statt save_id - s. Kommentar in _play_crn_game.
                    cur_save = (state.get("game") or {}).get("gameAge", prev_save)
                    if _is_draft_repick(state):
                        stuck = 0                 # Repick: kein Haenger
                        _pk = _repick_pool_key(state)
                        if _pk in repick_done:
                            # Diesen Pool bereits gedraftet -> NICHT erneut antworten, sonst
                            # behaelt der Server den Fokus auf diesem Spieler und der andere
                            # kommt nie dran (Live-Lock, 15.07.). Ueberspringen = auf Mitspieler
                            # warten. Ein NEUES Paeckchen (anderer _pk) wird wieder beantwortet.
                            continue
                        repick_done.add(_pk)
                    elif sig == prev_sig and cur_save == prev_save:
                        stuck += 1
                    else:
                        stuck = 0
                    prev_sig, prev_save = sig, cur_save

                    # Fix 1: MCTS nur auf der ersten or-Aktion des Turns.
                    mcts_allowed_now = is_mcts and mcts_first_action_pending

                    # Fix 2: Haengt dieselbe Entscheidung fest, erst Heuristik
                    # erzwingen, dann (falls auch das nicht hilft) Partie abbrechen,
                    # damit ein einzelner Input keine ganze Serie blockiert.
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
                        # MCTS-Budget des Turns ist mit der ersten or-Aktion
                        # verbraucht; ein Zug des Gegners eroeffnet einen neuen Turn.
                        if is_mcts and mcts_allowed_now and waiting.get("type") == "or":
                            mcts_first_action_pending = False
                        elif not is_mcts:
                            mcts_first_action_pending = True
                    break   # nach einem Zug neu pollen (wer ist jetzt dran?)

            if abort_game:
                break

            if game_over:
                res = _finalize_selfplay(base_url, color_to_id, mcts_color,
                                         transitions, data_file)
                break

            if not acted:
                idle += 1
                # Backup-Spielende: niemand aktiv + globale Parameter maximal
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
                # Diagnose: alle ~10s den vollen Serverzustand beider Spieler
                # ausgeben, damit sichtbar wird, worauf der Server wartet.
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

        # Ergebnis dieser Partie verbuchen (None = abgebrochen → verworfen).
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

        # Aufräumen (Lock-Datei, falls MCTS einen geschrieben hat)
        try:
            os.unlink(LOCK_FILE)
        except Exception:
            pass
        time.sleep(2.0)

    # ── Zusammenfassung ──
    if crn:
        _summarize_crn(crn_games, n_games)
    elif results:
        n_scored = len(results)
        log.info("══════════════════════════════════════════════════")
        log.info("📊 Auswertung: %d von %d Partien gewertet", n_scored, n_games)
        for color in all_colors:
            # "Sieg" = führend in dieser Partie (Gleichstand zählt für jede
            # führende Farbe – bei 2 Spielern praktisch immer eindeutig).
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
    """
    Automatischer Multi-Spiel-Modus.

    Der Bot mit der ersten Farbe (alphabetisch) erstellt das Spiel und
    schreibt die IDs. Alle anderen Bots lesen sie.
    """
    log.info("🎮 Auto-Modus | Farbe: %s | %d Spieler | %d Spiele",
             my_color, n_players, n_games)

    # Ersteller ist immer die erste Farbe in all_colors (deterministisch)
    creator_color = all_colors[0]
    is_creator    = (my_color == creator_color)
    log.info("   Ersteller: %s | Ich bin Ersteller: %s", creator_color, is_creator)

    # MCTS-Rolle auflösen: nur EIN Bot darf MCTS fahren, sonst Rollback-Race.
    enable_mcts = (mcts_role == "all") or (mcts_role == "creator" and is_creator)
    log.info("   MCTS-Rolle: %s | Dieser Bot fährt MCTS: %s", mcts_role, enable_mcts)
    if mcts_role == "all" and n_players > 1:
        log.warning("   ⚠ mcts-role=all: mehrere Bots rollen zurück – Rollback-Race möglich!")

    for game_num in range(1, n_games + 1):
        log.info("── Partie %d/%d ──", game_num, n_games)

        is_creator_this_round = is_creator
        my_player_id  = None
        all_player_ids = []

        # Alte Koordinierungsdatei löschen damit kein Spiel aus vorheriger Runde gelesen wird
        if is_creator:
            try:
                os.unlink(GAME_COORD_FILE)
            except FileNotFoundError:
                pass

        # Alle Bots signalisieren Bereitschaft
        signal_ready(my_color, game_num)

        if is_creator:
            # Ersteller wartet bis alle anderen bereit sind, dann erstellt er das Spiel
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
            # Nicht-Ersteller: warten bis Koordinierungsdatei geschrieben wird
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

        # Aufräumen nach Spiel
        if is_creator:
            try:
                os.unlink(GAME_COORD_FILE)
            except Exception:
                pass
        # Lock- und Ready-Datei bereinigen
        for f in [LOCK_FILE, READY_FILE]:
            try:
                os.unlink(f)
            except Exception:
                pass

        # Warte damit andere Bots das Spielende erkennen können
        # bevor die nächste Koordinierungsdatei geschrieben wird
        time.sleep(100.0)  # Warte bis alle Bots Spielende erkannt haben (Timeout=90s)

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
    """
    1v1: MCTS-Bot gegen einen menschlichen Spieler.

    Erstellt ein 2-Spieler-Spiel mit Undo (Voraussetzung für die Rollouts),
    gibt dem Menschen seinen Browser-Link und steuert die Bot-Farbe selbst.
    Die Zeit-Timeouts der Spielende-Erkennung sind abgeschaltet, damit der
    Mensch beliebig lange überlegen kann; das echte Spielende wird weiter über
    phase=="end" bzw. "alle inaktiv" erkannt.
    """
    colors = VALID_COLORS[:2]
    if bot_color not in colors:
        bot_color = colors[0]
    human_color = next(c for c in colors if c != bot_color)

    # Menschenpartien: zufaelliges offizielles Board (Tharsis/Hellas/Elysium ->
    # Varianz bei Meilensteinen/Awards) und Draft-Variante, wie im echten Match
    # und in der spaeteren Einsatz-Umgebung.
    # 23.07.: Das war als Absicht schon so kommentiert, der Aufruf hatte aber
    # board="tharsis" fest verdrahtet - jede Menschenpartie lief auf demselben Brett.
    # Ein tm_settings.json ueberschreibt die Wahl weiterhin (payload[k] = v unten).
    if board == "random":
        board = random.choice(OFFICIAL_BOARDS)
    game_id, color_to_id = create_mp_game_with_undo(
        base_url, 2, colors, draft=draft, board=board,
        human_color=human_color, expansions=expansions, settings=settings)
    bot_id   = color_to_id[bot_color]
    human_id = color_to_id[human_color]

    # Bestaetigung: was gesendet wurde + welches Board der Server WIRKLICH erstellt hat
    # (Board-Name steht nicht im State -> ueber die board-spezifischen Meilensteine erkennen)
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
    """
    Beitritt zu einer bestehenden (echten) Partie allein über die eigene
    Spieler-ID. Spielt rein über das Expertensystem – keine Rollouts, kein Undo,
    kein Zugriff auf fremde Spieler-IDs. Damit fair und live-tauglich: der Bot
    postet nur seine eigenen Züge und löst keine Rollback-/Benachrichtigungs-
    Effekte bei den Mitspielern aus.
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
        all_player_ids = [player_id],   # nur eigene ID → keine Fremd-Zugriffe
        n_rollouts     = n_rollouts,
        max_candidates = max_candidates,
        data_file      = data_file,
        simple_rollout = simple_rollout,
        enable_mcts    = False,         # fair: ausschließlich Expertensystem
        human_opponent = True,          # echte Mitspieler → kein Idle-Abbruch
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
                        help="Brett fuer --vs-human (Standard: random = zufaellig aus "
                             "Tharsis/Hellas/Elysium). Ein --settings-File hat Vorrang.")
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
    parser.add_argument("--post-reuse-wait", type=float, default=None,
                        help="Schutzpause (s) nach dem POST, wenn POST_REUSE aktiv ist "
                             "(Default 0.4). 0 = keine Pause, erhoeht das Risiko von "
                             "'Not waiting for anything'")
    parser.add_argument("--no-post-reuse", action="store_true",
                        help="POST-Antwort NICHT als Bestaetigung verwenden (alter Pfad "
                             "mit save_id-Nachkontrolle). Zum Gegenpruefen bei identischem Seed")
    parser.add_argument("--game-cap", type=int, default=None,
                        help="Zeitdeckel je Partie in Sekunden (Default 1800). "
                             "Reichhaltige Settings verlaengern Partien deutlich; unter "
                             "hoher --parallel-Last zusaetzlich erhoehen")
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

    # Echte Runden-Einstellungen laden (--settings tm_settings.json)
    if args.post_reuse_wait is not None:
        globals()["POST_REUSE_WAIT"] = args.post_reuse_wait

    if args.no_post_reuse:
        globals()["POST_REUSE"] = False
        print("[POST_REUSE] aus - alter Pfad mit save_id-Nachkontrolle")
    else:
        print("[POST_REUSE] aktiv - POST-Antwort ersetzt die save_id-Nachkontrolle")

    if args.game_cap:
        globals()["GAME_TIME_CAP"] = args.game_cap
        print(f"[Zeitdeckel] {args.game_cap}s je Partie")

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

    # Geschwindigkeit: POST_WAIT/POLL_INTERVAL in BEIDEN Modulen ueberschreiben
    # (tm_mcts_mp importiert die Namen by-value aus tm_bot)
    import tm_bot as _tb
    if args.post_wait is not None:
        _tb.POST_WAIT = args.post_wait
        globals()["POST_WAIT"] = args.post_wait
    if args.poll is not None:
        _tb.POLL_INTERVAL = args.poll
        globals()["POLL_INTERVAL"] = args.poll

    # Datei-Logging: Handler am Root-Logger, damit auch tm_bot/tm_mcts landen.
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
    # Champion laeuft auf eingefrorener card_db, falls angegeben -> card_db-Aenderungen
    # werden asymmetrisch (nur Challenger) und damit messbar.
    champ_db_path = args.champion_db or db_path
    if args.champion_db:
        log.info("Champion nutzt eingefrorene card_db: %s", champ_db_path)

    # Gepaartes CRN-A/B: Decks via clonedGamedId reproduziert, Rollen/Sitze
    # über das Paar ausbalanciert. --auto-games zählt Paare.
    if args.ab_crn:
        all_colors = args.colors or VALID_COLORS[:args.players]
        champ_decide = load_decide_variant(args.champion_module, champ_db_path)
        pairs   = args.auto_games if args.auto_games > 0 else 30
        n_games = pairs * 2
        # Master-Seed: aus --seed uebernehmen oder neu erzeugen. Pro Paar wird
        # daraus deterministisch ein Deck-Seed abgeleitet (parallel-fest).
        master_seed = args.seed if args.seed is not None else random.randrange(2**31)
        log.info("🧪 A/B-CRN | %d Paare (=%d Partien) | Champion-Modul=%s | "
                 "MCTS AUS | Decks gepaart via clonedGamedId | master_seed=%d",
                 pairs, n_games, args.champion_module, master_seed)
        if args.parallel > 1:
            run_ab_crn_parallel(
                base_url=args.url, n_players=args.players,
                all_colors=all_colors, draft=args.draft, n_pairs=pairs,
                champion_decide=champ_decide, workers=args.parallel,
                master_seed=master_seed, settings=_settings)
            _print_replay_command(master_seed)
            return
        run_sequential_selfplay(
            base_url       = args.url,
            n_players      = args.players,
            all_colors     = all_colors,
            mcts_color     = all_colors[0],   # nur Finalize-Label; MCTS ist aus
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
            settings           = _settings,
        )
        _print_replay_command(master_seed)
        return

    # A/B-Heuristikvergleich: Challenger (Live) vs. Champion (eingefroren),
    # MCTS global aus, feste Sitze. Reproduzierbar über --seed.
    if args.ab:
        all_colors = args.colors or VALID_COLORS[:args.players]
        if args.seed is not None:
            random.seed(args.seed)   # reproduzierbare Folge der Spiel-Seeds
        champ_color = args.champion_color or (
            all_colors[1] if len(all_colors) > 1 else all_colors[0])
        champ_decide = load_decide_variant(args.champion_module, champ_db_path)
        # Champion-Farbe nutzt die eingefrorene Variante; alle übrigen Farben
        # fallen auf die Live-Heuristik (Challenger) zurück.
        decide_by_color = {champ_color: champ_decide}
        roles_by_color  = {c: ("Champion" if c == champ_color else "Challenger")
                           for c in all_colors}
        n_games = args.auto_games if args.auto_games > 0 else 1
        log.info("🧪 A/B | Champion=%s (%s) vs. Challenger=übrige | MCTS AUS | "
                 "feste Sitze | %d Spiele%s", champ_color, args.champion_module,
                 n_games, f" | seed={args.seed}" if args.seed is not None else "",
settings       = _settings,
)
        run_sequential_selfplay(
            base_url       = args.url,
            n_players      = args.players,
            all_colors     = all_colors,
            mcts_color     = champ_color,   # nur fürs Finalize-Label; MCTS ist global aus
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

    # Sequenzieller Ein-Prozess-Modus: ein Prozess steuert alle Bots,
    # keine Farb-Registrierung / Koordinierungsdateien nötig.
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


    # 1v1 gegen Menschen: Spiel erstellen, Link ausgeben, Bot spielt automatisch.
    if args.vs_human:
        run_vs_human(
            base_url       = args.url,
            bot_color      = args.color or VALID_COLORS[0],
            draft          = args.draft or bool(_settings and _settings.get("draftVariant")),
            expansions     = {e.strip() for e in args.expansions.split(",") if e.strip()},
            settings       = _settings,
            board          = args.board,
            n_rollouts     = args.rollouts,
            max_candidates = args.candidates,
            data_file      = args.data,
            simple_rollout = args.simple_rollout,
            enable_mcts    = not args.no_mcts,
        )
        return

    # Beitritt zu echter Partie: nur eigene Player-ID, reine Heuristik, fair.
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

    # Farbe bestimmen: automatisch oder manuell
    if args.color:
        my_color = args.color
    else:
        my_color = register_color(allowed_colors=VALID_COLORS[:args.players])

    # Farben aller Spieler bestimmen
    if args.colors:
        all_colors = args.colors
    else:
        all_colors = VALID_COLORS[:args.players]

    # Sicherstellen dass my_color in all_colors ist
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
