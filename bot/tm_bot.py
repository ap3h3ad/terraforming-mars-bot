"""
Terraforming Mars Bot - Stufe 3 mit Kartendatenbank

Voraussetzung:
  pip install requests
  card_db.json im selben Verzeichnis (erzeugt von analyze_cards.py)

Verwendung:
  python tm_bot.py --player-id p562f7891afa7
  python tm_bot.py --player-id p562f7891afa7 --url http://remoteserver:8080 --poll 3
"""

import argparse
import json
import logging
import os
import random
import time
import requests

# ML-Modell (optional)
try:
    import torch
    from tm_model import game_state_features, card_features, load_model, build_input
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Parameter-Fortschritts-Feature für ML-Modell v2 (Input-Dim 61)
try:
    from train_mcts import param_progress_from_state as _param_progress_fn
except ImportError:
    def _param_progress_fn(game: dict) -> float:
        oxygen = game.get("oxygenLevel", 0)
        temp   = game.get("temperature", -30)
        oceans = game.get("oceans", 0)
        return (oxygen / 14.0 + (temp + 30) / 38.0 + oceans / 9.0) / 3.0

def param_progress_from_state(state: dict) -> float:
    return _param_progress_fn(state.get("game", {}))

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

DEFAULT_URL    = "http://localhost:9000"
POLL_INTERVAL  = 2
POST_WAIT      = 2
ERROR_WAIT     = 6
MAX_API_ERRORS = 5
MC_RESERVE     = 8
# Endgame-VP-Boden (Variante A): In der letzten Generation ist nicht ausgegebenes
# Geld bis auf den Tiebreaker verloren. 1 fester VP ist dann 3.5 wert (DB-Median
# brutto/vp ueber 96 reine VP-Karten), OHNE Kostenabzug. Nur feste vp>0.
VP_ENDGAME_VALUE = 3.5
# Skalenangleich Handkarten vs. Standardprojekte in handle_or:
# score_action() multipliziert M€-Werte mit 3.0-4.5 (Greenery-SP: 15M*3.0=45),
# _score_card_rules() liefert rohe M€-Werte. Ohne Angleich verlieren Karten
# fast immer (gemessen: 39 SP vs. 3 Kartenspiele pro Partie).
CARD_PLAY_SCALE = 3.0
# Kaufschwelle: Eine Karte wird im Ausspiel nur gewaehlt, wenn
# raw*CARD_PLAY_SCALE das schwaechste Standardprojekt (Asteroid-SP, 21)
# schlaegt -> Kauf lohnt erst ab raw > 21/3 = 7. Schwelle 0 fuehrte zu
# 62% toten Kaeufen (gemessen, Lauf 2026-06-05).
BUY_MIN_SCORE   = 0.5
# Grünflaechen-Untergrenze (Diagnose 2026-06-15): ungenutztes M€ fliesst sonst in
# SP-Grünflaechen (~23 M fuer ~2 VP, Netto ~-13 M-aequiv.). Hat der Bot M€, das er
# ohnehin so verbrennen wuerde, ist fast jede Karte der bessere M€-Einsatz. Darum
# wird die Kaufschwelle NUR bei grünflaechen-faehigem Ueberschuss auf diesen Wert
# abgesenkt (req-unspielbare Karten = -50 bleiben drausssen). Konservativ -10
# statt -13. WICHTIG: Tote Kaeufe (gekauft, nie gespielt) im A/B messen -
# Schwelle 0 *unbedingt* ergab frueher 62% tote Kaeufe; das Ueberschuss-Gate
# soll genau das verhindern.
GREENERY_SP_COST   = 23
GREENERY_BUY_FLOOR = -10.0
# ── Akquise-Lever (obs: Bot spielt ~22 Karten/Partie vs. Mensch ~36 -> Engine zu
# klein/langsam -> verliert Meilenstein-Rennen + Karten-VP). Ursache: die Research-
# Reserve koppelt den 3-M-Kauf an die SOFORT-Spielbarkeit der teuersten Karte
# (reserve = MC_RESERVE + best_cost) -> frueh/bei teuren Angeboten budget~0 -> kauft
# nichts. Der Lever entkoppelt das: nur ein kleiner Puffer + ANTEIL von best_cost.
# Dead-Buy-sicher: Qualitaetsschwelle bleibt BUY_MIN_SCORE (0.5) statt auf -10 zu
# droppen -> mehr DECENT Karten, kein Junk. Champion (Vergleich): LEVER_ACQUIRE=False.
LEVER_ACQUIRE        = False
ACQUIRE_RESERVE_FRAC = 0.0     # Anteil von best_cost in der Research-Reserve (0=voll entkoppelt)
ACQUIRE_BUY_BAR      = 0.0     # Kaufschwelle unter dem Lever (statt 0.5): breiter, aber kein Junk
# Greenery-Disziplin (obs 3): generationsskalierte Fruehbremse auf Greenery-SP. Frueh ist
# das Geld in Karten/Engine/Meilensteinen besser aufgehoben; spaet (urgency uebernimmt)
# darf der Bot ueberschuessiges Geld doch in Greeneries kippen. Plant-Greenery (gratis)
# bleibt unberuehrt. Analog zur Asteroid-SP_DISCIPLINE.
GREENERY_LATE_GEN      = 5     # ab tl<=5 keine Bremse mehr (Spielende-Ernte)
GREENERY_EARLY_PENALTY = 1.5   # Netto-M Abzug je Generation oberhalb der Schwelle
# Geldwert-Faktor fuer SP-Kosten (kombinierter Engine-Kandidat): 1.0 = volle,
# ehrliche Kosten -> SPs netto, damit Engine-Aktivierung gegen sie gewinnen kann.
SP_COST_WEIGHT  = 1.0   # SP tragen VOLLE Kosten wie Karten (fair; Faktor damit neutral).
                        # Das fruehere Horten/Meilenstein-Problem loesen wir NICHT ueber diesen
                        # Faktor, sondern ueber das Opportunitaetskosten-Signal (LEVER_IDLE)
                        # + Meilenstein-Abschluss (LEVER_MS_COMPLETE). last_gen bleibt 0.
# Anteil der Restgenerationen, in denen eine ACTIVE-Karten-Aktion realistisch
# genutzt wird. < 1.0 = Tote-Kaeufe-Waechter: der Bot aktiviert nicht jede
# Generation (Zug-Konkurrenz, fehlende Eingaberessourcen) -> abgezinst.
ACTION_ACTIVATION_RATE = 0.5

# Kuratierte Ausnahme zum pauschalen +8-"passiver-ACTIVE-Effekt"-Floor: Karten vom Typ
# ACTIVE mit action_once=0, die WEDER einen echten passiven Effekt haben NOCH eine lohnende
# Aktion (ihr Wert ist eine schwache/bedingte, in card_db nicht modellierte Aktion). Ohne
# diese Ausnahme spielt der Bot sie als toten +8-Play und aktiviert sie nie (z.B. Search
# For Life: 3 MC fuer 3 VP NUR bei 3 Wissenschafts-Ressourcen - reiner Gluecksspiel-Play).
# Generisch nicht erkennbar, weil gute passive Karten (Arctic Algae) in card_db gleich
# aussehen (action_once=0, keine Marker) - daher kuratiert. Erweiterbar.
_NO_PASSIVE_VALUE = {
    "Search For Life",
}
# Fuer FUETTERBARE Ressourcen-VP-Stapelkarten (vp_dyn.kind=="resources", z.B. Pets,
# Tardigrades, Decomposers, grosse Tier-/Wissenschaftskarten): hoehere Rate, weil der
# Bot sie zuverlaessig jede Generation stapelt (action_once schlaegt Pass). Der fuel-
# Faktor bleibt davor als Machbarkeits-Deckel -> voraussetzungsschwere/unfuetterbare
# Karten (Physics Complex ohne Energieprod) bleiben korrekt niedrig.
VP_STACK_ACTIVATION_RATE = 0.85
# Optionswert-Boden fuer spaet freischaltende, fuetterbare Ressourcen-VP-Karten mit
# reiner Globalparameter-Schranke: Mindest-Aktivierungen, weil die Karte irgendwann
# spielbar wird und schon 2-3 Trigger den gedeckelten Downside (~2 MC Verkauf) schlagen.
LATE_ENGINE_MIN_ACTIVATIONS = 2.5
# Planungs-Hebel: haelt der Bot eine hochdichte Ressourcen-VP-Engine auf der Hand,
# die NUR noch knapp hinter einem Globalparameter klemmt, lohnt es, genau diesen
# Parameter hochzutreiben (jede fruehere Freischaltung = mehr Stapel-Generationen).
PLAN_MIN_DENSITY = 1.0     # nur fette Engines (Fish/Livestock/Predators/Penguins/Birds/Whales)
PLAN_MAX_STEPS   = 4       # nur wenn die Schranke <= 4 Schritte entfernt ist (Endspurt)
PLAN_WEIGHT      = 0.6     # Bonus = action_once * PLAN_WEIGHT je gehaltener Engine
PLAN_BONUS_CAP   = 8.0     # Deckel pro Parameter
PLAN_MAX_PROD_GAP = 4      # Produktions-Treibstoff nur foerdern, wenn <= 4 Prod fehlen (Reichweite)
# Akquise fetter Spät-Engines (Spielerregel Damian): spekulativ kaufen, wenn sich >=2 Trigger
# ausrechnen lassen. Wert = erwartete Trigger * VP-pro-Trigger(=action_once) - Kosten*Gewicht.
ACQUIRE_MIN_TRIGGERS = 2.0
ACQUIRE_COST_WEIGHT  = 0.5  # Downside gedeckelt (Verkauf ~2 MC) -> Kosten nur halb gewichtet
# Meilenstein-Pursue (gap 2..PURSUE_MAX): nur Frontrunner-Rennen mit kleinem, fallendem Bonus.
PURSUE_MAX_GAP   = 3        # weiter als 3 Schritte: nicht verfolgen (zu spekulativ)
PURSUE_WEIGHT    = 0.4      # Bonus = (net/gap) * WEIGHT  -> klein, kippt nur gute Zuege
PURSUE_BONUS_CAP = 4.0

# ── Idle-Signal (Anti-Horten), vereinheitlicht ───────────────────────────────────
# Prinzip: Hat der Bot keine positive Handkarte spielbar UND Geld ueber dem Reserve-
# Puffer, laeuft das Geld leer -> die Kosten sind illusorisch. Dann werden SPs UND
# netto-negative Handkarten zu ihrem GROSS-Wert bewertet (Kostenabzug entfaellt) statt
# zu horten. Kein Magic-Bonus; die Staerke IST der reale Zugwert. Ein sweepbarer Param.
LEVER_IDLE   = True     # SP- UND Karten-Idle: gross-Wert bei Leerlauf
IDLE_RESERVE = 25.0     # so viel MC fuer kuenftige Karten behalten; darueber gilt Geld als leer
# Meilenstein-Abschluss (allein regressiv -> AUS; via _milestone_complete_bonus):
LEVER_MS_COMPLETE = False
MS_COMPLETE_BONUS = 17.0
# Meilenstein-/Award-Ausrichtung (obs 5/10): Kauf-Bias auf Karten, die ein in-Play und
# realistisch GEWINNBARES Ziel voranbringen. Klein gehalten (Tunnelblick vermeiden).
ALIGN_MAX_GAP     = 4       # Meilenstein nur als Ziel, wenn <= so viele Schritte entfernt
ALIGN_AWARD_SLACK = 2       # Award nur als Ziel, wenn Bot fuehrt oder <= so weit zurueck
ALIGN_BUY_BONUS   = 4.0     # flacher Kauf-Bonus je ausrichtungs-passender Karte (kippt, treibt nicht)
# Enabler-Gate (obs 1/8): wenige Karten, deren Wert von einem Enabler abhaengt, der NICHT
# in card_db kodiert ist. Ohne Enabler stark abwerten (Kauf UND Starthand-Keep).
ENABLER_PENALTY   = 20.0
_PARAM_STEP = {"temperature": 2, "oxygen": 1, "oceans": 1, "venus": 2}
# Deploy-Kapazitaet (Diagnose 1V1 2026-06-16): Aktionen sind knapp (~2/Gen, spaet
# fast alle an Greenery-Conversion gebunden). Liegen schon mehr unspielte Handkarten
# vor, als in der Restlaufzeit realistisch ausspielbar sind, ist jeder weitere Kauf
# totes Kapital (gemessen: Spaetkaeufe 5-6/Partie, davon 1 gespielt). Schaetzung:
# ~DEPLOY_CARDS_PER_GEN Handkarten-Plays je Restgeneration. Vorsichtig kalibriert -
# hochdrehen, falls tote Kaeufe im A/B weiter zu hoch. WICHTIG: gegen die tote-
# Kaeufe-Waechtermetrik testen, damit die Gesamt-Kaeufe nicht zurueckkippen.
DEPLOY_CARDS_PER_GEN      = 1.0   # geschaetzte ausspielbare Handkarten je Restgeneration
DEPLOY_OVERFLOW_PENALTY   = 0.6   # Abzug je Karte ueber der Kapazitaet (vorsichtig)
# Anreiz-Feld (Engine-Synergie, Increment 1): vorsichtige Tag-Nachfrage. Karten, deren
# Tags die eigene Engine fuettern (eigene vp_dyn-Tag-Karten) oder offene Tag-Bedingungen
# erfuellen, bekommen einen kleinen Bonus. Band bewusst eng - ueber die VP-pro-Gen-Kurve
# hochdrehen. TAG_DEMAND_CAP deckelt die Nachfrage je Tag. Wirkt ueber score_card auf
# Ausspiel UND Kauf. Spaeter: kuratierte Tile-/Produktions-Synergien (Lakefront-Klasse).
TAG_SYNERGY_UNIT          = 1.5
TAG_DEMAND_CAP            = 3.0
# Tile-Synergie (Increment 2): Karten, die einen nachgefragten Tile-Typ legen
# (Ozean bei Lakefront/Arctic-Algae-Engine, Stadt bei Tharsis/Pets), bekommen einen
# Bonus je gelegtem Tile. Ozean headroom-gegated (bei 9 Ozeanen wertlos).
TILE_SYNERGY_UNIT         = 1.5

# --- Feature-Flags zur sauberen A/B-Isolation einzelner Hebel ---------------
# Eine kanonische Datei; jeder Hebel einzeln schaltbar, damit der Grenzbeitrag
# isoliert messbar ist (statt Versions-Dateien zu jonglieren). Default = AN.
#   LEVER_BUY_DISCIPLINE: kapazitaetsbewusster Kauf-Abzug (DEPLOY_OVERFLOW_PENALTY)
#   LEVER_INCENTIVE_FIELD: Engine-Synergie (Tag-Nachfrage + Tile-Synergie)
# Hinweis: Die Tile-Synergie ist ZUSAETZLICH datengegated - sie wirkt nur mit einer
# card_db, die tile_reward-Felder enthaelt (neue card_db), unabhaengig vom Flag.
LEVER_BUY_DISCIPLINE      = True
LEVER_INCENTIVE_FIELD     = True
#   LEVER_ADAPTIVE_HORIZON: Restlaufzeit aus beobachtetem Parameter-Verlauf statt
#   fester 2P-Rate (spielerzahl-agnostisch; 6P-Partien sind kuerzer -> kuerzerer
#   Horizont, automatisch). AUS = fester 2P-Prior wie zuvor.
LEVER_ADAPTIVE_HORIZON    = True
#   LEVER_ENDGAME_RATE: erfasst die Endspiel-Beschleunigung (Heat-Dump->Temp,
#   Greenery-Schub->O2) ueber die JUENGSTE Parameter-Rate aus globalsPerGeneration.
#   Der Gesamt-Durchschnitt mittelt das zaehe Frueh- mit dem schnellen Spaetspiel weg
#   -> r_eff in der Schlussphase um Faktor 3-7 zu hoch (gemessen Tharsis 2026-06-26).
#   max(Durchschnitt, juengste) -> kuerzerer Horizont nur wenn die juengste Rate
#   hoeher ist; die Anlaufphase bleibt unberuehrt. AUS = reiner Durchschnitt wie zuvor.
LEVER_ENDGAME_RATE        = True
#   LEVER_PLANT_ENGINE: weiche Schwellen-Rampe fuer Pflanzen-Produktion statt
#   harter 0-Klippe. Ab Spielmitte wurde die ERSTE Pflanzen-Karte als wertlos
#   bewertet (eine +1-Karte allein erreicht 8 nicht mehr) -> Engine-Start
#   blockiert. Rampe = Teilkredit nach Naehe zur Schwelle. AUS = harte Schwelle.
LEVER_PLANT_ENGINE        = True
#   LEVER_SP_DISCIPLINE: Frueh-Abschlag fuer Asteroid-SP (reiner TR-Kauf, 14 MC/TR,
#   mieseste MC->VP-Rate). Im Fruehspiel gehoert das Geld in eine Engine; der
#   Abschlag faded zur Spielmitte, spaet uebernimmt urgency die Ernte. AUS = wie zuvor.
LEVER_SP_DISCIPLINE       = True
#   LEVER_VP_ENGINE: fuetterbare Ressourcen-VP-Stapelkarten nach projizierter
#   Akkumulation bewerten (hoehere Aktivierungsrate), fuel-gedeckelt. Hebt Kauf-/
#   Spielwert -> frueherer Erwerb + frueheres Ausspielen -> mehr Karten-VP. AUS = wie zuvor.
LEVER_VP_ENGINE           = True
#   LEVER_LATE_ENGINE: Optionswert spaet freischaltender VP-Engines (Fish/Livestock/
#   Predators/Penguins). Mindest-Aktivierungen, da Globalparameter zuverlaessig steigen
#   und 2-3 Trigger den 2-MC-Downside schlagen. fuel + Horizont-Gate bleiben. AUS = wie zuvor.
LEVER_LATE_ENGINE         = True
#   LEVER_PLAN: Planungs-Hebel (Voraussetzung einer gehaltenen Engine erzwingen). AUS,
#   weil siegraten-negativ (29->21 auf identischen Decks) + Tunnelblick: der eindimensionale
#   "maxe Ozeane fuer Penguins"-Vektor blendet bessere Alternativen aus. Stattdessen LEVER_ACQUIRE:
#   spekulativ kaufen, Parameter natuerlich steigen lassen.
LEVER_PLAN                = False
#   LEVER_ACQUIRE: fette, nur global-gesperrte Ressourcen-VP-Engines (Penguins/Birds/Fish)
#   im Kauf/Draft NICHT als unspielbar verwerfen (Globalparameter steigt natuerlich) -
#   der LATE_ENGINE-Triggerwert gegen die Kaufkosten entscheidet. Behebt die Akquise-Luecke.
#   LEVER_ACQUIRE (V1, VERWORFEN): Reserve KOMPLETT entkoppelt (Frac=0.0) UND gleichzeitig die
#   Kaufschwelle gesenkt (BUY_BAR 0.5->0.0). A/B (deckgenau, n=26): ΔVP -4.08 (17/26 schlechter).
#   Doppelt falsch: kaufte Junk (Schwelle) UND frass das Spielgeld (Reserve) -> tote Kaeufe.
LEVER_ACQUIRE             = False
#   LEVER_ACQUIRE2 (V2): Reserve-Anteil senken. GEMESSEN (11.07.): WIRKUNGSLOS - die Reserve
#   ist NICHT der Engpass (nur 1 von 9 Testfaellen aenderte sich). Darum AUS (Frac 1.0 = altes
#   Verhalten). DER ECHTE ENGPASS IST DIE KAUFSCHWELLE / DIE BEWERTUNG:
#     Score-Verteilung ueber 137 Base-Projektkarten (Gen 4, 40 M€):
#        20% REQ-gesperrt (<=-40) | 35% negativ | 3% knapp unter Schwelle
#        17% knapp drueber (0.5-5) | 26% klar gut (>5)     -> MEDIAN-SCORE: -2.5
#     Nur 42% aller Karten sind ueberhaupt kaufwuerdig; im 4-Karten-Draft kauft der Bot im
#     Schnitt nur 1.5 - bei 2-3 gespielten Karten/Gen laeuft die Hand zwangslaeufig leer
#     (gemessen: 7 -> 4 -> 1 -> 0). NAECHSTER SCHRITT: nicht die Reserve, sondern (a) die
#     harte REQ-Sperre (-50) fuer TEMPORAERE Requirements (globale Parameter erfuellen sich
#     im Spielverlauf!) und (b) die generelle Kalibrierung der Kartenbewertung pruefen.
LEVER_ACQUIRE2            = False
ACQUIRE2_RESERVE_FRAC     = 1.0   # 1.0 = altes Verhalten (volle best_cost-Reserve)
#   LEVER_MILESTONE: Meilenstein-Verhalten. Robuste Deckel-Erkennung aus game["milestones"],
#   flache Kosten (8), fenster-bewusste Dringlichkeit (qualifizierten Meilenstein sichern,
#   bevor die 3 Slots voll sind), und Pursue-Zweig (gap 2-3, Frontrunner, alignment-gated).
LEVER_MILESTONE           = True
#   LEVER_ALIGN: Kauf-Bias auf Karten, die ein in-Play & gewinnbares Ziel (Meilenstein/Award)
#   voranbringen - z.B. Legend->Events, Energizer->Energieprod, Builder->Bautags (obs 5/10).
LEVER_ALIGN               = True
#   LEVER_ENABLER: wertet Combo-Karten ohne ihren Enabler stark ab (Insulation ohne Waerme-
#   prod, Virus/Protected Habitats in 2P) - Kauf und Starthand-Keep (obs 1/8).
LEVER_ENABLER             = True
#   LEVER_GREENERY_DISCIPLINE: generationsskalierte Fruehbremse auf Greenery-SP gegen die
#   Greenery-Flut (obs 3) - frueh Karten/Engine/Meilensteine bevorzugen, spaet Ernte zulassen.
LEVER_GREENERY_DISCIPLINE = True

# ---------------------------------------------------------------------------
# Kartendatenbank laden
# ---------------------------------------------------------------------------

CARD_DB: dict = {}

def load_card_db(path: str = "card_db.json"):
    global CARD_DB
    if not os.path.exists(path):
        log.warning("card_db.json nicht gefunden – Bot läuft ohne Kartenbewertung")
        return
    with open(path, encoding="utf-8") as f:
        CARD_DB = json.load(f)
    log.info("Kartendatenbank geladen: %d Karten", len(CARD_DB))


ML_MODEL = None
ML_DEVICE = "cpu"


def load_ml_model(path: str = "tm_model.pt"):
    global ML_MODEL, ML_DEVICE
    if not TORCH_AVAILABLE:
        log.info("PyTorch nicht verfügbar – regelbasierte Bewertung")
        return
    if not os.path.exists(path):
        log.info("Kein ML-Modell gefunden – regelbasierte Bewertung")
        return
    ML_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ML_MODEL = load_model(path, ML_DEVICE)
    ML_MODEL.eval()
    log.info("ML-Modell geladen (device: %s)", ML_DEVICE)


def card_info(name: str) -> dict:
    """Gibt Karten-Features zurück, oder leeres Dict wenn unbekannt."""
    return CARD_DB.get(name, {})


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tm_bot")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def get_state(base_url: str, player_id: str) -> dict:
    r = requests.get(f"{base_url}/api/player", params={"id": player_id}, timeout=10)
    r.raise_for_status()
    return r.json()


def post_input(base_url: str, player_id: str, payload: dict) -> dict:
    r = requests.post(
        f"{base_url}/player/input",
        params={"id": player_id},
        json=payload,
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Spielzustand-Helfer
# ---------------------------------------------------------------------------

def get_plant_cost(state: dict) -> int:
    corp = state.get("pickedCorporationCard", [])
    if any(c.get("name") == "Ecoline" for c in corp):
        return 6
    return 8


def can_convert_plants(state: dict) -> bool:
    return state["thisPlayer"].get("plants", 0) >= get_plant_cost(state)


def can_convert_heat(state: dict) -> bool:
    player = state["thisPlayer"]
    temp   = state["game"].get("temperature", -30)
    return player.get("heat", 0) >= 8 and temp < 8


def thermalist_hold_value(state: dict) -> float:
    """Vorsprung des aktiven Spielers beim Award 'Thermalist' (meiste Hitze = 5 VP)
    -- ABER nur, wenn der Award ueberhaupt noch GEWINNBAR ist. Sonst ist die Hitze
    fuer den Award wertlos und darf gewandelt werden.

    Gewinnbar = Thermalist im Spiel UND (bereits gefundet ODER noch fundbar, d.h.
    weniger als 3 Awards gefundet -- max. 3 pro Spiel). Funded-Status steht im State:
    jeder Award traegt 'color'/'playerName' des Funders (FundedAwardModel), sonst
    weggelassen. 'color' gesetzt <=> gefundet.

    Rueckgabe: eigene - beste gegnerische Hitze-Wertung; >0 = fuehrt einen
    gewinnbaren Thermalist. 0, wenn nicht im Spiel / nicht mehr gewinnbar / nicht fuehrt."""
    game = state.get("game", {}) or {}
    awards = game.get("awards", []) or []
    therm = next((a for a in awards if a.get("name") == "Thermalist"), None)
    if therm is None:
        return 0.0                                   # nicht im Spiel
    funded = bool(therm.get("color") or therm.get("playerName"))
    if not funded:
        funded_count = sum(1 for a in awards if a.get("color") or a.get("playerName"))
        if funded_count >= 3:
            return 0.0                               # 3 Awards weg -> Thermalist tot
    me = (state.get("thisPlayer", {}) or {}).get("color")
    mine, opp = 0, 0
    for s in therm.get("scores", []) or []:
        if s.get("color") == me:
            mine = s.get("score", 0)
        else:
            opp = max(opp, s.get("score", 0))
    return mine - opp


def game_options(state: dict) -> dict:
    """Die Spieloptionen der Partie (Server: GameModel.gameOptions). Der Bot las sie bisher
    GAR NICHT - dabei aendern einige davon die REGELN:
      requiresVenusTrackCompletion  Mandatory Venus: das Spiel endet NICHT, bevor Venus 30 ist
                                    -> laengere Partie + Venus-Terraforming ist Pflicht
      solarPhaseOption              World Government: globale Parameter steigen jede Generation
                                    automatisch -> Requirements erfuellen sich schneller
      escapeVelocity / soloTR / twoCorpsVariant / fastModeOption / undoOption ...
    Board-Boni und Meilensteine/Awards liest der Bot ohnehin dynamisch vom Server, dafuer
    braucht er die Optionen nicht."""
    return (state.get("game", {}) or {}).get("gameOptions", {}) or {}


def turns_left(state: dict) -> int:
    game = state["game"]
    tl = game.get("lastSoloGeneration", 14) - game.get("generation", 1)
    # Mandatory Venus: das Spiel endet erst, wenn AUCH Venus voll ist -> die Partie dauert
    # laenger als die reine Generationen-Schaetzung. Konservativ um die fehlenden Venus-Schritte
    # verlaengern (1 Schritt ~ 1 Generation), damit spaete Engines/Requirements nicht
    # faelschlich als unerreichbar abgewertet werden.
    if game_options(state).get("requiresVenusTrackCompletion"):
        venus_missing = max(0, 30 - game.get("venusScaleLevel", 0)) // 2
        tl = max(tl, venus_missing)
    return tl


# ---------------------------------------------------------------------------
# Kartenbewertung
# ---------------------------------------------------------------------------

def score_card_ml(card: dict, state: dict) -> float:
    """Kartenbewertung via ML-Modell."""
    name = card.get("name", "")
    cost = card.get("calculatedCost", 0)
    info = card_info(name)
    s_feats = game_state_features(state)
    c_feats = card_features(info, calc_cost=cost)
    inp = build_input(s_feats, c_feats).unsqueeze(0).to(ML_DEVICE)
    with torch.no_grad():
        return ML_MODEL(inp).item()


# --- Reasoning-Log (ENV TMBOT_RLOG): Diagnose, WARUM (nicht) Engines. Kein Verhalten. ---
_RLOG = os.environ.get("TMBOT_RLOG")
_rlog_bd: dict = {}   # letzte Engine-Wert-Komponenten je Kartenname (aus score_card)
_DUMP_WF = os.environ.get("TM_DUMP_WF")   # Diagnose: vollen waitingFor dumpen (verhaltensneutral)

# Aus den TS-Kartendefinitionen extrahiert: Karten, deren Aktion REIN "+Ressource auf sich
# selbst" ist (kostenlos, kein Steal, keine Bedingung). Solche Aktivierungen sind IMMER
# besser als Pass -> duerfen nicht am Pass-Floor (score_action("pass")=4) scheitern.
# Betroffen waren nur die mit action_once<4 (Tardigrades 1.25, Small Animals 2.5, ...);
# Ants (Steal), Security Fleet/Water Import (Kosten), Regolith Eaters (Modus) sind bewusst
# NICHT enthalten und behalten den Pass-Floor.
_FREE_ACCUM = frozenset({
    "bio-sol", "birds", "celestic", "cloud tourism", "dirigibles", "extremophiles", "fish",
    "floater technology", "livestock", "luna archives", "lunar observation post",
    "main belt asteroids", "martian culture", "maxwell base", "penguins", "pollinators",
    "pride of the earth arkship", "psychrophiles", "small animals", "solarpedia",
    "stormcraft incorporated", "stratopolis", "stratospheric birds", "symbiotic fungus",
    "tardigrades", "venera base", "venusian insects", "vermin",
})


def _rlog_write(rec: dict) -> None:
    if not _RLOG:
        return
    try:
        with open(_RLOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _idle_engine_log(state: dict, options: list, player: dict) -> None:
    """Loggt ACTIVE-Engines, die der Bot BESITZT, deren Aktion aber NICHT angeboten wird
    (= gerade nicht aktivierbar) — plus Ressourcen/Parameter-Kontext. Damit laesst sich
    'warum nicht aktiviert' beantworten: Physics Complex nicht angeboten + Energie<6 =
    Ressourcenmangel; Hitze-Engine nicht angeboten + temp am Max = Parameter erschoepft.
    Verhaltensneutral (nur bei TMBOT_RLOG)."""
    if not _RLOG:
        return
    try:
        game = state.get("game", {}) or {}
        played = player.get("tableau") or player.get("playedCards") or []
        offered = set()
        for opt in options:
            _t = str(opt.get("title", "")).lower()
            _ot = opt.get("type")
            if _ot == "option" and _is_card_action(_t):
                offered.add(_t.strip())
            # Buendel-Option "Perform an action from a played card": die aktivierbaren
            # Karten stehen in opt["cards"] (NICHT als eigener Option-Titel). Ohne das
            # erfasste der Idle-Log den Live-Aktivierungspfad gar nicht -> feuerte nie.
            # Erkennung woertlich vom funktionierenden Handler (Z.~3145) uebernommen.
            elif _ot == "card" and ("perform an action" in _t or "played card" in _t
                                    or opt.get("selectBlueCardAction")):
                for c in opt.get("cards", []) or []:
                    nm = c.get("name") if isinstance(c, dict) else c
                    if nm:
                        offered.add(str(nm).strip().lower())
        idle = []
        for c in played:
            name = c.get("name") if isinstance(c, dict) else c
            if not name:
                continue
            ao = (_action_card_info(name) or {}).get("action_once")
            if ao and ao > 0 and str(name).strip().lower() not in offered:
                idle.append(name)
        if idle:
            _rlog_write({"phase": "idle_engine", "gen": game.get("generation"),
                         "engines": idle,
                         "energy": player.get("energy", 0), "titanium": player.get("titanium", 0),
                         "heat": player.get("heat", 0), "plants": player.get("plants", 0),
                         "steel": player.get("steel", 0), "mc": player.get("megacredits", 0),
                         "temp": game.get("temperature"), "oxygen": game.get("oxygenLevel"),
                         "oceans": game.get("oceans"),
                         "avail_actions": player.get("availableBlueCardActionCount")})
    except Exception:
        pass


def score_card(card: dict, state: dict) -> float:
    """
    Hybrid: max(ML-Score, regelbasierter Score).
    Verhindert dass ML alle Handkarten auf negativ drückt.
    """
    rules_score = _score_card_rules(card, state)
    if ML_MODEL is not None:
        try:
            ml_raw = score_card_ml(card, state)
            # ML gibt normalisierte Werte (~-3 bis +3), skaliere auf regelbasierte Skala
            ml_score = ml_raw * 8.0
            return max(ml_score, rules_score)
        except Exception as e:
            log.debug("ML-Score Fehler: %s", e)
    return rules_score


PROD_CAP = 6.0   # gedeckelter Produktions-Ernte-Horizont (Grenzertrag, ~BGG-Horizont)
# Hitze/Energie sind NICHT baukarten-limitiert wie Stahl/Titan (Ueberschuss-Energie
# wird automatisch zu Hitze, Hitze zu Temperatur). Daher eigener, hoeherer Horizont
# statt PROD_CAP=6: der realistische Anteil am ~19-stufigen Temperatur-Track in 2P.
# Kalibrierung (nicht self-play-neutral!) -> bei Bedarf justieren.
HEAT_GENS_CAP = 9.0


# Spielstart-Werte der globalen Parameter (Spielregel, NICHT spielerzahl-abhaengig).
_PARAM_START = {"temperature": -30, "oxygen": 0, "oceans": 0, "venus": 0}
_PARAM_FIELD = {"temperature": "temperature", "oxygen": "oxygenLevel",
                "oceans": "oceans", "venus": "venusScaleLevel"}


def _param_rate(game: dict, param: str) -> float:
    """Steigerungsrate eines globalen Parameters pro Generation - SELBST-KALIBRIEREND.
    Leitet die Rate aus dem BEOBACHTETEN Verlauf der laufenden Partie ab
    (Gesamtfortschritt seit Spielstart / verstrichene Generationen) und mischt sie mit
    dem 2P-Prior, bis genug beobachtet ist. Dadurch spielerzahl-agnostisch: in 6P
    steigen die Parameter schneller -> hoehere Rate -> kuerzerer Horizont, automatisch.
    Frueh (wenig Daten) dominiert der Prior, ab ~Gen 5 die Beobachtung.
    LEVER_ADAPTIVE_HORIZON=False -> fester 2P-Prior wie zuvor (fuer sauberen A/B)."""
    prior = _PARAM_RATE.get(param, 1.0)
    if not LEVER_ADAPTIVE_HORIZON:
        return prior
    elapsed = max(0, game.get("generation", 1) - 1)
    if elapsed <= 0:
        return prior
    steps = game.get(_PARAM_FIELD.get(param, param), _PARAM_START.get(param, 0)) \
            - _PARAM_START.get(param, 0)
    if steps <= 0:
        return prior
    obs = steps / elapsed
    w = min(1.0, elapsed / 4.0)          # Vertrauen in die Beobachtung waechst mit der Zeit
    blended = (1.0 - w) * prior + w * obs
    # Endspiel-Beschleunigung: die juengste Rate (letzte ~2 Gen aus der serverseitigen
    # globalsPerGeneration-Historie) erfasst Heat-Dump/Greenery-Schub, den der Gesamt-
    # Durchschnitt wegmittelt. max() -> kuerzerer Horizont NUR wenn juengst > Schnitt;
    # in der Anlaufphase ist juengst <= Schnitt, dort bleibt alles wie zuvor.
    if LEVER_ENDGAME_RATE:
        gpg = game.get("globalsPerGeneration") or []
        if len(gpg) >= 3:
            win = min(2, len(gpg) - 1)
            try:
                recent = (gpg[-1].get(param, 0) - gpg[-1 - win].get(param, 0)) / win
            except (AttributeError, TypeError):
                recent = 0.0
            if recent > blended:
                return recent
    return blended


def _remaining_gens(game: dict) -> tuple[float, float]:
    """Prognostiziert Restgenerationen aus dem Parameter-Stand statt fixer 14.
    Das Spiel endet, wenn der LANGSAMSTE Parameter sein Maximum erreicht -> max(...).
    Rate via _param_rate (selbst-kalibrierend, spielerzahl-agnostisch).
    Rueckgabe (R_eff, Gen_bis_Temp_max)."""
    temp   = game.get("temperature", -30)
    oxygen = game.get("oxygenLevel", 0)
    oceans = game.get("oceans", 0)
    gtt  = max(0.0, (8  - temp)   / _param_rate(game, "temperature"))
    gto2 = max(0.0, (14 - oxygen) / _param_rate(game, "oxygen"))
    gtoc = max(0.0, (9  - oceans) / _param_rate(game, "oceans"))
    r_eff = min(16.0, max(1.0, max(gtt, gto2, gtoc)))
    return r_eff, gtt


def _gens_to_global_req(info: dict, game: dict) -> float:
    """Prognostizierte Generationen bis die globalen Voraussetzungen einer Karte
    erfuellt sind (0 = keine/erfuellt). Geschlossenes max-Fenster -> faktisch nie."""
    cur = {"temperature": game.get("temperature", -30),
           "oxygen":      game.get("oxygenLevel", 0),
           "oceans":      game.get("oceans", 0),
           "venus":       game.get("venusScaleLevel", 0)}
    delay = 0.0
    for rg in (info.get("req_global") or []):
        p, v = rg.get("param"), rg.get("value", 0)
        if p not in cur:
            continue
        if rg.get("max"):
            if cur[p] > v:
                return 99.0
            continue
        dist = v - cur[p]
        if dist > 0:
            delay = max(delay, dist / _param_rate(game, p))
    return delay


def _contextual_prod_value(prod: dict, player: dict, oxygen: int,
                           r_eff: float, gtt: float, plant_threshold: int,
                           temp: int = -30) -> float:
    """Horizont- und senkenbewusster Wert der Produktions-DELTAS einer Karte.
    Ersetzt die statische BGG-Bewertung: jede Ressource zaehlt nur, solange sie
    bis zum prognostizierten Spielende real genutzt werden kann.
      M€/Stahl/Titan: nutzbar bis vorletzte Gen (letzte Gen nur Tie-Breaker).
      Waerme: bis Temperatur maximal. Energie: dito, minus 1 Gen (->Waerme-Verzug).
      Pflanzen: -> Greenery, inkl. der finalen Produktionsrunde nach dem Passen (+1);
                0, wenn die Schwelle (8 / Ecoline 7) bis dahin nicht erreichbar ist.
    Stahl-/Titanwert kommt aus dem State (Advanced Alloys etc.: bis 4 / 5).
    PROD_CAP deckelt den Ernte-Horizont: man kann nicht 14 Generationen Produktion
    effizient verbrauchen (es gehen die passenden Karten/Senken aus) -> Grenzertrag.
    Kappe NUR fuer Stahl/Titan (ausgabe-limitiert: man braucht Baukarten). M€ ist
    universelle Waehrung (immer ausgebbar) und Pflanzen werden zu Greenery (direkte VP)
    -> beide UNgekappt ueber den vollen Horizont. Waerme/Energie sind senken-limitiert
    (Temperaturleiste) und bleiben ueber gtt + Kappe begrenzt."""
    eff = min(r_eff, PROD_CAP)
    full_gens   = max(0.0, r_eff - 1.0)          # M€: voller Horizont (immer ausgebbar)
    capped_gens = max(0.0, eff - 1.0)            # Stahl/Titan: gekappt (Baukarten-limitiert)
    # Hitze/Energie: gtt (Generationen bis Temp-Max) kommt aus der BEOBACHTETEN Rate,
    # die den Gegner enthaelt. Rennt ein starker Gegner die Temperatur hoch, kollabiert
    # gtt -> Hitze wuerde wertlos bewertet, der Bot gaebe den Temperatur-Track auf
    # (selbst-verstaerkend). Die verbleibenden Stufen sind aber ein UMKAEMPFTER Pool,
    # um den der Bot rennt. Boden = Reststufen (8-temp)/2. WICHTIG: NICHT mit PROD_CAP
    # (Stahl/Titan-Baukarten-Limit) deckeln -- Energie/Hitze sind nicht baukarten-
    # limitiert -> eigener Deckel HEAT_GENS_CAP, sonst werden clean Energie-Karten
    # (Geothermal etc.) unter ihren BGG-Wert gedrueckt und nie gekauft.
    steps_left  = max(0.0, (8 - temp) / 2.0)
    heat_gens   = min(r_eff, max(gtt, steps_left), HEAT_GENS_CAP)
    # Energie = Waerme: Der 1-Gen-Verzug (Energie wird erst naechste Produktionsphase
    # zu Waerme) wird durch die OPTIONALITAET aufgewogen -- Energie kann Waerme werden
    # ODER teure Karten speisen (Physics Complex, Iron Works ...). Netto gleichwertig
    # -> kein Verzugsabzug. (Kein Bonus DARUEBER hinaus: der Karteneingangs-Wert ist
    # bedingt und teils schon ueber die Energieprod-Voraussetzung jener Karten erfasst.)
    energy_gens = heat_gens
    steel_v = player.get("steelValue", 2) or 2
    titan_v = player.get("titaniumValue", 3) or 3

    v  = prod.get("megacredits", 0) * 1.0     * full_gens
    v += prod.get("steel", 0)       * steel_v * capped_gens
    v += prod.get("titanium", 0)    * titan_v * capped_gens
    v += prod.get("heat", 0)        * 1.25    * heat_gens      # 1/8 TR * 10M
    v += prod.get("energy", 0)      * 1.25    * energy_gens

    dplant = prod.get("plants", 0)
    if dplant:
        cur_plants = player.get("plants", 0)
        plant_prod_total = (player.get("production", {}) or {}).get("plants", 0) + dplant
        reach = r_eff + 1.0                        # echter Horizont + finale Runde
        projected = cur_plants + plant_prod_total * reach
        greenery = 5.0 + (10.0 if oxygen < 14 else 0.0)   # 1 VP + ggf. 1 O2-TR
        if projected >= plant_threshold:
            v += dplant * (greenery / 8.0) * reach            # voll konvertierbar
        elif LEVER_PLANT_ENGINE:
            # Weiche Rampe statt harter 0: Teilkredit nach Naehe zur Schwelle.
            # Haelt die ERSTE Pflanzen-Karte ab Spielmitte kaufwuerdig (Engine-
            # Start). Faktor <= 1 -> bereits gueltige Faelle bleiben unveraendert.
            factor = max(0.0, projected / plant_threshold)
            v += dplant * (greenery / 8.0) * reach * factor
        # sonst (Flag aus): Pflanzenproduktion bleibt wertlos (0)
    return v


def _score_card_rules(card: dict, state: dict) -> float:
    """
    Berechnet den situativen Wert einer Karte.

    Kombiniert:
    - Basis-Score aus card_db (Produktion, TR, VP)
    - Situative Anpassungen (Generationen verbleibend, Parameter-Stand)
    - Korporation-Synergien
    - Anforderungsprüfung (Karte spielbar?)
    """
    name = card.get("name", "")
    cost = card.get("calculatedCost", 0)
    info = card_info(name)

    if not info:
        # Karte nicht in DB: neutraler Score basierend auf Kosten
        return -cost * 0.3

    game   = state["game"]
    player = state["thisPlayer"]
    tl     = turns_left(state)
    oxygen = game.get("oxygenLevel", 0)
    temp   = game.get("temperature", -30)
    oceans = game.get("oceans", 0)
    mc_prod = player.get("megacreditProduction", 0)

    # Basis-Score aus DB: bereits voll cost-bereinigt (Breakdown: -cost 1:1,
    # BGG-Einheitswerte fuer Produktion/TR/VP). KEIN weiterer Kostenabzug noetig.
    score = float(info.get("score_total", 0))

    # Rabatte gutschreiben: score_total rechnet mit dem DB-Druckpreis,
    # calculatedCost enthaelt Spieler-Rabatte (1 M pro M, BGG-konform)
    score += info.get("cost", cost) - cost

    # --- Situative Anpassungen (DB-Schema: production/global/oceans/tr/type) ---
    prod = info.get("production", {}) or {}
    glob = info.get("global", {}) or {}

    # Kontextuelle Produktionsbewertung: score_total enthaelt Produktion statisch zu
    # BGG-Werten (MC=5, Steel=8, Titan=10, Plant=10, Energy=7, Heat=6). Diesen
    # statischen Anteil durch einen horizont- und senkenbewussten Wert ersetzen
    # (_contextual_prod_value): jede Ressource zaehlt nur, solange sie bis zum
    # prognostizierten Spielende real genutzt werden kann. So sind idle/spaete
    # Produktion ~0 wert und das Opfern ungenutzter Produktion (z.B. Strip Mine)
    # wird korrekt guenstig.
    r_eff, gtt = _remaining_gens(game)
    _corp = state.get("pickedCorporationCard", []) or []
    plant_threshold = 7 if any(c.get("name") == "Ecoline" for c in _corp) else 8
    static_prod_bgg = (
        prod.get("megacredits", 0) * 5 +
        prod.get("steel", 0)       * 8 +
        prod.get("titanium", 0)    * 10 +
        prod.get("plants", 0)      * 10 +
        prod.get("energy", 0)      * 7 +
        prod.get("heat", 0)        * 6
    )
    new_prod = _contextual_prod_value(prod, player, oxygen, r_eff, gtt, plant_threshold, temp)
    score += new_prod - static_prod_bgg

    # Aktions-/Effektkarten (type ACTIVE): wiederholbare Aktion ueber die
    # Restlaufzeit bewerten statt pauschal +8. action_once = Netto-M pro
    # Aktivierung (BGG, aus card_db). Konservative Aktivierungsrate als
    # Tote-Kaeufe-Waechter (s. ACTION_ACTIVATION_RATE).
    if (info.get("type") or "").upper() == "ACTIVE":
        action_once = float(info.get("action_once", 0) or 0)
        if action_once > 0:
            # Aktivierungen ueber das Nach-Freischaltungs-Fenster: voraussetzungs-
            # gesperrte Engines (z.B. Penguins: 8 Ozeane) zaehlen nur Generationen
            # NACH erfuellter Bedingung. Frueh erfuellt + andere Parameter niedrig
            # -> grosses R_eff -> viele Aktivierungen -> Engine wird stark.
            gens_to_unlock = _gens_to_global_req(info, game)
            # Fuetterbare Ressourcen-VP-Stapelkarten aktiviert der Bot zuverlaessig
            # jede Generation (ihre Aktion schlaegt Pass) -> hoehere Rate, damit ihr
            # Engine-Potenzial beim Kauf nicht halbiert wird. fuel (unten) deckelt
            # die Machbarkeit; voraussetzungsschwere Karten bleiben darueber niedrig.
            rate = ACTION_ACTIVATION_RATE
            if LEVER_VP_ENGINE and (info.get("vp_dyn") or {}).get("kind") == "resources":
                rate = VP_STACK_ACTIVATION_RATE
            activations = max(0.0, r_eff - gens_to_unlock) * rate
            # Optionswert spaet freischaltender VP-Engines (Hinweis Damian): Eine
            # hochdichte Ressourcen-VP-Karte mit reiner Globalparameter-Schranke
            # (Fish +2C, Livestock/Predators O2, Penguins Ozeane) wird IRGENDWANN
            # spielbar - Globalparameter steigen im echten Spiel zuverlaessig, auch
            # durch den Gegner. Selbst 2-3 spaete Trigger lohnen gegen den gedeckelten
            # Downside (Verkauf ~2 MC). Daher Mindest-Aktivierungen als Optionswert,
            # sofern die Karte im Horizont (+Puffer fuer die pessimistische Solo-Rate)
            # ueberhaupt freischaltet. fuel deckelt weiter die Fuetterbarkeit;
            # tag-/produktionsgebundene Karten bekommen den Boden NICHT (nicht sicher
            # erreichbar). gens_to_unlock==99 = geschlossenes max-Fenster -> kein Boden.
            if (LEVER_LATE_ENGINE
                    and (info.get("vp_dyn") or {}).get("kind") == "resources"
                    and gens_to_unlock <= r_eff + 2.0):
                only_global = (info.get("req_global")
                               and not info.get("req_tags")
                               and not info.get("req_prod"))
                if only_global or not (info.get("requirements") or []):
                    activations = max(activations,
                                      min(LATE_ENGINE_MIN_ACTIVATIONS, r_eff))
            # Tote-Kaeufe-Waechter II: Engine, die der Bot nicht befeuern kann
            # (z.B. Ironworks spend energy 4 ohne Energieproduktion), liefert ihre
            # Aktion nicht -> Aktionswert anteilig abwerten. OR-Aktionen ({}) sind
            # flexibel und gelten als befeuerbar.
            fuel = 1.0
            prodp = player.get("production", {}) or {}
            for res, amt in (info.get("action_input") or {}).items():
                if not amt or amt <= 0:
                    continue
                supply = prodp.get(res, player.get(res + "Production", 0))
                if res != "energy":   # Stahl/Titan/Pflanzen: Lager anteilig dazu
                    supply += player.get(res, 0) / max(1.0, tl)
                fuel = min(fuel, supply / amt)
            fuel = max(0.0, min(1.0, fuel))
            # Schwellen-Realismus fuer Ressourcen-VP-Engines mit per>1 (Tardigrades
            # per=4): Ressourcen UNTERHALB der naechsten per-Stufe sind 0 VP wert.
            # Linear (action_once*activations) ueberbewertet den gestrandeten Rest
            # -> Bot kauft/spielt Tardigrades und strandet sie (0 VP in den Daten).
            # Stattdessen GEFLOORTE realisierte VP: floor(proj_res/per)*each.
            # Dichte Engines (Pets per=2, Security Fleet per=1) bleiben ~unberuehrt.
            vd  = info.get("vp_dyn") or {}
            per = int(vd.get("per", 1) or 1)
            if vd.get("kind") == "resources" and per > 1:
                each  = float(vd.get("each", 1) or 1)
                vp_mc = action_once * per / each       # MC pro VP (Tardigrades: 5)
                proj  = activations * fuel              # ~1 Ressource je Aktivierung
                realized_vp = (int(proj) // per) * each # gefloort: proj<per -> 0
                score += realized_vp * vp_mc
            else:
                score += action_once * activations * fuel
            if _RLOG and (info.get("vp_dyn") or {}).get("kind") == "resources":
                _rlog_bd[name] = {"action_once": round(action_once, 2),
                                  "gens_to_unlock": round(gens_to_unlock, 2),
                                  "r_eff": round(r_eff, 2),
                                  "activations": round(activations, 2),
                                  "fuel": round(fuel, 2)}
        else:
            # Kuratierte tote Plays (action_once=0, kein echter passiver Effekt) bekommen
            # den +8-Floor NICHT - sonst spielt der Bot sie und aktiviert sie nie.
            if name not in _NO_PASSIVE_VALUE:
                score += 8   # passiver ACTIVE-Effekt ohne bezifferte Aktion

    # Direkte TR-Erhoehungen (info["tr"]) stecken mit 10M bereits korrekt in
    # score_total -> kein Re-Add. Nur: Parameter weniger wert wenn schon maximal.
    if oxygen >= 14:
        score -= glob.get("oxygen", 0) * 12
    if temp >= 8:
        score -= glob.get("temperature", 0) * 12
    if oceans >= 9:
        score -= info.get("oceans", 0) * 12

    # Negative M€-Produktion: Malus abhängig von verbleibenden Generationen
    if prod.get("megacredits", 0) < 0:
        malus = abs(prod["megacredits"]) * tl * 1.5
        score -= malus
        # Extra-Malus wenn Produktion bereits negativ
        if mc_prod < 0:
            score -= 20

    # Pflanzenkarten besonders wertvoll für Ecoline
    corp = state.get("pickedCorporationCard", [])
    if any(c.get("name") == "Ecoline" for c in corp):
        score += prod.get("plants", 0) * 5

    # Pflanzenproduktion weniger wert wenn Oxygen schon maximal
    if oxygen >= 14:
        score -= prod.get("plants", 0) * 3

    # (Award-Fortschritts-Zuschlag wieder entfernt: Screening 2026-06-07
    #  zeigte -0.25 Claims/Partie durch verdraengte Meilenstein-Budgets,
    #  Marge -1.25. Helper _award_progress_bonus bleibt fuer einen spaeteren,
    #  gezaehmteren Anlauf erhalten, wird aber nicht aufgerufen.)

    # VP-Strafe direkt aus API-Karten-Objekt (überschreibt card_db wenn vorhanden)
    # Manche Karten haben neg VP: Nuclear Zone (-2), Hackers (-1), etc.
    api_vp = card.get('victoryPoint', card.get('vp', None))
    if api_vp is not None and api_vp < 0:
        db_vp = info.get('vp', 0)
        if abs(api_vp) > abs(db_vp):   # API-Wert ist schlimmer → nutze API
            extra_penalty = (abs(api_vp) - abs(db_vp)) * 5
            score -= extra_penalty

    # (kein zusätzlicher Kostenabzug: score_total ist bereits netto;
    #  das frühere 'score -= cost*0.4' rechnete Kosten zu 140% an)

    # Adjazenz-VP-Tiles: Commercial District (1 VP je angrenzende Stadt) und Capital
    # (1 VP je angrenzendem Ozean) haben vp=None/0 in card_db -> hier grob schaetzen.
    # Die Platzierung maximiert die Adjazenz (s. choose_best_space), darum ~erwartbare
    # Adjazenz aus dem aktuellen Board, diskontiert. PLATZHALTER bis zum vollstaendigen
    # Extraktions-Durchgang (echte Adjazenz-Regel + platzierungs-bewusste Schaetzung).
    _nl = card.get("name", "").lower()
    if "commercial district" in _nl or _nl == "capital":
        _spaces = game.get("spaces", []) or []
        _want = TILE_CITY if "commercial" in _nl else TILE_OCEAN
        _n = sum(1 for s in _spaces if s.get("tileType") == _want)
        score += min(_n, 3) * 5.0 * 0.6   # ~1-2 erreichbare Nachbarn, 1 VP = 5M

    # Dynamische VP (generisch, vp_dyn): Karten mit vp=0 in der DB, die pro Tag/
    # Stadt/Ressource skalieren (z.B. Io Mining Industries = 1 VP je Jovian-Tag).
    # KAUFSEITIG ist die Endzahl unbekannt -> prinzipielle Untergrenze: aktuelle
    # Zaehlung + die Tags, die DIESE Karte selbst mitbringt ("VP, wenn das Spiel
    # nach diesem Zug endet"). Unterschaetzt kuenftiges Wachstum bewusst, statt
    # mit einem willkuerlichen Faktor zu raten. 1 VP = 5 M.
    rule = info.get("vp_dyn")
    if rule:
        kind = rule.get("kind")
        count = 0
        if kind == "tag":
            tg = rule["tag"].upper()
            count = _tag_count(player, tg) + sum(1 for t in info.get("tags", []) if t.upper() == tg)
        elif kind == "cities":
            _spaces = game.get("spaces", []) or []
            count = sum(1 for s in _spaces if s.get("tileType") == TILE_CITY)
        # resources: Akkumulation ist beim Kauf 0 -> via action_once bereits bewertet
        if count:
            per = rule.get("per", 1) or 1
            score += (count // per) * rule.get("each", 1) * 5.0

    # Feeder-Synergie (kontextuell): Karten mit 'synergy_adds' kippen Ressourcen auf eine
    # ANDERE Karte (z.B. Large Convoy: +4 Animal). Wert nur, wenn eine kompatible vp_dyn-
    # Ressourcen-Engine besessen (Tableau) bzw. im Kauf/Keep-Kontext gehalten wird (Hand via
    # waitingFor-Optionen). Ohne Engine -> 0. 1 VP = 5 M.
    # WICHTIG: Die Daten liegen in 'synergy_adds' ({type:'Animal', count:N}), NICHT in 'feeds'
    # (das ist ueberall leer) - Code frueher an das falsche Feld gekoppelt -> Synergie war tot.
    synergy = info.get("synergy_adds")
    if synergy:
        eng_names = set()
        for c in (player.get("tableau") or []):
            n = c.get("name") if isinstance(c, dict) else c
            if n:
                eng_names.add(n)
        if state.get("_feed_include_hand"):
            for c in (player.get("cardsInHand") or []):
                n = c.get("name") if isinstance(c, dict) else c
                if n:
                    eng_names.add(n)
            # waitingFor-Optionen = Kaufkandidaten (zaehlen) bzw. Draft-Paeckchen (NICHT
            # zaehlen: davon bleibt nur 1 Karte -> Paket-Nachbar ist keine gehaltene Engine).
            if not state.get("_draft_ctx"):
                for opt in (state.get("waitingFor", {}) or {}).get("options", []) or []:
                    for c in (opt.get("cards") or []):
                        n = c.get("name") if isinstance(c, dict) else c
                        if n:
                            eng_names.add(n)
        # synergy_adds[].type ist kapitalisiert ("Animal"/"Microbe"/"Any"/...). Match auf eine
        # gehaltene vp_dyn-Ressourcen-Engine ueber den passenden Tag (ANIMAL/MICROBE); "Any"
        # (CEO) matcht jede vp_dyn-Engine. Floater/Data/Fighter: kein Tag-Match -> kein Bonus.
        _RES_TAG = {"animal": "ANIMAL", "microbe": "MICROBE", "any": None}

        def _best_vp_per_res(res: str) -> float:
            r = (res or "").lower()
            if r not in _RES_TAG:
                return 0.0
            tag = _RES_TAG[r]
            best = 0.0
            for nm in eng_names:
                ci = card_info(nm) or {}
                vd = ci.get("vp_dyn") or {}
                if vd.get("kind") != "resources":
                    continue
                if tag is not None and tag not in (ci.get("tags") or []):
                    continue
                best = max(best, (vd.get("each", 1) or 1) / (vd.get("per", 1) or 1))
            return best

        _vals = [a.get("count", 0) * _best_vp_per_res(a.get("type", "")) for a in synergy]
        if _vals:
            # synergy_mode unterscheidet OR-Karten (max: nur eine Option gewaehlt, z.B.
            # Imported Hydrogen) von AND-Karten (sum: beide Adds, z.B. Imported Nitrogen).
            mode = info.get("synergy_mode", "max")
            score += (sum(_vals) if mode == "sum" else max(_vals)) * 5.0

    # Engine-Synergie (Anreiz-Feld): Karten, deren Tags die eigene Engine fuettern
    # oder offene Tag-Bedingungen erfuellen, und Karten, die einen nachgefragten
    # Tile-Typ legen (Ozean fuer Lakefront/Arctic-Algae-Engine), werden hoeher
    # bewertet. Vorsichtig gedeckelt; wirkt ueber score_card auf Ausspiel UND Kauf.
    demand = _strategy_demand(state) if LEVER_INCENTIVE_FIELD else None
    if demand:
        syn = 0.0
        for t in info.get("tags", []):
            units = demand.get(str(t).lower(), 0.0)
            if units > 0:
                syn += min(units, TAG_DEMAND_CAP) * TAG_SYNERGY_UNIT
        # Tile-Platzierung erfuellt Tile-Nachfrage (on-play oceans/city/greenery
        # ODER Aktions-Platzierer wie Aquifer Pumping). Ozean headroom-gegated.
        n_ocean = (info.get("oceans", 0) or 0)
        if "oceans" in (info.get("action_places") or []):
            n_ocean = max(n_ocean, 1)
        if n_ocean and oceans < 9 and demand.get("oceans", 0) > 0:
            syn += min(demand["oceans"], TAG_DEMAND_CAP) * n_ocean * TILE_SYNERGY_UNIT
        if (info.get("city", 0) or 0) > 0 and demand.get("cities", 0) > 0:
            syn += min(demand["cities"], TAG_DEMAND_CAP) * TILE_SYNERGY_UNIT
        if (info.get("greenery", 0) or 0) > 0 and demand.get("greenery", 0) > 0:
            syn += min(demand["greenery"], TAG_DEMAND_CAP) * TILE_SYNERGY_UNIT
        score += syn

    # Planungs-Hebel: belohnt Karten, die einen Globalparameter ODER die Produktion
    # voranbringen, die eine GEHALTENE hochdichte VP-Engine zum Freischalten/Fuettern
    # braucht (Greenery/global->O2 fuer Predators; Energieprod fuer Physics Complex;
    # Venus-Karten fuer Venusian Animals). Ergaenzt den SP-Hook in score_action und das
    # Tag-Anreizfeld - zusammen decken sie alle Bedingungstypen ab.
    if LEVER_PLAN:
        score += _planning_card_bonus(info, state)

    # Endgame-VP-Boden (Variante A): In der ALLERLETZTEN Generation ist nicht
    # ausgegebenes Geld (bis auf den seltenen Tiebreaker) verloren. Eine leistbare
    # Karte mit FESTEN Siegpunkten ist dann ihren VP-Wert wert - der Kostenabzug in
    # score_total unterstellt faelschlich, das Geld haette Alternativnutzen. Nur
    # feste vp>0 (kein vp_dyn-Schaetzwert, kein Parameter-Effekt), damit vp=0-Karten
    # wie Towing A Comet/Deimos Down NICHT faelschlich aufgewertet werden. Boden,
    # kein Deckel -> bereits hoehere Bewertungen bleiben unveraendert. Signal:
    # _is_last_generation (parameter-getrieben, konsistent mit Reserve/Kosten-Logik).
    if _is_last_generation(state):
        _vp = card.get("victoryPoint", card.get("vp", info.get("vp", 0))) or 0
        if _vp > 0 and cost <= player.get("megacredits", 0):
            score = max(score, _vp * VP_ENDGAME_VALUE)

    return score



# Geschätzter Parameter-Fortschritt pro Generation (2P, aus Logs: ~14 Gen,
# Temp -30..+8, O2 0..14, Ozeane 0..9)
_PARAM_RATE = {"temperature": 2.7, "oxygen": 1.0, "oceans": 0.65, "venus": 1.0}
REQ_UNREACHABLE = -50.0


def _tag_count(player: dict, tag: str) -> int:
    tags = player.get("tags", [])
    if isinstance(tags, dict):
        return tags.get(tag, 0)
    return next((t.get("count", 0) for t in tags if t.get("tag") == tag), 0)


def _strategy_demand(state: dict) -> dict:
    """Engine-Nachfrage je Dimension (Tags kleingeschrieben + Tile-Dimensionen
    'oceans'/'cities'/'greenery') aus SAUBEREN Quellen:
    (1) eigene vp_dyn-Tag-Karten (Io Mining = VP je Jovian-Tag -> Jovian gefragt),
    (2) offene Tag-Voraussetzungen auf Hand/Tableau (Karte braucht 2 Science),
    (3) eigene tile_reward-Karten (Lakefront/Arctic Algae belohnen pro Ozean ->
        Ozean-Platzierung gefragt; Tharsis/Pets -> Staedte; Herbivores -> Greenery).
    Memoisiert auf dem State (pro Zug fuer alle Kandidaten gleich)."""
    cached = state.get("_strat_demand")
    if cached is not None:
        return cached
    player = state.get("thisPlayer", {})
    demand: dict = {}
    owned = list(player.get("tableau") or player.get("playedCards") or []) + list(player.get("cardsInHand") or [])
    for c in owned:
        nm = c.get("name") if isinstance(c, dict) else c
        info = card_info(nm or "")
        if not info:
            continue
        vd = info.get("vp_dyn") or {}
        if vd.get("kind") == "tag" and vd.get("tag"):
            demand[vd["tag"]] = demand.get(vd["tag"], 0.0) + 1.0
        for r in (info.get("requirements") or []):
            if r.get("type") == "tag" and not r.get("max") and r.get("value"):
                if _tag_count(player, r["value"]) < r.get("count", 1):
                    demand[r["value"]] = demand.get(r["value"], 0.0) + 0.5
        for dim in (info.get("tile_reward") or []):
            demand[dim] = demand.get(dim, 0.0) + 1.0
    state["_strat_demand"] = demand
    return demand


def _requirement_penalty(info: dict, state: dict, tl: int, skip_global: bool = False) -> float:
    """
    Kauf-Malus fuer unerfuellte Voraussetzungen (Quelle: Server-Repo).
    - Tags/Produktion unerfuellt: faktisch nicht kaufen (Bot baut kaum Tags auf)
    - Globale Parameter: Generationen bis zur Freischaltung schaetzen; nie
      erreichbar -> nicht kaufen, sonst Produktionswert-Verfall bis dahin anrechnen
    skip_global=True (Starthand-Keep): globale Parameter steigen im Spielverlauf
    ohnehin -> NICHT bestrafen. Tag-/Produktions-Req bleiben aber gegated, weil der
    Bot diese ggf. NIE erfuellt (sonst tote Keeps wie Beam=Jovian ohne Jovian-Plan).
    """
    player = state.get("thisPlayer", {})
    game   = state.get("game", {})
    pen    = 0.0

    for tag, need in (info.get("req_tags") or {}).items():
        if _tag_count(player, tag) < need:
            return REQ_UNREACHABLE

    prod_own = player.get("production", {}) or {}
    for res in (info.get("req_prod") or []):
        if prod_own.get(res, 0) <= 0:
            return REQ_UNREACHABLE

    # Implizite Produktions-Voraussetzung: eine Karte, die Produktion SENKT, verlangt genug
    # Produktion, um die Senkung zu absorbieren (so prueft es die echte Engine).
    # WICHTIG - TEMPORAERES Gate, kein permanentes: Der Bot BAUT Produktion im Spielverlauf
    # auf. Eine harte Sperre (REQ_UNREACHABLE) verhindert den KAUF fuer immer -> 60 Karten
    # (Ocean City, Electro Catapult, Cupola City, Business Network ...) wurden nie gekauft
    # -> Hand lief leer (gemessen 7->0). Der Server bietet unspielbare Karten ohnehin nicht
    # zum Ausspielen an, also genuegt hier ein Malus, der mit dem fehlenden Abstand waechst.
    # (Gleicher Denkfehler wie zuvor bei den Turmoil-party-Requirements.)
    for res, delta in (info.get("production", {}) or {}).items():
        if delta < 0:
            floor = -5 if res == "megacredits" else 0
            missing = floor - (prod_own.get(res, 0) + delta)   # >0 => noch nicht spielbar
            if missing > 0:
                pen -= 4.0 * missing        # je weiter weg, desto unattraktiver

    # Turmoil-Voraussetzungen (party/chairman/partyLeader). Diese stehen NUR in
    # 'requirements' (req_tags/req_global/req_prod sind dafuer leer).
    # WICHTIG: Die regierende Partei WECHSELT JEDE GENERATION -> das ist ein TEMPORAERES
    # Gate (wie globale Parameter), KEIN dauerhaftes (wie fehlende Tags). REQ_UNREACHABLE
    # (-50) waere falsch: es vergiftete 33 Karten dauerhaft, auch beim KAUFEN -> der Bot
    # kaufte/spielte sie nie mehr (beobachtet: nur 4 gespielte Karten bis Spielmitte).
    # Darum: nur ein milder Malus, wenn die Voraussetzung GERADE nicht erfuellt ist -
    # die Karte bleibt kaufbar und wird gespielt, sobald die Partei an die Macht kommt.
    # Der Server bietet unspielbare Karten ohnehin nicht zum Ausspielen an.
    _turm = (game.get("turmoil") or {})
    if _turm:
        for r in (info.get("requirements") or []):
            rt = r.get("type")
            if rt == "party":
                _ruling = _turm.get("ruling")
                if isinstance(_ruling, dict):
                    _ruling = _ruling.get("name", "")
                if str(_ruling or "").lower().replace(" ", "") != \
                   str(r.get("value", "")).lower().replace(" ", ""):
                    pen -= 6.0          # wartet auf den Regierungswechsel
            elif rt == "chairman":
                if _turm.get("chairman") != player.get("color"):
                    pen -= 6.0
            elif rt == "partyLeader":
                _leads = any(p.get("partyLeader") == player.get("color")
                             for p in (_turm.get("parties") or []))
                if not _leads:
                    pen -= 6.0

    if skip_global:
        return pen

    cur = {"temperature": game.get("temperature", -30),
           "oxygen":      game.get("oxygenLevel", 0),
           "oceans":      game.get("oceans", 0),
           "venus":       game.get("venusScaleLevel", 0)}
    delay = 0.0
    for rg in (info.get("req_global") or []):
        p, v = rg.get("param"), rg.get("value", 0)
        if p not in cur:
            continue
        if rg.get("max"):
            if cur[p] > v:
                return REQ_UNREACHABLE   # Fenster bereits geschlossen
            continue
        dist = v - cur[p]
        if dist > 0:
            delay = max(delay, dist / _PARAM_RATE.get(p, 1.0))
    if delay > 0:
        if delay >= tl - 1:
            return REQ_UNREACHABLE       # wird nie rechtzeitig spielbar
        # Produktionswert bei Freischaltung statt heute ansetzen
        prod = info.get("production", {}) or {}
        static_prod = (prod.get("megacredits", 0) * 5 + prod.get("steel", 0) * 8 +
                       prod.get("titanium", 0) * 10 + prod.get("plants", 0) * 10 +
                       prod.get("energy", 0) * 7 + prod.get("heat", 0) * 6)
        f_now    = max(0.3, tl / 7.0)
        f_unlock = max(0.3, (tl - delay) / 7.0)
        pen += static_prod * (f_unlock - f_now)   # <= 0
    return pen


def score_card_to_buy(card: dict, state: dict, for_initial_keep: bool = False) -> float:
    """
    Score für Kartenkauf.
    Berücksichtigt ob die Karte in verbleibenden Generationen spielbar ist.

    for_initial_keep=True (Starthand): die Karte ist eine Spieloption ueber die
    GESAMTE Restlaufzeit zum Preis von 3 M. Bezahlbarkeit-jetzt und (spaeter
    erreichbare) unerfuellte Voraussetzungen sind dann irrelevant -> beide Abzuege
    werden uebersprungen, damit req-gegatete Zukunfts-Plays (Birds, Great Dam) und
    Effekt-Karten nicht faelschlich verworfen werden.
    """
    state["_feed_include_hand"] = True     # Kauf/Keep: gehaltene Engines zaehlen fuer Feeder
    base = score_card(card, state)
    state.pop("_feed_include_hand", None)
    tl = turns_left(state)
    player = state["thisPlayer"]
    mc = player.get("megacredits", 0)
    mc_prod = player.get("production", {}).get("megacredits", 0)
    cost = card.get("calculatedCost", 0)

    # Voraussetzungen: beim Keep nur globale Parameter ueberspringen (steigen ohnehin),
    # Tag-/Produktions-Gate bleibt aktiv -> nie erfuellbare Keeps (Beam=Jovian,
    # Magnetic Field Generators=Energieprod.) werden korrekt verworfen.
    # LEVER_ACQUIRE: dieselbe Nachsicht auch beim KAUFEN/Draften fuer NUR global-gesperrte
    # Ressourcen-VP-Engines (Penguins/Birds/Fish). Der Globalparameter steigt natuerlich -
    # die Karte wird irgendwann spielbar; der LATE_ENGINE-Triggerwert gegen die Kaufkosten
    # entscheidet (Spielerregel: "kaufen, wenn >=2 Trigger drin"). Ohne das verwirft die
    # UNREACHABLE-Strafe (delay>=tl-1) genau diese Karten fruehzeitig. Tag-/Prod-gesperrte
    # Engines (Pride=Tags, Physics=Energie) bleiben gegated - die sind nicht "natuerlich".
    info = CARD_DB.get(card.get("name", ""), {})
    _is_glob_engine = ((info.get("vp_dyn") or {}).get("kind") == "resources"
                       and info.get("req_global")
                       and not info.get("req_tags") and not info.get("req_prod"))
    skip_g = for_initial_keep or (LEVER_ACQUIRE and _is_glob_engine)
    req_pen = _requirement_penalty(info, state, tl, skip_global=skip_g)
    if req_pen <= REQ_UNREACHABLE:   # harte Sperre: nie spielbar -> nicht kaufen/keepen
        return REQ_UNREACHABLE
    base += req_pen

    # LEVER_ACQUIRE: explizite Akquise-Bewertung fuer fette, nur global-gesperrte Ressourcen-
    # Engines (Penguins/Birds/Fish/Predators). Spielerregel: spekulativ kaufen, wenn sich
    # >=2 Trigger ausrechnen lassen. Wert = erwartete Trigger (bei NATUERLICHEM Parameter-
    # verlauf) * VP-pro-Trigger - Kosten*Gewicht. Greift nur als Untergrenze (max), damit die
    # Karte nicht durch die undurchsichtige Tiefenbewertung unter ihren Akquise-Wert faellt.
    if LEVER_ACQUIRE and _is_glob_engine and not for_initial_keep:
        r_eff, _ = _remaining_gens(state.get("game", {}))
        gtu      = _gens_to_global_req(info, state.get("game", {}))
        exp_trig = max(0.0, r_eff - gtu)
        if exp_trig >= ACQUIRE_MIN_TRIGGERS:    # nur wenn sich >=2 Trigger ausrechnen
            ao  = float(info.get("action_once", 0) or 0)
            acq = exp_trig * ao - cost * ACQUIRE_COST_WEIGHT
            base = max(base, acq)

    if not for_initial_keep:
        # Bezahlbarkeit nur im normalen Kauf: Starthand-Keep ist ein 3-M-Zukunfts-Play.
        mc_in_2_gens = mc + mc_prod * 2
        if mc_in_2_gens < cost:
            base -= (cost - mc_in_2_gens) * 0.5

    # Kaufgebühr abziehen (auch beim Keep: 3 M)
    base -= 1.5

    # Spät im Spiel: stärkerer Abzug für teure Karten die nie gespielt werden
    if not for_initial_keep and tl <= 3 and cost > mc:
        base -= (cost - mc) * 0.8

    # Deploy-Kapazitaet: liegen bereits mehr unspielte Handkarten vor, als in der
    # Restlaufzeit realistisch ausspielbar sind (~DEPLOY_CARDS_PER_GEN je Gen), ist
    # ein weiterer Kauf totes Kapital. Beisst durch das kleine tl automatisch spaet
    # staerker, ohne frueh global die Kaeufe zuzudrehen.
    if LEVER_BUY_DISCIPLINE and not for_initial_keep:
        hand = player.get("cardsInHandNbr", 0)
        capacity = tl * DEPLOY_CARDS_PER_GEN
        overflow = hand - capacity
        if overflow > 0:
            base -= overflow * DEPLOY_OVERFLOW_PENALTY

    # Enabler-Gate (obs 1/8): Combo-Karten ohne ihren Enabler stark abwerten - auch beim
    # Starthand-Keep, denn der Bot soll Insulation/Virus/Protected Habitats gar nicht erst
    # halten, wenn der Enabler fehlt.
    if LEVER_ENABLER:
        nm = card.get("name", "")
        if nm in _ENABLER_CARDS and not _enabler_ok(nm, state):
            base -= ENABLER_PENALTY

    # Meilenstein-/Award-Ausrichtung (obs 5/10): kleiner Bias auf Karten, die ein in-Play &
    # gewinnbares Ziel voranbringen (Legend->Events, Energizer->Energieprod, ...).
    base += _alignment_buy_bonus(card, state)

    return base


# ---------------------------------------------------------------------------
# Heuristiken
# ---------------------------------------------------------------------------

# Korporationspräferenz basierend auf Winrate-Statistiken (BGG-Thread, 2-5P gewichtet)
# Wert = gewichteter Vorteil gegenüber Erwartungs-Winrate (1/N)
# Spielerzahl-adaptiv: Saturn+Tharsis konstant stark; Ecoline gut ab 4P
CORP_PRIORITY = {
    "Saturn Systems":              +0.116,  # Stärkste Korporation über alle Spielerzahlen
    "Tharsis Republic":            +0.109,  # Konsistent stark, tile-basiert
    "Ecoline":                     +0.092,  # Besonders gut 4-5P, schwächer 2P
    "Credicor":                    +0.020,  # Gut 2-3P, schwächer 4-5P
    "Mining Guild":                +0.018,
    "Interplanetary Cinematics":   -0.003,
    "Teractor":                    -0.014,
    "Thorgate":                    -0.029,
    "Phobolog":                    -0.032,
    "Helion":                      -0.056,  # Stark 2P, schwach 3-4P
    "United Nations Mars Initiative": -0.107,
    "Inventrix":                   -0.121,  # Schwächste Korporation
}

def _estimate_corp_value(corp: dict) -> float:
    """
    Schätzt den Wert einer unbekannten Korporation anhand von Startkapital
    und Startproduktionen nach BGG-Guide Ressourcenwerten (Sektion 6).

    BGG: Empfehlung ab 60M Anfangswert (Startkapital + Produktionen + Ressourcen).
    Rückgabe: normierter Wert relativ zu 60M Schwelle → [−0.15, +0.15]
    """
    mc       = corp.get("startingMegaCredits", corp.get("megaCredits", 42))
    steel_p  = corp.get("startingSteel", 0) * 8       # 1 Stahl-Prod = 8M
    titan_p  = corp.get("startingTitanium", 0) * 10
    plant_p  = corp.get("startingPlants", 0) * 10
    energy_p = corp.get("startingEnergy", 0) * 7
    heat_p   = corp.get("startingHeat", 0) * 6
    cards    = corp.get("startingCards", 0) * 3       # 1 Karte ≈ 3M (card-rich)

    total = mc + steel_p + titan_p + plant_p + energy_p + heat_p + cards
    # Normierung: 60M = 0.0, 70M = +0.10, 50M = -0.10
    return (total - 60) / 100.0


def choose_corporation(options: list[dict], num_players: int = 2) -> str:
    """
    Wählt die stärkste verfügbare Korporation.

    Bekannte Korporationen: Winrate-Statistiken (BGG-Thread, 2-5P gewichtet).
    Unbekannte Korporationen: Startkapital-Schätzung nach BGG-Guide Sektion 6.
    Spielerzahl-adaptiv: Ecoline besser bei 4-5P, Credicor/Helion besser bei 2P.
    """
    adjustments = {}
    if num_players >= 4:
        adjustments["Ecoline"]  = +0.05
        adjustments["Credicor"] = -0.08
        adjustments["Helion"]   = -0.05
    elif num_players == 2:
        adjustments["Credicor"] = +0.06
        adjustments["Helion"]   = +0.06

    best_name  = None
    best_score = -999.0

    for corp in options:
        name = corp.get("name", "")
        # Case-insensitive Lookup (Server-Namen können abweichen: "ThorGate" vs "Thorgate")
        corp_key = next((k for k in CORP_PRIORITY if k.lower() == name.lower()), None)
        if corp_key:
            base = CORP_PRIORITY[corp_key]
        else:
            # Unbekannte Korporation: regelbasierte Schätzung
            base = _estimate_corp_value(corp)
            log.info("🏢 Unbekannte Korporation '%s' – Schätzwert: %.3f", name, base)

        score = base + adjustments.get(name, 0.0)
        if score > best_score:
            best_score = score
            best_name  = name

    chosen = best_name or options[0].get("name", "")
    log.info("🏢 Korporation gewählt: %s (score=%.3f, %dP)", chosen, best_score, num_players)
    return chosen


def choose_preludes(options: list[dict], count: int, state: dict | None = None) -> list[str]:
    """Waehlt die 'count' wertvollsten Preludes. Bevorzugt score_card (live, kontext-
    abhaengig - Synergie/vp_dyn/Feeder wie beim aktiven Spiel-Pfad); ohne state Fallback
    auf score_total aus CARD_DB. Ersetzt die fruehere hartcodierte Prioritaetsliste."""
    if state is not None:
        def _val(c) -> float:
            try:
                return score_card(c, state)
            except Exception:
                return (CARD_DB.get(c.get("name", ""), {}) or {}).get("score_total", 0.0) or 0.0
    else:
        def _val(c) -> float:
            return (CARD_DB.get(c.get("name", ""), {}) or {}).get("score_total", 0.0) or 0.0
    ranked = sorted(options, key=_val, reverse=True)
    return [c["name"] for c in ranked[:count]]


def choose_cards_to_buy(cards: list[dict], state: dict, card_cost: int = 3) -> list[str]:
    """
    Kaufe Karten basierend auf situativem Score.
    BGG-Guide: nur kaufen was man auch bald spielen kann.
    """
    player  = state["thisPlayer"]
    mc      = player.get("megacredits", 0)
    tl      = turns_left(state)

    # Spätes Spiel: weniger Karten kaufen
    if tl <= 2:
        return []  # Gen 13-14: keine neuen Karten mehr

    # Schätze beste Kartenkosten aus den angebotenen Karten
    best_cost = 0
    for c in cards:
        sc = score_card_to_buy(c, state)
        if sc > BUY_MIN_SCORE:
            best_cost = max(best_cost, c.get("calculatedCost", 0))

    # Research-Reserve. FRUEHERER BUG: reserve = MC_RESERVE + best_cost, wobei best_cost die
    # teuerste ANGEBOTENE Karte war - auch eine, die der Bot gar nicht kaufen will. Eine teure
    # Karte im Paeckchen (z.B. Soletta 35) trieb die Reserve so hoch, dass das Budget 0 wurde
    # und der Bot GAR NICHTS kaufte - obwohl 3-M€-Kaeufe moeglich und lohnend waren (gemessen:
    # 4 von 10 Draft-Angeboten -> 0 Kaeufe; Hand lief leer 7->0).
    # RICHTIG: Erst die lohnenden Karten bestimmen, dann Spielgeld fuer DIESE reservieren.
    # Konservativ: die BILLIGSTE lohnende Karte muss spielbar bleiben (nicht die teuerste
    # angebotene) - so bleibt Geld zum Ausspielen, ohne den Kauf zu blockieren.
    scored = [(score_card_to_buy(c, state), c["name"], c.get("calculatedCost", 0))
              for c in cards]
    scored.sort(reverse=True)

    # Qualitaetsschwelle: immer BUY_MIN_SCORE. Das frueher hier verwendete Grünflaechen-Floor
    # (-10 bei Ueberschuss) stammte aus der Zeit, als die best_cost-Reserve den Kauf blockierte
    # und M€ ungenutzt in SP-Grünflaechen verbrannte. Mit der korrigierten Reserve kauft der Bot
    # ohnehin regelmaessig; das Floor liess dann nur noch JUNK durch (gemessen: 30-43% der
    # Kaeufe unter der Schwelle). Junk-Kaeufe waren schon einmal die Ursache fuer 62% tote
    # Kaeufe -> Floor entfernt, Schwelle bleibt hart.
    buy_bar = BUY_MIN_SCORE
    worth = [(sc, nm, cost) for sc, nm, cost in scored if sc > buy_bar]
    if not worth:
        return []

    # Spielgeld-Reserve fuer die BILLIGSTE lohnende Karte (nicht fuer die teuerste ANGEBOTENE -
    # das war der Bug: eine teure Karte im Paeckchen blockierte den ganzen Kauf).
    play_reserve = min(cost for _, _, cost in worth)
    reserve = MC_RESERVE + play_reserve
    budget  = max(0, mc - reserve)
    max_buy = min(budget // card_cost, 3)

    if max_buy == 0:
        return []

    chosen = [nm for _sc, nm, _c in worth[:max_buy]]

    if _RLOG:
        _gen = (state.get("game") or {}).get("generation")
        _chosen = set(chosen)
        for _sc, _nm, _c in scored:
            _bd = _rlog_bd.get(_nm)
            if _bd is not None:                       # nur Ressourcen-VP-Engines
                _rlog_write({"phase": "buy", "gen": _gen, "name": _nm,
                             "score": round(_sc, 2), "bought": _nm in _chosen,
                             "buy_bar": round(buy_bar, 2), **_bd})

    if chosen:
        log.info("  📦 Kaufe: %s", ", ".join(chosen))
    return chosen


def _choose_cards_to_buy_old(cards: list[dict], state: dict) -> list[str]:
    """(unbenutzt, Referenz - alte Logik mit dem best_cost-Reserve-Bug)"""
    player    = state["thisPlayer"]
    mc        = player.get("megacredits", 0)
    card_cost = 3
    best_cost = 0
    for c in cards:
        if score_card_to_buy(c, state) > BUY_MIN_SCORE:
            best_cost = max(best_cost, c.get("calculatedCost", 0))

    reserve = MC_RESERVE + best_cost
    budget  = max(0, mc - reserve)
    max_buy = min(budget // card_cost, 3)

    if max_buy == 0:
        return []

    buy_bar = GREENERY_BUY_FLOOR if budget >= GREENERY_SP_COST else BUY_MIN_SCORE

    scored = [(score_card_to_buy(c, state), c["name"]) for c in cards]
    scored.sort(reverse=True)

    chosen = []
    for sc, name in scored:
        if len(chosen) >= max_buy:
            break
        if sc > buy_bar:
            chosen.append(name)

    if _RLOG:
        _gen = (state.get("game") or {}).get("generation")
        _chosen = set(chosen)
        for _sc, _nm in scored:
            _bd = _rlog_bd.get(_nm)
            if _bd is not None:                       # nur Ressourcen-VP-Engines
                _rlog_write({"phase": "buy", "gen": _gen, "name": _nm,
                             "score": round(_sc, 2), "bought": _nm in _chosen,
                             "buy_bar": round(buy_bar, 2), **_bd})

    if chosen:
        log.info("  📦 Kaufe: %s", ", ".join(chosen))
    else:
        log.info("  📦 Kein Kartenkauf sinnvoll")

    return chosen


def choose_card_to_play(cards: list[dict], state: dict) -> dict | None:
    """Wähle beste spielbare Karte nach situativem Score."""
    if not cards:
        return None

    player = state["thisPlayer"]
    mc     = player.get("megacredits", 0)

    # Nur Karten die wir uns leisten können
    affordable = [c for c in cards if c.get("calculatedCost", 999) <= mc]
    if not affordable:
        return None

    # Bevorzuge mit Reserve, Fallback ohne
    with_reserve = [c for c in affordable if c.get("calculatedCost", 999) <= mc - MC_RESERVE]
    pool = with_reserve if with_reserve else affordable

    scored = [(score_card(c, state), c) for c in pool]
    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        return None

    best_score, best_card = scored[0]

    # Spiele Karte nur wenn Score positiv
    if best_score <= 0:
        return None

    return best_card


def get_playable_cards(cards: list[dict], state: dict, max_cards: int = 5) -> list[tuple[float, dict]]:
    """
    Gibt bis zu max_cards spielbare Karten mit ihren Scores zurück.
    Für MCTS-Kandidaten-Auswahl: mehr Optionen = bessere Entscheidungen.
    """
    player = state.get("thisPlayer", {})
    mc     = player.get("megacredits", 0)

    affordable = [c for c in cards if c.get("calculatedCost", 0) <= mc]
    if not affordable:
        return []

    with_reserve = [c for c in affordable if c.get("calculatedCost", 999) <= mc - MC_RESERVE]
    pool = with_reserve if with_reserve else affordable

    scored = [(score_card(c, state), c) for c in pool]
    scored.sort(key=lambda x: x[0], reverse=True)

    # Nur positive Scores (spielenswerte Karten)
    playable = [(sc, c) for sc, c in scored if sc > 0]
    return playable[:max_cards]


def build_payment(card: dict, player: dict | None = None) -> dict:
    """
    Berechnet optimale Bezahlung für eine Karte.
    Nutzt Stahl (2 MC/Stück) für Building-Karten und
    Titan (3 MC/Stück) für Space-Karten (NUR SPACE-Tag - Titan zahlt in TM nicht
    fuer Jovian ohne Space; Cloud Tourism = JOVIAN+VENUS wurde sonst mit Titan
    bezahlt -> Server 400).

    player: state["thisPlayer"] – wenn angegeben, werden Ressourcen optimiert.
    """
    cost = card.get("calculatedCost", 0)
    steel_used    = 0
    titanium_used = 0

    if player and cost > 0:
        # Tags aus CARD_DB holen (uppercase), API gibt manchmal auch tags
        card_name = card.get("name", "")
        tags_raw  = CARD_DB.get(card_name, {}).get("tags", [])
        tags      = [t.upper() for t in tags_raw]

        steel_have    = player.get("steel", 0)
        titanium_have = player.get("titanium", 0)
        mc_have       = player.get("megacredits", 0)
        # Effektive Werte aus dem Server-State (Advanced Alloys: Stahl 3 / Titan 4,
        # PhoboLog: Titan 4). Fallback auf die Standardwerte 2 / 3.
        steel_value    = player.get("steelValue", 2) or 2
        titanium_value = player.get("titaniumValue", 3) or 3

        # Titan NUR fuer Space-Karten (nicht Jovian ohne Space -> Server lehnt ab)
        if "SPACE" in tags:
            # Nutze so viel Titan wie möglich ohne überzubezahlen
            max_titan = min(titanium_have, cost // titanium_value)
            # Nicht mehr Titan einsetzen als wir MC haben um den Rest zu zahlen
            while max_titan > 0 and (cost - max_titan * titanium_value) < 0:
                max_titan -= 1
            titanium_used = max_titan
            cost -= titanium_used * titanium_value

        # Stahl für Building-Karten
        if "BUILDING" in tags:
            max_steel = min(steel_have, (cost + 1) // steel_value)  # +1 damit wir nicht überbezahlen
            # Stahl auf genau cost abrunden (kein Überbezahlen bei MC)
            while max_steel > 0 and (cost - max_steel * steel_value) < 0:
                max_steel -= 1
            steel_used = max_steel
            cost -= steel_used * steel_value

        # Verbleibende Kosten in MC (nicht negativ)
        cost = max(0, cost)

    return {
        "auroraiData": 0, "floaters": 0, "graphene": 0, "heat": 0,
        "kuiperAsteroids": 0, "lunaArchivesScience": 0,
        "megacredits": cost,
        "microbes": 0, "plants": 0, "seeds": 0,
        "spireScience": 0, "steel": steel_used, "titanium": titanium_used,
    }


# SpaceBonus-Werte (aus TM-Quellcode: TITANIUM=0, STEEL=1, PLANT=2, DRAW_CARD=3, HEAT=4)
_SPACE_BONUS_VALUE = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.5}



# Server-TileType-Enum (src/common/TileType.ts)
TILE_GREENERY, TILE_OCEAN, TILE_CITY = 0, 1, 2


def _tile_adjacency_score(tile_type, own_cities, opp_cities,
                          own_greens, opp_greens, oceans, free_adj):
    """Adjazenz-Wert eines Feldes je Tile-Typ (gemeinsam fuer Heuristik UND MCTS,
    damit beide Pfade nicht auseinanderdriften). Platzierungsboni werden separat
    vom Aufrufer addiert. Spezial-Tiles:
      commercial: 1 VP je angrenzender Stadt (egal wessen)  -> Staedte maximieren
      capital:    1 VP je angrenzendem Ozean                -> Ozeane maximieren
      neutral:    kein Adjazenz-Nutzen (z.B. Nuclear Zone)   -> guten Greenery-/Stadt-
                  Platz NICHT verschwenden (war vorher der greenery-Default-Bug)."""
    s = oceans * 1.0   # Ozean-Nachbarn ~ +2 MC (halbgewichtet)
    if tile_type == "greenery":
        s += own_cities * 3.0 - opp_cities * 2.0 + own_greens * 0.5
    elif tile_type == "city":
        s += (own_greens + opp_greens) * 3.0 + free_adj * 0.8   # vorhandene Greenery (sichere VP)
                                                                # dominiert ueber freies Potenzial
    elif tile_type == "commercial":
        s += (own_cities + opp_cities) * 5.0                    # 1 VP = 5M je angrenzender Stadt
    elif tile_type == "capital":
        s += oceans * 4.0 + free_adj * 0.5                      # 1 VP je Ozean (zzgl. Ozean-Basis)
    elif tile_type == "neutral":
        s += -own_cities * 3.0 - own_greens * 1.5 - free_adj * 0.4
    elif tile_type == "ocean":
        s += own_cities * 0.5 + own_greens * 0.5
    return s


def board_adjacency(space_map: dict) -> dict[str, list[str]]:
    """
    Hex-Nachbarn aus x/y berechnen - der Server liefert KEIN adjacency-Feld
    in der Player-View. Offsets exakt wie Board.computeAdjacentSpaces im
    Server-Repo (middleRow-relativ, Kolonie-Felder ausgenommen).
    """
    spaces = [s for s in space_map.values()
              if s.get("spaceType") != "colony"
              and isinstance(s.get("y"), int) and s["y"] >= 0]
    if not spaces:
        return {}
    by_xy  = {(s["x"], s["y"]): s["id"] for s in spaces}
    middle = max(s["y"] for s in spaces) // 2
    adj: dict[str, list[str]] = {}
    for s in spaces:
        x, y = s["x"], s["y"]
        tl, tr = [x, y - 1], [x, y - 1]
        bl, br = [x, y + 1], [x, y + 1]
        if y < middle:
            bl[0] -= 1; tr[0] += 1
        elif y == middle:
            br[0] += 1; tr[0] += 1
        else:
            br[0] += 1; tl[0] -= 1
        coords = {tuple(tl), tuple(tr), (x + 1, y), tuple(br), tuple(bl), (x - 1, y)}
        adj[s["id"]] = [by_xy[c] for c in coords
                        if c in by_xy and by_xy[c] != s["id"]]
    return adj


def _neighbor_tiles(sid: str, space_map: dict, adjacency: dict):
    """(tileType, ownerColor) der Nachbarfelder; tileType None = frei."""
    for aid in adjacency.get(sid, []):
        a = space_map.get(aid, {})
        yield a.get("tileType"), a.get("color")


# ── Ares: Spezial-Tiles gewaehren dem daneben Platzierenden einen Bonus. Das SpaceModel
# liefert KEIN adjacencyBonus-Feld -> Tabelle aus den TS-Kartendefinitionen, keyed auf
# den numerischen tileType (TileType-Enum). Damit "sieht" der Bot die Boni beim Platzieren.
_ARES_ADJ_BONUS = {
    3:  ["mc", "mc"],           # CAPITAL
    4:  ["mc", "mc"],           # COMMERCIAL_DISTRICT
    5:  ["animal"],             # ECOLOGICAL_ZONE
    6:  ["steel"],              # INDUSTRIAL_CENTER
    7:  ["heat", "heat"],       # LAVA_FLOWS
    10: ["heat", "heat"],       # MOHOLE_AREA
    11: ["mc"],                 # NATURAL_PRESERVE
    13: ["draw_card"],          # RESTRICTED_AREA
    14: ["asteroid", "steel"],  # DEIMOS_DOWN
    15: ["energy", "energy"],   # GREAT_DAM
    16: ["plant", "microbe"],   # MAGNETIC_FIELD_GENERATORS
    17: ["plant", "microbe"],   # BIOFERTILIZER_FACILITY
    18: ["titanium"],           # METALLIC_ASTEROID
    19: ["energy", "energy"],   # SOLAR_FARM
    21: ["plant"],              # OCEAN_FARM
    22: ["animal"],             # OCEAN_SANCTUARY
    # NUCLEAR_ZONE (12): kein Adjazenz-Bonus (roter Rand).
    # MINING_AREA (8)/MINING_RIGHTS (9): Stahl/Titan-Bonus liegt auf dem Feld (space.bonus),
    #   nicht als Adjazenz -> wird ueber _SPACE_BONUS_VALUE bereits erfasst.
}
_ARES_RES_VALUE = {"mc": 1.0, "heat": 0.8, "plant": 2.0, "animal": 2.5, "microbe": 1.5,
                   "steel": 2.0, "titanium": 3.0, "energy": 1.0, "asteroid": 2.0, "draw_card": 3.0}

# Hazard-Tiles (TileType 23-26): eine eigene Markierung (Greenery/City) daneben zu legen
# kostet 1 Produktion (mild) bzw. 2 (severe). M-aequivalenter Malus ~ Wert der billigsten
# geopferten Produktion (Hitze/Energie ~4 je Schritt). Ozeane tragen keine Markierung -> kein Malus.
_ARES_HAZARD_PENALTY = {23: 4.0, 25: 4.0,   # DUST_STORM_MILD, EROSION_MILD  (1 Prod)
                        24: 8.0, 26: 8.0}   # DUST_STORM_SEVERE, EROSION_SEVERE (2 Prod)

def _ares_adj_value(ttype, is_marker: bool = True) -> float:
    """Ares-Adjazenz beim Platzieren: Bonus fuers Legen neben ein Spezial-Tile (immer);
    Malus fuers Legen einer EIGENEN Markierung (Greenery/City, NICHT Ocean) neben ein Hazard."""
    if ttype in _ARES_HAZARD_PENALTY:
        return -_ARES_HAZARD_PENALTY[ttype] if is_marker else 0.0
    return sum(_ARES_RES_VALUE.get(r, 1.0) for r in _ARES_ADJ_BONUS.get(ttype, []))


def get_top_spaces(
    valid_ids:  list[str],
    space_map:  dict,
    tile_type:  str = "greenery",
    player_id:  str | None = None,
    n:          int = 3,
) -> list[tuple[float, str]]:
    """
    Gibt die Top-N Felder als (score, spaceId) zurück.
    Ermöglicht MCTS zwischen verschiedenen Positionen zu wählen.
    """
    adjacency = board_adjacency(space_map)

    def score_space(sid: str) -> float:
        space = space_map.get(sid, {})
        score = 0.0
        bonuses = space.get("bonus", [])
        if isinstance(bonuses, list):
            for b in bonuses:
                score += _SPACE_BONUS_VALUE.get(b, 0.5)
        own_cities = opp_cities = own_greens = opp_greens = oceans = free_adj = 0
        for ttype, owner in _neighbor_tiles(sid, space_map, adjacency):
            score += _ares_adj_value(ttype, tile_type in ("greenery", "city"))   # Ares: Bonus/Hazard-Malus
            if ttype is None:
                free_adj += 1
            elif ttype == TILE_GREENERY:
                if owner == player_id: own_greens += 1
                else:                  opp_greens += 1
            elif ttype == TILE_CITY:
                if owner == player_id: own_cities += 1
                else:                  opp_cities += 1
            elif ttype == TILE_OCEAN:
                oceans += 1
        score += _tile_adjacency_score(tile_type, own_cities, opp_cities,
                                        own_greens, opp_greens, oceans, free_adj)
        return score

    scored = [(score_space(sid), sid) for sid in valid_ids]
    scored.sort(key=lambda x: x[0], reverse=True)

    # Dedupliziere ähnliche Scores – nicht alle Top-3 mit Score 0 zurückgeben
    result = []
    prev_score = None
    for sc, sid in scored:
        if len(result) >= n:
            break
        # Immer erste nehmen; danach nur wenn deutlich unterschiedlich
        if prev_score is None or abs(sc - prev_score) > 0.3 or len(result) == 0:
            result.append((sc, sid))
            prev_score = sc
        elif len(result) < n:
            result.append((sc, sid))
    return result[:n]


def choose_best_space(
    valid_ids:  list[str],
    space_map:  dict,
    tile_type:  str = "greenery",   # "greenery", "city", "ocean"
    player_id:  str | None = None,
) -> str:
    """
    Wählt das beste Feld für ein Tile.

    Bewertung (höher = besser):
    1. Placement-Boni (Stahl, Titan, Pflanze, Karte, Hitze)
    2. Nähe zu eigenen Tiles (Greenerys geben VP wenn neben Städten)
    3. Für Greenerys: Felder neben eigenen Städten bevorzugen
    4. Für Städte: Felder mit vielen freien Nachbarfeldern (zukünftige Greenerys)
    """
    def score_space(sid: str) -> float:
        space = space_map.get(sid, {})
        score = 0.0

        # Placement-Boni (gewichtet nach BGG-Werten)
        bonuses = space.get("bonus", [])
        if isinstance(bonuses, list):
            for b in bonuses:
                score += _SPACE_BONUS_VALUE.get(b, 0.5)

        # Nachbar-Tiles analysieren (flaches Server-Schema: tileType/color)
        own_cities = opp_cities = own_greens = opp_greens = oceans = free_adj = 0
        for ttype, owner in _neighbor_tiles(sid, space_map, adjacency):
            score += _ares_adj_value(ttype, tile_type in ("greenery", "city"))   # Ares: Bonus/Hazard-Malus
            if ttype is None:
                free_adj += 1
            elif ttype == TILE_GREENERY:
                if owner == player_id: own_greens += 1
                else:                  opp_greens += 1
            elif ttype == TILE_CITY:
                if owner == player_id: own_cities += 1
                else:                  opp_cities += 1
            elif ttype == TILE_OCEAN:
                oceans += 1

        score += _tile_adjacency_score(tile_type, own_cities, opp_cities,
                                        own_greens, opp_greens, oceans, free_adj)

        return score

    adjacency = board_adjacency(space_map)
    if not valid_ids:
        return None
    return max(valid_ids, key=score_space)


# ---------------------------------------------------------------------------
# Aktionsbewertung
# ---------------------------------------------------------------------------

def _placement_bonus(space_id: str, tile_type: str, state: dict) -> float:
    """
    Berechnet Placement-Bonus einer Tile-Platzierung in MC-Wert (BGG-Guide Sektion 4).

    tile_type: "greenery", "city", "ocean"

    Greenery neben eigener Stadt:  +5M (1 VP = 5M)
    Greenery/Ocean neben Ozean:    +2M pro benachbartem Ozean
    Stadt neben eigener Greenery:  +5M pro Greenery (1 VP)
    """
    if not space_id:
        return 0.0

    game     = state.get("game", {})
    player   = state["thisPlayer"]
    my_color = player.get("color")
    spaces   = {s["id"]: s for s in game.get("spaces", [])}

    space = spaces.get(space_id)
    if not space:
        return 0.0

    # Nachbarn aus x/y berechnen (Server liefert kein adjacency-Feld);
    # Belegung steht FLACH am Space (tileType/color), TileType-Enum:
    # GREENERY=0, OCEAN=1, CITY=2
    adjacency = board_adjacency(spaces)
    bonus = 0.0

    for ttype, owner in _neighbor_tiles(space_id, spaces, adjacency):
        if ttype is None:
            continue
        if ttype == TILE_OCEAN:
            bonus += 2.0            # +2 MC je Ozean-Nachbar (Board-Regel)
            continue
        if tile_type == "greenery":
            if ttype == TILE_CITY and owner == my_color:
                bonus += 5.0        # eigene Stadt punktet +1 VP
            elif ttype == TILE_CITY:
                bonus -= 5.0        # Gegner-Stadt punktet +1 VP (Geschenk)
        elif tile_type == "city":
            if ttype == TILE_GREENERY:
                bonus += 5.0        # Stadt punktet je Gruenflaeche, egal wessen

    return bonus


def _is_last_generation(state: dict) -> bool:
    """True wenn das Spiel sicher nach dieser Generation endet: alle drei
    globalen Parameter maximal (bzw. Server-Flag isTerraformed). Dann ist
    M€-Horten sinnlos -> Reserven aufloesen, in VPs umwandeln (gemessen:
    55 M€ + 6 Stahl totes Endkapital im 1v1 vom 06.06.)."""
    game = state.get("game", {})
    return bool(game.get("isTerraformed")) or (
        game.get("temperature", -30) >= 8
        and game.get("oxygenLevel", 0) >= 14
        and game.get("oceans", 0) >= 9)


def _action_card_info(title: str) -> dict:
    """Loest einen Aktions-Option-Titel zur card_db-Karte auf."""
    t = (title or "").strip()
    for pre in ("use the action of ","use action of ","action of ","activate ","use action "):
        if t.lower().startswith(pre): t = t[len(pre):].strip(); break
    if t in CARD_DB: return CARD_DB[t]
    tl_ = t.lower()
    for n, c in CARD_DB.items():
        if n.lower() == tl_: return c
    return {}


def _planning_needs(state: dict) -> dict:
    """Planungs-Hebel (generalisiert): scannt Hand (und fuers Fuettern auch das Tableau)
    nach hochdichten Ressourcen-VP-Engines, die noch hinter einer ERREICHBAREN Bedingung
    klemmen, und leitet daraus 'Bedarf' ab:
      param: Globalparameter-Minimum knapp unerfuellt (Predators=O2, Venusian Animals=Venus).
             Pro Schritt naeher = action_once*PLAN_WEIGHT Wert (frueher frei -> mehr Stapeln).
      prod:  Aktion braucht Produktion, die der Bot (noch) nicht hat (Physics Complex=6 Energie).
             Gilt fuer Hand UND Tableau (eine gespielte, hungernde Engine will gefuettert werden).
    Tags sind bewusst NICHT hier - die deckt das Incentive-Field (_strategy_demand) bereits ab.
    Liefert {'param': {p: wert-pro-schritt}, 'prod': {res: wert-pro-prod}}. Memoisiert."""
    if not LEVER_PLAN:
        return {"param": {}, "prod": {}}
    cached = state.get("_plan_needs")
    if cached is not None:
        return cached
    game = state.get("game", {}); player = state.get("thisPlayer", {})
    cur = {"temperature": game.get("temperature", -30),
           "oxygen":      game.get("oxygenLevel", 0),
           "oceans":      game.get("oceans", 0),
           "venus":       game.get("venusScaleLevel", 0)}
    prod_own = player.get("production", {}) or {}
    param: dict = {}; prod: dict = {}

    def _scan(cards, do_param):
        for c in (cards or []):
            nm = c.get("name") if isinstance(c, dict) else c
            info = card_info(nm or "")
            if not info:
                continue
            vd = info.get("vp_dyn") or {}
            if vd.get("kind") != "resources":
                continue
            if (vd.get("each", 1) / max(1, vd.get("per", 1))) < PLAN_MIN_DENSITY:
                continue
            ao = float(info.get("action_once", 0) or 0)
            if do_param:   # nur HAND: globale Schranke fuers Freischalten
                for rg in (info.get("req_global") or []):
                    p, t = rg.get("param"), rg.get("value", 0)
                    if rg.get("max") or p not in cur:
                        continue
                    steps = (t - cur[p]) / _PARAM_STEP.get(p, 1)
                    if 0 < steps <= PLAN_MAX_STEPS:
                        param[p] = min(PLAN_BONUS_CAP, param.get(p, 0.0) + ao * PLAN_WEIGHT)
            for res, amt in (info.get("action_input") or {}).items():  # Hand+Tableau: Treibstoff
                if amt and amt > 0:
                    gap = amt - prod_own.get(res, 0)
                    if 0 < gap <= PLAN_MAX_PROD_GAP:
                        prod[res] = min(PLAN_BONUS_CAP, prod.get(res, 0.0) + ao * PLAN_WEIGHT)

    _scan(player.get("cardsInHand", []), do_param=True)
    _scan(player.get("tableau") or player.get("playedCards") or [], do_param=False)
    needs = {"param": param, "prod": prod}
    state["_plan_needs"] = needs
    return needs


def _planning_card_bonus(info: dict, state: dict) -> float:
    """Wert, den ein KANDIDATEN-Kartenspiel zur Freischaltung/Fuetterung gehaltener
    hochdichter Engines beitraegt: hebt es einen gefragten Globalparameter (global/
    greenery->O2/oceans) oder liefert es gefragte Produktion? Pro Schritt/Prod-Einheit
    der hinterlegte Bedarfswert. Tag-Beitraege macht weiter das Incentive-Field."""
    needs = _planning_needs(state)
    if not needs["param"] and not needs["prod"]:
        return 0.0
    bonus = 0.0
    # Parameter-Anhebungen der Karte (global-Feld + Greenery->O2 + Ozeane)
    raises = dict(info.get("global") or {})
    if info.get("greenery"):
        raises["oxygen"] = raises.get("oxygen", 0) + int(info["greenery"])
    if info.get("oceans"):
        raises["oceans"] = raises.get("oceans", 0) + int(info["oceans"])
    for p, amt in raises.items():
        if amt and needs["param"].get(p):
            bonus += needs["param"][p] * (amt / _PARAM_STEP.get(p, 1))
    # Produktions-Beitraege der Karte
    for res, delta in (info.get("production") or {}).items():
        if delta and delta > 0 and needs["prod"].get(res):
            bonus += needs["prod"][res] * delta
    return min(bonus, PLAN_BONUS_CAP)   # Anschub, keine Uebernahme


def score_action(action_type: str, state: dict,
                 placement_bonus: float = 0.0, card_title: str = "") -> float:
    """
    Bewertet eine Aktion nach BGG-Guide Ressourcenwerten (Sektion 1).

    Referenzwerte (BGG):
      1 TR = 10M, 1 VP = 5M, 1 MC-Prod = 5M, Greenery = 19M, Ozean = 14M
      SP overpay = 4M gegenüber Kartendurchschnitt
      placement_bonus: extra MC-Wert durch Tile-Platzierung (Ozean-Nachbarschaft etc.)
    """
    game    = state["game"]
    player  = state["thisPlayer"]
    oxygen  = game.get("oxygenLevel", 0)
    temp    = game.get("temperature", -30)
    oceans  = game.get("oceans", 0)
    tl      = turns_left(state)
    urgency = max(0, 4 - tl) * 2   # M-Dringlichkeit (Netto-M-Skala)

    mc = player.get("megacredits", 0)

    last_gen    = _is_last_generation(state)
    reserve     = 0 if last_gen else MC_RESERVE
    cost_weight = 0.0 if (last_gen or state.get("_idle_money")) else SP_COST_WEIGHT  # letzte Gen / Leerlauf: Kosten illusorisch
    plan        = _planning_needs(state)["param"]  # {param: Bonus} fuer gehaltene Spät-Engines

    # ── Pflanzen → Greenery ──────────────────────────────────────────────────
    # BGG: Greenery = 19M Wert (10M TR + 5M VP + 4M Placement)
    # Pflanzen-Greenery ist effizienter als SP-Greenery (kein 4M Aufpreis)
    if action_type == "greenery":
        # Pflanzen->Greenery: 1 TR + 1 VP (15M), bei O2-Max nur VP. Pflanzen
        # sind Konvertierungs-Ressource -> keine Geldkosten.
        gross = (15 if oxygen < 14 else 5) + placement_bonus + urgency
        gross += plan.get("oxygen", 0.0)   # Planungs-Hebel: O2 fuer gehaltene Engine
        gross += max(_milestone_action_bonus("greenery", state),
                      _milestone_complete_bonus("greenery", state))   # Pursue/Abschluss: Gardener
        return max(0, gross) * 3

    # ── Wärme → Temperatur ───────────────────────────────────────────────────
    # BGG: 1 TR = 10M; Wärme-Konvertierung = immer gut wenn Temp nicht voll
    if action_type == "heat":
        if temp >= 8:
            # Temperatur voll -> Wandlung bringt 0 TR. Hitze nur dann behalten, wenn
            # sie ueber einen GEWINNBAREN Thermalist (meiste Hitze = 5 VP) noch zaehlt
            # -> negativ, damit Passen/andere Zuege vorgezogen werden (Partie 8: Bot
            # sponserte Thermalist und wandelte die Hitze dann weg). Ist Thermalist
            # nicht gewinnbar / fuehrt der Bot nicht, ist die Hitze wertlos und das
            # Wandeln ein harmloser Wartezug (Zugzwang) -> 0 (erlaubt).
            if thermalist_hold_value(state) > 0:
                return -min(player.get("heat", 0), 12) * 0.5
            return 0.0
        # Anti-Horten: Hitze hat keinen anderen sinnvollen Nutzen als die Wandlung.
        # Gehortete Hitze ueber die naechste Wandlung (8) hinaus ist wasch-gefaehrdet
        # (Track maxt / Spielende) -> je mehr Ueberschuss, desto dringender JETZT
        # wandeln, statt auf die Spaet-urgency (tl<=4) zu warten. Gedeckelt, damit
        # es starke Kartenplays nicht ueberlagert.
        # ABER Thermalist-Kohaerenz: Fuehrt der Bot einen GEWINNBAREN Thermalist
        # (im Spiel, gefundet oder noch fundbar; meiste Hitze = 5 VP), ist seine
        # Hitze ~5 VP wert -> nicht wegwandeln, sonst verschenkt er den Award
        # (Partie 8). Ist Thermalist nicht mehr fundbar (3 Awards weg) oder fuehrt
        # er nicht, liefert thermalist_hold_value 0 -> normale Wandlung (Hitze sonst
        # wertlos). Nur bei komfortablem Vorsprung (>=16, nach 8er-Wandlung noch
        # klar vorn) darf der Ueberschuss gewandelt werden.
        t_hold = thermalist_hold_value(state)
        if 0 < t_hold < 16:
            return 0.0
        excess = max(0, player.get("heat", 0) - 8)
        hoard_urgency = min(excess * 0.5, 8.0)
        return max(0, 10 + urgency + hoard_urgency + plan.get("temperature", 0.0)) * 3

    # ── Standard-Projekte ────────────────────────────────────────────────────
    # BGG: SP kosten immer 4M mehr als Karten → nur lohnend mit Placement-Bonus
    # oder wenn keine bessere Karte spielbar ist.

    if action_type == "ocean_sp":
        if oceans >= 9:
            return 0
        cost = 18
        if mc < cost + reserve:
            return 0
        # BGG: Ozean = 14M (1 TR=10M + 4M Placement-Erwartung)
        # SP Aufpreis = 4M → Nettowert = 10M ohne Bonus
        # Placement-Bonus neben 2 Ozeanen (+4M) negiert SP-Aufpreis komplett
        net = 10 + placement_bonus - cost * cost_weight + urgency + plan.get("oceans", 0.0)
        return max(0, net) * 3

    if action_type == "temp_sp":
        if temp >= 8:
            return 0
        cost = 14
        if mc < cost + reserve:
            return 0
        net = 10 + placement_bonus - cost * cost_weight + urgency
        if LEVER_SP_DISCIPLINE:
            # Asteroid-SP ist reiner TR-Kauf (14 MC -> 1 TR), die mieseste MC->VP-
            # Rate. Frueh gehoert das Geld in eine Engine -> abwerten, solange viele
            # Generationen fuer Engine-Aufbau bleiben (tl>6). Faded zur Spielmitte;
            # spaet uebernimmt urgency die Ernte.
            net -= max(0.0, tl - 6) * 2.0
        net += plan.get("temperature", 0.0)   # Planungs-Hebel: Temp fuer gehaltene Engine
        return max(0, net) * 3

    # Air Scrapping (Venus Next): 15 MC -> +1 Venus-Stufe (1 TR). In Vanilla bietet der
    # Server dieses SP nicht an -> Branch bleibt brach. Sobald Venus Next aktiv ist,
    # bewertet er es analog zu Asteroid/Aquifer und traegt den Planungs-Bonus (Venus
    # fuer gehaltene Venus-Engines wie Venusian Animals).
    if action_type == "venus_sp":
        if game.get("venusScaleLevel", 0) >= 30:
            return 0
        cost = 15
        if mc < cost + reserve:
            return 0
        net = 10 + placement_bonus - cost * cost_weight + urgency + plan.get("venus", 0.0)
        return max(0, net) * 3

    if action_type == "greenery_sp":
        cost = 23
        if mc < cost + reserve:
            return 0
        gross = (15 if oxygen < 14 else 5) + placement_bonus
        net   = gross - cost * cost_weight + urgency + plan.get("oxygen", 0.0)
        net  += max(_milestone_action_bonus("greenery_sp", state),
                    _milestone_complete_bonus("greenery_sp", state))   # Pursue/Abschluss: Gardener
        if LEVER_GREENERY_DISCIPLINE:
            # Greenery-SP (23 MC -> 1 TR + 1 VP, bei O2-Max nur 1 VP) ist die mieseste
            # MC->VP-Rate. Frueh gehoert das Geld in Karten/Engine/Meilensteine -> abwerten,
            # solange viele Generationen bleiben. Faded zur Spielmitte; spaet erntet urgency.
            net -= max(0.0, tl - GREENERY_LATE_GEN) * GREENERY_EARLY_PENALTY
        return max(0, net) * 3

    if action_type == "city_sp":
        cost = 25
        if mc < cost + reserve:
            return 0
        # BGG: Stadt SP = 25 M€ fuer ~9M Grundwert (Placement + MC-Prod)
        # -> Defizit ~16M, break-even bei ~3 angrenzenden Gruenflaechen (3x5M).
        # Basis daher BONUS-GETRIEBEN statt pauschal: ohne VP-Nachbarschaft ist
        # die Stadt den SP-Preis nicht wert (gemessen: 13.4 Staedte/Partie bei
        # pauschaler Basis -> -2.92 VP, Lauf 2026-06-07).
        # Abnehmende Ertraege: jede weitere eigene Stadt bindet 25 M€,
        # verknappt legale Plaetze (Stadt-Adjazenz-Verbot) und verschleppt
        # das Spielende (gemessen: 64 Stadt-Aktionen/Partie, Spiele bis
        # Gen 30 im A/A 2026-06-07) -> -3 je bereits eigener Stadt.
        gross = 9 + placement_bonus
        net   = gross - cost * cost_weight - 5 * player.get("citiesCount", 0)
        net += max(_milestone_action_bonus("city_sp", state),
                   _milestone_complete_bonus("city_sp", state))   # Pursue/Abschluss: Mayor
        return max(0, net) * 3

    if action_type == "sell":
        # Verkaufen bringt nur 1 M€ pro Karte und gibt deren Optionswert auf.
        # Der Score muss diesen geringen Ertrag abbilden und klar UNTER dem
        # Pass-Score (4) liegen – sonst verschleudert der Bot Karten, die er
        # momentan nur als schwach bewertet (z.B. teure Engine-Karten), für 1 M€,
        # statt schlicht zu passen. So bleibt Verkaufen nur theoretisch möglich,
        # wenn jede andere Option noch schlechter wäre.
        # AUSNAHME letzte Generation: es gibt kein "spaeter" mehr, ungespielte
        # Karten verfallen ungenutzt. Dann Verkaufen UEBER Pass (4) heben, um den
        # Rest als M€ (Tiebreaker) zu verwerten - aber unter Kartenspiel/SP, damit
        # zuerst alles Sinnvolle (inkl. Endgame-VP-Karten) gespielt wird.
        if _is_last_generation(state):
            return 5
        return 1

    if action_type == "card_action":
        # Aktivierung einer ACTIVE-/Konzern-Aktion. Identische Logik wie
        # handle_or._act_value, damit beide Aktivierungspfade gleich bewerten.
        info = _action_card_info(card_title) if card_title else {}
        # Heat-Sperre: bei gemaxter Temperatur ist eine Aktion, deren einziger
        # Produktions-Ertrag Hitze ist (Underground Detonations), wertlos.
        temp = state.get("game", {}).get("temperature", -30)
        apr = info.get("action_prod_res") or info.get("production") or {}
        if temp >= 8 and apr.get("heat", 0) > 0 and all(
                r == "heat" or v <= 0 for r, v in apr.items()):
            return -1.0
        once = float(info.get("action_once", 0) or 0)
        draw = float(info.get("action_draw", 0) or 0)
        val  = once + 2.0 * draw
        if val > 0:
            return val * CARD_PLAY_SCALE
        # val <= 0: Aktion ohne bezifferten Netto-Ertrag. KEIN Pauschal-Fallback
        # mehr (frueher 10.0 fuer "unquantifizierte" Aktionen). Der erzeugte einen
        # Wert UEBER dem Spielen einer Handkarte und liess den Bot wertlose Grab-/
        # Tausch-Aktionen (Search For Life, Space Mirrors = 0) jede Generation
        # aktivieren statt Karten zu spielen (Deploy-Loop). handle_or._act_value
        # gated den Zwilling-Branch bereits via `val > 0`; hier ziehen wir nach,
        # damit BEIDE Aktivierungspfade identisch bewerten (vgl. Z. 1704).
        return 0.0

    if action_type == "pass":
        return 4

    return 0


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


INITIAL_KEEP_MIN = GREENERY_BUY_FLOOR   # -10: Startkarte = 3-M-Spieloption ueber die
                         # GESAMTE Restlaufzeit, also weit grosszuegiger als der
                         # Kauf (0.5). Bewertung via score_card_to_buy(for_initial_keep=True),
                         # d.h. ohne Bezahlbarkeit-/Req-Strafe (Zukunfts-Plays). Deckel
                         # begrenzt die Anzahl. ACHTUNG: korrigiert nur die ANZAHL der
                         # behaltenen Karten (Breite), NICHT die Unterbewertung von
                         # Effekt-Karten (Arctic Algae etc.) -> deren Auswahl bleibt offen.
INITIAL_KEEP_MAX = 7     # starke menschliche Eroeffnung behaelt ~7/10

def choose_initial_cards(cards: list[dict], state: dict) -> list[str]:
    """
    Startkarten-Auswahl (Forschungsphase Gen 1). NICHT choose_cards_to_buy
    verwenden: zum Entscheidungszeitpunkt ist megacredits=0 (der Konzern wird
    in derselben Antwort gewaehlt) -> budget=0 -> es wurde IMMER 0 von 10
    behalten (gemessen in allen drei 1v1s). Wir rechnen stattdessen mit
    konservativem Konzern-Startkapital und kaufen bis zu 6 Karten a 3 M€.
    """
    player = state.get("thisPlayer", {})
    init_player = dict(player)
    init_player["megacredits"] = max(player.get("megacredits", 0), 40)
    init_state = dict(state)
    init_state["thisPlayer"] = init_player

    scored = sorted(((score_card_to_buy(c, init_state, for_initial_keep=True), c.get("name", ""))
                     for c in cards), reverse=True)
    chosen = [n for sc, n in scored if sc > INITIAL_KEEP_MIN][:INITIAL_KEEP_MAX]
    log.info("  📦 Startkarten: behalte %d/%d: %s",
             len(chosen), len(cards), ", ".join(chosen) if chosen else "-")
    return chosen


# ── CEOs (Erweiterung 'ceo'): 38 Karten mit je EINER einmaligen Faehigkeit (OPG = once per
# game). Zwei Dinge fehlten dem Bot komplett: (1) die AUSWAHL beim Spielstart (er nahm einfach
# die erste der 3 angebotenen), (2) die NUTZUNG der OPG-Aktion (kein Handler -> die Faehigkeit
# verfiel ungenutzt).
#
# _CEO_VALUE: M-aequivalenter Wert der OPG-Faehigkeit (VORSCHLAG - vom TM-Experten justierbar).
# _CEO_REQUIRES: benoetigtes Modul. Die Settings der Runde VARIIEREN -> der Wert wird zur
# LAUFZEIT auf 0 gesetzt, wenn das Modul nicht aktiv ist (Apollo ohne Moon ist wertlos).
_CEO_VALUE = {
    # starke, modul-unabhaengige Faehigkeiten
    "Karen":     28.0,   # Prelude-Karten = Generation ziehen, eine spielen (frueh sehr stark)
    "Will":      24.0,   # Ressourcen auf eigene Karten (2 je Typ)
    "Clarke":    22.0,   # +1 Pflanzen- UND Hitze-Produktion (dauerhaft)
    "Tate":      20.0,   # Karten mit gewaehltem Tag aus dem Deck ziehen
    "Ryu":       18.0,   # Produktion tauschen (X+2 Einheiten)
    "Ender":     16.0,   # Karten abwerfen -> nachziehen (Hand-Refresh)
    "Musk":      16.0,   # Earth-Karten abwerfen -> gleich viele ziehen
    "Stefan":    14.0,   # Handkarten fuer je 3 M€ verkaufen
    "Jansson":   16.0,   # Platzierungsboni unter eigenen Tiles nochmal kassieren
    "Hal 9000":  14.0,   # Produktion senken -> sofort Ressourcen
    "Greta":     18.0,   # TR-Erhoehungen geben Bonus (dauerhafter Effekt)
    "Faraday":   16.0,   # Tag-Meilensteine geben Boni (dauerhaft)
    "Ingrid":    14.0,   # Tile-Platzierungen diese Generation verstaerkt
    "Van Allen": 20.0,   # Meilensteine kosten 0 + 3 M€ je geclaimtem Meilenstein
    "Rogers":    12.0,   # Venus-Requirements ignorieren (nur mit Venus wertvoll)
    "Xavier":    12.0,
    "Co-leadership": 10.0,
    # modulabhaengig (Wert nur wenn Modul aktiv - sonst 0, s. _CEO_REQUIRES)
    "Apollo":    16.0,   # 3 M€ je Moon-Tile
    "Neil":      16.0,
    "Shara":     16.0,
    "Oscar":     18.0,   # Chairman ersetzen (Turmoil)
    "Petra":     18.0,   # Neutrale Delegaten ersetzen (Turmoil)
    "Zan":       14.0,   # Delegaten in Reds (Turmoil)
    "Maria":     16.0,   # Kolonie-Plaettchen ziehen
    "Naomi":     18.0,   # Kolonie-Tracks auf Maximum
    "Yvonne":    18.0,   # Kolonie-Boni doppelt
    "Huan":      14.0,   # Gegner koennen nicht handeln + Trade Fleet
    "Floyd":     12.0,
    "Ulrich":    12.0,
    "Quill":     14.0,   # 2 Floater auf Venus-Karten
    "Xu":        14.0,   # 2 M€ je Venus-Tag
    # ── Bei Damians Runde aktuell GEBANNT, aber trotzdem bewertet: der Server bietet sie dann
    # gar nicht an (der Bot sieht sie nie) - aendert sich die Bannliste, sind sie sofort korrekt
    # bewertet, statt auf den neutralen Default (10) zu fallen. Sie sind auffaellig STARK -
    # vermutlich genau der Grund fuer den Bann.
    "Asimov":    30.0,   # zieht Awards (10-Gen), darf einen GRATIS funden + '+2 auf alle Awards'
    "Duncan":    30.0,   # 7-X VP UND 4X M€ (frueh gespielt: ~6 VP + Geld = sehr stark)
    "Caesar":    22.0,   # X Hazards legen; jeder Gegner verliert 1-2 Produktion (Ares, Angriff)
    "Gaia":      20.0,   # Ares-Adjazenzboni ALLER Tiles auf dem Mars einsammeln (Ares)
    "Gordon":    20.0,   # Platzierungsregeln ignorieren + 2 M€ je Greenery/Stadt (dauerhaft)
    "Lowell":    18.0,   # 8 M€ -> 3 CEOs ziehen, einen spielen (praktisch ein zweiter CEO)
    "Bjorn":     14.0,   # X+2 M€ vom reichsten Gegner stehlen
}
_CEO_REQUIRES = {
    "Apollo": "moon", "Neil": "moon", "Shara": "pathfinders",
    "Oscar": "turmoil", "Petra": "turmoil", "Zan": "turmoil",
    "Maria": "colonies", "Naomi": "colonies", "Yvonne": "colonies", "Huan": "colonies",
    "Quill": "venus", "Xu": "venus", "Rogers": "venus",
    "Caesar": "ares", "Gaia": "ares",
}


def _module_active(state: dict, mod: str) -> bool:
    """Ist ein Modul in DIESER Partie aktiv? Primaer aus gameOptions.expansions; ersatzweise
    an den Spieldaten erkennbar (turmoil/colonies/aresData sind nur dann im State)."""
    exp = (game_options(state).get("expansions") or {})
    if mod in exp:
        return bool(exp[mod])
    game = state.get("game", {}) or {}
    return {
        "turmoil":  bool(game.get("turmoil")),
        "colonies": bool(game.get("colonies")),
        "venus":    game.get("venusScaleLevel") is not None,
        "moon":     bool(game.get("moon")),
        "ares":     bool(game.get("aresData")),
    }.get(mod, False)


def score_ceo(name: str, state: dict) -> float:
    """Wert eines CEO fuer DIESE Partie: Grundwert, aber 0, wenn das benoetigte Modul aus ist."""
    req = _CEO_REQUIRES.get(name)
    if req and not _module_active(state, req):
        return 0.0
    return _CEO_VALUE.get(name, 10.0)     # unbekannte CEOs: neutraler Mittelwert


def choose_ceo(cards: list[dict], state: dict) -> list[str]:
    """Waehlt den fuer diese Partie besten CEO (Server bietet i.d.R. 3 an, min=max=1)."""
    ranked = sorted(cards, key=lambda c: score_ceo(c.get("name", ""), state), reverse=True)
    best = ranked[0].get("name") if ranked else None
    if best:
        log.info("  👔 CEO: %s (Wert %.0f) aus %s", best, score_ceo(best, state),
                 [c.get("name") for c in cards])
    return [best] if best else []


def handle_initial_cards(state: dict) -> dict:
    waiting = state["waitingFor"]
    player  = state["thisPlayer"]

    responses = []
    for option in waiting.get("options", []):
        title = str(option.get("title", "")).lower()
        cards = option.get("cards", [])
        min_c = option.get("min", 0)

        if "ceo" in title:
            chosen = choose_ceo(cards, state)
        elif "corporation" in title:
            game_obj = state.get("game", {})
            num_players = (
                len(game_obj.get("players", []))
                or len(game_obj.get("spectators", []))  # Fallback
                or game_obj.get("playerCount", 2)       # direktes Feld
                or 2
            )
            chosen = [choose_corporation(cards, num_players=num_players)]
        elif "prelude" in title:
            chosen = choose_preludes(cards, min_c, state)
        elif any(k in title for k in ("initial", "buy", "select cards")):
            chosen = choose_initial_cards(cards, state)
        else:
            sample_size = min(min_c, len(cards))
            chosen = [c["name"] for c in random.sample(cards, sample_size)]
            log.warning("  initialCards: unbekannte Option '%s'", title)

        responses.append({"type": "card", "cards": chosen})

    return {"type": "initialCards", "runId": state["runId"], "responses": responses}


def _score_card_for_opponent(card: dict, state: dict) -> float:
    """
    Schätzt wie wertvoll diese Karte für den Gegner wäre.
    Vereinfacht: hoher Eigenproduktions-/TR-Score = gefährlich für Gegner.
    Karten die eigene Ressourcen klauen (Sabotage, Hackers) werden stark bewertet.
    """
    name = card.get("name", "")
    info = CARD_DB.get(name, {})
    if not info:
        return 0.0

    danger = 0.0

    # Ressourcen-Produktion ist für jeden Gegner wertvoll (DB-Schema)
    prod = info.get("production", {}) or {}
    danger += prod.get("megacredits", 0) * 3.0
    danger += prod.get("steel", 0)       * 5.0
    danger += prod.get("titanium", 0)    * 6.0
    danger += prod.get("plants", 0)      * 6.0
    danger += prod.get("energy", 0)      * 4.0

    # TR-/Terraforming-Karten sind immer gefährlich
    tr_like = (info.get("tr", 0) + info.get("oceans", 0)
               + sum((info.get("global", {}) or {}).values()))
    danger += tr_like * 8.0

    # VP-Karten sind gefährlich
    danger += max(0, info.get("vp", 0)) * 4.0

    # Karten mit Aktionen (ACTIVE) sind langfristig gefährlich
    if (info.get("type") or "").upper() == "ACTIVE":
        danger += 5.0

    return danger


def choose_draft_card(cards: list[dict], state: dict) -> str:
    """
    Wählt die beste Karte beim Draft.

    Strategie: maximiere (Eigenwert - Gegner-Wert der besten verbleibenden Karte).
    D.h.: wähle die Karte, bei der die Differenz zwischen dem was ich bekomme
    und dem was der Gegner als nächstes bekommen könnte am größten ist.

    Vereinfacht: sortiere nach (eigener_score * 0.7 + gegner_gefahr * 0.3),
    wobei gegner_gefahr = Score der besten verbleibenden Karte nach unserer Wahl.
    """
    if not cards:
        return cards[0]["name"] if cards else ""

    # Bewerte alle Karten
    state["_draft_ctx"] = True     # Draft: Feeder-Synergie nur ueber Tableau, NICHT ueber
    scored = []                    # das Paeckchen (davon bleibt nur 1 Karte -> keine Spekulation)
    for card in cards:
        own_val    = score_card_to_buy(card, state)
        opp_danger = _score_card_for_opponent(card, state)
        # Kombinierter Score: 60% Eigenwert + 40% Gegner-Gefahr (verhindert)
        combined = own_val * 0.6 + opp_danger * 0.4
        scored.append((combined, own_val, opp_danger, card))

    state.pop("_draft_ctx", None)
    scored.sort(key=lambda x: x[0], reverse=True)

    # Draft-Diagnose (gated via TM_DIAG_HAND): komplette Rangliste des Paeckchens,
    # damit sichtbar wird, WOMIT eine abgelehnte Karte (z.B. Fish/Birds) verlor.
    if os.environ.get("TM_DIAG_HAND") and len(scored) > 1:
        log.info("  DRAFT-DIAG (%d Karten, sortiert nach combined):", len(scored))
        for comb, own, opp, card in scored:
            log.info("      %-28s eigen=%6.1f gefahr=%5.1f -> combined=%6.1f",
                     str(card.get("name", "?"))[:28], own, opp, comb)

    best_combined, best_own, best_opp, best_card = scored[0]
    log.info("  🃏 Draft: %s (eigen=%.1f, gefahr=%.1f)",
             best_card["name"], best_own, best_opp)
    return best_card["name"]


def _choose_removal_target(cards: list, state: dict, title: str) -> str:
    """Ziel-Wahl bei 'Select card to remove N <resource>'. Server bietet Karten BEIDER
    Spieler an, traegt aber keinen Besitzer -> Besitz ueber eigene playedCards bestimmen.
    Regel: Gegnerkarte bevorzugen (ihm die Ressource nehmen), dort die mit hoechster
    vp_dyn-Dichte (denied VP). Nur wenn ausschliesslich eigene Karten waehlbar sind, die
    am wenigsten schaedliche (Fuel-Mikroben/Dichte 0 vor eigenen VP-Mikroben). Behebt den
    Ants-Selbstschaden: vorher lief das ueber choose_draft_card (= 'beste behalten')."""
    def _name(c): return c.get("name") if isinstance(c, dict) else c
    def _density(c):
        vd = (CARD_DB.get(_name(c), {}) or {}).get("vp_dyn") or {}
        return (vd.get("each", 0) / vd.get("per", 1)) if vd.get("kind") == "resources" else 0.0
    own = {(_name(c)) for c in (state.get("thisPlayer", {}).get("tableau") or state.get("thisPlayer", {}).get("playedCards") or [])
           if (_name(c))}
    opp = [c for c in cards if _name(c) not in own]
    if opp:
        chosen = max(opp, key=_density)            # dem Gegner die wertvollste Ressource nehmen
        log.info("  🎯 Entfern-Ziel: '%s' (Gegner)", _name(chosen))
    else:
        chosen = min(cards, key=_density)          # gezwungen -> eigene Fuel-Karte zuerst
        log.info("  🎯 Entfern-Ziel: '%s' (selbst, gezwungen)", _name(chosen))
    return _name(chosen)


_REMOVAL_RES = ("microbe", "animal", "plant", "resource", "floater", "science",
                "data", "fighter", "asset", "camp", "fleet", "preservation")


def handle_card(state: dict) -> dict:
    waiting = state["waitingFor"]
    cards   = waiting.get("cards", [])
    raw_title = waiting.get("title", "")
    min_c   = waiting.get("min", 1)
    max_c   = waiting.get("max", min_c)

    # Wenn title ein dict ist: prüfe ob es ein Draft (type=2 = Kartenrichtung)
    # oder ein echter and-Typ (type=1=Amount, type=3=Space) ist
    if isinstance(raw_title, dict):
        data = raw_title.get("data", [])
        types_in_data = {item.get("type") for item in data}
        # type=2 mit value=Farbe = Draft-Richtungsinfo → als normaler Draft behandeln
        if types_in_data == {2}:
            # Karten sind in waiting.cards
            if cards:
                chosen = choose_draft_card(cards, state)
                log.info("  📋 Draft (and-format): behalte '%s' aus %d Karten",
                         chosen, len(cards))
                return {"type": "card", "runId": state["runId"], "cards": [chosen]}
            return {"type": "card", "runId": state["runId"], "cards": []}        
        # type=1 = Amount. ACHTUNG: Steht zusätzlich ein type=0 (Ressourcenname) im data,
        # ist der Wert die RESSOURCEN-Menge (z.B. "add 2 Microbe"), NICHT die Kartenzahl –
        # die ergibt sich dann aus min/max (oft genau 1 Zielkarte). Sonst (reines "keep N")
        # ist der Wert die Kartenzahl. In jedem Fall auf [min, max] begrenzen.
        if 1 in types_in_data and cards:
            val = int(next(
                (item.get("value", "1") for item in data if item.get("type") == 1), "1"
            ))
            n = max_c if (0 in types_in_data) else val
            n = max(min_c, min(n, max_c))
            scored = sorted(cards, key=lambda c: score_card(c, state), reverse=True)
            chosen = [c["name"] for c in scored[:n]]
            log.info("  📋 Kartenauswahl (%d/%d, min=%d max=%d): %s",
                     n, len(cards), min_c, max_c, chosen)
            return {"type": "card", "runId": state["runId"], "cards": chosen}
        # Andere dict-Typen → and-Typ
        return handle_and(state)

    title = str(raw_title).lower()

    if not cards:
        return {"type": "card", "runId": state["runId"], "cards": []}

    # Kartenkauf (Research-Phase)
    if any(k in title for k in ("buy", "select cards to buy", "research")):
        chosen = choose_cards_to_buy(cards, state)
        return {"type": "card", "runId": state["runId"], "cards": chosen}

    # Entfern-Ziel (z.B. Ants: "Select card to remove 1 Microbe(s)"): NICHT als Draft
    # behandeln - sonst entfernt der Bot von seiner eigenen besten Karte. Gegner bevorzugen.
    if "remove" in title and any(r in title for r in _REMOVAL_RES):
        chosen = _choose_removal_target(cards, state, title)
        return {"type": "card", "runId": state["runId"], "cards": [chosen]}

    # Draft: genau eine Karte auswählen und behalten
    if any(k in title for k in ("draft", "keep", "select a card")) or (
        min_c == 1 and max_c == 1 and len(cards) > 1
    ):
        chosen = choose_draft_card(cards, state)
        log.info("  📋 Draft: behalte '%s' aus %d Karten", chosen, len(cards))
        return {"type": "card", "runId": state["runId"], "cards": [chosen]}

    # Prelude spielen
    if "prelude" in title:
        scored = sorted(cards, key=lambda c: score_card(c, state), reverse=True)
        return {"type": "card", "runId": state["runId"], "cards": [scored[0]["name"]]}

    # Generisch: min_c beste Karten
    if min_c > 0:
        scored = sorted(cards, key=lambda c: score_card_to_buy(c, state), reverse=True)
        chosen = [c["name"] for c in scored[:min_c]]
        return {"type": "card", "runId": state["runId"], "cards": chosen}

    return {"type": "card", "runId": state["runId"], "cards": []}


# ACTIVE-Karten-Namen die eine Aktion haben (aus card_db)
# ---------------------------------------------------------------------------
# Meilenstein- und Award-Bewertung
# ---------------------------------------------------------------------------

# Meilensteine kosten 8 MC und geben 5 VP = 25M Wert
# Nur claimen wenn: genug MC, noch nicht geclaimed, Bot führt klar
# Alle bekannten Meilenstein-Namen (aus TM-Quellcode MilestoneName.ts)
# Der Server schickt nur den Namen als Option-Titel (z.B. "Gardener"), kein "milestone" im Titel
MILESTONE_NAMES = {
    # Tharsis
    "terraformer", "mayor", "gardener", "planner", "builder",
    # Elysium
    "generalist", "specialist", "ecologist", "tycoon", "legend",
    # Hellas
    "diversifier", "tactician", "polar explorer", "energizer", "rim settler",
    # Venus / Ares / Moon
    "hoverlord", "networker", "one giant step", "lunarchitect",
    # Andere Boards
    "colonizer", "minimalist", "terran", "tropicalist",
    "economizer", "pioneer", "land specialist", "martian",
    "t. collector", "firestarter", "terra pioneer", "spacefarer", "gambler",
    "architect", "coastguard", "c. forester",
    "v. electrician", "smith", "tradesman", "irrigator", "capitalist",
    "agronomist", "engineer", "v. spacefarer", "geologist", "farmer",
    "tunneler", "risktaker", "purifier",
    # Modular
    "briber", "builder7", "forester", "fundraiser", "hydrologist",
    "landshaper", "legend4", "lobbyist", "merchant", "metallurgist",
    "philantropist", "pioneer4", "planetologist", "producer", "researcher",
    "spacefarer4", "sponsor", "tactician4", "terraformer29", "terran5",
    "thawer", "trader", "tycoon10",
}

_MILESTONE_COST = 8


def _count_event_type(cards) -> int:
    """Zaehlt Karten vom TYP EVENT (fuer Legend). NICHT der 'event'-Tag (nur 10 Karten) -
    Legend zaehlt gespielte Events = Typ EVENT (142 Karten)."""
    n = 0
    for c in cards or []:
        nm = c.get("name") if isinstance(c, dict) else c
        if ((CARD_DB.get(nm, {}) or {}).get("type") or "").upper() == "EVENT":
            n += 1
    return n


def _count_req_cards(cards) -> int:
    """Zaehlt Karten mit Requirement IN PLAY (fuer Tactician-Meilenstein): nicht-leeres
    'requirements' UND Typ != EVENT. Events sind nach dem Ausspielen nicht 'in play',
    zaehlen also nicht. 'requirements' ist die vollstaendige Req-Quelle (Tags, Global-
    parameter, Produktion, Erweiterungen) - kein Sonderfall fuer req_tags/req_prod noetig."""
    n = 0
    for c in cards or []:
        nm = c.get("name") if isinstance(c, dict) else c
        info = CARD_DB.get(nm, {}) or {}
        if info.get("requirements") and info.get("type") != "EVENT":
            n += 1
    return n


# Karten, die Floater-Ressourcen halten (aus TS: resourceType CardResource.FLOATER) -
# fuer die Hoverlord-Meilenstein-Zaehlung (7 Floater auf Karten).
_FLOATER_CARDS = {
    "Cloud Vortex Outpost", "Floating Refinery", "Floating Trade Hub", "Cloud Tourism",
    "Saturn Surfing", "Weather Balloons", "Robin Haulings", "Floating Habs",
    "Deuterium Export", "Extractor Balloons", "Dirigibles", "Local Shading", "Celestic",
    "Stratopolis", "Jet Stream Microscrappers", "Forced Precipitation", "Aerial Mappers",
    "Titan Air-scrapping", "Atmo Collectors", "Jovian Lanterns", "Titan Floating Launch-pad",
    "Stormcraft Incorporated", "Red Spot Observatory", "Jupiter Floating Station",
    "Titan Shuttles",
}


def _player_stats(state: dict) -> dict:
    """
    Berechnet spielerbezogene Statistiken die für Meilenstein- und
    Award-Bewertung benötigt werden (Tiles, Tags, Karten, Produktion).
    """
    player   = state["thisPlayer"]
    game     = state.get("game", {})
    spaces   = game.get("spaces", [])
    my_color = player.get("color")

    # Tiles zählen
    own_cities    = 0
    own_greeneries = 0
    for s in spaces:
        tile = s.get("tile")
        if not tile:
            continue
        pid = tile.get("playerId") or s.get("playerId")
        if pid != my_color:
            continue
        t = tile.get("tileType", -1)
        if t == 0:    own_greeneries += 1   # Greenery
        elif t == 1:  own_cities     += 1   # City

    # Gespielte Karten + Tags aus card_db
    played = player.get("tableau") or player.get("playedCards") or []
    tag_counts: dict[str, int] = {}
    for c in played:
        name = c.get("name", "") if isinstance(c, dict) else c
        info = CARD_DB.get(name, {})
        for tag in info.get("tags", []):
            t = tag.lower()
            tag_counts[t] = tag_counts.get(t, 0) + 1

    # Handkarten
    hand_size = len(player.get("cardsInHand", []))

    # Floater-Ressourcen auf Karten (fuer Hoverlord-Meilenstein). Das Tableau liefert pro
    # Karte 'resources' (Anzahl); Floater-Halter kommen aus _FLOATER_CARDS (aus TS extrahiert).
    floaters = 0
    for c in played:
        nm = c.get("name", "") if isinstance(c, dict) else c
        if nm in _FLOATER_CARDS and isinstance(c, dict):
            floaters += c.get("resources", 0) or 0

    # Ares-Meilenstein-Zaehler (Networker = Tiles neben Bonus-Tiles gelegt; Purifier =
    # entfernte Hazards) aus game.aresData.milestoneResults, per Spieler-id zugeordnet.
    networker = purifier = 0
    _mid = player.get("id")
    for _e in ((game.get("aresData") or {}).get("milestoneResults") or []):
        if _e.get("id") == _mid:
            networker = _e.get("networkerCount", 0) or 0
            purifier  = _e.get("purifierCount", 0) or 0
            break

    # Produktionen
    prods = {
        "mc":       player.get("megacreditProduction", 0),
        "steel":    player.get("steelProduction", 0),
        "titanium": player.get("titaniumProduction", 0),
        "plant":    player.get("plantProduction", 0),
        "energy":   player.get("energyProduction", 0),
        "heat":     player.get("heatProduction", 0),
    }

    # Turmoil: eigener EINFLUSS (TS getInfluence): +1 Chairman; in der DOMINANTEN Partei
    # +1 als Leader (+1 extra bei >1 Delegat) bzw. +1 als Nicht-Leader mit >=1 Delegat.
    _turm = game.get("turmoil") or {}
    influence = 0
    if _turm:
        _me = player.get("color")
        if _turm.get("chairman") == _me:
            influence += 1
        _parties = _turm.get("parties") or []
        if _parties:
            _dom = max(_parties, key=lambda p: len(p.get("delegates") or []))
            _n = sum(1 for d in (_dom.get("delegates") or [])
                     if (d.get("color") if isinstance(d, dict) else d) == _me)
            if _dom.get("partyLeader") == _me:
                influence += 1 + (1 if _n > 1 else 0)
            elif _n > 0:
                influence += 1
    # Blaue (ACTIVE) Karten + eigene Kolonien - fuer die Global-Event-Bewertung
    blue_cards = sum(1 for c in played
                     if str((card_info(c.get("name") if isinstance(c, dict) else c) or {})
                            .get("type", "")).upper() == "ACTIVE")
    n_colonies = sum(1 for c in (game.get("colonies") or [])
                     if player.get("color") in (c.get("colonies") or []))

    # Fuer die Fan-/Modular-Meilensteine (Trader/Tradesman, Farmer, Lobbyist):
    #   res_types = Anzahl VERSCHIEDENER Nicht-Standard-Ressourcentypen auf eigenen Karten
    #   bio_res   = Mikroben + Tiere auf eigenen Karten
    #   delegates = eigene Delegaten im Turmoil-Kongress (alle Parteien)
    _res_types, _bio = set(), 0
    for c in played:
        if not isinstance(c, dict):
            continue
        n = c.get("resources", 0) or 0
        if n <= 0:
            continue
        rt = str((c.get("resourceType") or
                  (card_info(c.get("name")) or {}).get("resourceType") or "")).upper()
        if rt:
            _res_types.add(rt)
        if rt in ("MICROBE", "ANIMAL"):
            _bio += n
    _delegates = 0
    if _turm:
        for _p in (_turm.get("parties") or []):
            _delegates += sum(1 for d in (_p.get("delegates") or [])
                              if (d.get("color") if isinstance(d, dict) else d)
                              == player.get("color"))

    return {
        "cities":      own_cities,
        "greeneries":  own_greeneries,
        "tags":        tag_counts,
        "hand":        hand_size,
        "played":      len(played),
        "events":      _count_event_type(played),
        "req_cards":   _count_req_cards(played),
        "tr":          player.get("terraformRating", 14),
        "prods":       prods,
        "mc":          player.get("megacredits", 0),
        "steel":       player.get("steel", 0),
        "titanium":    player.get("titanium", 0),
        "heat":        player.get("heat", 0),
        "floaters":    floaters,
        "networker":   networker,
        "purifier":    purifier,
        # Turmoil aktiv? -> Terraformer-Meilenstein braucht nur 26 TR statt 35
        "turmoil":     bool(game.get("turmoil")),
        "influence":   influence,
        "res_types":   len(_res_types),
        "bio_res":     _bio,
        "delegates":   _delegates,
        "blue_cards":  blue_cards,
        "colonies":    n_colonies,
        "plants":      player.get("plants", 0),
    }


def _milestone_gap(title: str, stats: dict) -> int:
    """
    Schritte bis zur Erfüllung eines Meilensteins, rein aus Statistiken berechnet –
    nutzbar für eigene UND gegnerische Spieler (für die Wegschnapp-Abschätzung).
    0 = erfüllt. Unbekannte Meilensteine → 1 (konservativ).
    """
    t     = title.lower()
    tags  = stats.get("tags", {})
    prods = stats.get("prods", {})
    # Verschiedene Tags = Typen mit count > 0 (der players-State listet alle Typen,
    # auch solche mit 0 – daher nicht len(tags) verwenden).
    distinct_tags = sum(1 for v in tags.values() if v > 0)

    # ── Tharsis ──
    # Terraformer: Schwelle 35, MIT TURMOIL nur 26 (TS: Terraformer.terraformRatingTurmoil).
    # Ohne diese Anpassung unterschaetzt der Bot seine Naehe massiv und verfolgt den
    # Meilenstein in Turmoil-Partien nie.
    if   t == "terraformer":
        _need = 26 if stats.get("turmoil") else 35
        return max(0, _need - stats.get("tr", 0))
    elif t == "mayor":          return max(0, 3 - stats.get("cities", 0))
    elif t == "gardener":       return max(0, 3 - stats.get("greeneries", 0))
    elif t == "builder":        return max(0, 8 - tags.get("building", 0))
    elif t == "planner":        return max(0, 16 - stats.get("hand", 0))
    # ── Elysium ──
    elif t == "generalist":     return sum(1 for v in prods.values() if v < 1)
    elif t == "specialist":     return max(0, 10 - max(prods.values(), default=0))
    elif t == "ecologist":
        bio = tags.get("plant", 0) + tags.get("microbe", 0) + tags.get("animal", 0)
        return max(0, 4 - bio)
    elif t == "tycoon":         return max(0, 15 - (stats.get("played", 0) - stats.get("events", 0)))
    elif t == "legend":         return max(0, 5 - stats.get("events", 0))
    # ── Hellas ──
    elif t == "diversifier":    return max(0, 8 - distinct_tags)
    elif t == "tactician":      return max(0, 5 - stats.get("req_cards", 0))
    elif t == "polar explorer":
        # NAEHERUNG: korrekt waeren nur Tiles in den untersten zwei Reihen; ohne Brett-
        # geometrie in stats zaehlen wir alle eigenen Tiles -> ueberschaetzt (gap zu klein).
        return max(0, 3 - (stats.get("cities", 0) + stats.get("greeneries", 0)))
    elif t == "energizer":      return max(0, 6 - prods.get("energy", 0))
    elif t == "rim settler":    return max(0, 3 - tags.get("jovian", 0))
    # ── Erweiterungs-/Modular-Meilensteine (randomMA), aus Stats berechenbar ──
    elif t == "agronomist":     return max(0, 4 - tags.get("plant", 0))       # 4 Pflanzen-Tags
    elif t == "architect":      return max(0, 3 - tags.get("city", 0))        # 3 Stadt-Tags
    elif t == "builder7":       return max(0, 7 - tags.get("building", 0))    # 7 Building-Tags
    elif t == "capitalist":     return max(0, 64 - stats.get("mc", 0))        # 64 M€
    elif t == "v. electrician": return max(0, 4 - tags.get("power", 0))       # 4 Power-Tags
    elif t == "v. spacefarer":  return max(0, 4 - tags.get("space", 0))       # 4 Space-Tags
    elif t in ("metallurgist", "smith"):                                       # 6 Stahl+Titan-Prod
        return max(0, 6 - (prods.get("steel", 0) + prods.get("titanium", 0)))
    elif t == "hoverlord":      return max(0, 7 - stats.get("floaters", 0))     # 7 Floater (Venus)
    elif t == "networker":      return max(0, 3 - stats.get("networker", 0))    # 3 Tiles neben Bonus-Tiles (Ares)
    elif t == "purifier":       return max(0, 3 - stats.get("purifier", 0))     # 3 Hazards entfernt (Ares)
    # ── FAN-/MODULAR-Meilensteine (Include Fan MA + randomMA). Aus dem TS-Manifest
    #    (milestones/modular/) extrahiert; nur die, die aus vorhandenen Stats berechenbar sind.
    elif t == "engineer":       return max(0, 10 - (prods.get("energy", 0) + prods.get("heat", 0)))
    elif t == "producer":       return max(0, 16 - sum(prods.values()))          # Gesamtproduktion
    elif t == "researcher":     return max(0, 4 - tags.get("science", 0))
    elif t == "terran5":        return max(0, 5 - tags.get("earth", 0))
    elif t == "terraformer29":  return max(0, 29 - stats.get("tr", 0))
    elif t == "tycoon10":       return max(0, 10 - stats.get("played", 0))
    elif t == "tactician4":     return max(0, 4 - stats.get("req_cards", 0))
    elif t == "legend4":        return max(0, 4 - stats.get("events", 0))
    elif t == "capitalist":     return max(0, 64 - stats.get("mc", 0))
    elif t == "fundraiser":     return max(0, 12 - stats.get("mc", 0))
    elif t in ("trader", "tradesman"):                                            # 3 versch. Nicht-Standard-Ressourcen
        return max(0, 3 - stats.get("res_types", 0))
    elif t == "farmer":         return max(0, 5 - stats.get("bio_res", 0))        # Mikroben + Tiere
    elif t == "lobbyist":       return max(0, 7 - stats.get("delegates", 0))      # Turmoil
    # ── Sonstige / unbekannt ──
    else:                       return 1


def _opponent_stats(state: dict) -> list[dict]:
    """
    Vereinfachte Statistiken aller Gegner aus state["players"] (öffentliche Daten),
    im selben Format wie _player_stats – für _milestone_gap der Gegner.
    """
    me  = state.get("thisPlayer", {}).get("color")
    game = state.get("game", {})
    out = []
    for p in state.get("players", []):
        if p.get("color") == me:
            continue
        vp = p.get("victoryPointsBreakdown", {})
        _fl = 0
        for _c in p.get("tableau", []):
            _nm = _c.get("name", "") if isinstance(_c, dict) else _c
            if _nm in _FLOATER_CARDS and isinstance(_c, dict):
                _fl += _c.get("resources", 0) or 0
        _nw = _pu = 0
        for _e in ((game.get("aresData") or {}).get("milestoneResults") or []):
            if _e.get("id") == p.get("id"):
                _nw = _e.get("networkerCount", 0) or 0
                _pu = _e.get("purifierCount", 0) or 0
                break
        out.append({
            "tr":         p.get("terraformRating", 0),
            "cities":     p.get("citiesCount", 0),
            "greeneries": vp.get("greenery", 0),
            "tags":       p.get("tags", {}),
            "hand":       p.get("cardsInHandNbr", 0),
            "heat":       p.get("heat", 0),
            "steel":      p.get("steel", 0),
            "titanium":   p.get("titanium", 0),
            "mc":         p.get("megacredits", 0),
            "played":     len(p.get("tableau", [])),
            "events":     _count_event_type(p.get("tableau", [])),
            "req_cards":  _count_req_cards(p.get("tableau", [])),
            "prods": {
                "mc":       p.get("megacreditProduction", 0),
                "steel":    p.get("steelProduction", 0),
                "titanium": p.get("titaniumProduction", 0),
                "plant":    p.get("plantProduction", 0),
                "energy":   p.get("energyProduction", 0),
                "heat":     p.get("heatProduction", 0),
            },
            "floaters":   _fl,
            "networker":  _nw,
            "purifier":   _pu,
            "turmoil":    bool(game.get("turmoil")),
        })
    return out


def _milestone_state(state: dict) -> tuple:
    """Robust aus game["milestones"] (NICHT claimedMilestones, das ist None): liefert
    (verfuegbare_namen, claimed_count, eigene_claims, frei_slots). Ein Eintrag mit
    'color'/'playerName' ist beansprucht. Globaler Deckel: max 3 beansprucht insgesamt."""
    game = state.get("game", {})
    me   = state.get("thisPlayer", {}).get("color")
    ms   = game.get("milestones", []) or []
    avail = []; claimed = 0; mine = 0
    for m in ms:
        nm = (m.get("name") or "").strip()
        owner = m.get("color") or m.get("playerName")
        if owner:
            claimed += 1
            if m.get("color") == me or m.get("playerName") == state.get("thisPlayer", {}).get("name"):
                mine += 1
        else:
            avail.append(nm)
    return avail, claimed, mine, max(0, 3 - claimed)


def _milestone_pursuit(state: dict):
    """Pursue-Logik: waehlt den EINEN Meilenstein, den der Bot am billigsten und mit
    Frontrunner-Vorsprung erreichen kann (gap 2..PURSUE_MAX). Rennen-bewusst (eigener gap
    <= schnellster Gegner), slot-bewusst (Deckel) und in Reichweite. Liefert (name_lower,
    gap, net) oder None. Memoisiert."""
    if not LEVER_MILESTONE:
        return None
    if "_ms_pursuit" in state:
        return state["_ms_pursuit"]
    avail, claimed, mine, free = _milestone_state(state)
    res = None
    if free >= 1 and avail:
        stats = _player_stats(state)
        opps  = _opponent_stats(state)
        tl    = turns_left(state)
        best  = None
        for nm in avail:
            g = _milestone_gap(nm, stats)
            if not (1 <= g <= PURSUE_MAX_GAP):
                continue
            opp_g = min([_milestone_gap(nm, o) for o in opps], default=99)
            if g > opp_g:                 # nicht Frontrunner -> nicht verfolgen
                continue
            if g > tl - 1:                # nicht rechtzeitig erreichbar
                continue
            if best is None or g < best[1]:   # billigster (kleinster gap) gewinnt
                best = (nm.lower(), g)
        if best:
            res = (best[0], best[1], 25 - _MILESTONE_COST)
    state["_ms_pursuit"] = res
    return res


# Welche score_action-Aktion schliesst welchen Meilenstein-gap? Bewusst nur die
# spezifischen, alignment-sauberen Faelle - NICHT Terraformer (das waere ein Freibrief
# fuer SP-Spam und genau der Tunnelblick/Ineffizienz, die wir vermeiden wollen).
_MILESTONE_PURSUE_ACTIONS = {
    "gardener": {"greenery", "greenery_sp"},
    "mayor":    {"city_sp"},
}


def _milestone_action_bonus(action_type: str, state: dict) -> float:
    """Kleiner Bonus auf eine Aktion, die den gap zum aktuell verfolgten Meilenstein
    schliesst. Bewusst klein (kippt gute Zuege, rechtfertigt keine schlechten) und faellt
    mit kleinerem gap groesser aus (naeher am Abschluss = lohnender)."""
    pur = _milestone_pursuit(state)
    if not pur:
        return 0.0
    name, gap, net = pur
    if action_type in _MILESTONE_PURSUE_ACTIONS.get(name, ()):
        return min(PURSUE_BONUS_CAP, (net / max(1, gap)) * PURSUE_WEIGHT)
    return 0.0


def _milestone_complete_bonus(action_type: str, state: dict) -> float:
    """Starkes Signal: eine Aktion, die einen Meilenstein DIESEN Zug abschliesst
    (Bot-gap == 1, unclaimed, freier Slot) -> voller Meilenstein-Nettowert. Bewusst
    RENNEN-AGNOSTISCH (bei unclaimed Meilenstein zaehlt nur, wer zuerst claimt - der
    Gegner kann qualifiziert sein, ohne geclaimt zu haben) und NUR gap 1 (kein Chasing
    ferner Meilensteine -> vermeidet die alte gap-basierte Self-Play-Regression)."""
    if not LEVER_MS_COMPLETE:
        return 0.0
    avail, claimed, mine, free = _milestone_state(state)
    if free < 1 or not avail:
        return 0.0
    stats = _player_stats(state)
    for nm in avail:
        if action_type in _MILESTONE_PURSUE_ACTIONS.get(nm.lower(), ()):
            if _milestone_gap(nm, stats) == 1:
                return MS_COMPLETE_BONUS
    return 0.0


# ── Meilenstein-/Award-Ausrichtung (obs 5/10) ────────────────────────────────
# Ziel -> Karten-Eigenschaften, die es voranbringen. Bewusst nur saubere, in card_db
# pruefbare Mappings (Typ/Tag/Produktion/Tile/Kosten). Board-abhaengig - es greifen nur
# die Ziele, die tatsaechlich in-Play UND realistisch gewinnbar sind.
_TARGET_PROPS = {
    # Meilensteine
    "legend":      {"type:event"},
    "builder":     {"tag:building"},
    "ecologist":   {"tag:bio"},
    "gardener":    {"tile:greenery"},
    "mayor":       {"tile:city"},
    "energizer":   {"prod:energy"},
    "rim settler": {"tag:jovian"},
    # Awards (je nach Board)
    "scientist":   {"tag:science"},
    "banker":      {"prod:megacredits"},
    "thermalist":  {"prod:heat"},
    "miner":       {"prod:steel", "prod:titanium"},    # Proxy: Prod statt Ist-Ressourcen
    "industrialist": {"prod:steel", "prod:energy"},     # Proxy: Prod statt Ist-Ressourcen
    "celebrity":   {"cost:high"},                       # >=20 MC, NUR gruen/blau (kein Event)
    "space baron": {"tag:space"},
    "cultivator":  {"tile:greenery"},
    "contractor":  {"tag:building"},                    # meiste Building-Tags (Hellas)
    "landlord":    {"tile:any"},                        # meiste Tiles
    "excentric":   {"holds:resources"},                 # meiste Ressourcen auf Karten (Server: 'Excentric')
    "venuphile":   {"tag:venus"},
    "magnate":     {"type:automated"},
    # Bewusst NICHT gemappt (kein sauberer Kauf-Bias moeglich):
    #   benefactor (TR - zu breit, fast alles), desert settler / estate dealer
    #   (brett-positions-abhaengig: Suedhalbkugel / ozean-angrenzend - nicht aus card_db ableitbar)
}
_BIO_TAGS = {"plant", "microbe", "animal"}


def _alignment_targets(state: dict) -> set:
    """In-Play UND realistisch gewinnbare Ziele -> Menge favorisierter Karten-Eigenschaften.
    Meilenstein: nicht geclaimt, Slot frei, Bot Frontrunner (eigener gap <= bester Gegner)
    und in Reichweite (<= ALIGN_MAX_GAP). Award: Bot fuehrt oder <= ALIGN_AWARD_SLACK zurueck.
    So entstehen i.d.R. nur 1-3 Ziele - kein Tunnelblick auf abstrakt wertvolle Ziele."""
    if not LEVER_ALIGN:
        return set()
    if "_align_tgt" in state:
        return state["_align_tgt"]
    props = set()
    game  = state.get("game", {})
    stats = _player_stats(state)
    opps  = _opponent_stats(state)
    me    = state.get("thisPlayer", {}).get("color")
    avail, claimed, mine, free = _milestone_state(state)
    if free >= 1:
        for nm in avail:
            key = nm.lower()
            if key not in _TARGET_PROPS:
                continue
            g  = _milestone_gap(nm, stats)
            og = min([_milestone_gap(nm, o) for o in opps], default=99)
            if g <= og and g <= ALIGN_MAX_GAP:
                props |= _TARGET_PROPS[key]
    for aw in game.get("awards", []):
        nm = (aw.get("name") or "").lower()
        if nm not in _TARGET_PROPS:
            continue
        scores = {s.get("color"): s.get("score", 0) for s in aw.get("scores", [])}
        my   = scores.get(me, 0)
        best = max([v for c, v in scores.items() if c != me], default=0)
        if my >= best - ALIGN_AWARD_SLACK:
            props |= _TARGET_PROPS[nm]
    state["_align_tgt"] = props
    return props


def _card_advances_alignment(info: dict, cost: float, props: set) -> bool:
    """Hat die Karte eine der favorisierten Eigenschaften?"""
    if not props:
        return False
    tags = {t.lower() for t in info.get("tags", [])}
    typ  = (info.get("type") or "").lower()
    prod = info.get("production", {}) or {}
    for p in props:
        if p == "type:event"     and typ == "event":              return True
        if p == "type:automated" and typ == "automated":          return True
        if p == "tag:bio"        and (tags & _BIO_TAGS):          return True
        if p == "cost:high"      and cost >= 20 and typ != "event":  return True  # Celebrity: kein Event
        if p == "holds:resources" and (info.get("vp_dyn") or {}).get("kind") == "resources":
            return True
        if p.startswith("tag:"):
            tg = p[4:]
            if tg in tags:                                        return True
        if p.startswith("prod:"):
            pr = p[5:]
            if pr == "any" and any(v > 0 for v in prod.values()): return True
            if prod.get(pr, 0) > 0:                               return True
        if p.startswith("tile:"):
            tl = p[5:]
            if tl == "any" and (info.get("greenery") or info.get("city") or info.get("oceans")):
                return True
            if tl == "greenery" and info.get("greenery"):         return True
            if tl == "city"     and info.get("city"):             return True
    return False


def _alignment_buy_bonus(card: dict, state: dict) -> float:
    """Kleiner Kauf-/Keep-Bonus, wenn die Karte ein gewinnbares Ziel voranbringt."""
    if not LEVER_ALIGN:
        return 0.0
    props = _alignment_targets(state)
    if not props:
        return 0.0
    info = CARD_DB.get(card.get("name", ""), {})
    cost = card.get("calculatedCost", card.get("cost", 0)) or 0
    return ALIGN_BUY_BONUS if _card_advances_alignment(info, cost, props) else 0.0


# ── Enabler-Gate (obs 1/8) ───────────────────────────────────────────────────
_ENABLER_CARDS = {"Insulation", "Virus", "Protected Habitats"}


def _enabler_ok(name: str, state: dict) -> bool:
    """Hat die Combo-Karte ihren (nicht in card_db kodierten) Enabler? False -> abwerten."""
    p    = state.get("thisPlayer", {})
    prod = p.get("production", {}) or {}
    if name == "Insulation":
        return prod.get("heat", 0) > 0          # Waerme-Prod -> MC-Prod: ohne Waermeprod wertlos
    if name == "Virus":
        me = p.get("color")
        for o in state.get("players", []):       # Angriff: nur wertvoll, wenn Gegner Pflanzen/Tiere hat
            if o.get("color") != me and (o.get("plants", 0) > 0 or o.get("animals", 0) > 0):
                return True
        return False
    if name == "Protected Habitats":
        return False                             # Defensiv: in 2P gegen nicht-aggressiv ~ wertlos
    return True


def _score_milestone(title: str, state: dict, known_claimable: bool = False) -> float:
    """
    Bewertet ob der Bot diesen Meilenstein claimen sollte.

    Logik:
      1. Ist er bereits erfüllt? → Unbedingt claimen (hoher Score).
      2. Ist er 1 Schritt entfernt? → Claimen wenn es sich lohnt.
      3. Weiter entfernt → 0 (nicht claimen).

    Wert eines Meilensteins: 5 VP = 25 MC minus Kosten (8/14/20).
    BGG-Guide: Meilensteine sind fast immer gut wenn erreichbar.
    """
    player  = state["thisPlayer"]
    game    = state.get("game", {})
    mc      = player.get("megacredits", 0)

    if LEVER_MILESTONE:
        _avail, claimed_count, _mine, free_slots = _milestone_state(state)
        cost = _MILESTONE_COST                      # flach 8 (Server: MILESTONE_COST=8)
    else:
        _avail = []
        claimed_count = len(game.get("claimedMilestones", []) or [])
        free_slots = 3 - claimed_count
        cost = 8 + claimed_count * 6                 # altes (falsches) 8/14/20-Modell
    if claimed_count >= 3:
        return 0
    if mc < cost + MC_RESERVE:
        return 0

    tl = turns_left(state)
    if tl < 1:
        return 0

    stats = _player_stats(state)
    # Kommt der Titel aus der Server-Option "Claim a milestone", ist er
    # serverseitig bereits als claimbar gefiltert -> gap=0 statt eigener Schaetzung
    gap   = 0 if known_claimable else _milestone_gap(title, stats)

    # ── Bewertung ─────────────────────────────────────────────────────────────
    if gap == 0:
        # Erfüllt → claimbar. Basiswert = 25 MC (5 VP) minus Kosten.
        net = 25 - cost
        opp_gaps = [_milestone_gap(title, o) for o in _opponent_stats(state)]
        opp_gap  = min(opp_gaps) if opp_gaps else 99
        urgency = 0
        if opp_gap <= 0:
            urgency = 45   # Gegner kann denselben Meilenstein sofort claimen
        elif opp_gap <= 1:
            urgency = 35   # Gegner 1 Schritt entfernt
        elif claimed_count >= 2:
            urgency = 35   # letzter freier Slot – knapp, jetzt sichern
        if LEVER_MILESTONE and urgency < 35:
            # Fenster-bewusst: auch ohne Gegner am SELBEN Meilenstein sichern, wenn der
            # globale 3er-Deckel sich schliesst. Bedrohung = wie viele freie Meilensteine
            # ein Gegner JETZT in einem Schritt greifen koennte (er kann mehrere Slots
            # ueber die naechsten Zuege fuellen). Bleibt danach <=1 Slot, jetzt sichern.
            # Das war der Fehler in Spiel 1: Bot erfuellte Gardener, zoegerte, der Mensch
            # fuellte alle Slots mit anderen Meilensteinen.
            threats = sum(1 for o in _opponent_stats(state)
                          for a in _avail if _milestone_gap(a, o) <= 1)
            if free_slots - threats <= 1:
                urgency = 35
        log.info("   🏆 Meilenstein '%s' ERFÜLLT (cost=%d, net=%.0f, opp_gap=%d, dringend=%d)",
                 title, cost, net, opp_gap, urgency)
        return net + 10 + urgency
    elif gap == 1:
        # 1 Schritt entfernt: lohnt sich wenn noch Generationen übrig
        net = 20 - cost   # Leicht abgewertet wegen Unsicherheit
        urg = 0
        if LEVER_MILESTONE:
            opp_gap = min([_milestone_gap(title, o) for o in _opponent_stats(state)], default=99)
            if opp_gap <= 1:
                urg = 15   # Gegner ebenfalls am letzten Schritt -> nicht trödeln
        if net > 0 and tl >= 2:
            log.info("   🏆 Meilenstein '%s' fast erfüllt (gap=1, cost=%d)", title, cost)
            return net + urg
        return 0
    else:
        return 0


# Distinktive Award-Namen-Bestandteile (Tharsis / Elysium / Hellas). Dienen sowohl
# der Erkennung der Funding-Option als auch der Wert-/Schwellen-Zuordnung.
_AWARD_KEYS = (
    # Basis + Erweiterungen (aus TS: server/awards/) - der Bot bewertet Awards ueber die
    # SERVER-SCORES (game.awards[].scores), braucht die Namen aber, um die Funding-OPTION
    # ueberhaupt als Award zu ERKENNEN. Fehlte ein Name, landete die Option im Pass-Fallback
    # und der Award wurde NIE gefunded (betraf u.a. Venuphile und ALLE Fan-Awards).
    "landlord", "banker", "scientist", "thermalist", "miner",
    "celebrity", "entrepreneur", "desert settler", "estate dealer", "benefactor",
    "contractor", "cultivator", "excentric", "magnate", "space baron", "rim contractor",
    "venuphile", "blacksmith", "industrialist", "naturalist", "voyager", "visionary",
    "forecaster", "edgedancer",
    # FAN / modular (server/awards/modular/) - kommen mit "Include Fan Milestones/Awards"
    "administrator", "collector", "constructor", "electrician", "founder", "highlander",
    "incorporator", "investor", "landscaper", "manufacturer", "metropolist", "mogul",
    "politician", "suburbian", "traveller",
)

_AWARD_THRESHOLDS = {
    "landlord": 4, "banker": 8, "scientist": 4, "thermalist": 8, "miner": 8,
    "celebrity": 8, "entrepreneur": 4, "desert settler": 3, "estate dealer": 4,
    "benefactor": 40, "contractor": 8, "cultivator": 4, "excentric": 30,
    "magnate": 8, "space baron": 6, "rim contractor": 3,
}


def _is_award_option(title: str) -> bool:
    """Erkennt eine Award-Funding-Option am enthaltenen Award-Namen.
    Robust gegen Titelvarianten ('Landlord' wie 'Fund Landlord award').
    ABER: Turmoil-Ruling-Policies enthalten Parteinamen, die sich mit Award-Namen ueberschneiden
    ('Pay 10 M€ to draw 3 cards (Turmoil SCIENTISTS)' enthaelt 'Scientist' -> war ein False
    Positive und wurde faelschlich als Award-Funding behandelt)."""
    t = title.lower()
    if "turmoil" in t:
        return False
    return any(k in t for k in _AWARD_KEYS)


def _is_fund_award_option(opt: dict) -> bool:
    """Erkennt die verschachtelte 'Fund an award'-Auswahl: eine äußere or-Option,
    deren Titel ein Message-Objekt ist und deren Sub-Optionen die einzelnen
    Awards sind (Scientist/Banker/…)."""
    if opt.get("type") != "or":
        return False
    title = opt.get("title", "")
    msg = title.get("message", "") if isinstance(title, dict) else str(title)
    if "fund" in msg.lower() and "award" in msg.lower():
        return True
    # Fallback: Sub-Optionen tragen Award-Namen
    subs = opt.get("options", [])
    return bool(subs) and any(_is_award_option(str(s.get("title", ""))) for s in subs)


def _award_value(title: str, stats: dict) -> int:
    """Wert eines Spielers in der Award-Kategorie – für eigene UND gegnerische
    Bewertung (Award-Ränge entscheiden sich relativ zum Gegner). -1 = unbekannt."""
    t     = title.lower()
    tags  = stats.get("tags", {})
    prods = stats.get("prods", {})
    if   "landlord" in t:       return stats.get("cities", 0) + stats.get("greeneries", 0)
    elif "banker" in t:         return prods.get("mc", 0)
    elif "scientist" in t:      return tags.get("science", 0)
    elif "thermalist" in t:     return stats.get("heat", 0)
    elif "miner" in t:          return stats.get("steel", 0) + stats.get("titanium", 0)
    elif "celebrity" in t:      return stats.get("played", 0)
    elif "entrepreneur" in t:   return tags.get("earth", 0)
    elif "desert settler" in t: return stats.get("cities", 0)
    elif "estate dealer" in t:  return stats.get("greeneries", 0)
    elif "benefactor" in t:     return stats.get("tr", 0)
    elif "contractor" in t:     return tags.get("building", 0)
    elif "cultivator" in t:     return stats.get("greeneries", 0)
    elif "excentric" in t:      return stats.get("mc", 0)
    elif "magnate" in t:        return prods.get("mc", 0)
    elif "space baron" in t:    return tags.get("space", 0) + tags.get("jovian", 0)
    elif "rim contractor" in t: return tags.get("jovian", 0)
    else:                       return -1


def _award_threshold(title: str) -> int:
    """Grober Reifegrad-Schwellwert der Kategorie (für die Vorsprung-Sicherheit)."""
    t = title.lower()
    for k, v in _AWARD_THRESHOLDS.items():
        if k in t:
            return v
    return 5


def _award_progress_bonus(info: dict, state: dict) -> float:
    """
    Zuschlag fuer Karten, die die eigene Metrik eines bereits GEFUNDETEN
    Awards verbessern (ein gefundeter Award ist ein 5-VP-Rennen; gemessen:
    Banker gefunded und dann nicht bedient -> 5 VP an den Gegner).
    Nur wenn das Rennen noch offen ist (eigener Stand >= opp_max - 2).
    """
    game = state.get("game", {})
    funded = [a for a in game.get("awards", []) if a.get("playerName")]
    if not funded:
        return 0.0

    my_color = state.get("thisPlayer", {}).get("color")
    tl   = turns_left(state)
    prod = info.get("production", {}) or {}
    stock = info.get("stock", {}) or {}
    tags = [str(t).lower() for t in info.get("tags", [])]

    def metric_delta(name: str) -> float:
        n = name.lower()
        if "banker" in n or "magnate" in n:
            return max(0, prod.get("megacredits", 0))
        if "miner" in n:
            return (stock.get("steel", 0) + stock.get("titanium", 0)
                    + (max(0, prod.get("steel", 0)) + max(0, prod.get("titanium", 0))) * min(3, tl))
        if "thermalist" in n:
            return stock.get("heat", 0) + max(0, prod.get("heat", 0)) * min(3, tl)
        if "scientist" in n:
            return tags.count("science")
        if "landlord" in n or "cultivator" in n or "estate dealer" in n:
            return info.get("greenery", 0) + info.get("city", 0)
        return 0.0

    bonus = 0.0
    for a in funded:
        own, opp_max = None, 0
        for s in a.get("scores", []):
            if s.get("color") == my_color:
                own = s.get("score", 0)
            else:
                opp_max = max(opp_max, s.get("score", 0))
        if own is None or own < opp_max - 2:
            continue   # Rennen verloren -> Metrik nicht mehr jagen
        bonus += metric_delta(a.get("name", "")) * 2.0
    return min(10.0, bonus)


def _award_scores_from_server(title: str, state: dict):
    """(own, opp_max) aus game.awards[].scores - der Server zaehlt jede
    Award-Metrik selbst (board-agnostisch, jede Erweiterung). None wenn der
    Award dort nicht gefunden wird (-> Fallback auf die Namens-Heuristik)."""
    t = title.lower()
    my_color = state.get("thisPlayer", {}).get("color")
    for a in state.get("game", {}).get("awards", []):
        if a.get("name", "").lower() in t or t in a.get("name", "").lower():
            own, opp_max = None, 0
            for s in a.get("scores", []):
                if s.get("color") == my_color:
                    own = s.get("score", 0)
                else:
                    opp_max = max(opp_max, s.get("score", 0))
            if own is not None:
                return own, opp_max
    return None


def _score_award(title: str, state: dict) -> float:
    """
    Bewertet ob der Bot diesen Award sponsern sollte.

    BGG-Guide: Awards lohnen sich nur wenn man in der Kategorie führt
    oder der Gegner noch weit entfernt ist. Kosten 8/14/20 MC.
    Erwartungswert: 5 VP (25 MC) wenn 1., 2 VP (10 MC) wenn 2.

    Strategie: nur sponsern wenn eigener Wert >= 60% des Schwellwerts
    für den 1. Platz (grobe Schätzung).
    """
    player = state["thisPlayer"]
    game   = state.get("game", {})
    mc     = player.get("megacredits", 0)

    funded     = game.get("fundedAwards", [])
    fund_count = len(funded)
    if fund_count >= 3:
        return 0

    cost = 8 + fund_count * 6
    if mc < cost + MC_RESERVE + 5:
        return 0

    tl = turns_left(state)
    if tl < 0:
        return 0   # Spiel vorbei

    # Stand bevorzugt vom SERVER (game.awards[].scores zaehlt jede Metrik
    # selbst, board-agnostisch); Namens-Heuristik nur als Fallback.
    sv = _award_scores_from_server(title, state)
    if sv is not None:
        own, opp_max = sv
    else:
        my_stats = _player_stats(state)
        own      = _award_value(title, my_stats)
        if own < 0:
            # Unbekannter Award (z.B. Expansion): sehr konservativer Score
            return max(0, 5 - cost)
        opp_vals = [_award_value(title, o) for o in _opponent_stats(state)]
        opp_max  = max(opp_vals) if opp_vals else 0
    lead = own - opp_max

    if own <= 0 or lead <= 0:
        return 0   # führt nicht / leere Kategorie → nicht funden

    # ZEIT-Konfidenz: Awards zaehlen am SPIELENDE. Ein Vorsprung ist nur so
    # viel wert, wie er ueber die Restgenerationen haltbar ist (gemessen:
    # Miner-Funding in Gen 1 bei lead=5, Banker-Geschenk in Gen 6).
    # ZEIT-Konfidenz: Awards zaehlen am SPIELENDE. Ein Vorsprung ist nur so
    # viel wert, wie er ueber die Restgenerationen haltbar ist. In der LETZTEN
    # Generation (tl<=1) ist spaet+fuehrend am SICHERSTEN -> nicht blocken,
    # aber Mindest-Vorsprung 2 verlangen (Gegner kann im letzten Zug noch ~1
    # Tile/Schritt aufholen). Frueherer Bug: tl<2 -> NIE gesponsert, selbst bei
    # grossem Vorsprung (z.B. Landlord +5 in der letzten Gen -> gepasst).
    floor       = 2 if tl <= 1 else 1
    need_tight  = max(floor, 1 + tl // 4)
    need_comf   = max(floor, 1 + tl // 2)
    if lead < need_tight:
        return 0   # Vorsprung ueber die Restzeit nicht verteidigbar
    comfortable = lead >= need_comf
    # Wert eines gefundeten Fuehrungs-Awards = realer Siegpunktwert, nicht der
    # alte 18/12-Deckel: 1. Platz = 5 VP (~25 MC). Bei nur knapp verteidigbarem
    # Vorsprung (tight, nicht comfortable) droht Platz 2 (2 VP) -> Erwartung
    # dazwischen (~16 MC). Die Fuehrungs- und Zeit-Gates oben verhindern weiter
    # das Zu-frueh-Sponsern; hier wird NUR der Wert eines bereits validen
    # Sponserings korrekt angesetzt (vorher massiv unterbewertet -> verlor jede
    # Runde gegen Kartenplays -> Fuehrungs-Awards blieben ungefundet).
    AWARD_VP_MC = 5.0
    gross = (5.0 if comfortable else 3.2) * AWARD_VP_MC   # 25 bzw. 16 MC
    net   = gross - cost
    # Now-or-never: ein sicher gefuehrter Award, der am Spielende ungefundet
    # bleibt, ist reiner VP-Verlust. In der letzten Generation anheben, damit das
    # Sponsern den letzten Zug gegen marginale Kartenplays gewinnt.
    if comfortable and tl <= 1:
        net *= 1.4
    log.info("   🥇 Award '%s' sponsern (own=%d, opp_max=%d, lead=%d, cost=%d, net=%.0f%s)",
             title, own, opp_max, lead, cost, net, ", komfortabel" if comfortable else "")
    return max(0, net)



def _get_active_card_names() -> set:
    """Gibt alle Namen von ACTIVE-Karten zurück die eine Aktion haben."""
    return {n for n, c in CARD_DB.items() if (c.get("type") or "").upper() == "ACTIVE"}

def _is_card_action(title: str) -> bool:
    """
    Erkennt ob ein Option-Titel eine Karten- oder Konzern-Aktion ist.
    TM-Server zeigt Aktionen als Option mit Titel = Kartenname oder
    "Use action of <Name>".
    """
    if not title:
        return False
    title_lower = title.lower()
    # Explizite Aktions-Phrasen
    if any(kw in title_lower for kw in ("use action", "action of", "activate", "use the action")):
        return True
    # Kartenname direkt als Titel (ACTIVE-Karte hat gleichen Namen wie Aktion)
    active_names_lower = {n.lower() for n in _get_active_card_names()}
    if title_lower in active_names_lower:
        return True
    return False


def _extract_msg_number(raw_title) -> int:
    """Erste ganzzahlig parsbare Zahl aus einem Message-Objekt
    ({"data": [{"type":..,"value":..}], "message": ..}) ziehen.
    Gegner-Namen u.Ä. sind nicht int-parsbar und werden übersprungen."""
    if isinstance(raw_title, dict):
        for d in raw_title.get("data", []):
            try:
                return int(d.get("value"))
            except (TypeError, ValueError):
                continue
    return 0


def _plant_attack_score(opt: dict):
    """Score für eine Option aus RemoveAnyPlants (removeAnyPlants-Effekt).
    Rückgabe:
      None  -> keine Pflanzen-Entfern-Option (anderer Zweig zuständig, z.B. Skip)
      -50   -> eigene Pflanzen entfernen (warnings=['removeOwnPlants']) -> niemals
      20+N  -> Gegner-Pflanzen entfernen, N = tatsächlich entfernte Menge (mehr = besser)
    Reihenfolge wichtig: erst Warnung prüfen, da die Self-Option denselben
    'Remove ... plants from ...'-Titel trägt wie die Gegner-Optionen."""
    warnings = [str(w).lower() for w in opt.get("warnings", [])]
    if "removeownplants" in warnings:
        return -50.0
    raw_title = opt.get("title", "")
    msg = raw_title.get("message", "") if isinstance(raw_title, dict) else str(raw_title)
    msg = msg.lower()
    # 'skip removing plants' enthält 'removing', nicht 'remove' -> faellt durch (None)
    if "remove" in msg and "plants" in msg and "from" in msg:
        return 20.0 + _extract_msg_number(raw_title)
    return None


_DIAG_LOGGED_GENS: set = set()
_DIAG_MS_LOGGED_GENS: set = set()


def _diag_milestones(state: dict) -> None:
    """Diagnose-Hook (gated via TM_DIAG_HAND): loggt einmal je Generation den
    Qualifikations-Stand des Bots fuer ALLE Meilensteine. gap=0 -> qualifiziert
    (Server wuerde claimen lassen). Trennt das Tempo-Problem (gap wird nie 0 ->
    Bot baut zu langsam) vom Claim-Fehler (gap=0, aber nicht geclaimt -> Bewertung/
    Timing). Rein additiv/verhaltensneutral."""
    if not os.environ.get("TM_DIAG_HAND"):
        return
    game = state.get("game", {})
    ms   = game.get("milestones", []) or []
    if not ms:
        return
    gen = game.get("generation", 0)
    key = (game.get("id") or game.get("gameId") or "", gen)
    if key in _DIAG_MS_LOGGED_GENS:
        return
    _DIAG_MS_LOGGED_GENS.add(key)
    me_name  = state.get("thisPlayer", {}).get("name")
    me_color = state.get("thisPlayer", {}).get("color")
    stats = _player_stats(state)
    parts = []
    for m in ms:
        nm = (m.get("name") or "?").strip()
        owner = m.get("color") or m.get("playerName")
        if owner:
            mine = (m.get("color") == me_color or m.get("playerName") == me_name)
            parts.append(f"{nm}=CLAIMED{'(ich)' if mine else '(Gegner)'}")
        else:
            gap = _milestone_gap(nm, stats)
            parts.append(f"{nm}={'QUALIFIZIERT' if gap <= 0 else f'gap{gap}'}")
    log.info("MEILENSTEIN-DIAG Gen %s | %s", gen, " | ".join(parts))


def _diag_holding(state: dict) -> None:
    """Diagnose-Hook (gated via Umgebungsvariable TM_DIAG_HAND). Wird zentral in
    decide() aufgerufen und sieht damit JEDEN waitingFor. Loggt pro Entscheidung
    eine kompakte Strukturzeile (wtype/title/option-types) und - falls eine
    projectCard-Option mit spielbaren Handkarten vorliegt - pro Karte score_card
    + Klasse. Rein additiv/verhaltensneutral. Einmal je (Spiel, Gen, wtype, title)."""
    if not os.environ.get("TM_DIAG_HAND"):
        return
    waiting = state.get("waitingFor") or {}
    options = waiting.get("options", []) or []
    wtype   = waiting.get("type")
    title   = str(waiting.get("title", ""))[:40]
    game = state.get("game", {})
    gen  = game.get("generation", 0)
    key  = (game.get("id") or game.get("gameId") or "", gen, wtype, title)
    if key in _DIAG_LOGGED_GENS:
        return
    _DIAG_LOGGED_GENS.add(key)
    otypes = [o.get("type") for o in options]
    SP_NAMES = {"Aquifer", "Greenery", "City", "Power Plant:SP", "Asteroid:SP"}
    pc_opt = next((o for o in options if o.get("type") == "projectCard"), None)
    hand_cards: list = []
    if pc_opt is not None:
        hand_cards = [c for c in pc_opt.get("cards", [])
                      if not c.get("name", "").endswith(":SP")
                      and c.get("name") not in SP_NAMES]
    if not hand_cards:
        log.info("DIAG Gen %s | wtype=%s '%s' | option-types=%s | keine spielbaren Handkarten",
                 gen, wtype, title, otypes)
        return
    player = state.get("thisPlayer", {})
    mc  = player.get("megacredits", 0)
    nbr = player.get("cardsInHandNbr", len(hand_cards))
    tl  = turns_left(state)
    counts: dict[str, int] = {}
    log.info("HAND-DIAGNOSE Gen %s | Hand gesamt %s | spielbar angeboten %d | %d M€ | wtype=%s",
             gen, nbr, len(hand_cards), mc, wtype)
    for c in hand_cards:
        name = c.get("name", "?")
        cost = c.get("calculatedCost", 999)
        try:
            sc = score_card(c, state)
        except Exception as e:
            log.info("    %-28s score=FEHLER (%s)", name[:28], e)
            continue
        try:
            req_pen = _requirement_penalty(card_info(name), state, tl)
        except Exception:
            req_pen = 0.0
        if req_pen <= REQ_UNREACHABLE:
            klass = "REQ-GATED"       # Requirement nicht erfuellt -> zu Recht gehalten
        elif cost > mc:
            klass = "UNAFFORDABLE"    # Geld fehlt
        elif sc <= 0:
            klass = "SCORE<=0"        # leistbar + Req ok, Bewertung negativ -> LECK-Kandidat
        else:
            klass = "PLAYABLE>0"      # spielenswert, aber nicht gespielt -> Ranking/Limit
        counts[klass] = counts.get(klass, 0) + 1
        log.info("    %-28s score=%6.1f cost=%3s %s", name[:28], sc, cost, klass)
    log.info("    -> %s", " | ".join(f"{k}={v}" for k, v in counts.items()))


def _is_ruling_policy(title: str) -> bool:
    """Turmoil-Ruling-Policy-Aktionen (bezahlbare Aktionen der regierenden Partei)."""
    return "turmoil" in title and ("pay" in title or "spend" in title)


def _ruling_policy_value(title: str, state: dict) -> float:
    """Netto-M€-Wert einer Ruling-Policy-Aktion. Die drei bezahlbaren Policies sind alle
    KARTENQUELLEN - genau das, woran es dem Bot chronisch mangelt (Hand laeuft leer):
      Scientists: 10 M€ -> 3 Karten   (1 gezogene Karte ~ 4-5 M wert: keine 3 M€ Kaufkosten,
                                       und der Bot ist kartenarm -> netto klar positiv)
      Mars First:  4 M€ -> 1 Building-Karte
      Unity:       4 M€ -> 1 Space-Karte
    Wert nur, wenn der Bot es sich leisten kann, ohne sein Spielgeld zu verbrennen."""
    p  = state.get("thisPlayer", {}) or {}
    mc = p.get("megacredits", 0)
    if "scientists" in title:
        cost, cards = 10, 3
    elif "mars first" in title or "unity" in title:
        cost, cards = 4, 1
    else:
        return 0.0
    if mc < cost + MC_RESERVE:
        return 0.0                       # Spielgeld schuetzen
    return cards * DRAW_CARD_VALUE - cost


def handle_or(state: dict) -> dict:
    """
    Bewertet alle verfügbaren Aktionen und wählt die beste.
    """
    waiting    = state["waitingFor"]
    options    = waiting.get("options", [])
    player     = state["thisPlayer"]
    mc         = player.get("megacredits", 0)

    candidates = []  # (score, index, payload)
    _idle_engine_log(state, options, player)

    for i, opt in enumerate(options):
        otype = opt.get("type", "")
        # Der Titel ist oft ein Message-TEMPLATE ({"data": [...], "message": "..."}), kein
        # String. str(dict).lower() liefert die Dict-Repraesentation -> KEIN Branch matcht ->
        # die Option landet im generischen Pass-Fallback. Beobachtet: die KORPORATIONS-
        # ERSTAKTION ("Take first action of ${0} corporation", z.B. Valley Trust: 3 Preludes
        # ziehen) wurde so wie "Pass" behandelt -> der Bot passte die GANZE Generation weg,
        # spielte nie eine Karte, TR blieb 20, M€ lief auf 148. (Gleicher Message-Dict-Bug wie
        # zuvor in handle_player/Biomass Combustors.)
        _raw_title = opt.get("title", "")
        title = (_raw_title.get("message", "") if isinstance(_raw_title, dict)
                 else str(_raw_title)).lower()

        # Pflicht-Erstaktion "Place a city tile" (Tharsis Republic u.a.): MUSS vor allen
        # anderen Zweigen geprueft werden, sonst faengt ein frueherer elif (z.B. der
        # card_action-Filter) opt 0 ab und filtert sie mit sc<=0 weg -> nur Pass bleibt
        # -> Bot passt dauerhaft (Total-Ausfall). Freie Pflichtstadt schlaegt Pass immer;
        # die konkrete Feld-Wahl macht danach handle_space (SelectSpace-Folgeprompt).
        if otype == "option" and "place a city" in str(opt.get("buttonLabel", "")).lower():
            try:
                sc = float(score_action("city", state))
            except Exception:
                sc = 0.0
            sc = max(sc, 12.0)   # freie Pflichtstadt schlaegt Pass sicher
            candidates.append((sc, i, {
                "type": "or", "runId": state["runId"], "index": i,
                "response": {"type": "option"},
                "_label": f"🏙 Freie Stadt platzieren (score={sc:.0f})",
            }))
            continue

        # Greenery (Pflanzen → Tile): Top-3 Positionen als separate MCTS-Kandidaten
        if otype == "space" and can_convert_plants(state):
            valid_ids = opt.get("spaces", [])
            if valid_ids:
                space_map = {s["id"]: s for s in state["game"]["spaces"]}
                for pos_score, spid in get_top_spaces(
                    valid_ids, space_map, "greenery",
                    player_id=player.get("color"), n=3,
                ):
                    pb    = _placement_bonus(spid, "greenery", state)
                    base_sc = score_action("greenery", state, placement_bonus=pb)
                    combined = base_sc + pos_score * 2.0
                    candidates.append((combined, i, {
                        "type": "or", "runId": state["runId"], "index": i,
                        "response": {"type": "space", "spaceId": spid},
                        "_label": f"🌿 Greenery auf Feld {spid} (pos={pos_score:.1f}, pb={pb:.0f})",
                    }))

        # Wärme → Temperatur
        elif otype == "option" and "heat" in title and can_convert_heat(state):
            sc = score_action("heat", state)
            candidates.append((sc, i, {
                "type": "or", "runId": state["runId"], "index": i,
                "response": {"type": "option"},
                "_label": f"🌡 Hitze→Temp",
            }))

        # Turmoil: Delegat entsenden. WICHTIG: Die Aktion erscheint im Aktionsmenue als
        # SelectParty (Typ 'party'), NICHT als 'option' - Turmoil.getSendDelegateInput()
        # liefert direkt den SelectParty-Prompt. Titel (TS): "Send a delegate in an area
        # (from lobby)" | "(5 M€)" | "(3 M€)" (Incite).
        # Wert = Chairman-Chance (kompensiert die TR-Revision, 1 TR/Gen) + Ruling-Bonus.
        elif otype == "party" and "delegate" in title:
            _turm = (state.get("game", {}) or {}).get("turmoil") or {}
            if _turm:
                val  = _delegate_action_value(state)
                free = "lobby" in title
                cost = 0.0 if free else (3.0 if "3" in title else DELEGATE_COST)
                # Budget-Deckel: bezahlte Delegaten duerfen das Kartengeld nicht auffressen
                # (beobachtet: Bot verdelegierte sein M€ und spielte kaum noch Karten).
                if not free and mc < cost + MC_RESERVE + 6:
                    net = 0.0
                else:
                    net = val - cost
                if net > 0:
                    _parties = opt.get("parties", []) or []
                    _best = max(_parties, key=lambda p: _party_choice_value(p, state)) \
                        if _parties else None
                    if _best:
                        candidates.append((net * CARD_PLAY_SCALE, i, {
                            "type": "or", "runId": state["runId"], "index": i,
                            "response": {"type": "party", "partyName": _best},
                            "_label": f"🏛 Delegat → {_best} "
                                      f"({'gratis' if free else f'{cost:.0f} M€'}, net={net:.0f})",
                        }))

        # Colonies: Bezahl-Untermenue des Trades (9 M€ / 3 Energie / 3 Titan) - billigste
        # reale Option waehlen (Energie meist Ueberschuss, Titan am teuersten).
        elif otype == "or" and _is_trade_payment_option(opt):
            sub = opt.get("options", []) or []
            j = _pick_trade_payment(state, sub)
            if j is not None:
                candidates.append((100.0, i, {   # hoch: der Trade wurde bereits beschlossen
                    "type": "or", "runId": state["runId"], "index": i,
                    "response": {"type": "or", "index": j, "response": {"type": "option"}},
                    "_label": f"🚀 Trade-Zahlung: {str(sub[j].get('title',''))[:24]}",
                }))

        # Colonies: Handels-Aktion (Trade). Wert = bester handelbarer Kolonie-Ertrag minus
        # Trade-Kosten; nur bei net>0 als Kandidat. Folge-Prompts (Bezahlung, Kolonie-Auswahl)
        # uebernehmen handle_payment/handle_colony.
        elif otype == "option" and "trade" in title and "free" not in title:
            net = _score_trade(state)
            if net > 0:
                candidates.append((net * CARD_PLAY_SCALE, i, {
                    "type": "or", "runId": state["runId"], "index": i,
                    "response": {"type": "option"},
                    "_label": f"🚀 Trade (net={net:.0f})",
                }))

        # Handkarte spielen oder Standard-Projekt
        elif otype == "projectCard":
            all_cards = opt.get("cards", [])

            # Standard-Projekt-Namen explizit ausschließen
            SP_NAMES = {"Aquifer", "Greenery", "City", "Power Plant:SP", "Asteroid:SP"}

            # Echte Handkarten
            hand_cards = [c for c in all_cards
                         if not c["name"].endswith(":SP")
                         and c["name"] not in SP_NAMES]
            played_positive = False
            for sc, card in get_playable_cards(hand_cards, state, max_cards=3):
                played_positive = True
                sc = sc * CARD_PLAY_SCALE   # auf SP-Skala heben (s. Konstante)
                candidates.append((sc, i, {
                    "type": "or", "runId": state["runId"], "index": i,
                    "response": {
                        "type": "projectCard",
                        "card": card["name"],
                        "payment": build_payment(card, player),
                    },
                    "_label": f"🃏 {card['name']} (score={sc:.1f})",
                }))

            # LEVER_IDLE: keine positive Karte spielbar -> das Geld laeuft leer.
            # Prinzip: ueber dem Reserve-Puffer sind die Kartenkosten illusorisch (das Geld
            # wuerde sonst gehortet). Bewerte die beste netto-negative, aber effektiv
            # positive Karte zu ihrem GROSS-Wert (Kostenabzug rueckgaengig: net + cost) und
            # biete sie an -> schlaegt Pass statt zu horten. Kein getunter Bonus; die Staerke
            # IST der reale Kartenwert. Nur was den Reserve-Puffer nicht antastet.
            # Leerlauf-Flag: keine positive Handkarte + Geld ueber Reserve -> Geld illusorisch.
            # Wirkt auf SPs (score_action -> cost_weight=0, gross-Wert) UND den Card-Idle unten.
            state["_idle_money"] = bool(LEVER_IDLE and not played_positive
                                        and mc > IDLE_RESERVE)

            if LEVER_IDLE and not played_positive:
                _best = None
                for _c in hand_cards:
                    _cost = _c.get("calculatedCost", 999)
                    if _cost > mc - IDLE_RESERVE:
                        continue
                    try:
                        if _requirement_penalty(card_info(_c["name"]), state,
                                                turns_left(state)) <= REQ_UNREACHABLE:
                            continue
                        _net = score_card(_c, state)
                    except Exception:
                        continue
                    if _net > 0:
                        continue
                    _idle = _net + _cost   # Kostenabzug rueckgaengig -> Gross-Wert (minus hold)
                    if _idle > 0 and (_best is None or _idle > _best[0]):
                        _best = (_idle, _c)
                if _best is not None:
                    _idle, _card = _best
                    _sc = _idle * CARD_PLAY_SCALE
                    candidates.append((_sc, i, {
                        "type": "or", "runId": state["runId"], "index": i,
                        "response": {
                            "type": "projectCard",
                            "card": _card["name"],
                            "payment": build_payment(_card, player),
                        },
                        "_label": f"🃏 {_card['name']} (idle, score={_sc:.1f})",
                    }))

            # Standard-Projekte: Aquifer (Ozean), Asteroid (Temp), Greenery (O2)
            # Budget-Planung: berechne was die beste spielbare Handkarte kostet
            # Wenn SP das Budget für eine gute Handkarte zerstört → SP-Score reduzieren
            best_hand_score = 0
            best_hand_cost  = 0
            for hc in hand_cards:
                hs = score_card(hc, state)
                hcost = hc.get("calculatedCost", 0)
                if hs * CARD_PLAY_SCALE > best_hand_score and hcost <= mc - MC_RESERVE:
                    best_hand_score = hs * CARD_PLAY_SCALE
                    best_hand_cost  = hcost

            # Placement-Bonus: bestes verfügbares Feld für Ozean/Greenery/Stadt
            space_map = {s["id"]: s for s in state["game"].get("spaces", [])}
            all_space_ids = [s["id"] for s in state["game"].get("spaces", [])
                             if not s.get("tile") and s.get("spaceType") != "colony"]
            def _best_pb(tile_type: str) -> float:
                if not all_space_ids:
                    return 0.0
                return max((_placement_bonus(sid, tile_type, state)
                            for sid in all_space_ids), default=0.0)

            pb_ocean   = _best_pb("ocean")
            pb_greenery = _best_pb("greenery")
            pb_city    = _best_pb("city")

            sp_scores = {
                "Aquifer":     score_action("ocean_sp",    state, placement_bonus=pb_ocean),
                "Greenery":    score_action("greenery_sp", state, placement_bonus=pb_greenery),
                "Asteroid:SP": score_action("temp_sp",     state),
                "City":        score_action("city_sp",     state, placement_bonus=pb_city),
                "Air Scrapping": score_action("venus_sp",   state),   # Venus Next (sonst brach)
            }
            for sp_card in all_cards:
                sp_name = sp_card["name"]
                if sp_name in sp_scores:
                    sc = sp_scores[sp_name]
                    if sc > 0:
                        mc   = state["thisPlayer"].get("megacredits", 0)
                        cost = sp_card.get("calculatedCost", 999)
                        if cost <= mc - (0 if _is_last_generation(state) else MC_RESERVE):
                            # Budget-Malus: wenn nach SP die beste Handkarte nicht mehr leistbar
                            mc_after_sp = mc - cost
                            if best_hand_score > sc and mc_after_sp < best_hand_cost + MC_RESERVE:
                                # SP würde gute Handkarte blockieren → stark abwerten
                                sc = sc * 0.4
                                log.debug("  Budget-Malus für %s: mc_nach_sp=%d, hand_kosten=%d",
                                          sp_name, mc_after_sp, best_hand_cost)
                            candidates.append((sc, i, {
                                "type": "or", "runId": state["runId"], "index": i,
                                "response": {
                                    "type": "projectCard",
                                    "card": sp_name,
                                    "payment": build_payment(sp_card, player),
                                },
                                "_label": f"🏗 {sp_name} SP (score={sc:.1f})",
                            }))

        # Karte verkaufen
        elif otype == "card" and "sell" in title:
            sell_cards = opt.get("cards", [])
            if sell_cards:
                # Verkaufe Karte mit niedrigstem Score
                worst = min(sell_cards, key=lambda c: score_card(c, state))
                worst_score = score_card(worst, state)
                # Normal nur klar wertlose Karten (<-2), damit momentan schwach
                # bewertete Engines nicht fuer 1 M€ verschleudert werden. In der
                # LETZTEN Generation alles, was nicht mehr gespielt wird (score<=0),
                # zu M€ machen statt verfallen lassen (eine Karte/Zug; die guten
                # holt zuvor der Endgame-VP-Boden via score_card).
                sell_threshold = 0.01 if _is_last_generation(state) else -2.0
                if worst_score < sell_threshold:
                    sc = score_action("sell", state)
                    candidates.append((sc, i, {
                        "type": "or", "runId": state["runId"], "index": i,
                        "response": {"type": "card", "cards": [worst["name"]]},
                        "_label": f"💰 Verkaufe {worst['name']} (score={worst_score:.1f})",
                    }))

        # ACTIVE-Karten-Aktion aktivieren: Server praesentiert dies als
        # SelectCard "Perform an action from a played card" (otype 'card',
        # cards = aktivierbare Karten). Bisher gab es KEINEN Handler dafuer
        # -> der Bot konnte seine Engines nie aktivieren. Wir waehlen die Karte
        # mit dem hoechsten Aktionswert (action_once) und aktivieren sie.
        elif otype == "card" and ("perform an action" in title or "played card" in title
                                   or opt.get("selectBlueCardAction")):
            act_cards = opt.get("cards", [])
            if act_cards:
                temp = state["game"].get("temperature", -30)
                def _act_value(c):
                    info = card_info(c.get("name", ""))
                    # Heat-Sperre: bei gemaxter Temperatur ist eine Aktion, deren
                    # einziger Produktions-Ertrag Hitze ist (z.B. Underground
                    # Detonations), wertlos. Primaer aus action_prod_res (Aktions-
                    # Produktion je Ressource, aus dem Repo extrahiert); Fallback
                    # auf das play-production-Feld fuer aeltere card_db-Staende.
                    apr = info.get("action_prod_res")
                    if apr is None:
                        apr = info.get("production") or {}
                    if temp >= 8 and apr.get("heat", 0) > 0 and all(
                            r == "heat" or v <= 0 for r, v in apr.items()):
                        return -1.0
                    # Netto-Ertrag der Aktivierung. action_once nettet bereits
                    # Produktionswert minus Kosten (Space Mirrors: 7 Prod - 7 = 0);
                    # der Karten-Zug fehlt darin und wird hier ergaenzt (~2/Karte).
                    return (float(info.get("action_once", 0) or 0)
                            + 2.0 * float(info.get("action_draw", 0) or 0))
                best = max(act_cards, key=_act_value)
                val  = _act_value(best)
                # Nur aktivieren, wenn die Aktion echten Netto-Ertrag hat. Damit
                # bekommen wertlose Grab-/Tausch-Aktionen (Search For Life = 0,
                # Space Mirrors = 0) KEINEN Kandidaten mehr und stehen nicht laenger
                # ueber dem Ausspielen von Handkarten (Deploy-Loop-Fix).
                if val > 0:
                    sc = val * CARD_PLAY_SCALE
                    # Freie Akkumulator-Aktion (Tardigrades etc., aus TS extrahiert) ist
                    # IMMER besser als Pass -> Boden knapp ueber Pass=4, damit der Bot sie
                    # aktiviert statt zu passen. Ohne das verlor Tardigrades (1.25*scale
                    # < Pass=4) gegen Pass -> 0/640 Bloecke aktiviert.
                    if str(best.get("name", "")).strip().lower() in _FREE_ACCUM:
                        sc = max(sc, 5.0)
                    candidates.append((sc, i, {
                        "type": "or", "runId": state["runId"], "index": i,
                        "response": {"type": "card", "cards": [best["name"]]},
                        "_label": f"⚡ Aktion: {best['name']} (a={val:.1f})",
                    }))

        # Meilenstein claimen - verschachtelte "Claim a milestone"-or:
        # Server bietet nur bereits claimbare Meilensteine an (claimableMilestones)
        elif otype == "or" and "milestone" in (
                (opt.get("title", {}) or {}).get("message", "")
                if isinstance(opt.get("title"), dict) else str(opt.get("title", ""))).lower():
            sub_opts = opt.get("options", [])
            best_j, best_sc = None, 0.0
            for j, sub in enumerate(sub_opts):
                stitle = str(sub.get("title", "")).lower()
                mname  = next((m for m in MILESTONE_NAMES if m in stitle), stitle)
                sc_j = _score_milestone(mname, state, known_claimable=True)
                if sc_j > best_sc:
                    best_sc, best_j = sc_j, j
            if best_j is not None and best_sc > 0:
                sc = best_sc * CARD_PLAY_SCALE   # auf Vergleichsskala (wie Karten)
                ms_name = str(sub_opts[best_j].get("title", "?"))
                candidates.append((sc, i, {
                    "type": "or", "runId": state["runId"], "index": i,
                    "response": {"type": "or", "index": best_j,
                                 "response": {"type": "option"}},
                    "_label": f"🏆 Claim {ms_name} (score={sc:.0f})",
                }))

        # Meilenstein claimen (flacher Fallback: Titel = Meilensteinname)
        elif otype == "option" and title in MILESTONE_NAMES:
            sc = _score_milestone(title, state) * CARD_PLAY_SCALE
            if sc > 0:
                candidates.append((sc, i, {
                    "type": "or", "runId": state["runId"], "index": i,
                    "response": {"type": "option"},
                    "_label": f"🏆 Meilenstein: {str(opt.get('title', '?'))[:30]}",
                }))

        # Award sponsern – verschachtelte "Fund an award"-or (äußere Ebene):
        # erst die Fund-Option wählen, dann in der inneren or den besten Award.
        elif otype == "or" and _is_fund_award_option(opt):
            sub_opts = opt.get("options", [])
            best_j, best_sc = None, 0.0
            for j, sub in enumerate(sub_opts):
                sc_j = _score_award(str(sub.get("title", "")), state)
                if sc_j > best_sc:
                    best_sc, best_j = sc_j, j
            if best_j is not None and best_sc > 0:
                best_sc *= CARD_PLAY_SCALE   # auf Vergleichsskala (wie Karten)
                award_name = str(sub_opts[best_j].get("title", "?"))
                log.info("   🥇 Award funden: %s (score=%.0f)", award_name, best_sc)
                candidates.append((best_sc, i, {
                    "type": "or", "runId": state["runId"], "index": i,
                    "response": {"type": "or", "index": best_j,
                                 "response": {"type": "option"}},
                    "_label": f"🥇 Fund {award_name}",
                }))

        # Award sponsern – direkte innere or (Fallback, falls der Server die
        # Award-Auswahl ohne äußere Hülle präsentiert; dann sind die Awards
        # otype=="option" mit Award-Namen als Titel).
        elif otype == "option" and _is_award_option(title):
            sc = _score_award(title, state) * CARD_PLAY_SCALE
            if sc > 0:
                candidates.append((sc, i, {
                    "type": "or", "runId": state["runId"], "index": i,
                    "response": {"type": "option"},
                    "_label": f"🥇 {str(title)[:40]}",
                }))

        # ACTIVE-Karten-Aktion oder Konzern-Aktion
        # KORPORATIONS-ERSTAKTION ("Take first action of <Korp> corporation"). Die Alternative
        # ist "Pass for this generation" - der Bot muss diese Aktion also unbedingt erkennen,
        # sonst passt er die ganze Generation weg (genau das passierte). Diese Erstaktionen
        # sind praktisch immer stark (Valley Trust: 3 Preludes ziehen; Point Luna: Karten;
        # Teractor/Vitor: Geld/VP) -> hoch bewerten, damit sie den Pass sicher schlaegt.
        elif otype == "option" and "first action of" in title and "corporation" in title:
            candidates.append((CORP_FIRST_ACTION_VALUE * CARD_PLAY_SCALE, i, {
                "type": "or", "runId": state["runId"], "index": i,
                "response": {"type": "option"},
                "_label": f"🏢 Korporations-Erstaktion "
                          f"({str(opt.get('buttonLabel', ''))[:34]})",
            }))

        elif otype == "option" and _is_card_action(title):
            sc = score_action("card_action", state, card_title=str(title))
            if sc > 0:   # wertlose/heat-gesperrte Aktionen (sc<=0) nicht aktivieren
                candidates.append((sc, i, {
                    "type": "or", "runId": state["runId"], "index": i,
                    "response": {"type": "option"},
                    "_label": f"⚡ Karten-Aktion: {str(opt.get('title', '?'))[:30]}",
                }))

        # Pflanzen-Angriff (removeAnyPlants): Gegner schwächen, eigene Pflanzen schützen
        elif otype == "option" and _plant_attack_score(opt) is not None:
            sc = _plant_attack_score(opt)
            n  = _extract_msg_number(opt.get("title", ""))
            candidates.append((sc, i, {
                "type": "or", "runId": state["runId"], "index": i,
                "response": {"type": "option"},
                "_label": (f"🌿✂ {n} Pflanzen entfernen" if sc > 0
                           else "🚫 eigene Pflanzen meiden"),
            }))

        # CEO: einmalige Faehigkeit (OPG - "Use CEO once per game action", Typ 'card').
        # Hatte KEINEN Handler -> die Faehigkeit verfiel ungenutzt. Der Wert ist einmalig und
        # oft frueh am staerksten (Karen: Preludes; Clarke: Produktion) - aber nicht in der
        # allerersten Generation verpulvern, wenn der Effekt mit der Generation skaliert.
        elif otype == "card" and "ceo" in title and "once per game" in title:
            _ceos = opt.get("cards", []) or []
            if _ceos:
                _nm  = _ceos[0].get("name", "")
                _val = score_ceo(_nm, state)
                if _val > 0:
                    candidates.append((_val * CARD_PLAY_SCALE, i, {
                        "type": "or", "runId": state["runId"], "index": i,
                        "response": {"type": "card", "cards": [_nm]},
                        "_label": f"👔 CEO-Aktion: {_nm} (Wert {_val:.0f})",
                    }))

        # Pass / End Turn / Undo / Ruling-Policy-Aktionen. WICHTIG: Frueher bekamen ALLE
        # 'option'-Eintraege denselben Pass-Score und das Label "Pass" - dadurch konnte der Bot
        # 'End Turn' (nur den ZUG beenden, man bleibt in der Generation) mit 'Pass for this
        # generation' (die GANZE Generation aufgeben) verwechseln, 'Undo last action' waehlen
        # und die Turmoil-Ruling-Policy-Aktionen ignorieren.
        elif otype == "option":
            if "undo" in title:
                continue                       # NIEMALS den eigenen Zug rueckgaengig machen
            if "end turn" in title:
                # Nur den Zug beenden: strikt besser als Pass (man bleibt in der Generation
                # und kann spaeter reagieren). Wird nur gewaehlt, wenn nichts Besseres da ist.
                candidates.append((score_action("pass", state) + 0.5, i, {
                    "type": "or", "runId": state["runId"], "index": i,
                    "response": {"type": "option"},
                    "_label": "⏹ End Turn",
                }))
            elif _is_ruling_policy(title):
                sc = _ruling_policy_value(title, state)
                if sc > 0:
                    candidates.append((sc * CARD_PLAY_SCALE, i, {
                        "type": "or", "runId": state["runId"], "index": i,
                        "response": {"type": "option"},
                        "_label": f"🏛 Policy: {title[:28]} (net={sc:.0f})",
                    }))
            else:
                sc = score_action("pass", state)
                candidates.append((sc, i, {
                    "type": "or", "runId": state["runId"], "index": i,
                    "response": {"type": "option"},
                    "_label": "⏭ Pass",
                }))

    if not candidates:
        log.warning("  handle_or: keine Kandidaten, sende Index 0")
        return {"type": "or", "runId": state["runId"], "index": 0,
                "response": {"type": "option"}}

    state.pop("_idle_money", None)   # Leerlauf-Flag nicht ueber die Entscheidung leaken

    candidates.sort(key=lambda x: x[0], reverse=True)
    score, idx, payload = candidates[0]
    log.info("  → %s", payload.get("_label", "?"))

    if _RLOG:
        # Strand-Detektor: welche Free-Accums bietet der Server in dieser Entscheidung
        # aktivierbar an (im Buendel opt["cards"])? Post-hoc-Query:
        #   chosen==Pass UND offered_fa != []  =>  Bot passt, obwohl ein Free-Accum
        #   aktivierbar war (= Maskierungs-Strand). 0 Faelle => Maskierung beisst nie.
        _offered_fa = set()
        for _opt in options:
            _t = str(_opt.get("title", "")).lower()
            if _opt.get("type") == "card" and ("perform an action" in _t or "played card" in _t
                                               or _opt.get("selectBlueCardAction")):
                for _c in _opt.get("cards", []) or []:
                    _nm = _c.get("name") if isinstance(_c, dict) else _c
                    if _nm and str(_nm).strip().lower() in _FREE_ACCUM:
                        _offered_fa.add(str(_nm).strip().lower())
        _rlog_write({"phase": "play",
                     "gen": (state.get("game") or {}).get("generation"),
                     "chosen": payload.get("_label", "?"),
                     "offered_fa": sorted(_offered_fa),
                     "cands": [[round(s, 1), c.get("_label", "?")]
                               for s, i2, c in candidates[:6]]})

    payload.pop("_label", None)
    return payload


def handle_option(state: dict) -> dict:
    return {"type": "option", "runId": state["runId"]}


def handle_space(state: dict) -> dict:
    """
    Wählt das beste Feld für ein Tile.
    Leitet Tile-Typ aus dem Kontext ab (Title oder letzter gesendeter Zug).
    """
    waiting   = state["waitingFor"]
    valid_ids = waiting.get("spaces", [])
    title     = str(waiting.get("title", "")).lower()

    if not valid_ids:
        all_spaces = state["game"]["spaces"]
        valid_ids  = [
            s["id"] for s in all_spaces
            if s.get("spaceType") == "land" and "tileType" not in s
        ]

    if not valid_ids:
        log.error("  handle_space: keine validen Felder!")
        return {"type": "space", "runId": state["runId"], "spaceId": "03"}

    # Tile-Typ aus Titel ableiten
    if any(k in title for k in ("ocean", "water", "aquifer")):
        tile_type = "ocean"
    elif "commercial" in title:
        tile_type = "commercial"   # Commercial District: VP je angrenzender Stadt
    elif any(k in title for k in ("capital", "special city")):
        tile_type = "capital"      # Capital: VP je angrenzendem Ozean
    elif any(k in title for k in ("city", "urban", "noctis")):
        tile_type = "city"
    elif any(k in title for k in ("greenery", "green", "forest", "plant")):
        tile_type = "greenery"
    elif any(k in title for k in ("steel", "titanium", "mining")):
        tile_type = "mining"   # Feld mit Stahl/Titan-Bonus bevorzugen
    else:
        # Spezial-Tile ohne erkannten Adjazenz-Nutzen (z.B. Nuclear Zone): NICHT als
        # greenery behandeln - das suchte eigene Staedte und verschwendete gute Plaetze.
        tile_type = "neutral"

    _player   = state.get("thisPlayer", {})
    space_map = {s["id"]: s for s in state["game"]["spaces"]}

    best = choose_best_space(
        valid_ids, space_map,
        tile_type=tile_type,
        player_id=_player.get("color"),
    )
    log.info("  🗺  Feld %s (%s)", best, tile_type)
    return {"type": "space", "runId": state["runId"], "spaceId": best}

def handle_amount(state: dict) -> dict:
    waiting = state["waitingFor"]
    min_val = waiting.get("min", 0)
    log.info("  💰 Betrag: %d", min_val)
    return {"type": "amount", "runId": state["runId"], "amount": min_val}


def handle_payment(state: dict) -> dict:
    """Zahlungsauswahl. Zahlt zuerst in M€ (bis verfuegbar), deckt den Rest mit
    erlaubten Ressourcen (heat 1:1, steel/titanium nach Wert). Verhindert Ueberzahlung."""
    w = state["waitingFor"]
    p = state.get("thisPlayer", {})
    amount = w.get("amount", 3) or 3
    opts = w.get("paymentOptions", {}) or {}
    mc, heat = p.get("megacredits", 0), p.get("heat", 0)
    steel, titanium = p.get("steel", 0), p.get("titanium", 0)
    sv, tv = p.get("steelValue", 2), p.get("titaniumValue", 3)

    pay = {k: 0 for k in ("auroraiData", "floaters", "graphene", "heat", "kuiperAsteroids",
           "lunaArchivesScience", "megacredits", "microbes", "plants", "seeds",
           "spireScience", "steel", "titanium")}
    rem = amount
    use = min(rem, mc); pay["megacredits"] = use; rem -= use
    if rem > 0 and opts.get("heat") and heat > 0:
        use = min(rem, heat); pay["heat"] = use; rem -= use
    if rem > 0 and opts.get("steel") and steel > 0:
        use = min(-(-rem // sv), steel); pay["steel"] = use; rem = max(rem - use * sv, 0)
    if rem > 0 and opts.get("titanium") and titanium > 0:
        use = min(-(-rem // tv), titanium); pay["titanium"] = use; rem = max(rem - use * tv, 0)
    return {"type": "payment", "runId": state["runId"], "payment": pay}


def handle_and(state: dict) -> dict | None:
    """
    Behandelt 'and'-Typ: mehrere Eingaben gleichzeitig (z.B. nach Kartenspielen).

    TM-Server schickt z.B. nach 'Research':
      waitingFor.type = 'and'
      waitingFor.options = [ {type: 'amount'}, {type: 'card'} ]

    Oder das title-Feld ist ein dict mit 'data' = Liste von InputResponses.
    Jede Sub-Antwort wird einzeln verarbeitet.

    InputType enum: 0=Option, 1=Amount, 2=Card, 3=Space, 4=Payment, 5=Player
    """
    waiting = state.get("waitingFor", {})
    runId   = state["runId"]

    # Manche Server-Versionen liefern options als Liste
    options = waiting.get("options", [])

    # ── Ressourcen-Verteilung (z.B. Global Event "Dry Deserts": 'Gain N resource(s) for
    # influence' -> and-Typ mit je einer 'amount'-Option pro Ressource). Der Server erwartet
    # eine Antwort PRO OPTION (Summe = N). Frueher gewann hier der title.data-Zweig und schickte
    # nur EINE Antwort -> Server 400 -> Spielabbruch.
    if options and all(o.get("type") == "amount" for o in options) and len(options) > 1:
        # Wieviel darf insgesamt verteilt werden? (title.data hat die Anzahl)
        total = 0
        _t = waiting.get("title", "")
        if isinstance(_t, dict):
            for d in (_t.get("data") or []):
                try:
                    total = max(total, int(d.get("value", 0)))
                except (TypeError, ValueError):
                    pass
        if total <= 0:
            total = sum(o.get("min", 0) for o in options) or 1

        # Ressourcen-Praeferenz (M-aequivalent): Titan > Stahl = Pflanze > Hitze > Energie > M€.
        # Energie verfaellt zu Hitze, M€ ist am schwaechsten pro Stueck.
        pref = {"titanium": 3.0, "steel": 2.0, "plants": 2.0,
                "heat": 1.0, "energy": 0.9, "megacredits": 1.0}
        order = sorted(range(len(options)),
                       key=lambda i: -pref.get(str(options[i].get("title", "")).lower(), 0.5))

        amounts = [0] * len(options)
        left = total
        for i in order:
            if left <= 0:
                break
            cap = options[i].get("max", left)
            take = min(left, cap if cap is not None else left)
            amounts[i] = take
            left -= take
        responses = [{"type": "amount", "amount": a} for a in amounts]
        log.info("  🔀 and-Ressourcen: %s",
                 ", ".join(f"{options[i].get('title')}={amounts[i]}"
                           for i in range(len(options)) if amounts[i]))
        return {"type": "and", "runId": runId, "responses": responses}

    # Fallback: title ist ein dict mit 'data'
    title = waiting.get("title", "")
    if isinstance(title, dict) and "data" in title:
        # Altes Format: title.data = [{'type': int, 'value': str}, ...]
        # Einfach alle akzeptieren mit Standardwerten
        responses = []
        for item in title["data"]:
            itype = item.get("type", 0)
            if itype == 1:   # Amount – Wert direkt aus title.data nehmen
                responses.append({"type": "amount", "amount": int(item.get("value", 0))})
            elif itype == 0: # Option
                responses.append({"type": "option"})
            elif itype == 2: # Card – aus verfügbaren Karten wählen
                # Karten aus waiting.options oder waiting.cards
                available = (waiting.get("options") or
                             waiting.get("cards") or [])
                if available:
                    # Wähle beste Karte via Draft-Logik
                    chosen = choose_draft_card(available, state)
                    responses.append({"type": "card", "cards": [chosen]})
                else:
                    responses.append({"type": "card", "cards": []})
            elif itype == 3: # Space – bestes Feld wählen
                space_ids = [s["id"] for s in state["game"].get("spaces", [])
                             if s.get("spaceType") == "land" and "tileType" not in s]
                if space_ids:
                    space_map = {s["id"]: s for s in state["game"]["spaces"]}
                    best = choose_best_space(space_ids, space_map,
                                            player_id=state["thisPlayer"].get("color"))
                    responses.append({"type": "space", "spaceId": best})
                else:
                    responses.append({"type": "space", "spaceId": "03"})
        if responses:
            log.info("  🔀 and-Typ (title.data): %d Antworten", len(responses))
            return {"type": "and", "runId": runId, "responses": responses}

    # Neues Format: options ist eine Liste von Sub-Waitings
    if options:
        responses = []
        for opt in options:
            otype = opt.get("type", "option")
            sub_state = dict(state)
            sub_state["waitingFor"] = opt
            sub_result = decide(sub_state)
            if sub_result:
                responses.append(sub_result)
            else:
                responses.append({"type": "option"})

        if responses:
            log.info("  🔀 and-Typ (options): %d Antworten", len(responses))
            return {"type": "and", "runId": runId, "responses": responses}

    # Fallback: leere Antwort
    log.warning("  and-Typ: kein Format erkannt, sende leer")
    return {"type": "and", "runId": runId, "responses": []}


def handle_unknown(state: dict) -> None:
    waiting = state.get("waitingFor", {})
    wtype = waiting.get("type")
    title = waiting.get("title", "")
    # Wenn title ein dict ist, ist es vermutlich ein and-Typ
    if isinstance(title, dict):
        return handle_and(state)
    log.warning("⚠️  Unbekannter Typ: '%s' | '%s'",
                wtype, str(title)[:60])
    return None


def handle_player(state: dict) -> dict | None:
    """
    Spielerauswahl (z.B. Cloud Seeding: 'Select player to decrease heat
    production'). Negative Effekte (decrease/remove/steal/lose) treffen
    bevorzugt einen Gegner; sonst – und wenn nur die eigene Farbe wählbar
    ist – die eigene Farbe (bisheriges Verhalten).
    """
    waiting = state.get("waitingFor", {})
    players = waiting.get("players", [])
    if not players:
        return None
    colors   = [p if isinstance(p, str) else p.get("color", "") for p in players]
    my_color = state.get("thisPlayer", {}).get("color", "")
    # Der Titel ist oft ein Message-TEMPLATE ({"data": [...], "message": "..."}), kein String.
    # str(dict).lower() findet die Schluesselwoerter NICHT -> negative=False -> der Bot griff
    # SICH SELBST an (beobachtet: Biomass Combustors senkte die EIGENE Pflanzenproduktion,
    # obwohl der Gegner welche hatte). Darum den Text sauber aus 'message' ziehen.
    raw_title = waiting.get("title", "")
    title = (raw_title.get("message", "") if isinstance(raw_title, dict)
             else str(raw_title)).lower()

    negative  = any(k in title for k in ("decrease", "remove", "steal", "lose"))
    opponents = [c for c in colors if c and c != my_color]

    if negative and opponents:
        chosen = opponents[0]
    elif my_color in colors:
        chosen = my_color
    else:
        chosen = colors[0]
    log.info("  👤 Spielerwahl: %s (%s)", chosen,
             "Gegner" if chosen != my_color else "selbst")
    return {"type": "player", "runId": state["runId"], "player": chosen}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def handle_production_to_lose(state: dict) -> dict:
    """Ares-Hazard-Strafe: beim Bauen neben einem Hazard-Feld muss 1-2 Produktion gesenkt
    werden (Server-Typ 'productionToLose'). Der Bot opfert die WERTLOSESTE Produktion zuerst.
    Floors: M€-Produktion darf bis -5, alle anderen nur bis 0.

    payProduction = {cost: N, units: {megacredits, steel, titanium, plants, energy, heat}}
    Antwort: {type:'productionToLose', units:{...}} mit Summe der Senkungen == cost.
    """
    w    = state.get("waitingFor", {}) or {}
    pp   = w.get("payProduction", {}) or {}
    cost = pp.get("cost", 1) or 1
    have = pp.get("units", {}) or {}     # aktuelle Produktionsstufen

    # Opfer-Reihenfolge: niedrigster M-aequivalenter Wert je Produktionsschritt zuerst.
    # heat < energy < M€ < steel < plants < titanium (spiegelt die Produktionswertung).
    order = ["heat", "energy", "megacredits", "steel", "plants", "titanium"]
    reduce = {k: 0 for k in ("megacredits", "steel", "titanium", "plants", "energy", "heat")}
    remaining = cost
    for k in order:
        if remaining <= 0:
            break
        floor      = -5 if k == "megacredits" else 0      # M€-Prod darf negativ, Rest nicht
        can_reduce = have.get(k, 0) - floor               # moegliche Senkungsschritte
        take = min(remaining, max(0, can_reduce))
        if take > 0:
            reduce[k] += take
            remaining -= take

    log.info("   ⚠️ Hazard-Strafe: senke Produktion %s",
             ", ".join(f"{k} -{v}" for k, v in reduce.items() if v))
    return {"type": "productionToLose", "runId": state["runId"], "units": reduce}


_COLONY_VALUE = {   # grober Ressourcen-Wert-Nudge je Kolonie (Ertragstyp). trackPosition ist
    "Pluto": 2.0,    # das Hauptsignal; dieser Nudge hebt wertvolle Ertraege (Karten/Tier/
    "Miranda": 1.5,  # Floater/Mikroben) leicht an. Grob - vom TM-Experten justierbar.
    "Titan": 1.5, "Enceladus": 1.5, "Triton": 1.5,
    "Ceres": 1.0, "Europa": 1.0, "Ganymede": 1.0,
    "Luna": 0.8, "Io": 0.8, "Callisto": 0.8, "Deimos": 0.5,
}
TRADE_COST = 7.0   # M-aequivalente Trade-Kosten (9 M€ oder 3 Energie/Titan; Energie oft Ueberschuss)
DELEGATE_COST = 5.0  # Turmoil: Standardaktion "Delegat entsenden" kostet 5 M€
CORP_FIRST_ACTION_VALUE = 20.0  # Korporations-Erstaktion (Valley Trust: 3 Preludes ziehen;
                                # Point Luna: Karten; ...) - praktisch immer stark, muss den
                                # Pass sicher schlagen.
DRAW_CARD_VALUE = 4.5  # Wert einer gezogenen Karte (spart 3 M€ Kauf + Optionswert; der
                       # Bot ist chronisch kartenarm -> Kartenquellen sind besonders wertvoll)
TR_VALUE = 10.0      # Bot-Konvention (BGG): 1 TR = 10 M (1 VP + Einkommen jede Generation)
INFLUENCE_VALUE = 4.0  # 1 Einfluss: mildert Global Events + Ruling-Boni. Konservativ bewertet -
                       # er lohnt nur, wenn ein Global Event den Bot ueberhaupt trifft.

def _score_trade(state: dict) -> float:
    """Netto-M€-Wert der Colonies-Trade-Aktion: bester handelbarer Kolonie-Ertrag minus
    Trade-Kosten. Ertrag ~ trackPosition * Ressourcenwert (grob); Kosten ~7 M€ (9 M€ ODER
    3 Energie/Titan - Energie oft Ueberschuss). Eigene Kolonie -> Zusatzertrag beim Handeln.
    Nur aktive, nicht besuchte Kolonien sind handelbar."""
    game     = state.get("game", {})
    colonies = game.get("colonies", []) or []
    my       = state.get("thisPlayer", {}).get("color")
    best = 0.0
    for c in colonies:
        if not c.get("isActive") or c.get("visitor"):
            continue
        track = c.get("trackPosition", 0) or 0
        nudge = _COLONY_VALUE.get(c.get("name", ""), 1.0)
        yv = track * nudge * 2.0
        if my in (c.get("colonies") or []):
            yv += 3.0
        best = max(best, yv)
    return best - TRADE_COST


def _is_trade_payment_option(opt: dict) -> bool:
    """Erkennt das Trade-Bezahl-Untermenue: ein 'or' mit Optionen wie '9 M€', '3 Energie',
    '3 Titan' (Titel-Varianten je nach Server-Lokalisierung tolerant gematcht)."""
    if opt.get("type") != "or":
        return False
    subs = [str(o.get("title", "")).lower() for o in (opt.get("options", []) or [])]
    if not subs:
        return False
    hits = sum(1 for t in subs
               if ("energy" in t or "titanium" in t or "m€" in t or "megacredit" in t))
    return hits >= 2 and len(subs) <= 4   # mind. 2 Ressourcen-Zahlwege, kleines Menue


def _pick_trade_payment(state: dict, options: list) -> int | None:
    """Trade-Bezahlung: 9 M€ ODER 3 Energie ODER 3 Titan. Waehlt die fuer den Bot BILLIGSTE
    reale Option (nur bezahlbare!). Energie ist meist Ueberschuss (verfaellt ohnehin zu Hitze),
    Titan ist wertvoll (Space-Karten), M€ ist universell. Gibt den Options-Index zurueck."""
    p    = state.get("thisPlayer", {}) or {}
    mc   = p.get("megacredits", 0) or 0
    en   = p.get("energy", 0) or 0
    ti   = p.get("titanium", 0) or 0
    # M-aequivalente Opportunitaetskosten der jeweiligen Zahlung
    COST = {"energy": 3 * 1.0,      # 3 Energie (~1 M/Stueck; verfaellt sonst zu Hitze)
            "megacredits": 9 * 1.0,  # 9 M€
            "titanium": 3 * 3.0}     # 3 Titan (~3 M/Stueck - teuerste Option)
    best_i, best_cost = None, None
    for i, opt in enumerate(options):
        t = str(opt.get("title", "")).lower()
        if "energy" in t and en >= 3:
            key = "energy"
        elif "titanium" in t and ti >= 3:
            key = "titanium"
        elif ("m€" in t or "mc" in t or "megacredit" in t) and mc >= 9:
            key = "megacredits"
        else:
            continue
        if best_cost is None or COST[key] < best_cost:
            best_i, best_cost = i, COST[key]
    return best_i


def handle_colony(state: dict) -> dict | None:
    """Colonies: Auswahl einer Kolonie (Typ 'colony') - zum HANDELN (welche Kolonie) oder
    BAUEN (worauf). Ertrag ~ trackPosition * Ressourcenwert. Beim Handeln: schon besuchte
    (visitor) meiden, eigene gebaute Kolonie gibt Zusatz-Bonus. Beim Bauen: schon gebaute
    meiden, wertvollen Ertrag bevorzugen. Antwort: {type:'colony', colonyName:...}."""
    w = state.get("waitingFor", {}) or {}
    colonies = w.get("coloniesModel", []) or []
    if not colonies:
        return None
    title    = str(w.get("title", "")).lower()
    my       = state.get("thisPlayer", {}).get("color")
    is_build = "build" in title

    def _score(c) -> float:
        track = c.get("trackPosition", 0) or 0
        nudge = _COLONY_VALUE.get(c.get("name", ""), 1.0)
        built = my in (c.get("colonies") or [])
        if is_build:
            s = track * 0.5 + nudge * 3.0
            if built:
                s -= 100.0                     # nicht erneut auf dieselbe Kolonie bauen
        else:
            s = track * nudge                  # Handels-Ertrag ~ Track * Ressourcenwert
            if c.get("visitor"):
                s -= 100.0                     # schon besucht -> nicht handelbar
            if built:
                s += 2.0                        # eigene Kolonie -> Zusatzbonus beim Handeln
        return s

    best = max(colonies, key=_score)
    log.info("   🚀 Kolonie: %s (%s, track=%s)",
             best.get("name"), "bauen" if is_build else "handeln", best.get("trackPosition"))
    return {"type": "colony", "runId": state["runId"], "colonyName": best.get("name")}


def _party_value(party: str, state: dict) -> float:
    """Wert einer Partei FUER DEN BOT (Ruling Bonus haengt vom eigenen Profil ab):
      Greens      1 M€/Pflanzen-,Mikroben-,Tier-Tag  (+2 M€/Greenery)
      Kelvinists  1 M€/Hitze-Produktion
      Scientists  1 M€/Science-Tag
      MarsFirst   1 M€/Building-Tag
      Unity       1 M€/Venus-,Earth-,Jovian-Tag
      Reds        TR-feindlich (bestraft Terraforming) -> fuer einen TR-Bot negativ
    Zusaetzlich: Karten auf der Hand, die genau diese Partei als Requirement brauchen."""
    st     = _player_stats(state)
    tags   = st.get("tags", {}) or {}
    prods  = st.get("prods", {}) or {}
    p      = (party or "").lower().replace(" ", "")
    if p == "greens":
        v = tags.get("plant", 0) + tags.get("microbe", 0) + tags.get("animal", 0) \
            + 2 * st.get("greeneries", 0)
    elif p == "kelvinists":
        v = float(prods.get("heat", 0))
    elif p == "scientists":
        v = float(tags.get("science", 0))
    elif p in ("marsfirst", "mars first"):
        v = float(tags.get("building", 0))
    elif p == "unity":
        v = tags.get("venus", 0) + tags.get("earth", 0) + tags.get("jovian", 0)
    elif p == "reds":
        v = -4.0                      # bestraft eigenes Terraforming
    else:
        v = 0.0
    # Handkarten, die genau diese Partei als Ruling-Requirement brauchen -> Bonus
    for c in (state.get("thisPlayer", {}).get("cardsInHand") or []):
        nm = c.get("name") if isinstance(c, dict) else c
        for r in ((card_info(nm) or {}).get("requirements") or []):
            if r.get("type") == "party" and \
               str(r.get("value", "")).lower().replace(" ", "") == p:
                v += 3.0
    return v


# ── Global Events (Turmoil): Wert EINES zusaetzlichen Einflusspunktes, in M€.
# Muster (aus TS extrahiert):
#   negativ: Verlust = min(max, Einheiten) - Einfluss  -> 1 Einfluss spart 1 Einheit,
#            ABER nur solange der Bot ueberhaupt betroffen ist (Einheiten > aktueller Einfluss).
#   positiv: Gewinn  = Einheiten + Einfluss            -> 1 Einfluss = 1 Einheit mehr (immer).
# "units" liefert, wie viele Einheiten den Bot betreffen (aus den Stats); bei positiven
# Events ist das unbegrenzt (Einfluss zahlt immer) -> 99.
# Ressourcenwerte: 1 TR = 10 M, 1 Karte = 3 M, 1 Titan = 3 M, 1 Stahl = 2 M, 1 Pflanze = 2 M,
# 1 M€-Produktion = 5 M.
_GLOBAL_EVENTS = {
    # --- negativ: Einfluss mildert ---
    "Riots":                        (4.0,  lambda s: min(5, s.get("cities", 0))),
    "Pandemic":                     (3.0,  lambda s: min(5, s["tags"].get("building", 0))),
    "Global Dust Storm":            (2.0,  lambda s: min(5, s["tags"].get("building", 0))),
    "Solar Flare":                  (3.0,  lambda s: min(5, s["tags"].get("space", 0))),
    "Solarnet Shutdown":            (3.0,  lambda s: min(5, s.get("blue_cards", 0))),
    "Mud Slides":                   (4.0,  lambda s: min(5, s.get("cities", 0) + s.get("greeneries", 0))),
    "Miners On Strike":             (3.0,  lambda s: min(5, s["tags"].get("jovian", 0))),
    "Microgravity Health Problems": (3.0,  lambda s: min(5, s.get("colonies", 0))),
    "Red Influence":                (5.0,  lambda s: 99),   # +1 M€-Produktion je Einfluss
    "War On Earth":                 (10.0, lambda s: 4),    # jeder Einfluss verhindert 1 TR!
    "Eco Sabotage":                 (2.0,  lambda s: min(5, max(0, s.get("plants", 0) - 3))),
    "Corrosive Rain":               (3.0,  lambda s: 99),   # 1 Karte je Einfluss
    "Paradigm Breakdown":           (2.0,  lambda s: 99),   # 2 M€ je Einfluss
    "Snow Cover":                   (3.0,  lambda s: 99),   # 1 Karte je Einfluss
    "Dry Deserts":                  (2.0,  lambda s: 99),   # 1 Standardressource je Einfluss
    "Sabotage":                     (2.0,  lambda s: 99),   # 1 Stahl je Einfluss
    # REVOLUTION: Einfluss wird ADDIERT -> mehr Einfluss = eher Verlierer (2 TR!). SCHAEDLICH.
    "Revolution":                   (-10.0, lambda s: 99),
    # --- positiv: Einfluss zahlt immer 1 Einheit extra ---
    "Homeworld Support":            (2.0,  lambda s: 99),
    "Celebrity Leaders":            (2.0,  lambda s: 99),
    "Interplanetary Trade":         (2.0,  lambda s: 99),
    "Spinoff Products":             (2.0,  lambda s: 99),
    "Strong Society":               (2.0,  lambda s: 99),
    "Venus Infrastructure":         (2.0,  lambda s: 99),
    "Generous Funding":             (2.0,  lambda s: 99),
    "Scientific Community":         (1.0,  lambda s: 99),
    "Asteroid Mining":              (3.0,  lambda s: 99),   # 1 Titan je Einfluss
    "Jovian Tax Rights":            (3.0,  lambda s: 99),
    "Productivity":                 (2.0,  lambda s: 99),   # 1 Stahl
    "Successful Organisms":         (2.0,  lambda s: 99),   # 1 Pflanze
    "Aquifer Released By Public Council": (4.0, lambda s: 99),  # 1 Pflanze + 1 Stahl
    "Cloud Societies":              (2.0,  lambda s: 99),
    "Sponsored Projects":           (3.0,  lambda s: 99),   # 1 Karte
    "Volcanic Eruptions":           (6.0,  lambda s: 99),   # 1 Hitze-Produktion
    "Improved Energy Templates":    (3.5,  lambda s: 99),   # zaehlt als Power-Tag
    "Election":                     (5.0,  lambda s: 99),   # TR-Rennen: Einfluss zaehlt
    "Diversity":                    (2.0,  lambda s: 99),   # zaehlt als Tag
}


def _global_event_influence_value(state: dict, stats: dict) -> float:
    """Was ist EIN zusaetzlicher Einflusspunkt wert, angesichts der anstehenden Global Events?
    Der Bot sieht 'current' (wird am Ende dieser Generation abgehandelt) und 'coming'/'distant'
    (die naechsten beiden) - alle oeffentlich. Naeher liegende Events zaehlen voller.
    Genau das ist der Kern von Turmoil: Einfluss lohnt NUR, wenn ein Event den Bot trifft."""
    turm = (state.get("game", {}) or {}).get("turmoil") or {}
    if not turm:
        return 0.0
    my_inf = stats.get("influence", 0)
    total  = 0.0
    for key, weight in (("current", 1.0), ("coming", 0.6), ("distant", 0.3)):
        name = turm.get(key)
        if not name:
            continue
        entry = _GLOBAL_EVENTS.get(str(name))
        if not entry:
            continue
        per_inf, units_fn = entry
        try:
            units = units_fn(stats)
        except Exception:
            units = 0
        # Einfluss zahlt nur, solange er noch Einheiten abdeckt (negative Events) bzw. immer
        # (positive, units=99). Negativer per_inf (Revolution) = Einfluss SCHADET.
        if per_inf < 0 or units > my_inf:
            total += per_inf * weight
    return total
def _turmoil_party_state(state: dict):
    """Hilfsdaten: (dominante Partei, meine Delegaten dort, bester Gegner dort, Reserve)."""
    turm = (state.get("game", {}) or {}).get("turmoil") or {}
    my   = state.get("thisPlayer", {}).get("color")
    parties = turm.get("parties") or []
    if not parties:
        return None, 0, 0, 0

    def _count(p, color):
        return sum(1 for d in (p.get("delegates") or [])
                   if (d.get("color") if isinstance(d, dict) else d) == color)

    # Dominant = die Partei mit den meisten Delegaten (so bestimmt der Server sie).
    dom = max(parties, key=lambda p: len(p.get("delegates") or []))
    mine = _count(dom, my)
    top  = 0
    for c in {(d.get("color") if isinstance(d, dict) else d)
              for d in (dom.get("delegates") or [])}:
        if c != my:
            top = max(top, _count(dom, c))
    reserve = len(turm.get("reserve") or []) or 5
    return dom, mine, top, reserve


def _delegate_action_value(state: dict) -> float:
    """M-aequivalenter Wert, JETZT einen Delegaten zu entsenden.

    TS-Regeln (Turmoil.getInfluence, Z.461):
      Einfluss = +1 Chairman
               + in der DOMINANTEN Partei: +1 als Partei-Leader (+1 extra bei >1 Delegat)
                                           bzw. +1 als Nicht-Leader mit >=1 Delegat
    => Delegaten zaehlen NUR in der DOMINANTEN Partei. Delegaten in leeren/anderen Parteien
       bringen NICHTS (beobachtet: der Bot verteilte Delegaten auf alle Parteien und in eine
       Partei, in der der Gegner uneinholbar fuehrte -> reine M€-Verschwendung).
    Chairman (=+1 TR) wird der LEADER DER DOMINANTEN PARTEI.

    Wert nur, wenn realistisch etwas zu holen ist:
      (a) Leader-Uebernahme in der dominanten Partei ERREICHBAR (mine+1 > top) -> Chairman-Chance
      (b) sonst: erster Delegat in der dominanten Partei -> +1 Einfluss (mildert Global Events)
      (c) sonst: 0 (nicht schicken!)
    """
    game = state.get("game", {}) or {}
    turm = game.get("turmoil") or {}
    if not turm:
        return 0.0
    dom, mine, top, _res = _turmoil_party_state(state)
    if dom is None:
        return 0.0

    gens_left = max(0, (game.get("lastSoloGeneration") or 12) - game.get("generation", 1))
    if gens_left <= 1:
        return 0.0                      # am Spielende bringt Einfluss/Chairman nichts mehr

    # Was ist EIN Einflusspunkt angesichts der anstehenden Global Events wert? (Kern von
    # Turmoil - Einfluss lohnt nur, wenn ein Event den Bot tatsaechlich trifft/nutzt.)
    stats   = _player_stats(state)
    inf_val = _global_event_influence_value(state, stats)

    # (a) Chairman-Weg: ein weiterer Delegat macht mich zum Leader der DOMINANTEN Partei
    #     -> Chairman (+1 TR/Gen) UND +1 Einfluss (Leader) -> beides zaehlt.
    if mine + 1 > top:
        return TR_VALUE * min(1.0, gens_left / 6.0) + max(0.0, inf_val)

    # (b) Einfluss-Weg: erster Delegat in der dominanten Partei gibt +1 Einfluss.
    if mine == 0 and inf_val > 0:
        return inf_val

    # (c) Aufstrebende Partei: eine Partei, die NAECHSTE Generation dominant werden koennte
    #     (gleichauf mit oder knapp hinter der dominanten), ist ein legitimes Investment -
    #     dort fuehrend zu sein, zahlt sich aus, sobald sie dominant wird.
    parties = turm.get("parties") or []
    my      = state.get("thisPlayer", {}).get("color")
    dom_n   = len(dom.get("delegates") or [])
    for p in parties:
        if p is dom:
            continue
        n = len(p.get("delegates") or [])
        if n >= dom_n - 1:                      # kann naechste Gen dominant werden
            p_mine = sum(1 for d in (p.get("delegates") or [])
                         if (d.get("color") if isinstance(d, dict) else d) == my)
            p_top  = 0
            for c in {(d.get("color") if isinstance(d, dict) else d)
                      for d in (p.get("delegates") or [])}:
                if c != my:
                    p_top = max(p_top, sum(1 for d in (p.get("delegates") or [])
                                           if (d.get("color") if isinstance(d, dict) else d) == c))
            if p_mine + 1 > p_top:              # dort fuehrend werden -> spaetere Chairman-Chance
                return TR_VALUE * 0.4 * min(1.0, gens_left / 6.0)

    # (d) Aussichtslos: nichts zu holen -> NICHT schicken.
    return 0.0


def _party_choice_value(party: str, state: dict) -> float:
    """Wert, einen Delegaten GENAU IN DIESE Partei zu setzen.

    Einfluss/Chairman zaehlen NUR in der DOMINANTEN Partei (TS: getInfluence) -> alles andere
    ist Geldverschwendung (beobachtet: Bot streute Delegaten ueber alle Parteien).
    Nur wenn dort nichts zu holen ist (Gegner uneinholbar), kann der Ruling-Bonus einer
    anderen Partei ein schwacher Trostpreis sein.
    """
    turm = (state.get("game", {}) or {}).get("turmoil") or {}
    dom, mine, top, _res = _turmoil_party_state(state)
    is_dom = dom is not None and \
        str(dom.get("name", "")).lower() == str(party).lower()

    v = _party_value(party, state) * 0.2      # Ruling-Bonus: schwaches Nebenkriterium
    if is_dom:
        if mine + 1 > top:
            v += TR_VALUE                      # Leader-Uebernahme -> Chairman (+1 TR/Gen)
        elif mine == 0:
            v += INFLUENCE_VALUE               # erster Delegat -> +1 Einfluss
    return v


def handle_party(state: dict) -> dict | None:
    """Turmoil: Partei waehlen (Typ 'party') - z.B. wohin ein Delegat geht oder welche
    Partei regieren soll. Waehlt die fuer das eigene Profil wertvollste."""
    w       = state.get("waitingFor", {}) or {}
    parties = w.get("parties", []) or []
    if not parties:
        return None
    best = max(parties, key=lambda p: _party_choice_value(p, state))
    log.info("   🏛 Partei: %s (Wert %.1f)", best, _party_choice_value(best, state))
    return {"type": "party", "runId": state["runId"], "partyName": best}


def handle_delegate(state: dict) -> dict | None:
    """Turmoil: Delegat waehlen (Typ 'delegate') - z.B. wen zum Vorsitzenden machen bzw.
    wessen Delegat entfernt wird. Titel entscheidet: negativ (remove) -> Gegner/neutral
    treffen; positiv -> die eigene Farbe. Titel kann ein Message-Template sein."""
    w      = state.get("waitingFor", {}) or {}
    props  = w.get("players", []) or []          # Farben bzw. 'NEUTRAL'
    if not props:
        return None
    my     = state.get("thisPlayer", {}).get("color", "")
    raw    = w.get("title", "")
    title  = (raw.get("message", "") if isinstance(raw, dict) else str(raw)).lower()
    negative = any(k in title for k in ("remove", "lose"))
    if negative:
        others = [c for c in props if c != my]
        chosen = others[0] if others else props[0]
    else:
        chosen = my if my in props else props[0]
    log.info("   🏛 Delegat: %s (%s)", chosen, "gegen" if negative else "eigen")
    return {"type": "delegate", "runId": state["runId"], "player": chosen}


HANDLERS = {
    "initialCards": handle_initial_cards,
    "card":         handle_card,
    "or":           handle_or,
    "option":       handle_option,
    "space":        handle_space,
    "amount":       handle_amount,
    "payment":      handle_payment,
    "and":          handle_and,      # Mehrfach-Eingaben (z.B. nach Kartenspielen)
    "selectAmount": handle_amount,   # Alias
    "player":       handle_player,   # Spielerauswahl (z.B. Cloud Seeding)
    "productionToLose": handle_production_to_lose,   # Ares-Hazard-Strafe
    "colony":       handle_colony,   # Colonies: Handel/Bau-Auswahl
    "party":        handle_party,     # Turmoil: Partei waehlen
    "delegate":     handle_delegate,  # Turmoil: Delegat waehlen
}


def decide(state: dict) -> dict | None:
    waiting = state.get("waitingFor")
    if not waiting:
        return None
    if _DUMP_WF:
        try:
            _g = state.get("game") or {}
            with open(_DUMP_WF, "a", encoding="utf-8") as _f:
                _f.write(json.dumps({"gen": _g.get("generation"),
                                     "title": str(waiting.get("title"))[:80],
                                     "type": waiting.get("type"),
                                     # thisPlayer + Global-Parameter, um "besitzt Engine X,
                                     # wird aber nicht angeboten" von "besitzt sie nicht" zu
                                     # trennen (Diagnose konditionaler Engines). Nur bei
                                     # gesetztem TM_DUMP_WF -> verhaltensneutral.
                                     "thisPlayer": state.get("thisPlayer"),
                                     "params": {"temperature": _g.get("temperature"),
                                                "oxygenLevel": _g.get("oxygenLevel"),
                                                "oceans": _g.get("oceans")},
                                     "waitingFor": waiting}, ensure_ascii=False) + "\n")
        except Exception:
            pass
    _diag_holding(state)      # Diagnose (gated via TM_DIAG_HAND), verhaltensneutral
    _diag_milestones(state)   # Meilenstein-Qualifikation (gated), verhaltensneutral
    wtype = waiting.get("type")
    handler = HANDLERS.get(wtype, handle_unknown)
    try:
        return handler(state)
    except Exception as e:
        log.error("  Handler '%s' Exception: %s", wtype, e, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Deduplizierung
# ---------------------------------------------------------------------------

def waiting_key(waiting: dict) -> tuple:
    cards   = tuple(c.get("name", "") for c in waiting.get("cards", []))
    options = tuple(o.get("type", "") for o in waiting.get("options", []))
    spaces  = tuple(waiting.get("spaces", []))
    return (waiting.get("type"), str(waiting.get("title", "")), cards, options, spaces)


# ---------------------------------------------------------------------------
# Hauptschleife
# ---------------------------------------------------------------------------

def run_bot(base_url: str, player_id: str, poll: float, debug_pause: float = 0):
    import os
    _TLOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transitions.jsonl")
    log.info("📝 Transitions-Log-Pfad: %s", _TLOG_PATH)

    consecutive_errors = 0
    last_key           = None

    while True:
        try:
            state = get_state(base_url, player_id)
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            log.error("API-Fehler (%d/%d): %s", consecutive_errors, MAX_API_ERRORS, e)
            if consecutive_errors >= MAX_API_ERRORS:
                log.error("Zu viele Fehler – Bot beendet.")
                break
            time.sleep(poll * 2)
            continue

        game  = state.get("game", {})
        phase = game.get("phase", "")

        if phase == "end":
            vp  = state["thisPlayer"]["victoryPointsBreakdown"]["total"]
            tr  = state["thisPlayer"]["terraformRating"]
            won = game.get("isSoloModeWin", False)
            log.info("🏁 VP: %d | TR: %d | Gewonnen: %s", vp, tr, won)
            # VP-Quellen-Aufschluesselung. KARTEN ist die Schluesselmetrik fuer den
            # VP-Engine-Hebel (1V1-Diagnose: Bot 4-7 vs Mensch 38 Karten-VP).
            _vpb = state["thisPlayer"]["victoryPointsBreakdown"]
            log.info("🎴 VP-Quellen | KARTEN:%d Greenery:%d City:%d Meilenst:%d Awards:%d TR:%d",
                     _vpb.get("victoryPoints", 0), _vpb.get("greenery", 0),
                     _vpb.get("city", 0), _vpb.get("milestones", 0),
                     _vpb.get("awards", 0), _vpb.get("terraformRating", 0))
            # Engine-Diagnose (Anreiz-Feld-Test): Produktion am Spielende. Pflanzen-
            # Produktion ist der Schluessel-Indikator fuer eine echte Engine.
            tpl = state["thisPlayer"]
            log.info("🏭 Produktion | MC:%d Stahl:%d Titan:%d PFLANZEN:%d Energie:%d Hitze:%d | Summe:%d | INCENTIVE=%s",
                     tpl.get("megacreditProduction", 0), tpl.get("steelProduction", 0),
                     tpl.get("titaniumProduction", 0), tpl.get("plantProduction", 0),
                     tpl.get("energyProduction", 0), tpl.get("heatProduction", 0),
                     tpl.get("megacreditProduction", 0) + tpl.get("steelProduction", 0)
                     + tpl.get("titaniumProduction", 0) + tpl.get("plantProduction", 0)
                     + tpl.get("energyProduction", 0) + tpl.get("heatProduction", 0),
                     LEVER_INCENTIVE_FIELD)
            break

        player  = state.get("thisPlayer", {})
        waiting = state.get("waitingFor")

        if not waiting:
            time.sleep(poll)
            continue

        key = waiting_key(waiting)
        if key == last_key:
            time.sleep(poll)
            continue

        # Status
        gen    = game.get("generation", "?")
        step   = game.get("step", "?")
        wtype  = waiting.get("type", "?")
        title  = str(waiting.get("title", ""))[:40]
        mc     = player.get("megacredits", "?")
        tr     = player.get("terraformRating", "?")
        plants = player.get("plants", 0)
        oxygen = game.get("oxygenLevel", 0)
        temp   = game.get("temperature", -30)
        oceans = game.get("oceans", 0)
        log.info("[Gen %s|Step %s] MC:%s TR:%s 🌿%d O₂:%d%% %d°C 🌊%d | %s",
                 gen, step, mc, tr, plants, oxygen, temp, oceans, title or wtype)

        payload = decide(state)
        if payload is None:
            last_key = key
            time.sleep(poll * 3)
            continue

        if debug_pause > 0:
            log.info("  ⏸  Debug-Pause %.0fs – jetzt State abrufen!", debug_pause)
            time.sleep(debug_pause)

        try:
            post_input(base_url, player_id, payload)
            log.info("  ✅ OK")
            # --- Transitions-Logging (absoluter Pfad, neben tm_bot.py) ---
            try:
                import os, json as _json
                _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transitions.jsonl")
                with open(_p, "a", encoding="utf-8") as _tlog:
                    _tlog.write(_json.dumps({
                        "state": state,
                        "move": {k: v for k, v in payload.items() if k != "_label"},
                    }) + "\n")
                globals()["_TLOG_N"] = globals().get("_TLOG_N", 0) + 1
                log.info("  📝 %d Zeilen -> %s", globals()["_TLOG_N"], _p)
            except Exception as _e:
                log.error("  ⚠ Logging-Fehler: %s", _e)
            # --- Ende ---
            except Exception as _e:
                log.error("  ⚠ Logging-Fehler: %s", _e)
            # --- Ende Logging ---
            last_key = None
            time.sleep(POST_WAIT)
        except requests.HTTPError as e:
            body = e.response.text[:300] if e.response is not None else ""
            log.error("  ❌ HTTP %s: %s",
                      e.response.status_code if e.response else "?", body)
            last_key = key
            time.sleep(ERROR_WAIT)
        except Exception as e:
            log.error("  ❌ Fehler: %s", e)
            last_key = key
            time.sleep(ERROR_WAIT)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Terraforming Mars Bot")
    parser.add_argument("--player-id", required=True)
    parser.add_argument("--url",  default=DEFAULT_URL)
    parser.add_argument("--poll", default=POLL_INTERVAL, type=float)
    parser.add_argument("--db",    default="card_db.json",
                        help="Pfad zur Kartendatenbank")
    parser.add_argument("--model", default="tm_model.pt",
                        help="Pfad zum ML-Modell (optional)")
    parser.add_argument("--debug-pause", default=0, type=float,
                        help="Pause in Sekunden vor jedem POST (zum Debuggen)")
    args = parser.parse_args()

    load_card_db(args.db)
    load_ml_model(args.model)
    run_bot(args.url, args.player_id, args.poll, args.debug_pause)
