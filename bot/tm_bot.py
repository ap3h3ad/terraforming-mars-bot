"""
Terraforming Mars expert bot - heuristic card and action evaluation.

WHAT THIS IS
    A rule-based opponent for the open-source Terraforming Mars server
    (github.com/terraforming-mars/terraforming-mars). It contains no machine
    learning: every decision comes from an explicit, readable value function
    denominated in megacredits (MC). The exchange rates are the ones used by
    experienced players - 1 VP = 5 MC, 1 TR = 10 MC, 1 MC production = 10 MC,
    1 plant production = 10 MC, 1 heat production = 6 MC, 1 drawn card = 2 MC.

HOW A DECISION IS MADE
    The server sends a `waitingFor` structure describing what it wants. The
    module `tm_mcts_mp.py` polls it and hands it to `decide()`, which routes
    to a handler by input type (HANDLERS at the bottom of this file). Every
    handler ranks its options with one of three scoring entry points:

        score_card(card, state)         value of PLAYING a card, in MC
        score_card_to_buy(card, state)  value of BUYING/DRAFTING it (cheaper
                                        bar: a bought card can wait on hand)
        score_action(action, state)     value of a standard project, of
                                        passing, of selling, of converting
                                        plants or heat

    Card values start from `card_db.json` (a static per-card appraisal) and
    are then adjusted for the live game state: remaining generations, own
    production, tiles on the board, opponent holdings, milestone and award
    progress. Card scores are multiplied by CARD_PLAY_SCALE so that playing a
    card and doing a standard project are comparable numbers.

    Effects that depend on context cannot live in a static database and are
    kept as runtime tables here: _ATTACK (cards that hurt the opponent),
    _DYNAMIC ("X per city/tag/colony"), _TAG_TRIGGER, _CITY_TRIGGER,
    _GLOBAL_EVENTS (Turmoil), _CEO_VALUE, _AWARD_KEYS.

BEHAVIOUR FLAGS ("levers")
    Constants named LEVER_* switch individual pieces of judgement on and off.
    They exist so that a change can be measured in isolation: freeze a copy of
    this file as the champion, flip exactly one flag in the challenger, and let
    the two play a paired match (below). Every flag in this file is currently
    on; discarded experiments have been removed rather than left on False.

HOW IT IS RUN (all commands via tm_mcts_mp.py)
    Play against a human:
        py -3.12 tm_mcts_mp.py --vs-human --no-mcts --draft \
                 --expansions venus,ares,ceo [--board random]
    Join a game that already exists:
        py -3.12 tm_mcts_mp.py --join --player-id <pid> --no-mcts
    A/B test against a frozen champion (common random numbers, paired):
        py -3.12 tm_mcts_mp.py --ab-crn --champion-module tm_bot_champion \
                 --auto-games 40 --parallel 6 --draft
        The reported margin is challenger minus champion in VP, with a 95 %
        confidence interval over the pairs. `--draft` matters: without it the
        bot sees far fewer cards and card-selection changes cannot show up.
        `--parallel` beyond 6 tends to overload the server.

    A/B caveat worth knowing before trusting a result: both sides share every
    weakness, so the paired margin is blind to anything that only hurts against
    a stronger opponent (tempo, collapses, missed opportunities). Narrow levers
    measure sharply, engine-wide ones do not.

DIAGNOSTICS (environment variables)
    TM_DIAG_HAND=1        log hand composition each generation
    TM_DUMP_WF=<file>     dump every raw waitingFor structure (first thing to
                          collect when the bot passes or stalls unexpectedly)
    TMBOT_RLOG=<file>     log buy decisions with scores
    TM_SHADOW=<file>      while the human is to move, evaluate the position
                          from their seat and log what the bot would do

REQUIREMENTS
    Python 3.10+, `requests`, and card_db.json next to this file.
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

# Parameter progress feature for the optional ML model
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


def hand_cards(state: dict) -> list:
    """The bot's own hand cards.

    The server delivers them on the TOP LEVEL of the PlayerViewModel, not under
    thisPlayer. `ViewModel.thisPlayer` is a PublicPlayerModel and carries ONLY
    `cardsInHandNbr`, the count; the field `cardsInHand` with the actual cards sits
    on the PlayerViewModel, i.e. state["cardsInHand"]. It is built for the REQUESTED
    player, so polling with a foreign player id returns that player's hand.

    Reading `cardsInHand` off thisPlayer always yields an empty list.
    """
    h = state.get("cardsInHand")
    if h:
        return list(h)
    # fallback in case a server version does carry it on the player object
    return list((state.get("thisPlayer") or {}).get("cardsInHand") or [])

# ---------------------------------------------------------------------------
# Connection and polling
# ---------------------------------------------------------------------------
DEFAULT_URL    = "http://localhost:9000"
POLL_INTERVAL  = 2
POST_WAIT      = 2
ERROR_WAIT     = 6
MAX_API_ERRORS = 5
MC_RESERVE     = 8          # cash kept back so a bought card can still be played

# ---------------------------------------------------------------------------
# Scale and thresholds
# ---------------------------------------------------------------------------
# In the final generation unspent money is lost except as a tiebreaker, so a card
# with fixed VP is worth its VP without any cost deduction. 3.5 is the median
# gross-per-VP over the 96 pure VP cards in the database.
VP_ENDGAME_VALUE = 3.5

# score_action() returns MC values multiplied by 3.0-4.5 (a greenery standard
# project is 15 MC * 3.0 = 45), while card scoring returns raw MC. Without this
# factor cards almost always lose the comparison in handle_or.
CARD_PLAY_SCALE = 3.0

# A card is only played if raw * CARD_PLAY_SCALE beats the weakest standard
# project (asteroid, 21), so buying pays off above raw 7. A threshold of 0 was
# measured to produce 62 % dead buys (bought, never played).
BUY_MIN_SCORE   = 0.5

# Buying is optionality, not immediate play: a card costs 3 MC now and may be
# played several generations later. The research reserve is therefore sized for
# the CHEAPEST worthwhile card on offer, not the most expensive one - reserving
# for the expensive one drove the budget to zero and blocked buying entirely.
# This flag additionally lifts the hard "stop buying in the last two turns" rule;
# BUY_MIN_SCORE still keeps junk out.
LEVER_BUY_VS_PASS = True

# Spare money otherwise drains into greenery standard projects (~23 MC for ~2 VP,
# a net loss). When the bot holds greenery-capable surplus the buy threshold drops
# to this floor instead - almost any card is a better use of that money.
# Requirement-blocked cards (-50) stay out regardless.
GREENERY_SP_COST   = 23
GREENERY_BUY_FLOOR = -10.0

# Early brake on greenery standard projects: early the money belongs in the
# engine, late the urgency term takes over and surplus may go into greeneries.
# Plant-driven greenery (free) is untouched.
GREENERY_LATE_GEN      = 5     # from tl <= 5 no brake (endgame harvest)
GREENERY_EARLY_PENALTY = 1.5   # net MC deducted per generation above that

# Standard projects carry their full cost, like cards. Hoarding is handled by
# LEVER_IDLE instead of by discounting projects here.
SP_COST_WEIGHT  = 1.0

# Share of remaining generations in which an ACTIVE card's action is actually
# used. Below 1.0 because the bot does not activate every generation (competing
# actions, missing input resources).
ACTION_ACTIVATION_RATE = 0.5

# ---------------------------------------------------------------------------
# Energy does not accumulate
# ---------------------------------------------------------------------------
# Server behaviour (Player.runProductionPhase): `heat += energy; energy = 0` -
# all energy is converted to heat at the end of every generation. A player with
# 1 energy production can therefore NEVER pay an action costing 6 energy
# (Physics Complex), not even by saving up for six generations. Treating this
# linearly (fuel = prod / cost) wrongly assumes accumulation. Affects 13 cards.
# This is a residual value, not a hard block - energy production can be built up.
ENERGY_RAMP_FUEL = 0.25   # residual value while the production is still missing
ENERGY_GAP_MAX   = 3      # missing production steps beyond which it is worthless
# Exception: this card lets its owner choose how much energy to convert, so with
# it energy does accumulate and the linear model is correct again.
_ENERGY_KEEPER = "Supercapacitors"

# Curated exception to the flat "+8 for a passive ACTIVE effect" floor: ACTIVE
# cards with action_once = 0 that have neither a real passive effect nor a
# worthwhile action. Not detectable from data - genuinely good passive cards look
# identical in the database (action_once = 0, no markers).
_NO_PASSIVE_VALUE = {
    "Search For Life",
}

# Feedable resource-VP stacking cards (vp_dyn.kind == "resources") get a higher
# activation rate: the bot stacks them reliably because the action beats passing.
VP_STACK_ACTIVATION_RATE = 0.85
# Option value for resource-VP cards that unlock late behind a pure global
# parameter: the parameter rises on its own, and 2-3 triggers already beat the
# capped downside of selling the card for 1 MC.
LATE_ENGINE_MIN_ACTIVATIONS = 2.5

# Chasing a milestone that is still 2-3 steps away: small, decaying bonus that
# tips an otherwise close decision rather than driving play.
PURSUE_MAX_GAP   = 3
PURSUE_WEIGHT    = 0.4
PURSUE_BONUS_CAP = 4.0

# Anti-hoarding: with no positive card playable and money above the reserve the
# money is going to waste, so its cost is illusory. Standard projects and even
# net-negative cards are then valued at their gross worth.
LEVER_IDLE   = True
IDLE_RESERVE = 25.0     # MC held back for future cards; above this money is idle

# Selling threshold in early generations: without an engine almost every card
# scores negative, and -2.0 would hit half the starting hand.
SELL_EARLY_GENS       = 4
SELL_THRESHOLD_EARLY  = -25.0

# Milestone and award alignment: small buy bias towards cards that advance an
# in-play, realistically winnable goal. Kept small to avoid tunnel vision.
ALIGN_MAX_GAP     = 4       # milestone counts as a goal within this many steps
ALIGN_AWARD_SLACK = 2       # award counts as a goal if leading or this far behind
ALIGN_BUY_BONUS   = 4.0     # flat buy bonus per matching card

# A few cards whose value depends on an enabler that card_db does not encode.
# Without the enabler they are heavily devalued, on buying and on keeping.
ENABLER_PENALTY   = 20.0
_PARAM_STEP = {"temperature": 2, "oxygen": 1, "oceans": 1, "venus": 2}

# Deploy capacity: actions are scarce (~2 per generation, late almost all bound to
# greenery conversion). Holding more unplayed cards than can realistically be
# played in the remaining game makes every further buy dead capital.
DEPLOY_CARDS_PER_GEN      = 1.0   # playable hand cards per remaining generation
DEPLOY_OVERFLOW_PENALTY   = 0.6   # deduction per card above that capacity

# Incentive field: cards whose tags feed the bot's own engine, or satisfy open tag
# requirements, get a small bonus. Deliberately narrow.
TAG_SYNERGY_UNIT          = 1.5
TAG_DEMAND_CAP            = 3.0
# Cards placing a tile type the bot currently wants (ocean for a Lakefront or
# Arctic Algae engine, city for Tharsis or Pets). Ocean is headroom-gated.
TILE_SYNERGY_UNIT         = 1.5

# ---------------------------------------------------------------------------
# Behaviour flags
# ---------------------------------------------------------------------------
# One canonical file; each piece of judgement switchable on its own so that its
# marginal contribution can be measured in isolation against a frozen champion.

# Capacity-aware buy deduction (DEPLOY_OVERFLOW_PENALTY).
LEVER_BUY_DISCIPLINE      = True
# Engine synergy: tag demand and tile synergy. The tile part is additionally
# data-gated - it only does anything with a card_db carrying tile_reward fields.
LEVER_INCENTIVE_FIELD     = True
# Remaining game length from the observed parameter history instead of a fixed
# two-player prior; six-player games are shorter and get a shorter horizon.
LEVER_ADAPTIVE_HORIZON    = True
# Endgame acceleration (heat dump into temperature, greenery surge into oxygen)
# via the most recent parameter rate. A whole-game average blends the slow early
# game with the fast late game and overstates the horizon by a factor of 3-7.
LEVER_ENDGAME_RATE        = True
# Soft ramp for plant production instead of a hard threshold. With a hard one the
# FIRST plant card scored as worthless from mid-game on, blocking the engine.
LEVER_PLANT_ENGINE        = True
# Early discount on the asteroid standard project - pure TR at 14 MC per TR, the
# worst MC-to-VP rate in the game. Fades towards mid-game.
LEVER_SP_DISCIPLINE       = True
# Value feedable resource-VP cards by projected accumulation, fuel-capped.
LEVER_VP_ENGINE           = True
# Option value of VP engines that unlock late (Fish, Livestock, Predators,
# Penguins) - global parameters rise reliably.
LEVER_LATE_ENGINE         = True
# Milestone behaviour: cap detection from game["milestones"], flat cost of 8,
# window-aware urgency (secure a qualified milestone before the three slots fill)
# and the pursue branch above.
LEVER_MILESTONE           = True
# City value comes from the SPACE, not from how many cities already stand. A flat
# per-city penalty used to eat the adjacency VP as well, so from the third city on
# the bot refused every city - even one bordering five of its own greeneries.
# City VP count per adjacent greenery, and one greenery serves several cities, so
# a cluster of cities is more efficient, not less.
LEVER_CITY_ADJACENCY      = True
# Value an adjacency VP at 5.0 when choosing a space, consistent with the bot's
# own convention of 1 VP = 5 MC. It was 3.0 for greenery and city but already 5.0
# for commercial - the same quantity priced 40 % lower in two of three places.
LEVER_ADJACENCY_VP        = True
# Upgrade MC production while the bot has almost none, so that income build-up
# does not depend on what the deck happens to offer.
LEVER_MC_SCARCITY         = True
MC_SCARCITY_FLOOR         = 5.0    # production at which the bonus has faded out
MC_SCARCITY_BONUS         = 0.75   # uplift at production 0
# Put award scores in the same unit as card scores (CARD_PLAY_SCALE). Without it
# every award from the second one on loses against an arbitrary card play.
LEVER_AWARD_SCALE         = True
# Charge the cost of a card action (`action_prod`) when valuing the card. It used
# to be read only on execution, so Refugee Camps looked worth 158 instead of zero.
LEVER_ACTION_COST         = True
# Evaluate redeem options ("spend resources -> gain something") in handle_or at
# all. Without this branch the bot fell through to option 0 and kept collecting
# forever - 16 unused microbes on a card is 48 MC left on the table. Redeeming
# waits for the late game because the bot has only one action per generation:
# collecting eight times and cashing once beats cashing four times.
LEVER_REDEEM              = True
REDEEM_PROGRESS           = 0.75   # redeem from 75 % terraforming progress
REDEEM_CASH_FLOOR         = 8.0    # ... or when holding less than 8 MC
_REDEEM_RE = __import__("re").compile(
    r"gain\s*(?:triple\s*amount\s*of\s*)?(\d+)\s*(?:m€|mc|megacredit)"
    r"|(\d+)\s*(?:m€|mc|megacredit).{0,20}per")

# Value a pure resource collector by context: full value once a second card of the
# same kind is on the table, damped while it stands alone. These cards are weak
# individually and strong in combination.
LEVER_RESOURCE_SYNERGY    = True
RES_SYNERGY_FLOOR         = 2.0
RES_SYNERGY_DAMPING       = 0.5

# PURE COLLECTORS - cards whose action only puts a resource on a card, without
# that resource turning into TR, money or a card draw by itself.
# This list is CURATED on purpose: three attempts to derive it from the server
# source failed (marker-based detection misreads Nitrite Reducing Bacteria as a
# collector and Dirigibles as a payer, because its effect text mentions
# megacredits without producing any). A card belongs here if its action only
# stacks resources and the payoff needs other cards.
# Deliberately NOT included: Jet Stream Microscrappers and Forced Precipitation
# (floaters convert to a Venus step, sink built in), Nitrite Reducing Bacteria
# (gives TR), Ecological Zone (tile plus VP), Livestock and Pollinators.
PURE_COLLECTORS = frozenset({
    "Dirigibles",            # floaters only placed/moved; payment use is situational
    "Aerial Mappers",        # 1 VP plus a card per two activations
    "Decomposers",           # microbes, 1 VP per 3
    "Venusian Animals",      # animals via science tags
    "Floating Habs",         # 0.5 MC yield per activation
    "Jovian Lanterns",       # 7 activations to break even
    "Ocean Sanctuary",       # 1 VP at a cost of 12
    "Extremophiles",         # 1 VP per 3 generations
    "Sub-Crust Measurements",# 7 activations to break even
    "Solarpedia",            # 1 VP per 3 activations
    "Pets",                  # only pays off from 6 cities on
})

# Bias buying towards cards that advance an in-play milestone or award.
LEVER_ALIGN               = True
# Devalue cards whose enabler is missing (see ENABLER_PENALTY).
LEVER_ENABLER             = True
# Early brake on greenery standard projects (see GREENERY_LATE_GEN).
LEVER_GREENERY_DISCIPLINE = True


CARD_DB: dict = {}

def load_card_db(path: str = "card_db.json"):
    global CARD_DB
    if not os.path.exists(path):
        log.warning("card_db.json not found - running without card evaluation")
        return
    with open(path, encoding="utf-8") as f:
        CARD_DB = json.load(f)
    log.info("Card database loaded: %d cards", len(CARD_DB))


ML_MODEL = None
ML_DEVICE = "cpu"


def load_ml_model(path: str = "tm_model.pt"):
    global ML_MODEL, ML_DEVICE
    if not TORCH_AVAILABLE:
        log.info("PyTorch not available - rule-based evaluation")
        return
    if not os.path.exists(path):
        log.info("No ML model found - rule-based evaluation")
        return
    ML_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ML_MODEL = load_model(path, ML_DEVICE)
    ML_MODEL.eval()
    log.info("ML-Modell geladen (device: %s)", ML_DEVICE)


# The server delivers production FLAT (megacreditProduction, steelProduction, ...);
# The field `production` does NOT exist on thisPlayer; the server schema is flat
# and silently received an empty dict: plant projection 0, production requirements never
# satisfied, MC production 0 when buying.
_PROD_FIELDS = {"megacredits": "megacreditProduction", "steel": "steelProduction",
                "titanium": "titaniumProduction", "plants": "plantProduction",
                "energy": "energyProduction", "heat": "heatProduction"}


def player_production(player: dict) -> dict:
    """Production dict of a player. Accepts either an already normalised `production`
    mapping (simulation) or the flat server schema.
    """
    p = player.get("production")
    if isinstance(p, dict) and p:
        return p
    return {res: player.get(field, 0) for res, field in _PROD_FIELDS.items()}


def card_info(name: str) -> dict:
    """Card features, or an empty dict if the card is unknown."""
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

# The updated server sometimes answers more slowly under --parallel load than the old
_HTTP_TIMEOUT = 30
_HTTP_RETRIES = 3

def get_state(base_url: str, player_id: str) -> dict:
    last = None
    for attempt in range(_HTTP_RETRIES):
        try:
            r = requests.get(f"{base_url}/api/player", params={"id": player_id},
                             timeout=_HTTP_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except (requests.Timeout, requests.ConnectionError) as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise last


def post_input(base_url: str, player_id: str, payload: dict) -> dict:
    last = None
    for attempt in range(_HTTP_RETRIES):
        try:
            r = requests.post(
                f"{base_url}/player/input",
                params={"id": player_id},
                json=payload,
                timeout=_HTTP_TIMEOUT,
            )
            r.raise_for_status()
            return r.json()
        except (requests.Timeout, requests.ConnectionError) as e:
            last = e                       # network/timeout only -> safe to resend
            time.sleep(1.5 * (attempt + 1))
    raise last


# ---------------------------------------------------------------------------
# Spielzustand-Helfer
# ---------------------------------------------------------------------------

def get_plant_cost(state: dict) -> int:
    corp = state.get("pickedCorporationCard", [])
    if any(c.get("name") == "Ecoline" for c in corp):
        return 7               # Ecoline lowers the greenery cost to 7 (not 6)
    return 8


def can_convert_plants(state: dict) -> bool:
    return state["thisPlayer"].get("plants", 0) >= get_plant_cost(state)


def can_convert_heat(state: dict) -> bool:
    player = state["thisPlayer"]
    temp   = state["game"].get("temperature", -30)
    return player.get("heat", 0) >= 8 and temp < 8


def thermalist_hold_value(state: dict) -> float:
    """Lead of the active player on the Thermalist award (most heat = 5 VP), but only
    while the award is still WINNABLE: in play AND either already funded or still
    fundable (fewer than three awards funded, at most three per game). A funded
    award carries 'color'/'playerName'.

    Returns own minus best opposing heat score; > 0 means leading a winnable
    Thermalist. 0 when it is not in play, no longer winnable, or the bot is behind.
    """
    game = state.get("game", {}) or {}
    awards = game.get("awards", []) or []
    therm = next((a for a in awards if a.get("name") == "Thermalist"), None)
    if therm is None:
        return 0.0                                   # not in play
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
    """The game options (server: GameModel.gameOptions). Some of them change the RULES:
      requiresVenusTrackCompletion  the game does not end before Venus is complete
                                    -> longer game, Venus terraforming is mandatory
      solarPhaseOption              World Government: global parameters rise every generation
                                    automatically -> requirements are met sooner
      escapeVelocity / soloTR / twoCorpsVariant / fastModeOption / undoOption ...
    Board bonuses, milestones and awards are read dynamically from the server anyway."""
    return (state.get("game", {}) or {}).get("gameOptions", {}) or {}


def turns_left(state: dict) -> int:
    game = state["game"]
    tl = game.get("lastSoloGeneration", 14) - game.get("generation", 1)
    # Mandatory Venus: the game only ends once Venus is complete as well, so the game runs
    # longer than the plain generation estimate. Extend conservatively by the missing Venus
    # steps (one step ~ one generation) so late engines are not undervalued.
    # are not wrongly written off as unreachable.
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


# --- Reasoning log (env TMBOT_RLOG): diagnosis of WHY an engine was or was not bought. ---
_RLOG = os.environ.get("TMBOT_RLOG")
_rlog_bd: dict = {}   # last engine value components per card name (from score_card)
_DUMP_WF = os.environ.get("TM_DUMP_WF")   # diagnostics: dump the full waitingFor structure

# --- A/B telemetry (env TM_TELEM): one line per player and generation. Purely additive. ---
_TELEM = os.environ.get("TM_TELEM")
_TELEM_ZERO = {"sp_spend": 0.0, "sp_n": 0, "cards_n": 0, "last_gen": 0, "card_spend": 0.0,
               "buy_n": 0, "offer_n": 0, "pass_n": 0, "pass_with_cards_n": 0,
               "action_n": 0, "sell_n": 0}
_telem: dict = {}          # pid -> counters (see _TELEM_ZERO)

_draft_choice_cache: dict = {}
# Cards the server REJECTED in a draft answer ("Card <name> not found"). The rejected
# card is named in the error text, so it is skipped and the next best one is taken.
# Cleared after every SUCCESSFUL post.
_draft_rejected: set = set()

def _draft_cache_key(state, cards):
    pid = state.get("id") or (state.get("thisPlayer") or {}).get("color")
    names = tuple(sorted(c.get("name", "") for c in cards))
    return (pid, names)


def _telem_note(kind: str, cost: float = 0.0, pid: str | None = None) -> None:
    """Count a chosen action (only when TM_TELEM is set)."""
    if not _TELEM or pid is None:
        return
    d = _telem.setdefault(pid, dict(_TELEM_ZERO))
    if kind == "sp":
        d["sp_spend"] += cost
        d["sp_n"]     += 1
    elif kind == "card":
        d["cards_n"]    += 1
        d["card_spend"] += cost           # printed price of the card played
    elif kind == "pass":
        d["pass_n"] += 1
    elif kind == "pass_with_cards":       # passed although a card was playable
        d["pass_n"]            += 1
        d["pass_with_cards_n"] += 1
    elif kind == "offer":
        d["offer_n"] += int(cost)         # cards offered in draft or research
    elif kind == "action":
        d["action_n"] += 1                # ACTIVE-Kartenaktion abgedrueckt
    elif kind == "sell":
        d["sell_n"] += 1
    elif kind == "buy":
        d["buy_n"]   += int(cost)         # cards actually bought


def _telem_gen(state: dict) -> None:
    """Writes one line per player at the GENERATION CHANGE. The aggregator takes the last
    line per (module, pid) as the end of the game."""
    if not _TELEM:
        return
    try:
        game = state.get("game") or {}
        me   = state.get("thisPlayer") or {}
        pid  = state.get("id")                     # unique per game, not the colour
        gen  = game.get("generation", 0)
        d = _telem.setdefault(pid, dict(_TELEM_ZERO))
        if gen == d["last_gen"]:
            return
        d["last_gen"] = gen
        prod_sum = sum(me.get(f, 0) for f in
                       ("megacreditProduction", "steelProduction", "titaniumProduction",
                        "plantProduction", "energyProduction", "heatProduction"))
        with open(_TELEM, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "module":   __name__,
                "pid":      pid,
                "color":    me.get("color"),
                "game":     game.get("name"),
                "gen":      gen,
                "tr":       me.get("terraformRating", 0),
                "mc_prod":  me.get("megacreditProduction", 0),
                "prod_sum": prod_sum,
                "tableau":  len(me.get("tableau") or []),
                "cards_played":  d["cards_n"],
                "card_spend":    round(d["card_spend"], 1),
                "cards_bought":  d["buy_n"],
                "cards_offered": d["offer_n"],
                "pass_n":            d["pass_n"],
                "pass_with_cards_n": d["pass_with_cards_n"],
                "actions_used":      d["action_n"],
                "cards_sold":        d["sell_n"],
                "hand":     me.get("cardsInHandNbr", len(state.get("cardsInHand") or [])),
                "sp_spend": round(d["sp_spend"], 1),
                "sp_n":     d["sp_n"],
                "r_eff":    round(_remaining_gens(game)[0], 1),
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass

# Extracted from the server card definitions: cards whose action is purely "add a
# resource to this card" (free, no steal, no condition). Such an activation is always
# better than passing, so it must not lose against the pass floor.
# Only those with action_once < 4 were affected; the others clear the floor anyway and
# Ants (steal), Security Fleet / Water Import (cost) and Regolith Eaters (mode) are
# are not listed here.
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
    """Logs ACTIVE engines the bot OWNS whose action is not on offer (= not activatable
    right now), plus the resource and parameter context. That answers "why was it not
    activated": Physics Complex not offered with energy < 6 means lack of resources;
    a heat engine not offered at maximum temperature means the parameter is exhausted.
    Behaviour-neutral (only with TMBOT_RLOG)."""
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
            # Bundled option "Perform an action from a played card": the activatable
            # cards are in opt["cards"], not as an option title of their own.
            # detection taken verbatim from the working handler below
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


# ATTACK CARDS - what a card TAKES AWAY from the opponent. The card database only
# carries the owner's own cost, so without this table an attack card scores as pure
# expense and is practically never played.
#
# The value is capped by what the opponent ACTUALLY holds: Deimos Down removes 8 plants,
# but against an opponent holding 2 it is worth 2 (= 4 MC). That is why this is computed
# at runtime instead of being written statically into card_db.
_ATTACK = {
    # remove opponent plants (once, on play)
    "Aerial Lenses":        {"plants": 2},
    "Asteroid":             {"plants": 3},
    "Big Asteroid":         {"plants": 4},
    "Comet":                {"plants": 3},
    "Deepnuking":           {"plants": 3},
    "Deimos Down":          {"plants": 8},
    "Giant Ice Asteroid":   {"plants": 6},
    "Impactor Swarm":       {"plants": 2},
    "Metallic Asteroid":    {"plants": 4},
    "Mining Expedition":    {"plants": 2},
    "Small Asteroid":       {"plants": 2},
    # Virus is an OR card: "Remove up to 2 animals OR 5 plants from any player."
    "Virus":                {"plants": 5, "cardres": {"ANIMAL": 2}, "or": True},
    # reduce opponent production (once, on play). For Birds/Fish/Herbivores/Small Animals
    # the attack is also a ONE-OFF on play - the animal engine itself runs via action_once,
    # the attack is a bonus on top.
    "Asteroid Mining Consortium": {"prod": {"titanium": 1}},
    "Biomass Combustors":   {"prod": {"plants": 1}},
    "Birds":                {"prod": {"plants": 2}},
    "Cloud Seeding":        {"prod": {"heat": 1}},
    "Earthquake Machine":   {"prod": {"plants": 1}},
    "Energy Tapping":       {"prod": {"energy": 1}},
    "Fish":                 {"prod": {"plants": 1}},
    "Great Escarpment Consortium": {"prod": {"steel": 1}},
    "Hackers":              {"prod": {"megacredits": 2}},
    "Hackers:u":            {"prod": {"megacredits": 2}},
    "Heat Trappers":        {"prod": {"heat": 2}},
    "Herbivores":           {"prod": {"plants": 1}},
    "Power Supply Consortium": {"prod": {"energy": 1}},
    "Small Animals":        {"prod": {"plants": 1}},
    "Sub-zero Salt Fish":   {"prod": {"plants": 1}},
}
ATTACK_PLANT_VALUE = 2.0   # 1 zerstoerte Pflanze  = 2 M  (Damians Tabelle)
ATTACK_PROD_VALUE  = 4.0   # 1 gesenkte Gegner-Produktion = 4 M


# ── DYNAMISCHE EFFEKTE ("X pro Tag/Stadt/Kolonie") ─────────────────────────────────────────
# For these cards card_db carries EMPTY production/stock and only a marker
# ('production:dynamic'), which the evaluation has to resolve at runtime - otherwise only
# the cost of the card is visible and it scores far too low.
#
# Semantics from the server's Counter class:
#   tags (default) -> OWN tags, including this card itself ("including this" -> +1)
#   tags + all     -> tags of ALL players | tags + others -> opponents only
#   cities         -> ALL city tiles in play, not just the bot's own
#   cities + where -> 'onmars' / 'offmars'
#   colonies       -> own colonies
#   per: N         -> je N gezaehlte Einheiten 1 Schritt (abgerundet)
_DYNAMIC = {
    # what: 'prod' (dauerhafte Produktion) | 'stock' (einmalige Ressourcen)
    "Advanced Power Grid":  {"what": "prod",  "res": "megacredits", "tags": ["power"]},
    "Cartel":               {"what": "prod",  "res": "megacredits", "tags": ["earth"]},
    "Luna Metropolis":      {"what": "prod",  "res": "megacredits", "tags": ["earth"]},
    "Miranda Resort":       {"what": "prod",  "res": "megacredits", "tags": ["earth"]},
    "Martian Monuments":    {"what": "prod",  "res": "megacredits", "tags": ["mars"]},
    "Satellites":           {"what": "prod",  "res": "megacredits", "tags": ["space"]},
    "Sulphur Exports":      {"what": "prod",  "res": "megacredits", "tags": ["venus"]},
    "Power Grid":           {"what": "prod",  "res": "energy",      "tags": ["power"]},
    "Insects":              {"what": "prod",  "res": "plants",      "tags": ["plant"]},
    "Medical Lab":          {"what": "prod",  "res": "megacredits", "tags": ["building"], "per": 2},
    "Parliament Hall":      {"what": "prod",  "res": "megacredits", "tags": ["building"], "per": 3},
    "Lunar Mining":         {"what": "prod",  "res": "titanium",    "tags": ["earth"],    "per": 2},
    "Worms":                {"what": "prod",  "res": "plants",      "tags": ["microbe"],  "per": 2},
    "Galilean Waystation":  {"what": "prod",  "res": "megacredits", "tags": ["jovian"], "all": True},
    "Toll Station":         {"what": "prod",  "res": "megacredits", "tags": ["space"],  "others": True},
    "Energy Saving":        {"what": "prod",  "res": "energy",      "cities": True},
    "Zeppelins":            {"what": "prod",  "res": "megacredits", "cities": True, "where": "onmars"},
    "Off-World City Living":{"what": "prod",  "res": "megacredits", "cities": True, "where": "offmars"},
    "Orbital Power Grid":   {"what": "prod",  "res": "energy",      "cities": True, "where": "offmars"},
    "Interplanetary Transport": {"what": "prod", "res": "megacredits", "cities": True, "where": "offmars"},
    "Cassini Station":      {"what": "prod",  "res": "energy",      "colonies": True},
    "Ecology Research":     {"what": "prod",  "res": "plants",      "colonies": True},
    # ── einmalige Ressourcen (stock) ──
    "Battery Factory":      {"what": "stock", "res": "megacredits", "tags": ["power"]},
    "Static Harvesting":    {"what": "stock", "res": "megacredits", "tags": ["building"]},
    "PR Office":            {"what": "stock", "res": "megacredits", "tags": ["earth"]},
    "Orbital Cleanup":      {"what": "stock", "res": "megacredits", "tags": ["science"]},
    "Protected Growth":     {"what": "stock", "res": "plants",      "tags": ["power"]},
    "Robot Pollinators":    {"what": "stock", "res": "plants",      "tags": ["plant"]},
    "Expedition to the Surface - Venus": {"what": "stock", "res": "megacredits", "tags": ["venus"]},
    "Diaspora Movement":    {"what": "stock", "res": "megacredits", "tags": ["jovian"], "all": True},
    "Greenhouses":          {"what": "stock", "res": "plants",      "cities": True},
    "Molecular Printing":   {"what": "stock", "res": "megacredits", "cities": True},
    "Aerosport Tournament": {"what": "stock", "res": "megacredits", "cities": True},
    "Luxury Estate":        {"what": "stock", "res": "titanium",    "cities": True},
    "Martian Rails":        {"what": "stock", "res": "megacredits", "cities": True, "where": "onmars"},
    "Weather Balloons":     {"what": "stock", "res": "megacredits", "cities": True, "where": "onmars"},
    "Ceres Tech Market":    {"what": "stock", "res": "megacredits", "colonies": True},
    "Colonial Representation": {"what": "stock", "res": "megacredits", "colonies": True},
    "Venus Allies":         {"what": "stock", "res": "megacredits", "colonies": True},
    # Modules that are rarely played are still listed for correctness
    # valued if they are activated:
    "HE3 Lobbyists":        {"what": "prod",  "res": "megacredits", "tags": ["moon"]},
    "Luna Senate":          {"what": "prod",  "res": "megacredits", "tags": ["moon"], "all": True},
    "Takonda Castle (VII)": {"what": "stock", "res": "megacredits", "tags": ["microbe", "animal"]},
    "Soil Studies":         {"what": "stock", "res": "plants",      "colonies": True},
    "Summit Logistics":     {"what": "stock", "res": "megacredits", "colonies": True},
}
# Effects deliberately left out of the tables above, so the health check does not report
# them as gaps. The reason is given per entry.
_EFFECT_IGNORE = {
    # triggers on a card's TAG COUNT, not on a tag TYPE - the per-tag-type probability
    # estimate does not fit structurally
    "Sagitta Frontier Services": "triggers on a card with EXACTLY 1 tag - tag COUNT, not type",
    "Spire":                     "triggers on a card with AT LEAST 2 tags - tag COUNT",
    # corporation with 5 tag alternatives; value highly game-dependent
    "Hecate Speditions":         "Underworld-Korporation, 5 Tag-Alternativen - Modul inaktiv",
}
# MC value per unit. Production is permanent, stock is one-off.
# TS src/common/TileType.ts: CITY_TILES
_CITY_TILE_TYPES = frozenset({2, 3, 20, 37, 43})
_DYN_PROD_M  = {"megacredits": 5.0, "steel": 8.0, "titanium": 10.0,
                "plants": 10.0, "energy": 7.0, "heat": 6.0}
_DYN_STOCK_M = {"megacredits": 1.0, "steel": 2.0, "titanium": 3.0,
                "plants": 2.0, "energy": 1.0, "heat": 1.0}


def _dynamic_value(name: str, state: dict) -> float:
    """MC value of a dynamic effect ("X per tag/city/colony"), computed at runtime."""
    d = _DYNAMIC.get(name)
    if not d:
        return 0.0
    me    = state.get("thisPlayer", {}) or {}
    game  = state.get("game", {}) or {}
    stats = _player_stats(state)
    n = 0

    if d.get("tags"):
        if d.get("others"):
            n = sum(sum((p.get("tags", {}) or {}).get(t, 0) for t in d["tags"])
                    for p in (state.get("players") or [])
                    if p.get("color") != me.get("color"))
        elif d.get("all"):
            n = sum(sum((p.get("tags", {}) or {}).get(t, 0) for t in d["tags"])
                    for p in (state.get("players") or []))
        else:
            n = sum((stats.get("tags", {}) or {}).get(t, 0) for t in d["tags"])
            # "including this": if the card itself carries one of the tags, it counts
            _own = [t.lower() for t in (card_info(name) or {}).get("tags", [])]
            if any(t in _own for t in d["tags"]):
                n += 1
    elif d.get("cities"):
        # ALL cities in play, not just the bot's own - optionally filtered to on/off Mars
        where = d.get("where")
        n = 0
        for s in (game.get("spaces") or []):
            # The player view has a FLAT schema (tileType/color), there is no `tile` subdict.
        # note tile type 1 is OCEAN, not city: CITY 2, CAPITAL 3, OCEAN_CITY 20
            # RED_CITY 37, NEW_HOLLAND 43} (src/common/TileType.ts).
            if s.get("tileType") not in _CITY_TILE_TYPES:
                continue
            offmars = s.get("spaceType") == "colony"
            if where == "onmars" and offmars:
                continue
            if where == "offmars" and not offmars:
                continue
            n += 1
    elif d.get("colonies"):
        n = stats.get("colonies", 0)

    per = d.get("per", 1)
    units = n // per
    if units <= 0:
        return 0.0

    if d["what"] == "stock":
        return units * _DYN_STOCK_M.get(d["res"], 1.0)

    # Permanent production: valued by the same CONTEXTUAL rules as any other production
    # (horizon- and sink-aware: a resource only counts while it can still be spent)
    # up to the projected end of the game can be used.
    game  = state.get("game", {}) or {}
    r_eff, gtt = _remaining_gens(game)
    _corp = state.get("pickedCorporationCard", []) or []
    plant_threshold = 7 if any(c.get("name") == "Ecoline" for c in _corp) else 8
    return _contextual_prod_value({d["res"]: units}, me, game.get("oxygenLevel", 0),
                                  r_eff, gtt, plant_threshold,
                                  game.get("temperature", -30))


# ── PLATZIERUNGS-ABHAENGIGE PRODUKTION (Mining Area / Mining Rights) ───────────────────────
# Server: MiningCard - the tile goes on a space with a steel OR titanium bonus and raises
# that production by 1. card_db cannot hold this statically (which resource depends on the
# board), so it is resolved at runtime, like the dynamic effects above.
# Mining Area additionally requires an adjacent OWN tile (not an ocean).
_MINING_CARDS = {
    "Mining Area":        {"adjacent_own": True},
    "Mining Area:ares":   {"adjacent_own": True},
    "Mining Rights":      {"adjacent_own": False},
    "Mining Rights:ares": {"adjacent_own": False},
}
# SpaceBonus-Enum (src/common/boards/SpaceBonus.ts)
SB_TITANIUM, SB_STEEL, SB_PLANT, SB_DRAW, SB_HEAT, SB_OCEAN, SB_MC = 0, 1, 2, 3, 4, 5, 6
# MC value of the ONE-OFF placement bonuses of a space (stock, not production).
_SPACE_BONUS_M = {SB_TITANIUM: 3.0, SB_STEEL: 2.0, SB_PLANT: 2.0,
                  SB_DRAW: 4.5, SB_HEAT: 1.0, SB_MC: 1.0}


def _free_land_spaces(game: dict) -> list[dict]:
    return [s for s in (game.get("spaces") or [])
            if s.get("spaceType") == "land" and s.get("tileType") is None]


def _mining_prod_value(name: str, state: dict) -> float:
    """MC value of Mining Area / Mining Rights: best reachable space with a steel or
    titanium bonus. Value = contextual production (+1 step of that resource) plus the
    placement bonuses of the space. No valid space -> 0."""
    cfg = _MINING_CARDS.get(name)
    if not cfg:
        return 0.0
    game   = state.get("game", {}) or {}
    me     = state.get("thisPlayer", {}) or {}
    spaces = game.get("spaces") or []
    if not spaces:
        return 0.0

    space_map = {s["id"]: s for s in spaces}
    adjacency = board_adjacency(space_map) if cfg["adjacent_own"] else {}
    my_color  = me.get("color")

    r_eff, gtt = _remaining_gens(game)
    _corp = state.get("pickedCorporationCard", []) or []
    plant_threshold = 7 if any(c.get("name") == "Ecoline" for c in _corp) else 8
    oxygen = game.get("oxygenLevel", 0)
    temp   = game.get("temperature", -30)

    best = 0.0
    for s in _free_land_spaces(game):
        bonus = s.get("bonus") or []
        if SB_TITANIUM not in bonus and SB_STEEL not in bonus:
            continue
        if cfg["adjacent_own"]:
            # server rule: adjacent to an OWN tile that is not an ocean
            own_adj = any(t is not None and t != TILE_OCEAN and c == my_color
                          for t, c in _neighbor_tiles(s["id"], space_map, adjacency))
            if not own_adj:
                continue
        # if both bonuses apply the player chooses -> titanium (3 MC per unit vs 2)
        res = "titanium" if SB_TITANIUM in bonus else "steel"
        v  = _contextual_prod_value({res: 1}, me, oxygen, r_eff, gtt, plant_threshold, temp)
        v += sum(_SPACE_BONUS_M.get(b, 1.0) for b in bonus)   # placement bonuses of the space
        best = max(best, v)
    return best


# ── PRODUKTIONS-KOPIE (Robotic Workforce) ──────────────────────────────────────────────────
# Server: RoboticWorkforceBase - copies the production box of ONE own building card.
# Value = best copyable box, evaluated in context. Boxes with negative parts the bot
# cannot cover are not copyable (server: getPlayableBuildingCards).
_COPY_PROD_CARDS = {"Robotic Workforce"}
_PROD_FIELD = {"megacredits": "megacreditProduction", "steel": "steelProduction",
               "titanium": "titaniumProduction", "plants": "plantProduction",
               "energy": "energyProduction", "heat": "heatProduction"}


def _copy_prod_value(name: str, state: dict) -> float:
    if name not in _COPY_PROD_CARDS:
        return 0.0
    me   = state.get("thisPlayer", {}) or {}
    game = state.get("game", {}) or {}
    r_eff, gtt = _remaining_gens(game)
    _corp = state.get("pickedCorporationCard", []) or []
    plant_threshold = 7 if any(c.get("name") == "Ecoline" for c in _corp) else 8
    oxygen = game.get("oxygenLevel", 0)
    temp   = game.get("temperature", -30)

    # server: isCardApplicable - events are excluded (unless Odyssey), WILD counts as building
    tableau = me.get("tableau") or []
    has_odyssey = any(c.get("name") == "Odyssey" for c in tableau)

    best = 0.0
    for c in tableau:
        ci = card_info(c.get("name", ""))
        if not ci:
            continue
        if (ci.get("type") or "").upper() == "EVENT" and not has_odyssey:
            continue
        tags = [str(t).upper() for t in (ci.get("tags") or [])]
        if "BUILDING" not in tags and "WILD" not in tags:
            continue
        prod = ci.get("production") or {}
        if not prod:
            continue
        # negative parts must be covered (MC production may go to -5)
        ok = True
        for res, delta in prod.items():
            if delta < 0:
                have  = me.get(_PROD_FIELD.get(res, ""), 0)
                floor = -5 if res == "megacredits" else 0
                if have + delta < floor:
                    ok = False
                    break
        if not ok:
            continue
        best = max(best, _contextual_prod_value(prod, me, oxygen, r_eff, gtt,
                                                plant_threshold, temp))
    return best


# ── TRIGGER-EFFEKTE auf Stadt-Platzierungen ────────────────────────────────────────────────
# "When a city tile is placed, ..." - the value depends on how many cities are STILL TO COME.
# City rate taken from real game logs (two players): about 0.25 cities per generation.
VP_VALUE = 5.0            # 1 VP = 5 M (Bot-Konvention)

# ── TR-HORIZONT ────────────────────────────────────────────────────────────────────────────
# card_db values every TR step flat at 10 MC, which is horizon-blind. A TR step is worth
# more early than late: it pays out once per remaining production phase.
# True value of a TR step: 1 VP at the end plus 1 MC income per remaining
# Early (r_eff ~11) that is ~15 MC, in the last generation only ~5-6 MC.
# Applies to CARD evaluation only; score_action (heat->temperature, ocean project ...)
# deliberately stays at 10 MC, where the observed behaviour is already right.
# Robust against horizon error: TR and production scale with the same horizon, so their
# RATIO stays stable. If the horizon is too long, TR is if anything slightly undervalued.
LEVER_TR_HORIZON = True
TR_BGG_M = 10.0          # flacher Satz in card_db (score_breakdown: tr / global_* / ocean / greenery)


def _tr_value(r_eff: float) -> float:
    """MC value of ONE TR step, horizon-dependent. Consistent with _contextual_prod_value:
    income counts over the full horizon minus the last generation (tiebreaker only)."""
    return VP_VALUE + 1.0 * max(0.0, r_eff - 1.0)


def _tr_bgg_in_card(info: dict) -> float:
    """MC share of score_total that comes from TR STEPS, at the flat card database rate.
    NOT included: placement bonuses (ocean = 14 -> 10 TR plus 4 adjacency).
    """
    bd  = info.get("score_breakdown") or {}
    tr  = float(bd.get("tr", 0.0))
    tr += float(bd.get("global_temperature", 0.0))
    tr += float(bd.get("global_oxygen", 0.0))
    tr += float(bd.get("global_venus", 0.0))               # Venus discount - scales with
    tr += TR_BGG_M * float(info.get("oceans", 0) or 0)     # je Ozean 1 TR
    tr += TR_BGG_M * float(info.get("greenery", 0) or 0)   # je Greenery 1 O2-Schritt
    return tr
CITY_RATE_PER_GEN = 0.25
_CITY_TRIGGER = {
    # MC value per future city
    "Immigrant City":    {"prod": {"megacredits": 1}},   # +1 M€-PRODUKTION je Stadt (dauerhaft!)
    "Rover Construction": {"mc": 2},                     # +2 M€ je Stadt (einmalig)
    "Pets":              {"vp_per": 2},                  # +1 Tier je Stadt -> 1 VP je 2 Tiere
}


def _city_trigger_value(name: str, state: dict) -> float:
    """Value of a "whenever a city is placed" trigger: expected FUTURE cities x value.
    A city the card places itself counts too."""
    trg = _CITY_TRIGGER.get(name)
    if not trg:
        return 0.0
    game  = state.get("game", {}) or {}
    r_eff, _gtt = _remaining_gens(game)
    # expected cities until the end of the game (all players) plus the one this card places
    exp_cities = r_eff * CITY_RATE_PER_GEN
    if name == "Immigrant City":
        exp_cities += 1.0                     # places a city itself ("including this")
    if exp_cities <= 0:
        return 0.0

    if "prod" in trg:
        # permanent production, but it only starts once the tile exists -> on average only
        # half of the remaining game is usable
        me = state.get("thisPlayer", {}) or {}
        _corp = state.get("pickedCorporationCard", []) or []
        plant_threshold = 7 if any(c.get("name") == "Ecoline" for c in _corp) else 8
        val = 0.0
        for res, per_city in trg["prod"].items():
            steps = exp_cities * per_city
            val += _contextual_prod_value({res: steps}, me, game.get("oxygenLevel", 0),
                                          r_eff * 0.5, _gtt, plant_threshold,
                                          game.get("temperature", -30))
        return val
    if "mc" in trg:
        return exp_cities * trg["mc"]
    if "vp_per" in trg:
        return (exp_cities / trg["vp_per"]) * VP_VALUE
    return 0.0


# ── TAG-TRIGGER ("Whenever you play a X tag, ...") ─────────────────────────────────────────
# Cards whose ENGINE is the playing of one's own tag cards. card_db does not capture the
# -> alle stark negativ bewertet (Decomposers -8, Martian Zoo -10, Venusian Animals -18,
# trigger, so without this table they score as pure cost and are practically never played.
#
# Expected triggers = (cards per generation) x (share of that tag in the deck) x remaining gens.
# Beide Basiswerte GEMESSEN:
#   cards per generation: 1.8, a deliberately conservative estimate from game logs
#   tag share: from card_db (building .30, space .22, science .17, earth .13, power .11,
#               venus .11, plant .11, microbe .07, city .07, jovian .06, animal .05)
CARDS_PER_GEN = 1.8
_TAG_SHARE = {"building": .30, "space": .22, "science": .17, "earth": .13, "power": .11,
              "venus": .11, "plant": .11, "microbe": .07, "city": .07, "jovian": .06,
              "animal": .05, "mars": .10, "moon": .02, "crime": .01}

# gain: what ONE trigger yields.
#   vp_per: N  -> the resource gives 1 VP per N units  |  mc: direct megacredits
#   res: Ressourcenwert in M (Pflanze 2, Hitze 1, ...)  |  saving: Kartenkosten-Rabatt
_TAG_TRIGGER = {
    # A) resource collectors (resource onto this card -> VP)
    "Decomposers":              {"tags": ["animal", "plant", "microbe"], "vp_per": 3},
    "Martian Zoo":              {"tags": ["earth"],   "vp_per": 2},
    "Venusian Animals":         {"tags": ["science"], "vp_per": 1},
    "Carbon Nanosystems":       {"tags": ["science"], "vp_per": 2},
    "Titan Manufacturing Colony": {"tags": ["jovian"], "res": 2.0},   # Werkzeug -> Floater-artig
    "Space Privateers":         {"tags": ["crime"],   "res": 2.0},
    "Terraforming Robots":      {"tags": ["mars"],    "res": 2.0},
    "Stem Field Subsidies":     {"tags": ["science"], "res": 2.0},
    "Collegium Copernicus":     {"tags": ["science"], "res": 2.0},
    "Robin Haulings":           {"tags": ["venus"],   "res": 2.0},
    "The Archaic Foundation Institute": {"tags": ["moon"], "res": 2.0},
    # B) cost reducers (discount on the next card carrying the tag)
    "Earth Office":             {"tags": ["earth"],  "saving": 3.0},
    "Solar Logistics":          {"tags": ["earth"],  "saving": 2.0},
    "Venus Waystation":         {"tags": ["venus"],  "saving": 2.0},
    "Space Lanes":              {"tags": ["jovian", "earth", "venus"], "saving": 2.0},
    "Terraforming Control Station": {"tags": ["venus", "mars"], "saving": 2.0},
    # C) direkter Gewinn je Trigger
    "Albedo Plants":            {"tags": ["plant"],  "res": 3.0},     # 3 Hitze ~ 3 M
    "Viral Enhancers":          {"tags": ["plant", "microbe", "animal"], "res": 2.0},
    "EcoTec":                   {"tags": ["plant", "microbe", "animal"], "res": 2.0},
    "Ambient":                  {"tags": ["venus"],  "res": 3.0},
    "Space Relay":              {"tags": ["jovian"], "res": 3.0},     # 1 Karte ziehen
    "Mars University":          {"tags": ["science"], "res": 1.5},    # Handkarten-Tausch
}


def _tag_trigger_value(name: str, state: dict) -> float:
    """Value of a "whenever you play an X tag" trigger: expected triggers x value each."""
    trg = _TAG_TRIGGER.get(name)
    if not trg:
        return 0.0
    game = state.get("game", {}) or {}
    r_eff, _ = _remaining_gens(game)
    if r_eff <= 0:
        return 0.0
    # probability that a played card carries one of the trigger tags
    p = min(0.9, sum(_TAG_SHARE.get(t, 0.05) for t in trg["tags"]))
    triggers = CARDS_PER_GEN * p * r_eff + 1.0     # +1 for the card itself
    if triggers <= 0:
        return 0.0

    if "vp_per" in trg:
        return (triggers / trg["vp_per"]) * VP_VALUE
    if "saving" in trg:
        return triggers * trg["saving"]
    return triggers * trg.get("res", 1.0)


def _attack_value(name: str, state: dict) -> float:
    """MC value of an attack, CAPPED by what the OPPONENT actually holds.
    (Deimos Down removes 8 plants; against an opponent holding 2 it is worth 4 MC.)

    The two kinds of attack behave DIFFERENTLY:
      * PLANTS (RemoveAnyPlants): there is a skip option, so the bot never has to
        destroy its own plants. With no opponent plants the value is simply 0.
      * PRODUCTION (DecreaseAnyProduction): there is NO skip option. If the bot is
        the ONLY valid target it MUST hit itself, so the 'attack' is damage and
        counts negative. If nobody can be hit at all, nothing happens -> 0.
    """
    atk = _ATTACK.get(name)
    if not atk:
        return 0.0
    me   = state.get("thisPlayer", {}) or {}
    opps = [p for p in (state.get("players") or [])
            if p.get("color") != me.get("color")]

    val = 0.0
    # ── plants: upside only, never self-damage (the server offers a skip option) ──
    plant_val = 0.0
    if "plants" in atk and opps:
        best = max((p.get("plants", 0) or 0) for p in opps)
        plant_val = min(atk["plants"], best) * ATTACK_PLANT_VALUE

    # ── card resources (animals, microbes) taken off an opponent CARD ──
    # Removing them destroys victory points: most animal cards score 1 VP per animal,
    # some 1 VP per two (vp_dyn.per). The server removes from ONE card, so the best
    # single target decides. Skippable as well, hence no self-damage.
    cardres_val = 0.0
    for _rt, _cnt in (atk.get("cardres") or {}).items():
        for _p in opps:
            for _c in (_p.get("tableau") or []):
                _have = _c.get("resources", 0) or 0
                _info = CARD_DB.get(_c.get("name", ""), {}) or {}
                if _have <= 0 or (_info.get("res_type") or "").upper() != _rt:
                    continue
                _per = float(((_info.get("vp_dyn") or {}).get("per") or 1)) or 1.0
                cardres_val = max(cardres_val, min(_cnt, _have) * VP_VALUE / _per)

    # An OR card grants only ONE of its branches, so the bot takes the better one;
    # everything else stacks its effects.
    val += max(plant_val, cardres_val) if atk.get("or") else (plant_val + cardres_val)

    # ── production: bonus taken from the opponent OR forced damage to oneself ──
    _FIELD = {"megacredits": "megacreditProduction", "steel": "steelProduction",
              "titanium": "titaniumProduction", "plants": "plantProduction",
              "energy": "energyProduction", "heat": "heatProduction"}
    for res, cnt in (atk.get("prod") or {}).items():
        field = _FIELD.get(res, res)
        floor = -5 if res == "megacredits" else 0      # M€-Prod darf bis -5
        opp_room = max(((p.get(field, 0) or 0) - floor for p in opps), default=0)
        if opp_room > 0:
            val += min(cnt, opp_room) * ATTACK_PROD_VALUE          # Gegner treffbar -> Bonus
        else:
            # opponent not targetable. If the bot is itself a valid target it MUST hit
            # itself -> full damage. Otherwise the effect fizzles (0).
            own_room = max(0, (me.get(field, 0) or 0) - floor)
            val -= min(cnt, own_room) * ATTACK_PROD_VALUE
    return val


def score_card(card: dict, state: dict) -> float:
    """
    Hybrid: max(ML-Score, regelbasierter Score).
    Prevents the model from pushing every hand card negative.
    """
    rules_score = _score_card_rules(card, state)
    if ML_MODEL is not None:
        try:
            ml_raw = score_card_ml(card, state)
            # the model returns normalised values (~-3..+3); scale onto the rule-based scale
            ml_score = ml_raw * 8.0
            return max(ml_score, rules_score)
        except Exception as e:
            log.debug("ML-Score Fehler: %s", e)
    return rules_score


PROD_CAP = 6.0   # gedeckelter Produktions-Ernte-Horizont (Grenzertrag, ~BGG-Horizont)

# ── MC PRODUCTION: THE ONLY UNCAPPED HORIZON ───────────────────────────────────────────────
#
# ═══ VERWORFEN (A/B 13.07.: -2.02 VP [95%-CI -2.97 .. -1.08], n=280 Paare) ══════════════════
# ════════════════════════════════════════════════════════════════════════════════════════════
# Heat and energy are NOT building-card limited the way steel and titanium are
# converts to heat on its own, heat to temperature). Hence a separate, longer horizon
# than PROD_CAP: the realistic share of the ~19-step temperature track in a two-player game.
HEAT_GENS_CAP = 9.0


# Starting values of the global parameters (game rule, independent of player count).
_PARAM_START = {"temperature": -30, "oxygen": 0, "oceans": 0, "venus": 0}
_PARAM_FIELD = {"temperature": "temperature", "oxygen": "oxygenLevel",
                "oceans": "oceans", "venus": "venusScaleLevel"}


def _param_rate(game: dict, param: str) -> float:
    """Steigerungsrate eines globalen Parameters pro Generation - SELBST-KALIBRIEREND.
    Derives the rate from the OBSERVED course of the running game (total progress since
    the start divided by elapsed generations) and blends it with the two-player prior
    until enough has been observed. That makes it player-count agnostic: with six players
    the parameters rise faster -> higher rate -> shorter horizon, automatically.
    Early on the prior dominates, from about generation 5 the observation does.
    With the flag off, the fixed two-player prior applies."""
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
    w = min(1.0, elapsed / 4.0)          # confidence in the observation grows over time
    blended = (1.0 - w) * prior + w * obs
    # Endgame acceleration: the most recent rate (last ~2 generations from the server's
    # globalsPerGeneration history) captures the heat dump and greenery surge that a
    # average washes out. max() -> a shorter horizon ONLY when the recent rate is higher;
    # whole-game average washes out. During the slow start it is <= the average anyway.
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


# ── LENGTH CAP FOR THE HORIZON ─────────────────────────────────────────────────────────────
# Pure parameter extrapolation alone is a poor estimator of the remaining game length:
# it systematically overshoots and sticks to its upper bound for most of the game.
# Cause: OCEANS. They are placed only by card effects and arrive in bursts near the end
# (typically ten generations stuck at 1-3, then 1 -> 9 within two generations), whereas
# heat and oxygen partly terraform themselves (energy -> heat -> temperature, plants ->
# greenery -> oxygen). Extrapolating a linear rate from a bursty quantity overestimates.
# This matters because the horizon multiplies the ENTIRE production valuation, while TR
# is priced flat - too long a horizon prices production too high relative to TR.
# Gemessen (4 Partien vs. apehead, 50 Datenpunkte): Bias +6.12 -> -0.02, MAE 6.12 -> 0.46.
#
# ═══ VERWORFEN (A/B 13.07.: -12.06 VP [95%-CI -15.05 .. -9.06], n=35 Paare) ═════════════════
# TWO reasons, both fundamental:
#  2) SKALEN-KOPPLUNG. r_eff deflationiert JEDEN Kartenscore um ~30 %. PASS_SCORE,
#     0.7 vs 1.2, Prod-Summe 16 vs 32, M€ in SPs 336 vs 258. TR blieb UNVERAENDERT (23.0 vs
# ════════════════════════════════════════════════════════════════════════════════════════════


def _remaining_gens(game: dict) -> tuple[float, float]:
    """Projects the remaining generations from the parameter state instead of a fixed 14.
    The game ends when the SLOWEST parameter reaches its maximum -> max(...).
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
    """Projected generations until a card's global requirements are met (0 = none or
    already met). A closed max window means effectively never."""
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
    """up to the projected end of the game."""
    eff = min(r_eff, PROD_CAP)
    full_gens   = max(0.0, r_eff - 1.0)          # MC: full horizon (always spendable)
    mc_gens     = full_gens                       # M€: universal currency, full horizon
    capped_gens = max(0.0, eff - 1.0)            # Stahl/Titan: gekappt (Baukarten-limitiert)
    # Heat/energy: the generations-to-temperature-max figure comes from the OBSERVED rate,
    # which includes the opponent. If a fast opponent runs the temperature up, that figure
    # collapses and heat would be valued at nothing - the bot would concede the temperature
    # (self-reinforcing). The remaining steps are a CONTESTED pool, so a floor applies.
    # track, which is self-reinforcing. The remaining steps are a CONTESTED pool, so a floor
    # of (8 - temp) / 2 applies. Do NOT cap this with PROD_CAP (the steel/titanium building
    # card limit): energy and heat are not building-card limited, hence their own cap.
    steps_left  = max(0.0, (8 - temp) / 2.0)
    heat_gens   = min(r_eff, max(gtt, steps_left), HEAT_GENS_CAP)
    # Energy counts as heat: the one-generation delay (energy becomes heat only in the next
    # production phase) is offset by OPTIONALITY - energy can become heat or feed expensive
    # card actions (Physics Complex, Ironworks). Net equivalent, so no delay deduction and
    # no bonus beyond it either.
    energy_gens = heat_gens
    steel_v = player.get("steelValue", 2) or 2
    titan_v = player.get("titaniumValue", 3) or 3

    # The FIRST unit of MC production is existential, the sixteenth almost irrelevant, so
    # scarcity scales its value. Applies to the MC SHARE of a card only - scaling the whole
    # range (Marketing Experts 45.0 whether own MC production is 0 or 16).
    # card would drag VP and TR parts along with it.
    # cannot be bought. The factor applies to the MC SHARE of a card only.
    _mc_scarcity = 1.0
    if LEVER_MC_SCARCITY:
        _mc_now = player.get("megacreditProduction", 0) or 0
        if _mc_now < MC_SCARCITY_FLOOR:
            _mc_scarcity = 1.0 + MC_SCARCITY_BONUS * (
                (MC_SCARCITY_FLOOR - _mc_now) / MC_SCARCITY_FLOOR)
    v  = prod.get("megacredits", 0) * 1.0     * mc_gens * _mc_scarcity
    v += prod.get("steel", 0)       * steel_v * capped_gens
    v += prod.get("titanium", 0)    * titan_v * capped_gens
    v += prod.get("heat", 0)        * 1.25    * heat_gens      # 1/8 TR * 10M
    v += prod.get("energy", 0)      * 1.25    * energy_gens

    dplant = prod.get("plants", 0)
    if dplant:
        cur_plants = player.get("plants", 0)
        plant_prod_total = player_production(player).get("plants", 0) + dplant
        reach = r_eff + 1.0                        # echter Horizont + finale Runde
        projected = cur_plants + plant_prod_total * reach
        greenery = 5.0 + (10.0 if oxygen < 14 else 0.0)   # 1 VP + ggf. 1 O2-TR
        if projected >= plant_threshold:
            v += dplant * (greenery / 8.0) * reach            # voll konvertierbar
        elif LEVER_PLANT_ENGINE:
            # Soft ramp instead of a hard 0: partial credit by closeness to the threshold.
            # keeps the FIRST plant card worth buying from mid-game on (engine start)
            # Factor <= 1, so already valid cases stay unchanged.
            factor = max(0.0, projected / plant_threshold)
            v += dplant * (greenery / 8.0) * reach * factor
        # otherwise (flag off): plant production stays worthless (0)
    return v


def _score_card_rules(card: dict, state: dict) -> float:
    """
    Situational value of a card.

    Kombiniert:
    - base score from card_db (production, TR, VP)
    - Situative Anpassungen (Generationen verbleibend, Parameter-Stand)
    - Korporation-Synergien
    - requirement check (is the card playable?)
    """
    name = card.get("name", "")
    cost = card.get("calculatedCost", 0)
    info = card_info(name)

    if not info:
        # card not in the database: neutral score based on cost
        return -cost * 0.3

    game   = state["game"]
    player = state["thisPlayer"]
    tl     = turns_left(state)
    oxygen = game.get("oxygenLevel", 0)
    temp   = game.get("temperature", -30)
    oceans = game.get("oceans", 0)
    mc_prod = player.get("megacreditProduction", 0)

    # Base score from the database, already fully cost-adjusted
    # (BGG unit values for production/TR/VP). No further cost deduction needed.
    score = float(info.get("score_total", 0))

    # credit discounts: score_total uses the printed price from the database,
    # calculatedCost already includes player discounts
    score += info.get("cost", cost) - cost

    # --- Situative Anpassungen (DB-Schema: production/global/oceans/tr/type) ---
    prod = info.get("production", {}) or {}
    glob = info.get("global", {}) or {}

    # Kontextuelle Produktionsbewertung: score_total enthaelt Produktion statisch zu
    # BGG-Werten (MC=5, Steel=8, Titan=10, Plant=10, Energy=7, Heat=6). Diesen
    # replace the static share with a horizon- and sink-aware value
    # (_contextual_prod_value): a resource only counts while it can still be spent
    # can actually be used up to the projected end of the game.
    # production is worth ~0 and sacrificing unused production (e.g. Strip Mine)
    # is correctly cheap.
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

    # TR HORIZON: same step as above for production, now for TR. card_db assumes a flat
    # 10 MC per step; the true value depends on the remaining game (see _tr_value).
    if LEVER_TR_HORIZON:
        _tr_bgg = _tr_bgg_in_card(info)
        if _tr_bgg:
            score += _tr_bgg * (_tr_value(r_eff) / TR_BGG_M - 1.0)

    # Action cards (type ACTIVE): repeatable action over the remaining game.
    # Restlaufzeit bewerten statt pauschal +8. action_once = Netto-M pro
    # activation (from card_db). Conservative activation rate as a
    # Tote-Kaeufe-Waechter (s. ACTION_ACTIVATION_RATE).
    if (info.get("type") or "").upper() == "ACTIVE":
        action_once = float(info.get("action_once", 0) or 0)
        # The action's CARD DRAW has to be counted here, both in the value and in the gate
        # below - otherwise cards whose whole point is opening the draw channel
        # (Inventors' Guild, Business Network, Development Center, AI Central) either fall
        # through the gate or never get credited for the draw.
        if LEVER_DRAW_VALUE:
            action_once += DRAW_CARD_VALUE * float(info.get("action_draw", 0) or 0)
        # action_once is NOT net: if the action costs something, that cost sits in the
        # action_once is NOT net: if the action costs something, that cost sits in the
        # SEPARATE field `action_prod`, which must be charged here. Without it a card whose
        # action yields 5 and costs 5 (Refugee Camps) looks strong instead of worthless.
        if LEVER_ACTION_COST:
            action_once += float(info.get("action_prod", 0) or 0)
        # ★ LEVER_RESOURCE_SYNERGY (20.07., apeheads Caveat): "Einzelne Floater-/
        # Microbe cards are weak alone, but once the resources can be collected AND
        # A pure resource collector is weak on its own and strong in combination, so a FIXED
        # single value is necessarily wrong - too high while the card stands alone, too low
        # sobald mehrere zusammenkommen. Deshalb kontextabhaengig statt kuratiert:
        # once several of them are on the table. Hence the count of cards of the SAME
        # resource type already in play decides how much the collecting action is worth.
        # Applies only to cards with their own res_type; cards with a direct payout
        # (money, TR, card draw) are unaffected.
        if LEVER_RESOURCE_SYNERGY and action_once > 0 and name in PURE_COLLECTORS:
            _rt = info.get("res_type")
            if _rt:
                _eigene = 0
                for _c in (state.get("thisPlayer", {}).get("tableau") or []):
                    _ci = _action_card_info(_c.get("name") if isinstance(_c, dict) else _c)
                    if _ci and _ci.get("res_type") == _rt:
                        _eigene += 1
                if _eigene < RES_SYNERGY_FLOOR:
                    _fehlt = (RES_SYNERGY_FLOOR - _eigene) / RES_SYNERGY_FLOOR
                    action_once *= (1.0 - RES_SYNERGY_DAMPING * _fehlt)
        if action_once > 0:
            # activations over the window after unlocking
            # blocked engines (Penguins: 8 oceans) only count generations
            # after the condition is met. Met early with other parameters still low
            gens_to_unlock = _gens_to_global_req(info, game)
            # Feedable resource-VP stacking cards are activated reliably, so their engine
            # every generation (their action beats passing) -> higher rate, so that
            # potential is not halved on buying. The fuel factor below still caps
            # feasibility; requirement-heavy cards stay low.
            rate = ACTION_ACTIVATION_RATE
            if LEVER_VP_ENGINE and (info.get("vp_dyn") or {}).get("kind") == "resources":
                rate = VP_STACK_ACTIVATION_RATE
            activations = max(0.0, r_eff - gens_to_unlock) * rate
            # Option value of VP engines that unlock late: such a card WILL become
            # A dense resource-VP card behind a pure global parameter (Fish +2C,
            # Livestock/Predators oxygen, Penguins oceans) WILL unlock eventually,
            # playable - global parameters rise reliably in a real game, including
            # by the opponent. Even 2-3 late triggers pay against the capped downside.
            # playable, and the downside is capped (selling for ~2 MC). Hence a minimum
            # provided it unlocks within the horizon. Fuel still caps feedability;
            # tag- or production-gated cards do NOT get this floor, because those
            # requirements are not guaranteed to arrive.
            # reachable). gens_to_unlock == 99 = closed max window -> no floor.
            if (LEVER_LATE_ENGINE
                    and (info.get("vp_dyn") or {}).get("kind") == "resources"
                    and gens_to_unlock <= r_eff + 2.0):
                only_global = (info.get("req_global")
                               and not info.get("req_tags")
                               and not info.get("req_prod"))
                if only_global or not (info.get("requirements") or []):
                    activations = max(activations,
                                      min(LATE_ENGINE_MIN_ACTIVATIONS, r_eff))
            # Dead-buy guard: an engine the bot cannot fuel is worth less. If the input
            # (e.g. Ironworks spending 4 energy without energy production), their
            # resource is missing, the action cannot run, so its value is scaled down.
            # OR-actions ({}) are flexible and count as fuelable.
            fuel = 1.0
            prodp = player_production(player)
            _keeps_energy = any((c.get("name") == _ENERGY_KEEPER)
                                for c in (player.get("tableau") or []))
            for res, amt in (info.get("action_input") or {}).items():
                if not amt or amt <= 0:
                    continue
                supply = prodp.get(res, player.get(res + "Production", 0))
                if res == "energy" and not _keeps_energy:
                    # Energy does NOT accumulate (see ENERGY_RAMP_FUEL): either production
                    # alone covers the cost - then the action runs every generation - or it
                    # never runs at all. The current stock is good for at most ONE
                    # activation, this generation.
                    if supply >= amt:
                        f = 1.0
                    else:
                        f = ENERGY_RAMP_FUEL if (amt - supply) <= ENERGY_GAP_MAX else 0.0
                        if player.get("energy", 0) >= amt:
                            f = max(f, 1.0 / max(1.0, r_eff))
                    fuel = min(fuel, f)
                    continue
                if res != "energy":   # Stahl/Titan/Pflanzen: Lager anteilig dazu
                    supply += player.get(res, 0) / max(1.0, tl)
                fuel = min(fuel, supply / amt)
            fuel = max(0.0, min(1.0, fuel))
            # Threshold realism for resource-VP engines with per > 1 (Tardigrades per = 4):
            # resources BELOW the next full step are worth 0 VP, otherwise the bot buys
            # linear (action_once * activations) overvalues the stranded remainder
            # such a card and strands it just short of scoring.
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
            # Curated dead plays (action_once = 0, no real passive effect) do not
            # do NOT get the passive floor - otherwise the bot plays them and never acts
            if name not in _NO_PASSIVE_VALUE:
                score += 8   # passive ACTIVE effect with no quantified action

    # Direct TR increases (info["tr"]) are already counted at 10 MC in
    # score_total, so no re-add. Only: a parameter is worth less once maxed.
    if oxygen >= 14:
        score -= glob.get("oxygen", 0) * 12
    if temp >= 8:
        score -= glob.get("temperature", 0) * 12
    if oceans >= 9:
        score -= info.get("oceans", 0) * 12

    # Negative MC production: penalty depending on remaining generations
    if prod.get("megacredits", 0) < 0:
        malus = abs(prod["megacredits"]) * tl * 1.5
        score -= malus
        # extra penalty when production is already negative
        if mc_prod < 0:
            score -= 20

    # plant cards are especially valuable for Ecoline
    corp = state.get("pickedCorporationCard", [])
    if any(c.get("name") == "Ecoline" for c in corp):
        score += prod.get("plants", 0) * 5

    # plant production is worth less once oxygen is maxed
    if oxygen >= 14:
        score -= prod.get("plants", 0) * 3

    # (Award-Fortschritts-Zuschlag wieder entfernt: Screening 2026-06-07
    #  crowded-out milestone budgets, so it is not called.

    # VP penalty straight from the API card object (overrides card_db when present)
    # Some cards have negative VP: Nuclear Zone (-2), Hackers (-1), ...
    api_vp = card.get('victoryPoint', card.get('vp', None))
    if api_vp is not None and api_vp < 0:
        db_vp = info.get('vp', 0)
        if abs(api_vp) > abs(db_vp):   # API value is worse -> use the API
            extra_penalty = (abs(api_vp) - abs(db_vp)) * 5
            score -= extra_penalty

    # (no extra cost deduction: score_total is already net)

    # Adjacency VP tiles: Commercial District (1 VP per adjacent city) and Capital
    # (1 VP per adjacent ocean) have vp = None/0 in the card database -> estimated here.
    # Placement maximises adjacency (see choose_best_space), so the expected value is used.
    # adjacency from the current board, discounted. Placeholder until the full
    # Extraktions-Durchgang (echte Adjazenz-Regel + platzierungs-bewusste Schaetzung).
    _nl = card.get("name", "").lower()
    if "commercial district" in _nl or _nl == "capital":
        _spaces = game.get("spaces", []) or []
        _want = TILE_CITY if "commercial" in _nl else TILE_OCEAN
        _n = sum(1 for s in _spaces if s.get("tileType") == _want)
        score += min(_n, 3) * 5.0 * 0.6   # ~1-2 erreichbare Nachbarn, 1 VP = 5M

    # Dynamic VP (generic, vp_dyn): cards with vp = 0 in the database that score per tag,
    # Stadt/Ressource skalieren (z.B. Io Mining Industries = 1 VP je Jovian-Tag).
    # ON BUYING the final count is unknown, so use a principled lower bound: the current
    # count plus the tags THIS card brings itself, rather than guessing a factor. 1 VP = 5 MC.
    # ends after this move"). Deliberately underestimates future growth rather than
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
        # resources: accumulation is 0 at buying time -> already covered via action_once
        if count:
            per = rule.get("per", 1) or 1
            score += (count // per) * rule.get("each", 1) * 5.0

    # Feeder synergy (contextual): cards with 'synergy_adds' put resources on ANOTHER card
    # (e.g. Large Convoy: +4 animal). Only worth something if a compatible resource engine
    # is owned (tableau) or held in the buy/keep context (hand).
    # waitingFor options). Without an engine -> 0. 1 VP = 5 MC.
    # NOTE: the data lives in 'synergy_adds' ({type:'Animal', count:N}), NOT in 'feeds'
    # Note the data lives in 'synergy_adds', not in 'feeds' - the latter is empty everywhere.
    synergy = info.get("synergy_adds")
    if synergy:
        eng_names = set()
        for c in (player.get("tableau") or []):
            n = c.get("name") if isinstance(c, dict) else c
            if n:
                eng_names.add(n)
        if state.get("_feed_include_hand"):
            for c in (hand_cards(state) or []):
                n = c.get("name") if isinstance(c, dict) else c
                if n:
                    eng_names.add(n)
            # waitingFor options = buy candidates (counted) or draft packs (NOT counted,
            # only one card would remain, so a pack neighbour is not a held engine
            if not state.get("_draft_ctx"):
                for opt in (state.get("waitingFor", {}) or {}).get("options", []) or []:
                    for c in (opt.get("cards") or []):
                        n = c.get("name") if isinstance(c, dict) else c
                        if n:
                            eng_names.add(n)
        # synergy_adds[].type is capitalised ("Animal"/"Microbe"/"Any"/...). Match against a
        # held vp_dyn resource engine via the matching tag; "Any" matches all.
        # (CEO) matches any vp_dyn engine. Floater/data/fighter: no tag match, no bonus.
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
            # synergy_mode distinguishes OR cards (max: only one option is taken, e.g.
            # Imported Hydrogen) from AND cards (sum: both adds, e.g. Imported Nitrogen).
            mode = info.get("synergy_mode", "max")
            score += (sum(_vals) if mode == "sum" else max(_vals)) * 5.0

    # Engine synergy (incentive field): cards whose tags feed the bot's own engine or
    # satisfy open tag requirements, and cards placing a wanted tile type, score higher.
    demand = _strategy_demand(state) if LEVER_INCENTIVE_FIELD else None
    if demand:
        syn = 0.0
        for t in info.get("tags", []):
            units = demand.get(str(t).lower(), 0.0)
            if units > 0:
                syn += min(units, TAG_DEMAND_CAP) * TAG_SYNERGY_UNIT
        # Tile-Platzierung erfuellt Tile-Nachfrage (on-play oceans/city/greenery
        # or action placers such as Aquifer Pumping). Ocean is headroom-gated.
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

    # Endgame VP floor: in the very last generation unspent money is lost, so an affordable
    # money is lost apart from the rare tiebreaker. An affordable
    # card with FIXED victory points is worth its VP - the cost deduction in score_total
    # wrongly assumes the money has an alternative use. Only fixed vp > 0, so that vp = 0
    # cards are not wrongly promoted.
    # cards with vp = 0 are not wrongly promoted. A floor, not a cap.
    # a floor, not a cap -> higher valuations stay untouched. Signal:
    if _is_last_generation(state):
        _vp = card.get("victoryPoint", card.get("vp", info.get("vp", 0))) or 0
        if _vp > 0 and cost <= player.get("megacredits", 0):
            score = max(score, _vp * VP_ENDGAME_VALUE)

    # ATTACK on the opponent (destroy plants / reduce production), capped by what the
    # opponent ACTUALLY holds (see _attack_value).
    score += _attack_value(card.get("name", ""), state)

    # DYNAMIC EFFECTS ("1 MC production per Earth tag"). There card_db holds EMPTY
    # It is computed at runtime from tags, cities and colonies.
    score += _dynamic_value(card.get("name", ""), state)

    # TRIGGER-EFFEKTE auf Stadt-Platzierungen (Immigrant City / Rover Construction / Pets):
    # expected future cities x value (rate from real logs: ~0.25 cities per generation).
    score += _city_trigger_value(card.get("name", ""), state)

    # TAG TRIGGERS ("whenever you play an X tag"): expected future triggers x value.
    score += _tag_trigger_value(card.get("name", ""), state)

    # PLACEMENT/COPY PRODUCTION: Mining Area and Mining Rights (the space bonus decides the
    # resource) and Robotic Workforce (copies an own building production box).
    # card_db (score_total 0, no production field), so neither effect nor cost was seen.
    score += _mining_prod_value(card.get("name", ""), state)
    score += _copy_prod_value(card.get("name", ""), state)

    # KARTENZIEHEN: card_db rechnet flach 1.0 M€ je gezogener Karte (s. LEVER_DRAW_VALUE).
    # Delta to the true value - like production, this replaces rather than adds.
    if LEVER_DRAW_VALUE:
        _dc = float(info.get("draw_cards", 0) or 0)
        if _dc:
            score += _dc * (DRAW_CARD_VALUE - DRAW_BGG_M)

    return score



# Estimated parameter progress per generation (two players, from logs: ~14 generations,
# Temp -30..+8, O2 0..14, Ozeane 0..9)
_PARAM_RATE = {"temperature": 2.7, "oxygen": 1.0, "oceans": 0.65, "venus": 1.0}
REQ_UNREACHABLE = -50.0
# Requirements that fulfil themselves over the course of the game must not be blocked hard,
# or the bot will never buy the card. Penalty instead of block - adjustable:
REQ_TAG_MALUS  = 5.0   # je fehlendem Tag  (ab 5 fehlenden Tags: doch unerreichbar)
REQ_PROD_MALUS = 6.0   # missing production requirement (production gets built up)


def _tag_count(player: dict, tag: str) -> int:
    tags = player.get("tags", [])
    if isinstance(tags, dict):
        return tags.get(tag, 0)
    return next((t.get("count", 0) for t in tags if t.get("tag") == tag), 0)


def _strategy_demand(state: dict) -> dict:
    """Engine-Nachfrage je Dimension (Tags kleingeschrieben + Tile-Dimensionen
    'oceans'/'cities'/'greenery') from CLEAN sources:
    (1) own vp_dyn tag cards (Io Mining scores per Jovian tag -> Jovian is in demand),
    (2) offene Tag-Voraussetzungen auf Hand/Tableau (Karte braucht 2 Science),
    (3) own tile_reward cards (Lakefront and Arctic Algae reward oceans ->
        Ozean-Platzierung gefragt; Tharsis/Pets -> Staedte; Herbivores -> Greenery).
    Memoised on the state (identical for all candidates within a turn)."""
    cached = state.get("_strat_demand")
    if cached is not None:
        return cached
    player = state.get("thisPlayer", {})
    demand: dict = {}
    owned = list(player.get("tableau") or player.get("playedCards") or []) + hand_cards(state)
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
    """Buy penalty for unmet requirements.
    - tags/production unmet: effectively do not buy (the bot builds few tags)
    - global parameters: estimate the generations until they unlock; never
      reachable -> do not buy, otherwise charge the production decay until then
    Global parameters rise on their own, so they are not penalised outright. Tag and
    production requirements stay gated, because the bot may never meet them.
    """
    player = state.get("thisPlayer", {})
    game   = state.get("game", {})
    pen    = 0.0

    # TAG requirements (e.g. AI Central: 3 science tags). TEMPORARY - tags are COLLECTED
    # Tags are COLLECTED over the game. A hard block (REQ_UNREACHABLE) poisons such cards
    # permanently, including for BUYING, so the bot never acquires them even though the
    # tags would arrive within a few generations. The penalty grows with the gap: one
    # missing tag barely matters, five are close to unreachable.
    for tag, need in (info.get("req_tags") or {}).items():
        _gap = need - _tag_count(player, tag)
        if _gap > 0:
            if _gap >= 5:
                return REQ_UNREACHABLE          # realistically no longer catchable
            pen -= REQ_TAG_MALUS * _gap

    # Production requirements (the card DEMANDS running production). Also temporary:
    # Production gets built up. Penalty instead of a block.
    prod_own = player_production(player)
    for res in (info.get("req_prod") or []):
        if prod_own.get(res, 0) <= 0:
            pen -= REQ_PROD_MALUS

    # Implicit production requirement: a card that REDUCES production needs enough of that
    # production to absorb the reduction (this is how the server checks it).
    # IMPORTANT - a TEMPORARY gate, not a permanent one: the bot BUILDS production
    # A hard block would prevent BUYING forever, so a penalty growing with the missing
    # (Ocean City, Electro Catapult, Cupola City, Business Network ...) wurden nie gekauft
    # margin is used instead. The server does not offer unplayable cards for play anyway.
    # (Same reasoning as for the Turmoil party requirements.)
    for res, delta in (info.get("production", {}) or {}).items():
        if delta < 0:
            floor = -5 if res == "megacredits" else 0
            missing = floor - (prod_own.get(res, 0) + delta)   # > 0 => not yet playable
            if missing > 0:
                pen -= 4.0 * missing        # je weiter weg, desto unattraktiver

    # Turmoil requirements (party/chairman/partyLeader). These live ONLY in
    # 'requirements'; req_tags, req_global and req_prod are empty for them.
    # NOTE: the ruling party CHANGES EVERY GENERATION, so this is a TEMPORARY requirement.
    # gate (like global parameters), NOT a permanent one (like missing tags).
    # A hard block would poison such cards permanently, including for buying.
    # A mild penalty keeps the card buyable; it gets played once its party is in power.
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
                    pen -= 6.0          # waiting for the change of government
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
                return REQ_UNREACHABLE   # window already closed
            continue
        dist = v - cur[p]
        if dist > 0:
            delay = max(delay, dist / _PARAM_RATE.get(p, 1.0))
    if delay > 0:
        if delay >= tl - 1:
            return REQ_UNREACHABLE       # will never become playable in time
        # charge the production value at unlock time rather than today
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
    Score for buying a card.
    Takes into account whether the card is playable in the remaining generations.

    for_initial_keep=True (starting hand): the card is a play option over the WHOLE
    remaining game for a price of 3 MC. Affordability now and (later reachable) unmet
    requirements are then irrelevant, so both deductions are skipped, and future plays
    gated by requirements (Birds, Great Dam) are not wrongly discarded.
    """
    state["_feed_include_hand"] = True     # buy/keep: held engines count for feeders
    base = score_card(card, state)
    state.pop("_feed_include_hand", None)
    tl = turns_left(state)
    player = state["thisPlayer"]
    mc = player.get("megacredits", 0)
    mc_prod = player_production(player).get("megacredits", 0)
    cost = card.get("calculatedCost", 0)

    # Requirements: when keeping the starting hand, skip global parameters only (they rise
    # Tag-/Produktions-Gate bleibt aktiv -> nie erfuellbare Keeps (Beam=Jovian,
    # Magnetic Field Generators = energy production) are discarded correctly.
    # anyway); tag and production gates stay active, so keeps that can never be fulfilled
    # resource-VP engines (Penguins/Birds/Fish). The global parameter rises on its own -
    # are still discarded correctly.
    info = CARD_DB.get(card.get("name", ""), {})
    skip_g  = for_initial_keep
    req_pen = _requirement_penalty(info, state, tl, skip_global=skip_g)
    if req_pen <= REQ_UNREACHABLE:   # hard block: never playable -> do not buy or keep
        return REQ_UNREACHABLE
    base += req_pen
    if not for_initial_keep:
        # affordability only for a normal buy: keeping a starting card is a 3 MC future play
        mc_in_2_gens = mc + mc_prod * 2
        if mc_in_2_gens < cost:
            base -= (cost - mc_in_2_gens) * 0.5

    # deduct the buying fee (3 MC, also when keeping)
    base -= 1.5

    # late game: stronger deduction for expensive cards that will never be played
    if not for_initial_keep and tl <= 3 and cost > mc:
        base -= (cost - mc) * 0.8

    # Deploy capacity: if more unplayed cards are already in hand than can realistically be
    # played in the remaining game, a further buy is dead capital. Bites automatically
    # harder late through the small remaining-turn count, without throttling early buys.
    if LEVER_BUY_DISCIPLINE and not for_initial_keep:
        hand = player.get("cardsInHandNbr", 0)
        capacity = tl * DEPLOY_CARDS_PER_GEN
        overflow = hand - capacity
        if overflow > 0:
            base -= overflow * DEPLOY_OVERFLOW_PENALTY

    # Enabler gate: heavily devalue combo cards without their enabler, including when
    # keeping the starting hand - the bot should not hold Insulation, Virus or Protected
    # Habitats at all while the enabler is missing.
    if LEVER_ENABLER:
        nm = card.get("name", "")
        if nm in _ENABLER_CARDS and not _enabler_ok(nm, state):
            base -= ENABLER_PENALTY

    # Milestone and award alignment: small bias towards cards that advance an in-play and
    # gewinnbares Ziel voranbringen (Legend->Events, Energizer->Energieprod, ...).
    base += _alignment_buy_bonus(card, state)

    return base


# ---------------------------------------------------------------------------
# Heuristiken
# ---------------------------------------------------------------------------

# Corporation preference based on win-rate statistics (weighted over 2-5 players)
# Value = weighted advantage over the expected win rate (1/N)
# Spielerzahl-adaptiv: Saturn+Tharsis konstant stark; Ecoline gut ab 4P
CORP_PRIORITY = {
    "Saturn Systems":              +0.116,  # strongest corporation across player counts
    "Tharsis Republic":            +0.109,  # Konsistent stark, tile-basiert
    "Ecoline":                     +0.092,  # especially good with 4-5 players
    "Credicor":                    +0.020,  # good with 2-3 players, weaker with 4-5
    "Mining Guild":                +0.018,
    "Interplanetary Cinematics":   -0.003,
    "Teractor":                    -0.014,
    "Thorgate":                    -0.029,
    "Phobolog":                    -0.032,
    "Helion":                      -0.056,  # Stark 2P, schwach 3-4P
    "United Nations Mars Initiative": -0.107,
    "Inventrix":                   -0.121,  # weakest corporation
}

def _estimate_corp_value(corp: dict) -> float:
    """Estimate the value of an unknown corporation from its starting money and
    starting production, using the standard resource values.

    Returns a value normalised against a 60 MC threshold -> [-0.15, +0.15]
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
    """Pick the strongest available corporation.

    Unknown corporations are estimated from their starting money. Player-count
    aware: Ecoline is better with 4-5 players, Credicor and Helion with 2.
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
        # case-insensitive lookup (server names vary: "ThorGate" vs "Thorgate")
        corp_key = next((k for k in CORP_PRIORITY if k.lower() == name.lower()), None)
        if corp_key:
            base = CORP_PRIORITY[corp_key]
        else:
            # unknown corporation: rule-based estimate
            base = _estimate_corp_value(corp)
            log.info("🏢 Unknown corporation '%s' - estimated: %.3f", name, base)

        score = base + adjustments.get(name, 0.0)
        if score > best_score:
            best_score = score
            best_name  = name

    chosen = best_name or options[0].get("name", "")
    log.info("🏢 Corporation chosen: %s (score=%.3f, %dP)", chosen, best_score, num_players)
    return chosen


def choose_preludes(options: list[dict], count: int, state: dict | None = None) -> list[str]:
    """Pick the 'count' most valuable preludes. Prefers score_card (live and context
    aware); without a state it falls back to score_total from the card database."""
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
    Buy cards based on their situational score.
    Only buy what can realistically be played soon.
    """
    player  = state["thisPlayer"]
    mc      = player.get("megacredits", 0)
    tl      = turns_left(state)

    # Late game: buy fewer cards. LEVER_BUY_VS_PASS does not honour this hard ban - it
    # keeps buying late, but only cards that are net positive (below).
    if tl <= 2 and not LEVER_BUY_VS_PASS:
        if _TELEM:                                  # count the offer, buys = 0
            _telem_note("offer", len(cards), state.get("id"))
        return []  # last generations: no new cards

    # estimate the best card cost from the cards on offer
    best_cost = 0
    for c in cards:
        sc = score_card_to_buy(c, state)
        if sc > BUY_MIN_SCORE:
            best_cost = max(best_cost, c.get("calculatedCost", 0))

    # Research reserve: first determine which cards are worth buying, then reserve play
    # money for THOSE. Conservatively, the CHEAPEST worthwhile card must stay playable -
    # reserving for the most expensive card on offer drove the budget to zero and blocked
    # buying entirely, even though 3 MC buys were possible and worthwhile.
    scored = [(score_card_to_buy(c, state), c["name"], c.get("calculatedCost", 0))
              for c in cards]
    scored.sort(reverse=True)

    # Quality threshold: always BUY_MIN_SCORE. A lower floor while surplus money existed
    # only let junk through and produced dead buys (bought, never played).
    buy_bar = BUY_MIN_SCORE
    worth = [(sc, nm, cost) for sc, nm, cost in scored if sc > buy_bar]
    if not worth:
        return []

    # Reserve play money for the CHEAPEST worthwhile card, not the most expensive on offer.
    play_reserve = min(cost for _, _, cost in worth)
    reserve = MC_RESERVE + play_reserve
    budget  = max(0, mc - reserve)
    # No hard cap on the number of buys - there is no such rule, and in a draft noticeably
    # more worthwhile cards are passed around. The budget is the only limit.
    max_buy = int(budget // card_cost)

    if max_buy == 0:
        return []

    chosen = [nm for _sc, nm, _c in worth[:max_buy]]

    if _RLOG:
        _gen = (state.get("game") or {}).get("generation")
        _chosen = set(chosen)
        for _sc, _nm, _c in scored:
            _bd = _rlog_bd.get(_nm)
            if _bd is not None:                       # resource-VP engines only
                _rlog_write({"phase": "buy", "gen": _gen, "name": _nm,
                             "score": round(_sc, 2), "bought": _nm in _chosen,
                             "buy_bar": round(buy_bar, 2), **_bd})

    if chosen:
        log.info("  📦 Kaufe: %s", ", ".join(chosen))
    if _TELEM:
        _pid = state.get("id")
        _telem_note("offer", len(cards), _pid)
        _telem_note("buy",   len(chosen), _pid)
    return chosen


def _choose_cards_to_buy_old(cards: list[dict], state: dict) -> list[str]:
    """(unused, kept for reference - old logic with the best_cost reserve bug)"""
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
            if _bd is not None:                       # resource-VP engines only
                _rlog_write({"phase": "buy", "gen": _gen, "name": _nm,
                             "score": round(_sc, 2), "bought": _nm in _chosen,
                             "buy_bar": round(buy_bar, 2), **_bd})

    if chosen:
        log.info("  📦 Kaufe: %s", ", ".join(chosen))
    else:
        log.info("  📦 No worthwhile card purchase")

    return chosen


def choose_card_to_play(cards: list[dict], state: dict) -> dict | None:
    """Pick the best playable card by situational score."""
    if not cards:
        return None

    player = state["thisPlayer"]
    mc     = player.get("megacredits", 0)

    # only cards we can afford
    affordable = [c for c in cards if c.get("calculatedCost", 999) <= mc]
    if not affordable:
        return None

    # prefer with reserve, fall back without
    with_reserve = [c for c in affordable if c.get("calculatedCost", 999) <= mc - MC_RESERVE]
    pool = with_reserve if with_reserve else affordable

    scored = [(score_card(c, state), c) for c in pool]
    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        return None

    best_score, best_card = scored[0]

    # only play a card with a positive score
    if best_score <= 0:
        return None

    return best_card


def get_playable_cards(cards: list[dict], state: dict, max_cards: int = 5) -> list[tuple[float, dict]]:
    """Return up to max_cards playable cards with their scores.
    For MCTS candidate selection: more options mean better decisions.
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

    # only positive scores (cards worth playing)
    playable = [(sc, c) for sc, c in scored if sc > 0]
    return playable[:max_cards]


def build_payment(card: dict, player: dict | None = None) -> dict:
    """Work out the optimal payment for a card.
    Uses steel (2 MC each) for building cards and titanium (3 MC each) for SPACE
    cards. Titanium does not pay for Jovian without space - Cloud Tourism
    (JOVIAN+VENUS) was otherwise paid with titanium and rejected by the server.

    player: state["thisPlayer"] - when given, resources are optimised.
"""
    cost = card.get("calculatedCost", 0)
    steel_used    = 0
    titanium_used = 0

    if player and cost > 0:
        # tags from the card database (uppercase); the API sometimes sends tags too
        card_name = card.get("name", "")
        tags_raw  = CARD_DB.get(card_name, {}).get("tags", [])
        tags      = [t.upper() for t in tags_raw]

        steel_have    = player.get("steel", 0)
        titanium_have = player.get("titanium", 0)
        mc_have       = player.get("megacredits", 0)
        # effective values from the server state (Advanced Alloys: steel 3 / titanium 4,
        # PhoboLog: titanium 4). Falls back to the defaults 2 / 3.
        steel_value    = player.get("steelValue", 2) or 2
        titanium_value = player.get("titaniumValue", 3) or 3

        # titanium ONLY for space cards (Jovian without space is rejected by the server)
        if "SPACE" in tags:
            # use as much titanium as possible without overpaying
            max_titan = min(titanium_have, cost // titanium_value)
            # do not spend more titanium than leaves enough MC for the rest
            while max_titan > 0 and (cost - max_titan * titanium_value) < 0:
                max_titan -= 1
            titanium_used = max_titan
            cost -= titanium_used * titanium_value

        # steel for building cards
        if "BUILDING" in tags:
            max_steel = min(steel_have, (cost + 1) // steel_value)  # +1 to avoid overpaying
            # round steel down to exactly cost (no overpaying with MC)
            while max_steel > 0 and (cost - max_steel * steel_value) < 0:
                max_steel -= 1
            steel_used = max_steel
            cost -= steel_used * steel_value

        # remaining cost in MC (never negative)
        cost = max(0, cost)

    return {
        "auroraiData": 0, "floaters": 0, "graphene": 0, "heat": 0,
        "kuiperAsteroids": 0, "lunaArchivesScience": 0,
        "megacredits": cost,
        "microbes": 0, "plants": 0, "seeds": 0,
        "spireScience": 0, "steel": steel_used, "titanium": titanium_used,
    }


# Space bonus values (from the server: TITANIUM=0, STEEL=1, PLANT=2, DRAW_CARD=3, HEAT=4)
_SPACE_BONUS_VALUE = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.5}


def _bonus_weight(b: int, tile_type: str) -> float:
    """Weight of a placement bonus when choosing a SPACE.
    Normal case (greenery/city/ocean): flat, so the bonus does not outrank
    adjacency VP. Mining tiles are the exception: there the space bonus decides
    WHICH production the card gives (titanium 3 MC per unit vs steel 2), so it is
    weighted by real MC value - otherwise the tile lands on a steel space although
    the value was computed with titanium.
"""
    if tile_type == "mining":
        return _SPACE_BONUS_M.get(b, 1.0)
    return _SPACE_BONUS_VALUE.get(b, 0.5)



# Server-TileType-Enum (src/common/TileType.ts)
TILE_GREENERY, TILE_OCEAN, TILE_CITY = 0, 1, 2


def _tile_adjacency_score(tile_type, own_cities, opp_cities,
                          own_greens, opp_greens, oceans, free_adj):
    """Adjacency value of a space per tile type, shared by the heuristic and the MCTS
    path so the two cannot drift apart. Placement bonuses are handled separately.
      neutral: no adjacency benefit (e.g. Nuclear Zone) -> do not waste a good
               greenery or city space on it.
"""
    s = oceans * 1.0   # Ozean-Nachbarn ~ +2 MC (halbgewichtet)
    # An adjacency VP is worth 5.0 here, consistent with the bot's convention of 1 VP = 5 MC
    # and with the weighting already used for commercial tiles. Undervaluing it let space
    # convention of the whole bot (1 VP = 5 MC). The same quantity 40 % cheaper
    # bonuses outrank adjacency, which is the cheapest VP source on the board.
    # A city next to 5 greeneries costs 5 MC per VP, a heat project 8 MC, an ocean 18 MC.
    _adj = 5.0 if LEVER_ADJACENCY_VP else 3.0
    if tile_type == "greenery":
        s += own_cities * _adj - opp_cities * 2.0 + own_greens * 0.5
    elif tile_type == "city":
        s += (own_greens + opp_greens) * _adj + free_adj * 0.8  # vorhandene Greenery (sichere VP)
                                                                # dominates free potential
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
    """Compute hex neighbours from x/y - the server sends NO adjacency field in the
    player view. Offsets exactly as in the server's Board.computeAdjacentSpaces.
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
    """(tileType, ownerColor) of the neighbouring spaces; tileType None = free."""
    for aid in adjacency.get(sid, []):
        a = space_map.get(aid, {})
        yield a.get("tileType"), a.get("color")


# ── Ares: special tiles grant a bonus to whoever places next to them. The SpaceModel carries
# NO adjacencyBonus field, so this table comes from the card definitions, keyed on the
# numeric tile type. It is what lets the bot see those bonuses when placing.
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
    # NUCLEAR_ZONE (12): no adjacency bonus (red border).
    # MINING_AREA (8) / MINING_RIGHTS (9): the steel/titanium bonus sits on the space itself,
    #   not as adjacency, and is already covered via _SPACE_BONUS_VALUE.
}
_ARES_RES_VALUE = {"mc": 1.0, "heat": 0.8, "plant": 2.0, "animal": 2.5, "microbe": 1.5,
                   "steel": 2.0, "titanium": 3.0, "energy": 1.0, "asteroid": 2.0, "draw_card": 3.0}

# Hazard tiles: placing an own greenery or city next to one costs 1 production (mild) or
# 2 (severe). The MC-equivalent penalty is the value of the cheapest such production.
# sacrificed production (heat/energy ~4 per step). Oceans carry no marker -> no penalty.
_ARES_HAZARD_PENALTY = {23: 4.0, 25: 4.0,   # DUST_STORM_MILD, EROSION_MILD  (1 Prod)
                        24: 8.0, 26: 8.0}   # DUST_STORM_SEVERE, EROSION_SEVERE (2 Prod)

def _ares_adj_value(ttype, is_marker: bool = True) -> float:
    """Ares adjacency when placing: a bonus for placing next to a special tile
    (always); a penalty for placing an OWN marker (greenery or city, not ocean)
    next to a hazard.
"""
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
    """Return the top N spaces as (score, spaceId).
    Lets MCTS choose between different positions.
"""
    adjacency = board_adjacency(space_map)

    def score_space(sid: str) -> float:
        space = space_map.get(sid, {})
        score = 0.0
        bonuses = space.get("bonus", [])
        if isinstance(bonuses, list):
            for b in bonuses:
                score += _bonus_weight(b, tile_type)
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

    # deduplicate similar scores - do not return three entries all scoring 0
    result = []
    prev_score = None
    for sc, sid in scored:
        if len(result) >= n:
            break
        # always take the first; after that only if clearly different
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
    """Choose the best space for a tile.

    Scoring (higher is better):
    1. placement bonuses of the space
    2. proximity to own tiles (greeneries score VP next to cities)
    3. for greeneries: prefer spaces next to own cities
    4. for cities: spaces with many free neighbours (future greeneries)
"""
    def score_space(sid: str) -> float:
        space = space_map.get(sid, {})
        score = 0.0

        # placement bonuses (weighted by their MC values)
        bonuses = space.get("bonus", [])
        if isinstance(bonuses, list):
            for b in bonuses:
                score += _bonus_weight(b, tile_type)

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
    """Placement bonus of a tile placement, in MC."""
    if not space_id:
        return 0.0

    game     = state.get("game", {})
    player   = state["thisPlayer"]
    my_color = player.get("color")
    spaces   = {s["id"]: s for s in game.get("spaces", [])}

    space = spaces.get(space_id)
    if not space:
        return 0.0

    # neighbours computed from x/y (the server sends no adjacency field);
    # Occupancy is FLAT on the space (tileType/color):
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
                bonus += 5.0        # an own city scores +1 VP
            elif ttype == TILE_CITY:
                bonus -= 5.0        # Gegner-Stadt punktet +1 VP (Geschenk)
        elif tile_type == "city":
            if ttype == TILE_GREENERY:
                bonus += 5.0        # Stadt punktet je Gruenflaeche, egal wessen

    return bonus







def _is_last_generation(state: dict) -> bool:
    """True when the game certainly ends after this generation: all three global
    parameters at maximum (or the server flag isTerraformed). Hoarding money is
    pointless then - reserves should be turned into victory points.
"""
    game = state.get("game", {})
    return bool(game.get("isTerraformed")) or (
        game.get("temperature", -30) >= 8
        and game.get("oxygenLevel", 0) >= 14
        and game.get("oceans", 0) >= 9)


def _action_card_info(title: str) -> dict:
    """Resolve an action option title to its card in the card database."""
    t = (title or "").strip()
    for pre in ("use the action of ","use action of ","action of ","activate ","use action "):
        if t.lower().startswith(pre): t = t[len(pre):].strip(); break
    if t in CARD_DB: return CARD_DB[t]
    tl_ = t.lower()
    for n, c in CARD_DB.items():
        if n.lower() == tl_: return c
    return {}






def score_action(action_type: str, state: dict,
                 placement_bonus: float = 0.0, card_title: str = "") -> float:
    """Value an action using the standard resource values.

    A standard project costs 4 MC more than the same effect from a card.
    placement_bonus: extra MC from placing a tile (ocean adjacency and so on).
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
    if last_gen or state.get("_idle_money"):
        cost_weight = 0.0            # letzte Gen / Leerlauf: Kosten voellig illusorisch
    else:
        cost_weight = SP_COST_WEIGHT

    # ── Pflanzen → Greenery ──────────────────────────────────────────────────
    # greenery = 19 MC of value (10 TR + 5 VP + 4 placement)
    # plant greenery is more efficient than the standard project (no 4 MC surcharge)
    if action_type == "greenery":
        # plants -> greenery: 1 TR + 1 VP (15 MC), at max oxygen only the VP. Plants
        # are a conversion resource, so there is no money cost.
        gross = (15 if oxygen < 14 else 5) + placement_bonus + urgency
        gross += _milestone_action_bonus("greenery", state)   # Pursue/Abschluss: Gardener
        return max(0, gross) * 3

    # ── heat -> temperature ──────────────────────────────────────────────────
    # 1 TR = 10 MC; converting heat is always good while temperature is not maxed
    if action_type == "heat":
        if temp >= 8:
            # temperature full -> converting yields no TR. Only keep heat when
            # it still counts through a WINNABLE Thermalist award (most heat = 5 VP)
            # -> negative, so passing or other moves are preferred
            # otherwise the heat is worthless and converting it is free
            # converting is a harmless waiting move -> 0 (allowed).
            if thermalist_hold_value(state) > 0:
                return -min(player.get("heat", 0), 12) * 0.5
            return 0.0
        # Anti-hoarding: heat has no sensible use other than conversion. Heat hoarded
        # beyond the next conversion step (8) is at risk of being washed out, so convert
        # (track maxing out / end of game) -> the more surplus, the more urgent NOW
        # rather than waiting for late urgency. Capped so it does not drown out strong
        # card plays.
        # Thermalist coherence: if the bot leads a WINNABLE Thermalist (in play, funded or
        # still fundable; most heat = 5 VP), its heat is worth about 5 VP and should NOT be
        # converted away. If Thermalist is no longer fundable or the bot is not leading,
        # the hold value is 0 and normal conversion applies.
        # worthless). Only with a comfortable lead (>= 16, still ahead after one
        t_hold = thermalist_hold_value(state)
        if 0 < t_hold < 16:
            return 0.0
        excess = max(0, player.get("heat", 0) - 8)
        hoard_urgency = min(excess * 0.5, 8.0)
        return max(0, 10 + urgency + hoard_urgency) * 3

    # ── Standard-Projekte ────────────────────────────────────────────────────
    # Standard projects always cost 4 MC more than the same effect from a card, so they only
    # pay off with a placement bonus or when no better card is playable.

    if action_type == "ocean_sp":
        if oceans >= 9:
            return 0
        cost = 18
        if mc < cost + reserve:
            return 0
        # BGG: Ozean = 14M (1 TR=10M + 4M Placement-Erwartung)
        # standard project surcharge = 4 MC -> net value 10 MC without a bonus
        # Placement-Bonus neben 2 Ozeanen (+4M) negiert SP-Aufpreis komplett
        net = 10 + placement_bonus - cost * cost_weight + urgency
        return max(0, net) * 3

    if action_type == "temp_sp":
        if temp >= 8:
            return 0
        cost = 14
        if mc < cost + reserve:
            return 0
        net = 10 + placement_bonus - cost * cost_weight + urgency
        if LEVER_SP_DISCIPLINE:
            # The asteroid project is pure TR at 14 MC per TR, the worst MC-to-VP rate.
            # Early the money belongs in an engine, so discount it while many generations
            # remain for engine building. Fades towards mid-game; late, urgency takes over.
            net -= max(0.0, tl - 6) * 2.0
        return max(0, net) * 3

    # Air Scrapping (Venus Next): 15 MC for one Venus step (1 TR). Without Venus Next the
    # server never offers it and this branch stays dormant.
    if action_type == "venus_sp":
        if game.get("venusScaleLevel", 0) >= 30:
            return 0
        cost = 15
        if mc < cost + reserve:
            return 0
        net = 10 + placement_bonus - cost * cost_weight + urgency
        return max(0, net) * 3

    if action_type == "greenery_sp":
        cost = 23
        if mc < cost + reserve:
            return 0
        gross = (15 if oxygen < 14 else 5) + placement_bonus
        net   = gross - cost * cost_weight + urgency
        net  += _milestone_action_bonus("greenery_sp", state)   # Pursue/Abschluss: Gardener
        if LEVER_GREENERY_DISCIPLINE:
            # The greenery project (23 MC for 1 TR + 1 VP, at max oxygen only 1 VP) is the
            # worst MC-to-VP rate. Early the money belongs in cards, engine or milestones.
            # while many generations remain. Fades towards mid-game; late, urgency harvests.
            net -= max(0.0, tl - GREENERY_LATE_GEN) * GREENERY_EARLY_PENALTY
        return max(0, net) * 3

    if action_type == "city_sp":
        cost = 25
        if mc < cost + reserve:
            return 0
        # A city standard project costs 25 MC for about 9 MC of base value (placement plus
        # -> deficit ~16 MC, break-even at about 3 adjacent greeneries (3 x 5 MC).
        # MC production), so the base is BONUS-DRIVEN rather than flat: without valuable
        # neighbours a city is not worth the price.
        # pauschaler Basis -> -2.92 VP, Lauf 2026-06-07).
        # Diminishing returns: every further own city ties up 25 MC,
        # -> -3 per city already owned.
        if LEVER_CITY_ADJACENCY:
            net = 9 + placement_bonus - cost * cost_weight
            gross = 9 + placement_bonus
            net   = gross - cost * cost_weight - 5 * player.get("citiesCount", 0)
        net += _milestone_action_bonus("city_sp", state)   # Pursue/Abschluss: Mayor
        return max(0, net) * 3

    if action_type == "sell":
        # Selling yields only 1 MC per card and gives up its option value. The score must
        # reflect that low yield, so the bot does not dump cards it currently rates as weak
        # (expensive engine cards, say) for 1 MC.
        # In the last generation cards expire unused, so selling is raised ABOVE passing -
        # but only after everything worthwhile has been played.
        if _is_last_generation(state):
            return 5
        return 1

    if action_type == "card_action":
        # handle_or._act_value, so both activation paths score identically.
        info = _action_card_info(card_title) if card_title else {}
        # Heat block: at maximum temperature an action whose only production yield is heat
        # (Underground Detonations) is worthless.
        temp = state.get("game", {}).get("temperature", -30)
        apr = info.get("action_prod_res") or info.get("production") or {}
        if temp >= 8 and apr.get("heat", 0) > 0 and all(
                r == "heat" or v <= 0 for r, v in apr.items()):
            return -1.0
        once = float(info.get("action_once", 0) or 0)
        draw = float(info.get("action_draw", 0) or 0)
        _dv  = DRAW_CARD_VALUE if LEVER_DRAW_VALUE else DRAW_ACTION_OLD
        val  = once + _dv * draw
        if val > 0:
            return val * CARD_PLAY_SCALE
        # An unquantified action must not outrank playing a hand card, otherwise the bot
        # activates worthless grab actions instead of playing cards.
        return 0.0

    if action_type == "pass":
        # Passing is only right without money or without playable cards. The pass value
        # win against it; with neither money nor cards the old flat value applies.
        _p    = state.get("thisPlayer", {}) or {}
        _mc   = _p.get("megacredits", 0) or 0
        _hand = _p.get("cardsInHandNbr", len(_p.get("cardsInHand") or []))
        if _mc >= PASS_IDLE_MC and _hand >= PASS_IDLE_HAND:
            return PASS_SCORE_IDLE          # money and cards available -> passing wastes them
        return PASS_SCORE

    return 0


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


INITIAL_KEEP_MIN = GREENERY_BUY_FLOOR   # -10: a starting card is a 3 MC play option over
                         # the ENTIRE remaining game, so far more generous than
                         # Kauf (0.5). Bewertung via score_card_to_buy(for_initial_keep=True),
                         # i.e. without the affordability and requirement penalties
                         # the count is limited. NOTE: this corrects the NUMBER of
                         # cards kept, not the valuation of effect cards themselves.
INITIAL_KEEP_MAX = 7     # starke menschliche Eroeffnung behaelt ~7/10

def choose_initial_cards(cards: list[dict], state: dict) -> list[str]:
    """At decision time megacredits is 0 (the corporation is chosen in the same
    answer), so a budget check would always keep 0 of 10 cards. A conservative
    corporate starting balance is assumed instead, buying up to 6 cards at 3 MC.
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


# ── CEOs (expansion 'ceo'): 38 cards with ONE once-per-game ability each. Two things are
# needed: the CHOICE at game start, and the USE of the once-per-game action.
# verfiel ungenutzt).
#
# _CEO_VALUE: MC-equivalent value of the ability. _CEO_REQUIRES: the module it needs -
# the value is set to 0 at RUNTIME when that module is inactive (Apollo without the Moon
# expansion is worthless).
_CEO_VALUE = {
    # starke, modul-unabhaengige Faehigkeiten
    "Karen":     28.0,   # draw a generation of preludes, play one (very strong early)
    "Will":      24.0,   # resources onto own cards (2 per type)
    "Clarke":    22.0,   # +1 plant AND heat production (permanent)
    "Tate":      20.0,   # draw cards with a chosen tag from the deck
    "Ryu":       18.0,   # Produktion tauschen (X+2 Einheiten)
    "Ender":     16.0,   # discard cards, then draw again (hand refresh)
    "Musk":      16.0,   # discard Earth cards, draw the same number
    "Stefan":    14.0,   # sell hand cards for 3 MC each
    "Jansson":   16.0,   # collect the placement bonuses under own tiles again
    "Hal 9000":  14.0,   # Produktion senken -> sofort Ressourcen
    "Greta":     18.0,   # TR-Erhoehungen geben Bonus (dauerhafter Effekt)
    "Faraday":   16.0,   # Tag-Meilensteine geben Boni (dauerhaft)
    "Ingrid":    14.0,   # tile placements this generation are boosted
    "Van Allen": 20.0,   # Meilensteine kosten 0 + 3 M€ je geclaimtem Meilenstein
    "Rogers":    12.0,   # ignore Venus requirements (only valuable with Venus)
    "Xavier":    12.0,
    "Co-leadership": 10.0,
    # module-dependent (value only when the module is active, see _CEO_REQUIRES)
    "Apollo":    16.0,   # 3 M€ je Moon-Tile
    "Neil":      16.0,
    "Shara":     16.0,
    "Oscar":     18.0,   # Chairman ersetzen (Turmoil)
    "Petra":     18.0,   # Neutrale Delegaten ersetzen (Turmoil)
    "Zan":       14.0,   # Delegaten in Reds (Turmoil)
    "Maria":     16.0,   # Kolonie-Plaettchen ziehen
    "Naomi":     18.0,   # Kolonie-Tracks auf Maximum
    "Yvonne":    18.0,   # Kolonie-Boni doppelt
    "Huan":      14.0,   # opponents cannot trade, plus a trade fleet
    "Floyd":     12.0,
    "Ulrich":    12.0,
    "Quill":     14.0,   # 2 floaters onto Venus cards
    "Xu":        14.0,   # 2 M€ je Venus-Tag
    "Asimov":    30.0,   # draw awards, fund one for free, +2 on all awards
    "Duncan":    30.0,   # 7-X VP AND 4X MC (played early: ~6 VP plus money, very strong)
    "Caesar":    22.0,   # place X hazards; every opponent loses 1-2 production (Ares)
    "Gaia":      20.0,   # collect the Ares adjacency bonuses of ALL tiles on Mars
    "Gordon":    20.0,   # Platzierungsregeln ignorieren + 2 M€ je Greenery/Stadt (dauerhaft)
    "Lowell":    18.0,   # 8 MC -> draw 3 CEOs, play one (effectively a second CEO)
    "Bjorn":     14.0,   # steal X+2 MC from the richest opponent
}
_CEO_REQUIRES = {
    "Apollo": "moon", "Neil": "moon", "Shara": "pathfinders",
    "Oscar": "turmoil", "Petra": "turmoil", "Zan": "turmoil",
    "Maria": "colonies", "Naomi": "colonies", "Yvonne": "colonies", "Huan": "colonies",
    "Quill": "venus", "Xu": "venus", "Rogers": "venus",
    "Caesar": "ares", "Gaia": "ares",
}


def _module_active(state: dict, mod: str) -> bool:
    """Is a module active in THIS game? Primarily from gameOptions.expansions,
    otherwise recognisable from the game data (turmoil, colonies and aresData
    only exist when their module is on).
"""
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
    """Value of a CEO for this game: its base value, but 0 when its module is off."""
    req = _CEO_REQUIRES.get(name)
    if req and not _module_active(state, req):
        return 0.0
    return _CEO_VALUE.get(name, 10.0)     # unbekannte CEOs: neutraler Mittelwert


def choose_ceo(cards: list[dict], state: dict) -> list[str]:
    """Pick the best CEO for this game (the server usually offers 3, min = max = 1)."""
    ranked = sorted(cards, key=lambda c: score_ceo(c.get("name", ""), state), reverse=True)
    best = ranked[0].get("name") if ranked else None
    if best:
        log.info("  👔 CEO: %s (value %.0f) from %s", best, score_ceo(best, state),
                 [c.get("name") for c in cards])
    return [best] if best else []


def handle_initial_cards(state: dict) -> dict:
    waiting = state["waitingFor"]
    player  = state["thisPlayer"]

    responses = []
    for option in waiting.get("options", []):
        title = str(option.get("title", "")).lower()
        cards = _playable(option.get("cards", []))
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
    """Estimate how valuable this card would be for the opponent.
    Simplified: a high production or TR score is dangerous in their hands.
    Cards that steal resources (Sabotage, Hackers) score high.
"""
    name = card.get("name", "")
    info = CARD_DB.get(name, {})
    if not info:
        return 0.0

    danger = 0.0

    # resource production is valuable for any opponent
    prod = info.get("production", {}) or {}
    danger += prod.get("megacredits", 0) * 3.0
    danger += prod.get("steel", 0)       * 5.0
    danger += prod.get("titanium", 0)    * 6.0
    danger += prod.get("plants", 0)      * 6.0
    danger += prod.get("energy", 0)      * 4.0

    # TR and terraforming cards are always dangerous
    tr_like = (info.get("tr", 0) + info.get("oceans", 0)
               + sum((info.get("global", {}) or {}).values()))
    danger += tr_like * 8.0

    # VP cards are dangerous
    danger += max(0, info.get("vp", 0)) * 4.0

    # cards with actions (ACTIVE) are dangerous long term
    if (info.get("type") or "").upper() == "ACTIVE":
        danger += 5.0

    return danger


def choose_draft_card(cards: list[dict], state: dict) -> str:
    """Pick the best card in a draft.

    Strategy: maximise (own value - the danger of the best card left behind),
    i.e. take the card with the largest gap between what it gives us and what
    the opponent could pick up next.
"""
    if not cards:
        return cards[0]["name"] if cards else ""

    # score all cards
    state["_draft_ctx"] = True     # draft: feeder synergy via the tableau only, NOT via
    scored = []                    # the pack (only one card of it stays with us)
    for card in cards:
        own_val    = score_card_to_buy(card, state)
        opp_danger = _score_card_for_opponent(card, state)
        # Kombinierter Score: 60% Eigenwert + 40% Gegner-Gefahr (verhindert)
        combined = own_val * 0.6 + opp_danger * 0.4
        scored.append((combined, own_val, opp_danger, card))

    state.pop("_draft_ctx", None)
    scored.sort(key=lambda x: x[0], reverse=True)

    # Draft diagnostics (gated by TM_DIAG_HAND): full ranking of the pack,
    # so it is visible WHAT a rejected card lost against
    if os.environ.get("TM_DIAG_HAND") and len(scored) > 1:
        log.info("  DRAFT DIAG (%d cards, sorted by combined):", len(scored))
        for comb, own, opp, card in scored:
            log.info("      %-28s eigen=%6.1f gefahr=%5.1f -> combined=%6.1f",
                     str(card.get("name", "?"))[:28], own, opp, comb)

    best_combined, best_own, best_opp, best_card = scored[0]
    log.info("  🃏 Draft: %s (eigen=%.1f, gefahr=%.1f)",
             best_card["name"], best_own, best_opp)
    return best_card["name"]


def _choose_removal_target(cards: list, state: dict, title: str) -> str:
    """Target choice for 'select card to remove N <resource>'. The server offers cards
    of BOTH players but carries no owner, so ownership is derived from our own
    played cards. Rule: prefer an opponent card (denying them the resource), there
    the one with the highest vp_dyn density. Only if solely own cards are
    selectable, take the least harmful one (fuel microbes before own VP microbes).
"""
    def _name(c): return c.get("name") if isinstance(c, dict) else c
    def _density(c):
        vd = (CARD_DB.get(_name(c), {}) or {}).get("vp_dyn") or {}
        return (vd.get("each", 0) / vd.get("per", 1)) if vd.get("kind") == "resources" else 0.0
    own = {(_name(c)) for c in (state.get("thisPlayer", {}).get("tableau") or state.get("thisPlayer", {}).get("playedCards") or [])
           if (_name(c))}
    opp = [c for c in cards if _name(c) not in own]
    if opp:
        chosen = max(opp, key=_density)            # take the opponent's most valuable resource
        log.info("  🎯 Entfern-Ziel: '%s' (Gegner)", _name(chosen))
    else:
        chosen = min(cards, key=_density)          # forced -> own fuel card first
        log.info("  🎯 Entfern-Ziel: '%s' (selbst, gezwungen)", _name(chosen))
    return _name(chosen)


_REMOVAL_RES = ("microbe", "animal", "plant", "resource", "floater", "science",
                "data", "fighter", "asset", "camp", "fleet", "preservation")


def _playable(cards: list) -> list:
    """Filter out cards with isDisabled = True. The server marks options that must NOT
    be chosen (in a draft e.g. cards already taken, for standard projects
    unaffordable ones). Choosing one produces HTTP 400 and aborts the game.
"""
    return [c for c in (cards or []) if not (isinstance(c, dict) and c.get("isDisabled"))]


def handle_card(state: dict) -> dict:
    waiting = state["waitingFor"]
    cards   = _playable(waiting.get("cards", []))
    raw_title = waiting.get("title", "")
    min_c   = waiting.get("min", 1)
    max_c   = waiting.get("max", min_c)

    # If the title is a dict, check whether it is a draft (type 2 = card direction)
    # or a genuine and-type (type 1 = amount, type 3 = space)
    if isinstance(raw_title, dict):
        data = raw_title.get("data", [])
        types_in_data = {item.get("type") for item in data}
        # type 2 with value = colour is draft direction info (card flow between players).
        # This is the DRAFT, including the repick phase. Do NOT rely on thisPlayer's
        # players have selected" (repick = true). NOTE: in this phase thisPlayer carries
        # `needsToResearch` - that is a FUTURE flag, not the current state. What counts is
        # min/max of the waitingFor structure: the server wants EXACTLY `max` cards (here
        # cardsToKeep, usually 1). Sending a buy response instead yields HTTP 400.
        # min/max-Beachtung -> Endlosschleife. Jetzt: choose_draft_card, streng auf max gekappt.
        if types_in_data == {2}:
            k = max(1, int(max_c))                    # cardsToKeep from the request (min == max)
            # exclude cards the server rejected (see _draft_rejected), but only while
            # something remains - better to retry the full pool than to send an empty answer
            if _draft_rejected:
                _filtered = [c for c in cards if c.get("name") not in _draft_rejected]
                if len(_filtered) >= k:
                    if len(_filtered) != len(cards):
                        log.info("  📋 Draft: skipping rejected card(s) %s",
                                 sorted(_draft_rejected & {c.get("name") for c in cards}))
                    cards = _filtered
            if not cards:
                return {"type": "card", "runId": state["runId"], "cards": []}
            # REPICK STABILITY: decide once per draft round and stick to it. The danger
            # term fluctuates between requests, so without this the bot oscillates between
            ckey = _draft_cache_key(state, cards)
            cached = _draft_choice_cache.get(ckey)
            available = {c.get("name") for c in cards}
            if cached and all(nm in available for nm in cached) and len(cached) == k:
                # repeated request for the same pool (repick): repeat the same choice.
                # Logged on purpose - this path used to be silent, which made a crash log
                log.info("  📋 Draft: repeating choice %s (cache, %d cards in pool)",
                         list(cached), len(cards))
                return {"type": "card", "runId": state["runId"], "cards": list(cached)}
            picks = []
            pool  = list(cards)
            while pool and len(picks) < k:             # k times the best card still available
                nm = choose_draft_card(pool, state)
                picks.append(nm)
                pool = [c for c in pool if c.get("name") != nm]
            if len(_draft_choice_cache) > 4000:        # defensive against unbounded growth
                _draft_choice_cache.clear()
            _draft_choice_cache[ckey] = tuple(picks)
            log.info("  📋 Draft: keeping %s (max=%d) out of %d cards", picks, k, len(cards))
            return {"type": "card", "runId": state["runId"], "cards": picks}
        # type 1 = amount. NOTE: if a type 0 (resource name) is also present in data, the
        # value is the RESOURCE amount ("add 2 microbe"), not the card count - that then
        # follows from min/max (often exactly 1 target card). Otherwise the value is the
        # card count. Either way, clamp to [min, max].
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

    # Removal target (e.g. Ants: "Select card to remove 1 Microbe(s)") - NOT a draft
    # otherwise the bot removes resources from its own best card. Prefer opponents.
    if "remove" in title and any(r in title for r in _REMOVAL_RES):
        chosen = _choose_removal_target(cards, state, title)
        return {"type": "card", "runId": state["runId"], "cards": [chosen]}

    # draft: pick exactly one card and keep it
    if any(k in title for k in ("draft", "keep", "select a card")) or (
        min_c == 1 and max_c == 1 and len(cards) > 1
    ):
        chosen = choose_draft_card(cards, state)
        log.info("  📋 Draft: keeping '%s' out of %d cards", chosen, len(cards))
        return {"type": "card", "runId": state["runId"], "cards": [chosen]}

    # Prelude spielen
    if "prelude" in title:
        scored = sorted(cards, key=lambda c: score_card(c, state), reverse=True)
        return {"type": "card", "runId": state["runId"], "cards": [scored[0]["name"]]}

    # generic: the min_c best cards
    if min_c > 0:
        scored = sorted(cards, key=lambda c: score_card_to_buy(c, state), reverse=True)
        chosen = [c["name"] for c in scored[:min_c]]
        return {"type": "card", "runId": state["runId"], "cards": chosen}

    return {"type": "card", "runId": state["runId"], "cards": []}


# names of ACTIVE cards that have an action (from card_db)
# ---------------------------------------------------------------------------
# Milestone and award evaluation
# ---------------------------------------------------------------------------

# A milestone costs 8 MC and gives 5 VP = 25 MC of value.
# Only claim with enough money, not yet claimed, and a clear lead.
# All known milestone names (from the server's MilestoneName)
# The server sends only the name as the option title (e.g. "Gardener"), with no "milestone".
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
    """Count cards of TYPE EVENT (for Legend). NOT the 'event' tag (only 10 cards) -
    Legend counts played events, i.e. type EVENT (142 cards).
    """
    n = 0
    for c in cards or []:
        nm = c.get("name") if isinstance(c, dict) else c
        if ((CARD_DB.get(nm, {}) or {}).get("type") or "").upper() == "EVENT":
            n += 1
    return n


def _count_req_cards(cards) -> int:
    """Count cards with a requirement IN PLAY (for the Tactician milestone): non-empty
    'requirements' AND type != EVENT. Events are not 'in play' after being played.
    'requirements' is the complete source (tags, global parameters, production,
    expansions), so no special case for req_tags or req_prod is needed.
    """
    n = 0
    for c in cards or []:
        nm = c.get("name") if isinstance(c, dict) else c
        info = CARD_DB.get(nm, {}) or {}
        if info.get("requirements") and info.get("type") != "EVENT":
            n += 1
    return n


# Cards holding floater resources (server: resourceType FLOATER) - for counting the
# Hoverlord milestone (7 floaters on cards).
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
    """Player statistics needed for milestone and award evaluation
    (tiles, tags, cards, production).
    """
    player   = state["thisPlayer"]
    game     = state.get("game", {})
    spaces   = game.get("spaces", [])
    my_color = player.get("color")

    # count tiles
    own_cities    = 0
    own_greeneries = 0
    for s in spaces:
        # The SpaceModel carries `tileType` and `color` FLAT on the space - there is no
        # nested `tile` object. Reading one yields None for every space, which would
        # The space model is flat, so reading a nested `tile` object yields None
        # silently leave own_cities and own_greeneries at 0 and make the Mayor and
        # Gardener milestones unplannable.
        # Annahmen standen unbemerkt nebeneinander.
        t = s.get("tileType")
        if t is None:
            continue
        if s.get("color") != my_color:
            continue
        # TileType (src/common/TileType.ts, im Repo verifiziert 18.07.):
        #   0 = GREENERY, 1 = OCEAN, 2 = CITY, 3 = CAPITAL, 20 = OCEAN_CITY ...
        # Tile type 1 is OCEAN, not city (CITY 2, CAPITAL 3, OCEAN_CITY 20), so cities
        # must be matched against _CITY_TILE_TYPES.
        # the Mayor milestone (3 cities) was scored completely wrong
        if t == 0:
            own_greeneries += 1
        elif t in _CITY_TILE_TYPES:
            own_cities += 1

    # played cards and tags from the card database
    played = player.get("tableau") or player.get("playedCards") or []
    tag_counts: dict[str, int] = {}
    for c in played:
        name = c.get("name", "") if isinstance(c, dict) else c
        info = CARD_DB.get(name, {})
        for tag in info.get("tags", []):
            t = tag.lower()
            tag_counts[t] = tag_counts.get(t, 0) + 1

    # Handkarten
    hand_size = len(hand_cards(state))

    # Floater resources on cards (for the Hoverlord milestone). The tableau gives a
    # 'resources' count per card; floater holders come from _FLOATER_CARDS.
    floaters = 0
    for c in played:
        nm = c.get("name", "") if isinstance(c, dict) else c
        if nm in _FLOATER_CARDS and isinstance(c, dict):
            floaters += c.get("resources", 0) or 0

    # Ares-Meilenstein-Zaehler (Networker = Tiles neben Bonus-Tiles gelegt; Purifier =
    # removed hazards) from game.aresData.milestoneResults, keyed by player id.
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

    # Turmoil: own INFLUENCE (server getInfluence): +1 as chairman; in the DOMINANT party
    # +1 as leader (+1 more with more than one delegate), or +1 as non-leader with >= 1.
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
    # Blue (ACTIVE) cards and own colonies - for evaluating global events
    blue_cards = sum(1 for c in played
                     if str((card_info(c.get("name") if isinstance(c, dict) else c) or {})
                            .get("type", "")).upper() == "ACTIVE")
    n_colonies = sum(1 for c in (game.get("colonies") or [])
                     if player.get("color") in (c.get("colonies") or []))

    # For the fan and modular milestones (Trader/Tradesman, Farmer, Lobbyist):
    #   res_types = number of DIFFERENT non-standard resource types on own cards
    #   bio_res   = microbes plus animals on own cards
    #   delegates = own delegates in the Turmoil congress (all parties)
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
        # Turmoil active? -> the Terraformer milestone needs 26 TR instead of 35
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
    """Steps until a milestone is met, computed purely from statistics - usable for the
    bot AND for opponents (to estimate who might claim first).
    0 = met. Unknown milestones -> 1 (conservative).
    """
    t     = title.lower()
    tags  = stats.get("tags", {})
    prods = stats.get("prods", {})
    # Different tags = types with count > 0 (the player state lists every type,
    # including those at 0, so len(tags) must not be used).
    distinct_tags = sum(1 for v in tags.values() if v > 0)

    # ── Tharsis ──
    # Terraformer: threshold 35, WITH TURMOIL only 26. Without this the bot badly
    # underestimates how close it is.
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
        # APPROXIMATION: strictly only tiles in the bottom two rows count; without board
        # geometry in the statistics all own tiles are counted -> overestimates (gap too small).
        return max(0, 3 - (stats.get("cities", 0) + stats.get("greeneries", 0)))
    elif t == "energizer":      return max(0, 6 - prods.get("energy", 0))
    elif t == "rim settler":    return max(0, 3 - tags.get("jovian", 0))
    # ── expansion and modular milestones (randomMA), computable from the statistics ──
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
    # ── FAN and MODULAR milestones. Extracted from the server manifest; only those
    #    that can be computed from the statistics available here.
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
    elif t in ("trader", "tradesman"):                                            # 3 different non-standard resources
        return max(0, 3 - stats.get("res_types", 0))
    elif t == "farmer":         return max(0, 5 - stats.get("bio_res", 0))        # Mikroben + Tiere
    elif t == "lobbyist":       return max(0, 7 - stats.get("delegates", 0))      # Turmoil
    # ── Sonstige / unbekannt ──
    else:                       return 1


def _opponent_stats(state: dict) -> list[dict]:
    """Simplified statistics of all opponents from state["players"] (public data), in
    the same format as _player_stats - for _milestone_gap on the opponent side.
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
    """Read from game["milestones"] (NOT claimedMilestones, which is None): a milestone
    with 'color'/'playerName' set is claimed. Global cap: at most 3 claimed.
    """
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
    """Pursue logic: pick the ONE milestone the bot can reach most cheaply and with the
    best chance (own gap <= fastest opponent), slot-aware and within range.
    """
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
            if g > opp_g:                 # not the front runner -> do not pursue
                continue
            if g > tl - 1:                # not reachable in time
                continue
            if best is None or g < best[1]:   # billigster (kleinster gap) gewinnt
                best = (nm.lower(), g)
        if best:
            res = (best[0], best[1], 25 - _MILESTONE_COST)
    state["_ms_pursuit"] = res
    return res


# Which score_action closes which milestone gap? Deliberately only the specific,
# alignment-clean cases - NOT Terraformer, which would license standard-project spam.
_MILESTONE_PURSUE_ACTIONS = {
    "gardener": {"greenery", "greenery_sp"},
    "mayor":    {"city_sp"},
}


def _milestone_action_bonus(action_type: str, state: dict) -> float:
    """Small bonus for an action that closes the gap to the milestone currently being
    pursued. Deliberately small (it tips good moves, it does not justify bad ones)
    and larger the smaller the gap.
    """
    pur = _milestone_pursuit(state)
    if not pur:
        return 0.0
    name, gap, net = pur
    if action_type in _MILESTONE_PURSUE_ACTIONS.get(name, ()):
        return min(PURSUE_BONUS_CAP, (net / max(1, gap)) * PURSUE_WEIGHT)
    return 0.0




# ── Meilenstein-/Award-Ausrichtung (obs 5/10) ────────────────────────────────
# Goal -> card properties that advance it. Deliberately only clean mappings that can
# be checked in card_db (type, tag, production, tile, cost). Board-dependent: only
# goals that are actually in play AND realistically winnable take effect.
_TARGET_PROPS = {
    # Meilensteine
    "legend":      {"type:event"},
    "builder":     {"tag:building"},
    "ecologist":   {"tag:bio"},
    "gardener":    {"tile:greenery"},
    "mayor":       {"tile:city"},
    "energizer":   {"prod:energy"},
    "rim settler": {"tag:jovian"},
    # awards (depending on the board)
    "scientist":   {"tag:science"},
    "banker":      {"prod:megacredits"},
    "thermalist":  {"prod:heat"},
    "miner":       {"prod:steel", "prod:titanium"},    # proxy: production, not actual resources
    "industrialist": {"prod:steel", "prod:energy"},     # proxy: production, not actual resources
    "celebrity":   {"cost:high"},                       # >= 20 MC, green/blue only (no event)
    "space baron": {"tag:space"},
    "cultivator":  {"tile:greenery"},
    "contractor":  {"tag:building"},                    # meiste Building-Tags (Hellas)
    "landlord":    {"tile:any"},                        # meiste Tiles
    "excentric":   {"holds:resources"},                 # most resources on cards
    "venuphile":   {"tag:venus"},
    "magnate":     {"type:automated"},
    # Deliberately NOT mapped (no clean buy bias possible):
    #   benefactor (TR - zu breit, fast alles), desert settler / estate dealer
    #   (depends on board position: southern hemisphere / ocean-adjacent)
}
_BIO_TAGS = {"plant", "microbe", "animal"}


def _alignment_targets(state: dict) -> set:
    """In-play AND realistically winnable goals -> set of favoured card properties.
    Milestone: unclaimed, a slot free, bot is front runner (own gap <= best
    opponent) and within ALIGN_MAX_GAP. Award: bot leads or is at most
    ALIGN_AWARD_SLACK behind. That usually leaves 1-3 goals, avoiding tunnel vision.
    """
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
    """Does the card have one of the favoured properties?"""
    if not props:
        return False
    tags = {t.lower() for t in info.get("tags", [])}
    typ  = (info.get("type") or "").lower()
    prod = info.get("production", {}) or {}
    for p in props:
        if p == "type:event"     and typ == "event":              return True
        if p == "type:automated" and typ == "automated":          return True
        if p == "tag:bio"        and (tags & _BIO_TAGS):          return True
        if p == "cost:high"      and cost >= 20 and typ != "event":  return True  # no event
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
    """Small buy/keep bonus when the card advances a winnable goal."""
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
    """Does the combo card have its enabler (which card_db does not encode)?
    False -> devalue.
    """
    p    = state.get("thisPlayer", {})
    prod = p.get("production", {}) or {}
    if name == "Insulation":
        return prod.get("heat", 0) > 0          # heat production -> MC production
    if name == "Virus":
        me = p.get("color")
        for o in state.get("players", []):   # attack: only valuable when an opponent has plants or animals
            if o.get("color") != me and (o.get("plants", 0) > 0 or o.get("animals", 0) > 0):
                return True
        return False
    if name == "Protected Habitats":
        return False                             # defensive: worth ~nothing in a 2P game
    return True


def _score_milestone(title: str, state: dict, known_claimable: bool = False) -> float:
    """Should the bot claim this milestone?

    1. Already met -> claim (high score).
    2. One step away -> claim if worthwhile.
    3. Further away -> 0.

    Value of a milestone: 5 VP = 25 MC minus the cost (8/14/20).
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

    stats = _player_stats(state)
    # When the title comes from the server option "Claim a milestone", it is
    # already filtered server-side as claimable -> gap = 0 instead of our own estimate
    gap   = 0 if known_claimable else _milestone_gap(title, stats)

    # ── Bewertung ─────────────────────────────────────────────────────────────
    if gap == 0:
        # Met -> claimable. Base value = 25 MC (5 VP) minus the cost.
        net = 25 - cost
        opp_gaps = [_milestone_gap(title, o) for o in _opponent_stats(state)]
        opp_gap  = min(opp_gaps) if opp_gaps else 99
        urgency = 0
        if opp_gap <= 0:
            urgency = 45   # the opponent can claim the same milestone immediately
        elif opp_gap <= 1:
            urgency = 35   # Gegner 1 Schritt entfernt
        elif opp_gap <= 3:
            urgency = 20   # Gegner nah dran - Abstaende schrumpfen stetig
        elif opp_gap <= 6:
            urgency = 10   # opponent within reach in a few generations
        elif claimed_count >= 2:
            urgency = 35   # letzter freier Slot – knapp, jetzt sichern
        if LEVER_MILESTONE and urgency < 35:
            # Window-aware: secure a qualified milestone even without an opponent on the
            # SAME one, once the global cap of three is about to close. Threat = how many
            # free milestones an opponent could grab in one step right now.
            # filled every slot with other milestones.
            threats = sum(1 for o in _opponent_stats(state)
                          for a in _avail if _milestone_gap(a, o) <= 1)
            if free_slots - threats <= 1:
                urgency = 35
        log.info("   🏆 Milestone '%s' MET (cost=%d, net=%.0f, opp_gap=%d, urgent=%d)",
                 title, cost, net, opp_gap, urgency)
        return net + 10 + urgency
    elif gap == 1:
        # one step away: worth it while generations remain
        net = 20 - cost   # Leicht abgewertet wegen Unsicherheit
        urg = 0
        if LEVER_MILESTONE:
            opp_gap = min([_milestone_gap(title, o) for o in _opponent_stats(state)], default=99)
            if opp_gap <= 1:
                urg = 15   # opponent is also one step away -> do not dawdle
        if net > 0 and tl >= 2:
            log.info("   🏆 Milestone '%s' almost met (gap=1, cost=%d)", title, cost)
            return net + urg
        return 0
    else:
        return 0


# Distinktive Award-Namen-Bestandteile (Tharsis / Elysium / Hellas). Dienen sowohl
# both recognising the funding option and mapping values and thresholds.
_AWARD_KEYS = (
    # Base plus expansions (server: server/awards/). The bot scores awards from the
    # SERVER SCORES (game.awards[].scores) but needs the names to RECOGNISE a funding
    # option at all. A missing name sends the option into the pass fallback and the
    # award is never funded.
    "landlord", "banker", "scientist", "thermalist", "miner",
    "celebrity", "entrepreneur", "desert settler", "estate dealer", "benefactor",
    "contractor", "cultivator", "excentric", "magnate", "space baron", "rim contractor",
    "venuphile", "blacksmith", "industrialist", "naturalist", "voyager", "visionary",
    "forecaster", "edgedancer",
    # FAN / modular awards - available with "Include Fan Milestones/Awards"
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
    """Recognise an award funding option by the award name it contains.
    Note: Turmoil ruling policies contain party names that overlap with award names
    (Unity, Kelvinists, Reds), so those must not be treated as award funding.
    """
    t = title.lower()
    if "turmoil" in t:
        return False
    return any(k in t for k in _AWARD_KEYS)


def _is_fund_award_option(opt: dict) -> bool:
    """Recognise the nested 'Fund an award' selection: an outer or-option whose title
    is a message object and whose sub-options are the individual awards.
    """
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
    """Value of a player in an award category - for the bot AND for opponents (award
    ranks are decided relative to the opponent). -1 = unknown.
    """
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
    """Rough maturity threshold of the category (for how safe a lead is)."""
    t = title.lower()
    for k, v in _AWARD_THRESHOLDS.items():
        if k in t:
            return v
    return 5


def _award_progress_bonus(info: dict, state: dict) -> float:
    """Bonus for cards that improve the bot's own metric in an already FUNDED award
    (a funded award is a 5 VP race). Only while the race is still open
    (own score >= opp_max - 2).
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
            continue   # race lost -> stop chasing the metric
        bonus += metric_delta(a.get("name", "")) * 2.0
    return min(10.0, bonus)


def _award_scores_from_server(title: str, state: dict):
    """(own, opp_max) from game.awards[].scores - the server counts every award metric
    itself, board-agnostic and for every expansion. None when the award is not found
    there, which falls back to the name heuristic.
    """
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
    """Should the bot fund this award?

    Awards only pay off while leading the category or with the opponent far behind.
    Cost 8/14/20 MC, expected 5 VP (25 MC) for first place, 2 VP (10 MC) for second.
    """
    player = state["thisPlayer"]
    game   = state.get("game", {})
    mc     = player.get("megacredits", 0)

    # NOTE: the server has no `fundedAwards` field; it provides `awards`, and an award
    # counts as funded when `playerName` is set. Reading a non-existent field yields an
    # always-empty list, which would make the bot treat every award as the first one
    # (cost 8 instead of 14/20).
    # Sperre `fund_count >= 3` nie ausgeloest. Zwanzig Zeilen weiter oben in derselben
    # (_neighbor_tiles flach vs. _player_stats verschachtelt).
    funded     = [a for a in game.get("awards", []) if a.get("playerName")]
    fund_count = len(funded)
    if fund_count >= 3:
        return 0

    cost = 8 + fund_count * 6
    if mc < cost + MC_RESERVE + 5:
        return 0

    tl = turns_left(state)
    if tl < 0:
        return 0   # Spiel vorbei

    # Prefer the SERVER standings (game.awards[].scores counts every metric itself,
    # board-agnostic); the name heuristic is only a fallback.
    sv = _award_scores_from_server(title, state)
    if sv is not None:
        own, opp_max = sv
    else:
        my_stats = _player_stats(state)
        own      = _award_value(title, my_stats)
        if own < 0:
            # unknown award (e.g. from an expansion): very conservative score
            return max(0, 5 - cost)
        opp_vals = [_award_value(title, o) for o in _opponent_stats(state)]
        opp_max  = max(opp_vals) if opp_vals else 0
    lead = own - opp_max

    if own <= 0 or lead <= 0:
        return 0   # not leading / empty category -> do not fund

    # TIME CONFIDENCE: awards are scored at the END of the game, so a lead is only worth
    # what can be held over the remaining generations. In the LAST generation being ahead
    # is SAFEST -> do not block, but require a minimum lead of 2, since an opponent can
    # still catch up by about one tile or step in their final turn.
    floor       = 2 if tl <= 1 else 1
    need_tight  = max(floor, 1 + tl // 4)
    need_comf   = max(floor, 1 + tl // 2)
    if lead < need_tight:
        return 0   # lead not defensible over the remaining time
    comfortable = lead >= need_comf
    # Value of a funded award while leading = its real victory point value: first place
    # is 5 VP (~25 MC). With only a narrowly defensible lead second place (2 VP) is a
    # real risk, so the expectation sits in between (~16 MC). The leading and timing
    # gates above still prevent funding too early.
    AWARD_VP_MC = 5.0
    gross = (5.0 if comfortable else 3.2) * AWARD_VP_MC   # 25 bzw. 16 MC
    net   = gross - cost
    # Now or never: a safely led award left unfunded at the end of the game is pure VP
    # loss. Raise it in the last generation so funding wins the final turn against
    # marginal card plays.
    if comfortable and tl <= 1:
        net *= 1.4
    # Award scores have to be in the SAME UNIT as card scores: cards are multiplied by
    # CARD_PLAY_SCALE, so without this a second award (net 11) loses against almost any
    # card play. The leading and timing gates above stay untouched; only the value of an
    # (Decomposers scales to 26.0, Marketing Experts to 18.6), a third one (net 5) anyway.
    # already valid funding decision is rescaled.
    if LEVER_AWARD_SCALE:
        net *= CARD_PLAY_SCALE
    log.info("   🥇 Award '%s' sponsern (own=%d, opp_max=%d, lead=%d, cost=%d, net=%.0f%s)",
             title, own, opp_max, lead, cost, net, ", komfortabel" if comfortable else "")
    return max(0, net)



def _get_active_card_names() -> set:
    """All names of ACTIVE cards that have an action."""
    return {n for n, c in CARD_DB.items() if (c.get("type") or "").upper() == "ACTIVE"}

def _is_card_action(title: str) -> bool:
    """Recognise whether an option title is a card or corporation action."""
    if not title:
        return False
    title_lower = title.lower()
    # Explizite Aktions-Phrasen
    if any(kw in title_lower for kw in ("use action", "action of", "activate", "use the action")):
        return True
    # card name used directly as the title (an ACTIVE card shares its action's name)
    active_names_lower = {n.lower() for n in _get_active_card_names()}
    if title_lower in active_names_lower:
        return True
    return False


def _extract_msg_number(raw_title) -> int:
    """First integer-parsable number from a message object.
    Opponent names and the like are not int-parsable and are skipped.
    """
    if isinstance(raw_title, dict):
        for d in raw_title.get("data", []):
            try:
                return int(d.get("value"))
            except (TypeError, ValueError):
                continue
    return 0


def _plant_attack_score(opt: dict):
    """Score for an option coming from RemoveAnyPlants.
    Returns:
      None  -> not a plant removal option (another branch handles it, e.g. skip)
      20+N  -> remove opponent plants, N = amount actually removed (more is better)
    Order matters: check the warning first, because the self-option carries the same
    'Remove ... plants from ...' title as the opponent options.
    """
    warnings = [str(w).lower() for w in opt.get("warnings", [])]
    if "removeownplants" in warnings:
        return -50.0
    raw_title = opt.get("title", "")
    msg = raw_title.get("message", "") if isinstance(raw_title, dict) else str(raw_title)
    msg = msg.lower()
    # 'skip removing plants' contains 'removing', not 'remove' -> falls through (None)
    if "remove" in msg and "plants" in msg and "from" in msg:
        return 20.0 + _extract_msg_number(raw_title)
    return None


_DIAG_LOGGED_GENS: set = set()
_DIAG_MS_LOGGED_GENS: set = set()


def _diag_milestones(state: dict) -> None:
    """Diagnostic hook (gated by TM_DIAG_HAND): logs once per generation how close the
    bot is to EVERY milestone. gap = 0 means qualified. Separates the tempo problem
    (the gap never reaches 0 -> the bot builds too slowly) from a claim error
    (gap = 0 but not claimed -> evaluation).
    """
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
    """Diagnostic hook (gated by the TM_DIAG_HAND environment variable). Logs once per
    generation how close the bot is to EVERY milestone. gap = 0 means qualified.
    Separates the tempo problem (the gap never reaches 0) from a claim error
    (gap = 0 but not claimed).
    """
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
        hand_cards = [c for c in _playable(pc_opt.get("cards", []))
                      if not c.get("name", "").endswith(":SP")
                      and c.get("name") not in SP_NAMES]
    if not hand_cards:
        log.info("DIAG Gen %s | wtype=%s '%s' | option types=%s | no playable hand cards",
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
            klass = "REQ-GATED"       # requirement unmet -> rightly held back
        elif cost > mc:
            klass = "UNAFFORDABLE"    # Geld fehlt
        elif sc <= 0:
            klass = "SCORE<=0"        # leistbar + Req ok, Bewertung negativ -> LECK-Kandidat
        else:
            klass = "PLAYABLE>0"      # worth playing but not played -> ranking or limit
        counts[klass] = counts.get(klass, 0) + 1
        log.info("    %-28s score=%6.1f cost=%3s %s", name[:28], sc, cost, klass)
    log.info("    -> %s", " | ".join(f"{k}={v}" for k, v in counts.items()))


def _is_ruling_policy(title: str) -> bool:
    """Turmoil ruling policy actions (payable actions of the governing party)."""
    return "turmoil" in title and ("pay" in title or "spend" in title)


def _ruling_policy_value(title: str, state: dict) -> float:
    """Net MC value of a ruling policy action. All three payable policies are CARD
    SOURCES, which is what the bot is chronically short of:
      Scientists: 10 MC -> 3 cards (a drawn card is worth ~4-5 MC here, since the
                  3 MC buying fee is skipped and the bot is card-poor)
    Only valuable when the bot can afford it without burning its playing money.
    """
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
    """Evaluate all available actions and pick the best."""
    waiting    = state["waitingFor"]
    options    = waiting.get("options", [])
    player     = state["thisPlayer"]
    mc         = player.get("megacredits", 0)

    candidates = []  # (score, index, payload)
    _idle_engine_log(state, options, player)

    for i, opt in enumerate(options):
        otype = opt.get("type", "")
        # The title is often a message TEMPLATE ({"data": [...], "message": "..."}), not a
        # string. str(dict).lower() yields the dict representation, so NO branch matches
        # and the option falls into the generic pass fallback. The corporation's first
        # ERSTAKTION ("Take first action of ${0} corporation", z.B. Valley Trust: 3 Preludes
        # action was treated like "pass" that way, costing a whole generation.
        # zuvor in handle_player/Biomass Combustors.)
        _raw_title = opt.get("title", "")
        title = (_raw_title.get("message", "") if isinstance(_raw_title, dict)
                 else str(_raw_title)).lower()

        # ★ EINLOESE-OPTION (20.07., apeheads Befund: "16 Bakterien auf Sulphur-Eating
        # handle_or had NO branch for "trade resources for a yield": the option fell into
        # the generic fallback and the bot took index 0, i.e. "add 1 microbe", collecting
        # forever. Affects 20 cards with a redeem action. The yield is in the title
        # ("gain 3 MC per microbe removed"), the available amount in the SelectAmount max.
        if LEVER_REDEEM and otype in ("amount", "selectAmount"):
            _m = _REDEEM_RE.search(title)
            _max = opt.get("max") or 0
            if _m and _max > 0:
                _je = float(_m.group(1) or _m.group(2))
                _ertrag = _je * _max
                # Redeem in the END PHASE only. The bot has ONE action per generation:
                # collecting for eight generations and cashing 24 MC beats cashing 3 MC
                # cashing 3 MC four times. Redeeming too early would be the mirror error.
                # four times.
                # Exception: an acute cash shortage - then liquidity beats collecting.
                _prog = param_progress_from_state(state)
                _knapp = (player.get("megacredits", 0) or 0) < REDEEM_CASH_FLOOR
                if not (_prog >= REDEEM_PROGRESS or _knapp
                        or _is_last_generation(state)):
                    continue
                # Competing against the alternative branch "add a resource" is not enough:
                # its value sits in action_once and is not computed here. The yield is real
                # money, so it is valued directly in MC.
                candidates.append((_ertrag, i,
                                   {"type": "or", "runId": state["runId"], "index": i,
                                    "response": {"type": "amount", "amount": _max}}))
                log.info("  💰 Einloesen: %d Ressourcen x %.0f M = %.0f M",
                         _max, _je, _ertrag)
                continue

        # Mandatory first action "Place a city tile" (Tharsis Republic and others): MUST
        # other branches, otherwise an earlier elif (the card_action filter) catches
        # option 0 and filters it out with sc <= 0, leaving only pass.
        # come before everything else, otherwise the bot passes permanently.
        # the actual space choice is made afterwards by handle_space.
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

        # Greenery (plants -> tile): top 3 positions as separate MCTS candidates
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

        # heat -> temperature
        elif otype == "option" and "heat" in title and can_convert_heat(state):
            sc = score_action("heat", state)
            candidates.append((sc, i, {
                "type": "or", "runId": state["runId"], "index": i,
                "response": {"type": "option"},
                "_label": f"🌡 Hitze→Temp",
            }))

        # Turmoil: send a delegate. NOTE: the action appears in the action menu as
        # SelectParty (type 'party'), NOT an 'option' - Turmoil.getSendDelegateInput()
        # sends the SelectParty prompt directly. Title: "Send a delegate in an area"
        # (from lobby)" | "(5 M€)" | "(3 M€)" (Incite).
        # Value = chance of chairman (offsets the TR revision, 1 TR per generation)
        elif otype == "party" and "delegate" in title:
            _turm = (state.get("game", {}) or {}).get("turmoil") or {}
            if _turm:
                val  = _delegate_action_value(state)
                free = "lobby" in title
                cost = 0.0 if free else (3.0 if "3" in title else DELEGATE_COST)
                # Budget cap: paid delegates must not eat the money meant for cards
                # (observed: the bot delegated its MC away and hardly played cards).
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

        # Colonies: payment submenu of a trade (9 MC / 3 energy / 3 titanium) - cheapest
        # reale Option waehlen (Energie meist Ueberschuss, Titan am teuersten).
        elif otype == "or" and _is_trade_payment_option(opt):
            sub = opt.get("options", []) or []
            j = _pick_trade_payment(state, sub)
            if j is not None:
                candidates.append((100.0, i, {   # high: the trade has already been decided
                    "type": "or", "runId": state["runId"], "index": i,
                    "response": {"type": "or", "index": j, "response": {"type": "option"}},
                    "_label": f"🚀 Trade-Zahlung: {str(sub[j].get('title',''))[:24]}",
                }))

        # Colonies trade action. Value = best tradable colony yield minus the trade cost;
        # only a candidate when net > 0. Follow-up prompts (payment, colony choice)
        # uebernehmen handle_payment/handle_colony.
        elif otype == "option" and "trade" in title and "free" not in title:
            net = _score_trade(state)
            if net > 0:
                candidates.append((net * CARD_PLAY_SCALE, i, {
                    "type": "or", "runId": state["runId"], "index": i,
                    "response": {"type": "option"},
                    "_label": f"🚀 Trade (net={net:.0f})",
                }))

        # play a hand card or a standard project
        elif otype == "projectCard":
            all_cards = _playable(opt.get("cards", []))

            # explicitly exclude standard project names
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

            # LEVER_IDLE: no positive card playable -> the money is going to waste.
            # Above the reserve buffer the card cost is illusory (the money would just be
            # hoarded), so the best net-negative but effectively positive card is valued at
            # its GROSS worth (cost deduction undone) and offered - which beats passing.
            # No tuned bonus; the strength IS the real card value. Only what leaves the
            # reserve buffer untouched.
            # Idle flag: no positive hand card plus money above the reserve.
            # Affects standard projects (cost_weight = 0) AND the card idle below.
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
                    _idle = _net + _cost   # undo the cost deduction -> gross value
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
            # Budget planning: what does the best playable hand card cost?
            # If a standard project would destroy that budget, lower its score.
            best_hand_score = 0
            best_hand_cost  = 0
            for hc in hand_cards:
                hs = score_card(hc, state)
                hcost = hc.get("calculatedCost", 0)
                if hs * CARD_PLAY_SCALE > best_hand_score and hcost <= mc - MC_RESERVE:
                    best_hand_score = hs * CARD_PLAY_SCALE
                    best_hand_cost  = hcost

            # Placement bonus: best available space for ocean/greenery/city
            space_map = {s["id"]: s for s in state["game"].get("spaces", [])}
            # note the space model is flat, so every space would look free
            # Note the space model is flat; treating every space as free would search the
            # best placement bonus on occupied spaces and overrate the projects.
            # Standardprojekte Stadt/Greenery/Ozean systematisch ueberbewertet.
            all_space_ids = [s["id"] for s in state["game"].get("spaces", [])
                             if s.get("tileType") is None
                             and s.get("spaceType") != "colony"]
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
                "Air Scrapping": score_action("venus_sp",   state),   # Venus Next only
            }
            for sp_card in all_cards:
                sp_name = sp_card["name"]
                if sp_name in sp_scores:
                    sc = sp_scores[sp_name]
                    if sc > 0:
                        mc   = state["thisPlayer"].get("megacredits", 0)
                        cost = sp_card.get("calculatedCost", 999)
                        if cost <= mc - (0 if _is_last_generation(state) else MC_RESERVE):
                            # Budget penalty: if the best hand card is no longer affordable
                            mc_after_sp = mc - cost
                            if best_hand_score > sc and mc_after_sp < best_hand_cost + MC_RESERVE:
                                # the project would block a good hand card -> devalue
                                sc = sc * 0.4
                                log.debug("  Budget penalty for %s: mc_after_sp=%d, hand_cost=%d",
                                          sp_name, mc_after_sp, best_hand_cost)
                            candidates.append((sc, i, {
                                "type": "or", "runId": state["runId"], "index": i,
                                "response": {
                                    "type": "projectCard",
                                    "card": sp_name,
                                    "payment": build_payment(sp_card, player),
                                },
                                "_label": f"🏗 {sp_name} SP (score={sc:.1f})",
                                "_cost": cost,          # telemetry only
                            }))

        # Karte verkaufen
        elif otype == "card" and "sell" in title:
            sell_cards = _playable(opt.get("cards", []))
            if sell_cards:
                # Normally only clearly worthless cards (< -2), so that currently weak
                # engines are not dumped for 1 MC. In the LAST generation everything that
                # will not be played any more (score <= 0).
                # zu M€ machen statt verfallen lassen.
                last_gen = _is_last_generation(state)
                # ★ FIX 20.07. (apeheads Beobachtung, richtige Stellschraube):
                # The -2.0 threshold is far too loose early, because without an engine
                # almost every card scores negative - in generation 1 about 284 of 956
                # cards fall below it, statistically three of ten starting cards. At -25.0
                # only 21 cards qualify, which are the genuinely unplayable ones. From
                # SELL_EARLY_GENS on the normal threshold applies again, because by then
                # the bot's own engine makes the valuation meaningful.
                # aussagekraeftig.
                early = (state.get("game", {}).get("generation", 1) <= SELL_EARLY_GENS)
                sell_threshold = (0.01 if last_gen
                                  else (SELL_THRESHOLD_EARLY if early else -2.0))
                # candidates: every card below the threshold, ascending by score
                to_sell = sorted(
                    (c for c in sell_cards if score_card(c, state) < sell_threshold),
                    key=lambda c: score_card(c, state))
                if to_sell:
                    if last_gen:
                        # Server erlaubt {max: Handkartenzahl} -> ALLE verwertlosen
                        # sell in ONE move (saves round trips)
                        # Karte war reine Zeitverschwendung, apehead 17.07.).
                        names = [c["name"] for c in to_sell]
                        worst_score = score_card(to_sell[0], state)
                        sc = score_action("sell", state)
                        candidates.append((sc, i, {
                            "type": "or", "runId": state["runId"], "index": i,
                            "response": {"type": "card", "cards": names},
                            "_label": f"💰 Selling {len(names)} cards (last generation)",
                        }))
                    else:
                        # normal case: only the single most worthless card
                        worst = to_sell[0]
                        worst_score = score_card(worst, state)
                        sc = score_action("sell", state)
                        candidates.append((sc, i, {
                            "type": "or", "runId": state["runId"], "index": i,
                            "response": {"type": "card", "cards": [worst["name"]]},
                            "_label": f"💰 Verkaufe {worst['name']} (score={worst_score:.1f})",
                        }))

        # Activate an ACTIVE card action: the server presents this as an option bundle
        # SelectCard "Perform an action from a played card" (otype 'card',
        # (cards = the activatable cards). The card with the highest action value
        # (action_once) is chosen.
        elif otype == "card" and ("perform an action" in title or "played card" in title
                                   or opt.get("selectBlueCardAction")):
            act_cards = _playable(opt.get("cards", []))
            if act_cards:
                temp = state["game"].get("temperature", -30)
                def _act_value(c):
                    info = card_info(c.get("name", ""))
                    # Heat block: at maximum temperature an action whose only production
                    # yield is heat (Underground Detonations) is worthless. Primarily from
                    # action_prod_res, falling back to the play production field for older
                    # card_db versions.
                    apr = info.get("action_prod_res")
                    if apr is None:
                        apr = info.get("production") or {}
                    if temp >= 8 and apr.get("heat", 0) > 0 and all(
                            r == "heat" or v <= 0 for r, v in apr.items()):
                        return -1.0
                    # Net yield of the activation. action_once is already net, but the
                    # Produktionswert minus Kosten (Space Mirrors: 7 Prod - 7 = 0);
                    # card draw is missing from it and is added here.
                    _dv = DRAW_CARD_VALUE if LEVER_DRAW_VALUE else DRAW_ACTION_OLD
                    return (float(info.get("action_once", 0) or 0)
                            + _dv * float(info.get("action_draw", 0) or 0))
                best = max(act_cards, key=_act_value)
                val  = _act_value(best)
                # Only activate when the action has a real net yield. Swap actions
                # bekommen wertlose Grab-/Tausch-Aktionen (Search For Life = 0,
                # (Search For Life, Space Mirrors = 0) are therefore no longer candidates
                # and no longer outrank playing a hand card.
                if val > 0:
                    sc = val * CARD_PLAY_SCALE
                    # A free accumulator action is ALWAYS better than passing -> floor just
                    # above the pass score, so the bot activates instead of passing.
                    # below the pass score, so such blocks were never activated.
                    if str(best.get("name", "")).strip().lower() in _FREE_ACCUM:
                        sc = max(sc, 5.0)
                    candidates.append((sc, i, {
                        "type": "or", "runId": state["runId"], "index": i,
                        "response": {"type": "card", "cards": [best["name"]]},
                        "_label": f"⚡ Aktion: {best['name']} (a={val:.1f})",
                    }))

        # Meilenstein claimen - verschachtelte "Claim a milestone"-or:
        # the server only offers milestones that are already claimable
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
                sc = best_sc * CARD_PLAY_SCALE   # onto the comparison scale (like cards)
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

        # Fund an award - nested "Fund an award" or (outer level): first pick the fund
        # option, then the best award in the inner or.
        elif otype == "or" and _is_fund_award_option(opt):
            sub_opts = opt.get("options", [])
            best_j, best_sc = None, 0.0
            for j, sub in enumerate(sub_opts):
                sc_j = _score_award(str(sub.get("title", "")), state)
                if sc_j > best_sc:
                    best_sc, best_j = sc_j, j
            if best_j is not None and best_sc > 0:
                best_sc *= CARD_PLAY_SCALE   # onto the comparison scale (like cards)
                award_name = str(sub_opts[best_j].get("title", "?"))
                log.info("   🥇 Award funden: %s (score=%.0f)", award_name, best_sc)
                candidates.append((best_sc, i, {
                    "type": "or", "runId": state["runId"], "index": i,
                    "response": {"type": "or", "index": best_j,
                                 "response": {"type": "option"}},
                    "_label": f"🥇 Fund {award_name}",
                }))

        # Fund an award - direct inner or (fallback, when the server presents the award
        # selection without the outer wrapper).
        # otype == "option" with the award name as the title).
        elif otype == "option" and _is_award_option(title):
            sc = _score_award(title, state) * CARD_PLAY_SCALE
            if sc > 0:
                candidates.append((sc, i, {
                    "type": "or", "runId": state["runId"], "index": i,
                    "response": {"type": "option"},
                    "_label": f"🥇 {str(title)[:40]}",
                }))

        # ACTIVE card action or corporation action
        # CORPORATION FIRST ACTION ("Take first action of <corp> corporation"). The
        # is "Pass for this generation", so this action must be recognised, otherwise the
        # bot passes the whole generation away. These first actions are practically always
        # strong (Valley Trust: draw 3 preludes; Point Luna: cards; Teractor/Vitor: money
        # or VP) -> score them high so they safely beat passing.
        elif otype == "option" and "first action of" in title and "corporation" in title:
            candidates.append((CORP_FIRST_ACTION_VALUE * CARD_PLAY_SCALE, i, {
                "type": "or", "runId": state["runId"], "index": i,
                "response": {"type": "option"},
                "_label": f"🏢 Korporations-Erstaktion "
                          f"({str(opt.get('buttonLabel', ''))[:34]})",
            }))

        elif otype == "option" and _is_card_action(title):
            sc = score_action("card_action", state, card_title=str(title))
            if sc > 0:   # do not activate worthless or heat-blocked actions (sc <= 0)
                candidates.append((sc, i, {
                    "type": "or", "runId": state["runId"], "index": i,
                    "response": {"type": "option"},
                    "_label": f"⚡ Card action: {str(opt.get('title', '?'))[:30]}",
                }))

        # Plant attack (removeAnyPlants): weaken the opponent, protect own plants
        elif otype == "option" and _plant_attack_score(opt) is not None:
            sc = _plant_attack_score(opt)
            n  = _extract_msg_number(opt.get("title", ""))
            candidates.append((sc, i, {
                "type": "or", "runId": state["runId"], "index": i,
                "response": {"type": "option"},
                "_label": (f"🌿✂ {n} Pflanzen entfernen" if sc > 0
                           else "🚫 avoid own plants"),
            }))

        # CEO: einmalige Faehigkeit (OPG - "Use CEO once per game action", Typ 'card').
        # The value is one-off and often strongest early (Karen: preludes; Clarke:
        # production) - but not to be burned in the very first generation when the
        # effect scales with the generation.
        elif otype == "card" and "ceo" in title and "once per game" in title:
            _ceos = _playable(opt.get("cards", []))
            if _ceos:
                _nm  = _ceos[0].get("name", "")
                _val = score_ceo(_nm, state)
                if _val > 0:
                    candidates.append((_val * CARD_PLAY_SCALE, i, {
                        "type": "or", "runId": state["runId"], "index": i,
                        "response": {"type": "card", "cards": [_nm]},
                        "_label": f"👔 CEO action: {_nm} (value {_val:.0f})",
                    }))

        # Pass / End Turn / Undo / Ruling-Policy-Aktionen. WICHTIG: Frueher bekamen ALLE
        # 'option' entries share the pass score and the label "Pass", so the bot could
        # confuse 'End Turn' (end only the TURN, staying in the generation) with 'Pass for
        # this generation' (giving up the WHOLE generation), pick 'Undo last action', and
        # ignore the Turmoil ruling policy actions.
        elif otype == "option":
            if "undo" in title:
                continue                       # NEVER undo our own move
            if "end turn" in title:
                # End the turn only: strictly better than passing (we stay in the
                # generation). Chosen only when nothing better is available.
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
        log.warning("  handle_or: no candidates, sending index 0")
        return {"type": "or", "runId": state["runId"], "index": 0,
                "response": {"type": "option"}}

    state.pop("_idle_money", None)   # do not leak the idle flag past the decision

    candidates.sort(key=lambda x: x[0], reverse=True)
    score, idx, payload = candidates[0]
    log.info("  → %s", payload.get("_label", "?"))

    if _TELEM:
        _lbl = payload.get("_label", "")
        _pid = state.get("id")
        if _lbl.startswith("🏗"):
            _telem_note("sp", payload.get("_cost", 0.0), _pid)
        elif _lbl.startswith("🃏"):
            _nm = _lbl[2:].split(" (score=")[0].strip()
            _telem_note("card", float((card_info(_nm) or {}).get("cost", 0) or 0), _pid)
        elif _lbl.startswith("⚡"):
            _telem_note("action", 0.0, _pid)      # Kartenaktion (Expertendaten: BOB 159 vs K4rlchen 70)
        elif _lbl.startswith("💰"):
            _telem_note("sell", 0.0, _pid)
        elif "Pass" in _lbl:
            # Did the bot pass although a card was playable (= a candidate)?
            _had_card = any(str(_l).startswith("🃏") for _s, _i, _p in candidates
                            for _l in [(_p or {}).get("_label", "")])
            _telem_note("pass_with_cards" if _had_card else "pass", 0.0, _pid)

    if _RLOG:
        # Stranded detector: which free accumulators does the server offer in this
        # aktivierbar an (im Buendel opt["cards"])? Post-hoc-Query:
        #   chosen == pass AND offered_fa != []  =>  the bot passed although a free
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
    payload.pop("_cost", None)
    return payload


def handle_option(state: dict) -> dict:
    return {"type": "option", "runId": state["runId"]}


def handle_space(state: dict) -> dict:
    """Choose the best space for a tile.
    The tile type is derived from the context (title or last move sent).
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
        log.error("  handle_space: no valid spaces!")
        return {"type": "space", "runId": state["runId"], "spaceId": "03"}

    # derive the tile type from the title
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
        tile_type = "mining"   # prefer a space with a steel or titanium bonus
    else:
        # Special tile with no recognised adjacency benefit (e.g. Nuclear Zone): NOT
        # as greenery - that searched for own cities and wasted good spaces.
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
    """Payment selection. Pays in MC first (as far as available) and covers the rest with
    the allowed resources (heat 1:1, steel and titanium by value). Avoids overpaying.
    """
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
    """Or the title field is a dict with 'data' = a list of input responses.
    Each sub-answer is processed individually.
    """
    waiting = state.get("waitingFor", {})
    runId   = state["runId"]

    # some server versions deliver options as a list
    options = waiting.get("options", [])

    # ── Ressourcen-Verteilung (z.B. Global Event "Dry Deserts": 'Gain N resource(s) for
    # influence' -> an and-type with one 'amount' option per resource). The server expects
    # one answer PER OPTION (summing to N). Sending only ONE answer yields HTTP 400.
    if options and all(o.get("type") == "amount" for o in options) and len(options) > 1:
        # How much may be distributed in total? (title.data carries the count)
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
        # Energy decays into heat, MC is the weakest per unit.
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

    # Fallback: title is a dict with 'data'
    title = waiting.get("title", "")
    if isinstance(title, dict) and "data" in title:
        # Altes Format: title.data = [{'type': int, 'value': str}, ...]
        # simply accept everything with default values
        responses = []
        for item in title["data"]:
            itype = item.get("type", 0)
            if itype == 1:   # amount - take the value straight from title.data
                responses.append({"type": "amount", "amount": int(item.get("value", 0))})
            elif itype == 0: # Option
                responses.append({"type": "option"})
            elif itype == 2: # card - choose from the available cards
                # cards from waiting.options or waiting.cards
                available = (waiting.get("options") or
                             waiting.get("cards") or [])
                if available:
                    # pick the best card via the draft logic
                    chosen = choose_draft_card(available, state)
                    responses.append({"type": "card", "cards": [chosen]})
                else:
                    responses.append({"type": "card", "cards": []})
            elif itype == 3: # space - choose the best space
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

    # New format: options is a list of sub-waitings
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
    log.warning("  and-type: no known format, sending empty")
    return {"type": "and", "runId": runId, "responses": []}


def handle_unknown(state: dict) -> None:
    waiting = state.get("waitingFor", {})
    wtype = waiting.get("type")
    title = waiting.get("title", "")
    # If the title is a dict it is probably an and-type
    if isinstance(title, dict):
        return handle_and(state)
    log.warning("⚠️  Unbekannter Typ: '%s' | '%s'",
                wtype, str(title)[:60])
    return None


def handle_player(state: dict) -> dict | None:
    """Prefers an opponent; otherwise - and when only the bot's own colour is
    selectable - its own colour.
    """
    waiting = state.get("waitingFor", {})
    players = waiting.get("players", [])
    if not players:
        return None
    colors   = [p if isinstance(p, str) else p.get("color", "") for p in players]
    my_color = state.get("thisPlayer", {}).get("color", "")
    # The title is often a message TEMPLATE, not a string. str(dict).lower() does NOT find
    # the keywords -> negative = False -> the bot attacked ITSELF (observed: Biomass
    # Combustors reducing the bot's own plant production). Hence the text is taken
    # cleanly from 'message'.
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
    """Ares hazard penalty: building next to a hazard space costs 1-2 production (server
    type 'productionToLose'). The bot sacrifices its least valuable production first.
    Floors: MC production may go to -5, everything else only to 0.
    Answer: {type:'productionToLose', units:{...}} with the sum of reductions == cost.
    """
    w    = state.get("waitingFor", {}) or {}
    pp   = w.get("payProduction", {}) or {}
    cost = pp.get("cost", 1) or 1
    have = pp.get("units", {}) or {}     # aktuelle Produktionsstufen

    # Sacrifice order: lowest MC-equivalent value per production step first.
    # heat < energy < MC < steel < plants < titanium (mirrors the production valuation).
    order = ["heat", "energy", "megacredits", "steel", "plants", "titanium"]
    reduce = {k: 0 for k in ("megacredits", "steel", "titanium", "plants", "energy", "heat")}
    remaining = cost
    for k in order:
        if remaining <= 0:
            break
        floor      = -5 if k == "megacredits" else 0      # MC production may go negative
        can_reduce = have.get(k, 0) - floor               # moegliche Senkungsschritte
        take = min(remaining, max(0, can_reduce))
        if take > 0:
            reduce[k] += take
            remaining -= take

    log.info("   ⚠️ Hazard-Strafe: senke Produktion %s",
             ", ".join(f"{k} -{v}" for k, v in reduce.items() if v))
    return {"type": "productionToLose", "runId": state["runId"], "units": reduce}


_COLONY_VALUE = {   # rough resource-value nudge per colony (yield type). trackPosition is
    "Pluto": 2.0,    # the main signal; this nudge lifts valuable yields (cards, animals,
    "Miranda": 1.5,  # floaters/microbes) slightly. Rough - adjustable by a TM expert.
    "Titan": 1.5, "Enceladus": 1.5, "Triton": 1.5,
    "Ceres": 1.0, "Europa": 1.0, "Ganymede": 1.0,
    "Luna": 0.8, "Io": 0.8, "Callisto": 0.8, "Deimos": 0.5,
}
TRADE_COST = 7.0   # MC-equivalent trade cost (9 MC or 3 energy/titanium; energy is often surplus)
DELEGATE_COST = 5.0  # Turmoil: Standardaktion "Delegat entsenden" kostet 5 M€
PASS_SCORE      = 4.0   # normal pass value (no money / no cards -> passing is right)
PASS_SCORE_IDLE = 0.5   # pass value with money AND playable cards -> almost never pass
PASS_IDLE_MC    = 12    # from this much MC ...
PASS_IDLE_HAND  = 3     # ... and this many hand cards, the idle value applies
CORP_FIRST_ACTION_VALUE = 20.0  # Korporations-Erstaktion (Valley Trust: 3 Preludes ziehen;
                                # Point Luna: cards) - practically always strong, so it must
                                # Pass sicher schlagen.
# ── CARD DRAW: ONE VALUE, NOT THREE ────────────────────────────────────────────────────────
# The same effect used to be valued differently in three places:
#   card_db  `draw_cards`  = 1.0 MC per card (one-off "draw N cards")
#   DRAW_CARD_VALUE        = 4.5 MC, which was only used in the Turmoil policy handler.
#   _action_value/_act_value = 2.0 M€ je Karte  (wiederholbare Zieh-AKTION, hartcodiert)
# -- dieselbe Code/Daten-Entkopplung wie feeds/synergy_adds.
# LEVER_DRAW_VALUE makes DRAW_CARD_VALUE the SINGLE source of truth (33 cards).
# Anchor for the value: a card in research costs 3 MC, so that is what a draw is worth
# is worth at least. 4.5 = 3 MC saved plus the option value.
LEVER_DRAW_VALUE = True
DRAW_CARD_VALUE = 4.5  # value of a drawn card. 3.0 would be conservative (research price),
                       # 4.5 = Kaufpreis + Optionswert, 2.0/1.0 = altes (inkonsistentes) Verhalten
DRAW_BGG_M      = 1.0  # flacher Satz in card_db (score_breakdown: draw_cards)
DRAW_ACTION_OLD = 2.0  # previous hardcoded rate for action_draw
TR_VALUE = 10.0      # 1 TR = 10 MC (1 VP plus income every generation)
INFLUENCE_VALUE = 4.0  # 1 Einfluss: mildert Global Events + Ruling-Boni. Konservativ bewertet -
                       # it only pays when a global event actually hits the bot.

def _score_trade(state: dict) -> float:
    """Net MC value of the Colonies trade action: best tradable colony yield minus the
    trade cost (9 MC, or 3 energy/titanium - energy is often surplus). An own colony
    yields extra when trading. Only active, unvisited colonies are tradable.
    """
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
    """Recognise the trade payment submenu: an 'or' with options like '9 MC',
    '3 energy', '3 titanium'.
    """
    if opt.get("type") != "or":
        return False
    subs = [str(o.get("title", "")).lower() for o in (opt.get("options", []) or [])]
    if not subs:
        return False
    hits = sum(1 for t in subs
               if ("energy" in t or "titanium" in t or "m€" in t or "megacredit" in t))
    return hits >= 2 and len(subs) <= 4   # mind. 2 Ressourcen-Zahlwege, kleines Menue


def _pick_trade_payment(state: dict, options: list) -> int | None:
    """Trade payment: 9 MC OR 3 energy OR 3 titanium. Picks the option that is CHEAPEST
    for the bot (and affordable). Energy is usually surplus (it decays into heat
    anyway), titanium is valuable for space cards, MC is universal.
    Returns the option index.
    """
    p    = state.get("thisPlayer", {}) or {}
    mc   = p.get("megacredits", 0) or 0
    en   = p.get("energy", 0) or 0
    ti   = p.get("titanium", 0) or 0
    # MC-equivalent opportunity cost of each payment
    COST = {"energy": 3 * 1.0,      # 3 energy (~1 MC each; decays into heat otherwise)
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
    """Colonies: selecting a colony (type 'colony') - either to TRADE with (which one) or
    to BUILD on. Yield ~ trackPosition x resource value. When trading, avoid colonies
    already visited; an own colony gives an extra bonus. When building, avoid ones
    already built on.
    """
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
                s -= 100.0                     # do not build on the same colony twice
        else:
            s = track * nudge                  # Handels-Ertrag ~ Track * Ressourcenwert
            if c.get("visitor"):
                s -= 100.0                     # already visited -> not tradable
            if built:
                s += 2.0                        # own colony -> extra bonus when trading
        return s

    best = max(colonies, key=_score)
    log.info("   🚀 Kolonie: %s (%s, track=%s)",
             best.get("name"), "bauen" if is_build else "handeln", best.get("trackPosition"))
    return {"type": "colony", "runId": state["runId"], "colonyName": best.get("name")}


def _party_value(party: str, state: dict) -> float:
    """Value of a party FOR THE BOT (the ruling bonus depends on its own profile):
      Reds        hostile to TR (punishes terraforming) -> negative for a TR bot
    Also counts hand cards that need exactly this party as a requirement.
    """
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
    # hand cards that need exactly this party as a ruling requirement -> bonus
    for c in hand_cards(state):
        nm = c.get("name") if isinstance(c, dict) else c
        for r in ((card_info(nm) or {}).get("requirements") or []):
            if r.get("type") == "party" and \
               str(r.get("value", "")).lower().replace(" ", "") == p:
                v += 3.0
    return v


# ── Global events (Turmoil): value of ONE extra influence point, in MC.
# Patterns extracted from the server:
#   negativ: Verlust = min(max, Einheiten) - Einfluss  -> 1 Einfluss spart 1 Einheit,
#            but only while the bot is actually affected (units > current influence).
#   positive: gain = units + influence            -> 1 influence = 1 unit more (always).
# "units" gives how many units affect the bot (from the statistics); for positive events
# that is unlimited (influence always pays) -> 99.
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
    "War On Earth":                 (10.0, lambda s: 4),    # every influence prevents 1 TR
    "Eco Sabotage":                 (2.0,  lambda s: min(5, max(0, s.get("plants", 0) - 3))),
    "Corrosive Rain":               (3.0,  lambda s: 99),   # 1 Karte je Einfluss
    "Paradigm Breakdown":           (2.0,  lambda s: 99),   # 2 M€ je Einfluss
    "Snow Cover":                   (3.0,  lambda s: 99),   # 1 Karte je Einfluss
    "Dry Deserts":                  (2.0,  lambda s: 99),   # 1 Standardressource je Einfluss
    "Sabotage":                     (2.0,  lambda s: 99),   # 1 Stahl je Einfluss
    # REVOLUTION: influence is ADDED -> more influence makes losing more likely (2 TR). BAD.
    "Revolution":                   (-10.0, lambda s: 99),
    # --- positive: influence always pays one extra unit ---
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
    "Improved Energy Templates":    (3.5,  lambda s: 99),   # counts as a power tag
    "Election":                     (5.0,  lambda s: 99),   # TR race: influence counts
    "Diversity":                    (2.0,  lambda s: 99),   # counts as a tag
}


def _global_event_influence_value(state: dict, stats: dict) -> float:
    """What is ONE extra influence point worth, given the upcoming global events?
    The bot sees 'current' (resolved at the end of this generation) plus 'coming' and
    'distant' - all public. Nearer events count in full.
    This is the core of Turmoil: influence only pays when an event actually hits.
    """
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
        # influence only pays while it still covers units (negative events)
        # (positive, units=99). Negativer per_inf (Revolution) = Einfluss SCHADET.
        if per_inf < 0 or units > my_inf:
            total += per_inf * weight
    return total
def _turmoil_party_state(state: dict):
    """Helper data: (dominant party, own delegates there, best opponent there, reserve)."""
    turm = (state.get("game", {}) or {}).get("turmoil") or {}
    my   = state.get("thisPlayer", {}).get("color")
    parties = turm.get("parties") or []
    if not parties:
        return None, 0, 0, 0

    def _count(p, color):
        return sum(1 for d in (p.get("delegates") or [])
                   if (d.get("color") if isinstance(d, dict) else d) == color)

    # Dominant = the party with the most delegates (as the server determines it).
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
    """MC-equivalent value of sending a delegate NOW."""
    game = state.get("game", {}) or {}
    turm = game.get("turmoil") or {}
    if not turm:
        return 0.0
    dom, mine, top, _res = _turmoil_party_state(state)
    if dom is None:
        return 0.0

    gens_left = max(0, (game.get("lastSoloGeneration") or 12) - game.get("generation", 1))
    if gens_left <= 1:
        return 0.0                      # at the end of the game influence no longer pays

    # What is ONE influence point worth, given the upcoming global events? (The core of
    # Turmoil - influence only pays when an event actually hits or helps the bot.)
    stats   = _player_stats(state)
    inf_val = _global_event_influence_value(state, stats)

    # (a) Chairman route: another delegate makes the bot leader of the DOMINANT party
    #     -> chairman (+1 TR per generation) AND +1 influence (leader) - both count.
    if mine + 1 > top:
        return TR_VALUE * min(1.0, gens_left / 6.0) + max(0.0, inf_val)

    # (b) Influence route: the first delegate in the dominant party gives +1 influence.
    if mine == 0 and inf_val > 0:
        return inf_val

    # (c) Rising party: a party that could become dominant NEXT generation (level with
    #     or just behind the dominant one) is a legitimate investment - leading there
    #     pays off as soon as it takes over.
    parties = turm.get("parties") or []
    my      = state.get("thisPlayer", {}).get("color")
    dom_n   = len(dom.get("delegates") or [])
    for p in parties:
        if p is dom:
            continue
        n = len(p.get("delegates") or [])
        if n >= dom_n - 1:                      # could become dominant next generation
            p_mine = sum(1 for d in (p.get("delegates") or [])
                         if (d.get("color") if isinstance(d, dict) else d) == my)
            p_top  = 0
            for c in {(d.get("color") if isinstance(d, dict) else d)
                      for d in (p.get("delegates") or [])}:
                if c != my:
                    p_top = max(p_top, sum(1 for d in (p.get("delegates") or [])
                                           if (d.get("color") if isinstance(d, dict) else d) == c))
            if p_mine + 1 > p_top:              # lead there -> later chairman chance
                return TR_VALUE * 0.4 * min(1.0, gens_left / 6.0)

    # (d) Hopeless: nothing to gain -> do not send.
    return 0.0


def _party_choice_value(party: str, state: dict) -> float:
    """Value of putting a delegate into EXACTLY this party.

    Influence and the chairmanship only count in the DOMINANT party (server
    getInfluence), so anything else is a waste of money. Only when nothing can be
    won there (an opponent is out of reach) may another party's ruling bonus count.
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
    """party should govern. Picks the one most valuable for the bot's own profile."""
    w       = state.get("waitingFor", {}) or {}
    parties = w.get("parties", []) or []
    if not parties:
        return None
    best = max(parties, key=lambda p: _party_choice_value(p, state))
    log.info("   🏛 Party: %s (value %.1f)", best, _party_choice_value(best, state))
    return {"type": "party", "runId": state["runId"], "partyName": best}


def handle_delegate(state: dict) -> dict | None:
    """whose delegate is removed. The title decides: negative (remove) -> hit an
    opponent or a neutral delegate; positive -> the bot's own colour. The title may
    be a message template.
    """
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
    log.info("   🏛 Delegate: %s (%s)", chosen, "opponent" if negative else "own")
    return {"type": "delegate", "runId": state["runId"], "player": chosen}


def handle_ares_global_parameters(state: dict) -> dict:
    """The bot benefits from a LONGER game (its engine arrives late, see the TR
    horizon). Pushing every parameter down delays the maxima, so the game lasts
    longer. Choosing -1 also avoids the hazard ESCALATIONS that come with pushing
    up (severe erosions and dust storms).

    SAFE: the server only validates inRange(-1..1) and shifts a parameter only when
    it is `available`. -1 is always in range and is never rejected; unavailable
    parameters are ignored.
    """
    waiting = state.get("waitingFor", {})
    runId = waiting.get("runId") or state.get("runId")
    resp = {
        "lowOceanDelta":    -1,
        "highOceanDelta":   -1,
        "temperatureDelta": -1,
        "oxygenDelta":      -1,
    }
    log.info("  🦋 Ares-Parameter: alle -1 (Spiel verlaengern, Hazards vermeiden)")
    out = {"type": "aresGlobalParameters", "response": resp}
    if runId:
        out["runId"] = runId
    return out


HANDLERS = {
    "initialCards": handle_initial_cards,
    "card":         handle_card,
    "or":           handle_or,
    "option":       handle_option,
    "space":        handle_space,
    "amount":       handle_amount,
    "payment":      handle_payment,
    "and":          handle_and,      # multiple inputs (e.g. after playing a card)
    "selectAmount": handle_amount,   # Alias
    "player":       handle_player,   # Spielerauswahl (z.B. Cloud Seeding)
    "productionToLose": handle_production_to_lose,   # Ares-Hazard-Strafe
    "aresGlobalParameters": handle_ares_global_parameters,  # Ares: Butterfly Effect etc.
    "colony":       handle_colony,   # Colonies: Handel/Bau-Auswahl
    "party":        handle_party,     # Turmoil: Partei waehlen
    "delegate":     handle_delegate,  # Turmoil: Delegat waehlen
}


def decide(state: dict) -> dict | None:
    waiting = state.get("waitingFor")
    if not waiting:
        return None
    _telem_gen(state)
    if _DUMP_WF:
        try:
            _g = state.get("game") or {}
            with open(_DUMP_WF, "a", encoding="utf-8") as _f:
                _f.write(json.dumps({"gen": _g.get("generation"),
                                     "title": str(waiting.get("title"))[:80],
                                     "type": waiting.get("type"),
                                     # thisPlayer + Global-Parameter, um "besitzt Engine X,
                                     # but not offered" from "does not own it" (diagnosing
                                     # conditional engines). Only with
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
            # VP source breakdown. CARDS is the key metric for the VP engine.
            _vpb = state["thisPlayer"]["victoryPointsBreakdown"]
            log.info("🎴 VP sources | CARDS:%d greenery:%d city:%d milestones:%d awards:%d TR:%d",
                     _vpb.get("victoryPoints", 0), _vpb.get("greenery", 0),
                     _vpb.get("city", 0), _vpb.get("milestones", 0),
                     _vpb.get("awards", 0), _vpb.get("terraformRating", 0))
            # Engine-Diagnose (Anreiz-Feld-Test): Produktion am Spielende. Pflanzen-
            # Production is the key indicator of a real engine.
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
                        help="path to the card database")
    parser.add_argument("--model", default="tm_model.pt",
                        help="path to the ML model (optional)")
    parser.add_argument("--debug-pause", default=0, type=float,
                        help="pause in seconds before every POST (for debugging)")
    args = parser.parse_args()

    load_card_db(args.db)
    load_ml_model(args.model)
    run_bot(args.url, args.player_id, args.poll, args.debug_pause)
