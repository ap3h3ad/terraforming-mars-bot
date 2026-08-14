import * as responses from '../server/responses';
import {IPlayer} from '../IPlayer';
import {Handler} from './Handler';
import {Context} from './IHandler';
import {isPlayerId} from '../../common/Types';
import {Request} from '../Request';
import {Response} from '../Response';
import {Resource} from '../../common/Resource';
import {CardName} from '../../common/cards/CardName';
import {IProjectCard} from '../cards/IProjectCard';
import {Database} from '../database/Database';

/**
 * DEBUG-ENDPUNKT: veraendert live den Zustand eines Spielers.
 *
 * NUR fuer lokale Tests gedacht - er umgeht saemtliche Spielregeln. Deshalb ist er
 * standardmaessig AUS und muss ueber die Umgebungsvariable DEBUG_STATE_API=1
 * eingeschaltet werden. Ohne sie antwortet er mit 404, als gaebe es ihn nicht.
 *
 * Aufruf (POST):
 *   /api/debug/state?id=<playerId>
 *   Body: {"megacredits": 10, "steel": -2, "addCard": "Birds"}
 *
 * Ressourcen sind DELTAS (koennen negativ sein), der Bestand faellt nie unter 0.
 *
 * addCard prueft die Konsistenz: Die Karte muss im Nachzieh- oder Ablagestapel liegen.
 * Hat sie jemand auf der Hand oder gespielt, kommt ein Fehler - sonst gaebe es sie
 * zweimal im Spiel.
 *
 * removeCard nimmt eine Karte von der Hand und legt sie auf den Ablagestapel.
 *
 * WICHTIG - warum am Ende neu gefragt wird:
 * Der Server baut das Aktionsmenue EINMAL auf, wenn der Spieler an die Reihe kommt
 * (getActions()), mit den Ressourcen von genau diesem Moment. Wer danach Geld oder
 * Waerme dazubekommt, sieht die neuen Moeglichkeiten nicht - die Karte bleibt
 * unspielbar, die Waerme nicht umwandelbar. Deshalb ruft dieser Endpunkt zum Schluss
 * takeAction() auf, was das Menue neu erzeugt.
 * Das geschieht NUR, wenn der Spieler gerade auf genau dieses Aktionsmenue wartet -
 * bei jeder anderen offenen Eingabe (Feldwahl, Zahlung, Draft) wuerde ein takeAction()
 * die haengende Eingabe verwerfen und die Partie zerstoeren. In dem Fall wird die
 * Aenderung trotzdem uebernommen und im Ergebnis ein Hinweis zurueckgegeben.
 */
export class ApiDebugState extends Handler {
  public static readonly INSTANCE = new ApiDebugState();

  private static enabled(): boolean {
    const v = process.env.DEBUG_STATE_API;
    return v !== undefined && v !== '' && v !== '0' && v !== 'false';
  }

  public override async post(req: Request, res: Response, ctx: Context): Promise<void> {
    if (!ApiDebugState.enabled()) {
      responses.notFound(req, res, 'debug api disabled (set DEBUG_STATE_API=1)');
      return;
    }

    const playerId = ctx.url.searchParams.get('id');
    if (playerId === null || !isPlayerId(playerId)) {
      responses.badRequest(req, res, 'missing or invalid id parameter');
      return;
    }

    const game = await ctx.gameLoader.getGame(playerId);
    if (game === undefined) {
      responses.notFound(req, res, 'cannot find game for that player');
      return;
    }
    let player: IPlayer | undefined;
    try {
      player = game.getPlayerById(playerId);
    } catch (err) {
      console.warn(`unable to find player ${playerId}`, err);
    }
    if (player === undefined) {
      responses.notFound(req, res, 'player not found');
      return;
    }

    // Body wie in PlayerInput.ts einlesen - Request bietet keine body()-Methode.
    const raw = await new Promise<string>((resolve) => {
      let buf = '';
      req.on('data', (data) => {
        buf += data.toString();
      });
      req.once('end', () => resolve(buf));
    });
    let body: any;
    try {
      body = JSON.parse(raw);
    } catch (err) {
      responses.badRequest(req, res, 'body is not valid JSON');
      return;
    }

    const applied: Array<string> = [];

    // ---- Ressourcen (Deltas) --------------------------------------------------
    const resources: Array<[string, Resource]> = [
      ['megacredits', Resource.MEGACREDITS],
      ['steel', Resource.STEEL],
      ['titanium', Resource.TITANIUM],
      ['plants', Resource.PLANTS],
      ['energy', Resource.ENERGY],
      ['heat', Resource.HEAT],
    ];
    for (const [key, resource] of resources) {
      const raw = body[key];
      if (raw === undefined) continue;
      const delta = Number(raw);
      if (!Number.isFinite(delta) || !Number.isInteger(delta)) {
        responses.badRequest(req, res, `${key} must be an integer`);
        return;
      }
      const before = player.stock[resource];
      // Stock.add deckelt negative Deltas selbst auf den vorhandenen Bestand.
      player.stock.add(resource, delta, {log: true});
      applied.push(`${key} ${before} -> ${player.stock[resource]}`);
    }

    // ---- Karte auf die Hand ---------------------------------------------------
    const cardName = body['addCard'];
    if (cardName !== undefined) {
      const name = String(cardName) as CardName;
      const deck = game.projectDeck;

      const inDraw = deck.drawPile.findIndex((c) => c.name === name);
      const inDiscard = deck.discardPile.findIndex((c) => c.name === name);

      if (inDraw < 0 && inDiscard < 0) {
        // Wo steckt sie? Fuer eine brauchbare Fehlermeldung nachsehen.
        let holder = '';
        for (const p of game.players) {
          if (p.cardsInHand.some((c) => c.name === name)) {
            holder = `in hand of ${p.color}`;
            break;
          }
          if (p.playedCards.has(name)) {
            holder = `already played by ${p.color}`;
            break;
          }
        }
        if (holder !== '') {
          responses.badRequest(req, res, `card '${name}' is not available: ${holder}`);
          return;
        }
        // Nicht im Spiel. Haeufigster Grund: die falsche VARIANTE. Karten wie
        // 'Deimos Down' existieren als ':ares', ':promo', ':SP' - und mit aktivem Ares
        // ersetzt die Ares-Fassung die Grundkarte komplett
        // (AresCardManifest.cardsToRemove). Welche im Deck liegt, haengt also von den
        // Modulen dieser Partie ab. Deshalb hier nachsehen, welche Varianten es
        // TATSAECHLICH gibt, statt nur 'unknown' zu melden.
        const base = String(name).split(':')[0].toLowerCase();
        const inDeck = [...deck.drawPile, ...deck.discardPile]
          .map((c) => String(c.name))
          .filter((n) => n.split(':')[0].toLowerCase() === base);
        if (inDeck.length > 0) {
          responses.badRequest(req, res,
            `card '${name}' is not in this game - available variant(s): ` +
            inDeck.join(', '));
        } else {
          responses.badRequest(req, res,
            `card '${name}' is not a project card in this game ` +
            '(wrong name, or its module is disabled)');
        }
        return;
      }

      const card: IProjectCard = inDraw >= 0 ?
        deck.drawPile.splice(inDraw, 1)[0] :
        deck.discardPile.splice(inDiscard, 1)[0];
      player.cardsInHand.push(card);
      applied.push(`card '${name}' from ${inDraw >= 0 ? 'draw pile' : 'discard pile'} to hand`);
    }

    // ---- Karte von der Hand entfernen ----------------------------------------
    const removeName = body['removeCard'];
    if (removeName !== undefined) {
      const name = String(removeName) as CardName;
      const idx = player.cardsInHand.findIndex((c) => c.name === name);
      if (idx < 0) {
        responses.badRequest(req, res,
          `card '${name}' is not in hand of ${player.color}`);
        return;
      }
      const card = player.cardsInHand.splice(idx, 1)[0];
      game.projectDeck.discard(card);
      applied.push(`card '${name}' from hand to discard pile`);
    }

    if (applied.length === 0) {
      responses.badRequest(req, res, 'nothing to do - no known field in body');
      return;
    }

    game.log('DEBUG: state changed for ${0} (' + applied.join('; ') + ')',
      (b) => b.player(player as IPlayer));

    // Aktionsmenue neu aufbauen, damit die Aenderung nutzbar wird (s. Kopfkommentar).
    let refreshed = false;
    let note: string | undefined;
    const waitingFor = player.getWaitingFor();
    const title = (waitingFor as any)?.title;
    const isActionMenu = typeof title === 'string' &&
      (title === 'Take your first action' || title === 'Take your next action');
    if (waitingFor === undefined) {
      note = 'player is not waiting for input - change applies when it is their turn';
    } else if (isActionMenu) {
      player.takeAction(false);
      refreshed = true;
    } else {
      note = `player is waiting for '${String(title)}' - menu NOT refreshed; ` +
        'finish that input first, then the change becomes usable';
    }

    await Database.getInstance().saveGame(game);

    res.setHeader('Content-Type', 'application/json');
    res.write(JSON.stringify({
      ok: true, player: player.color, applied: applied,
      refreshed: refreshed, note: note,
    }));
    res.end();
  }
}
