#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patcht die Bot-Integration in ein terraforming-mars-Repo.

Warum ein PATCH-Skript und kein Kopier-Skript:
Fertige Dateien zu kopieren geht nur gut, wenn Quell- und Ziel-Repo denselben Stand haben.
Das war hier nicht der Fall (lokales Repo != upstream/main -> fehlender Import
CreateGameSettingsStorage, SubnauticPirates-Typfehler). Dieses Skript sucht stattdessen
ANKER-Stellen und fuegt nur die noetigen Zeilen ein - unabhaengig vom Repo-Stand.

Eigenschaften:
  * IDEMPOTENT   - mehrfaches Ausfuehren aendert nichts (erkennt bereits gepatchte Dateien)
  * SICHER       - legt von jeder geaenderten Datei ein .bak an
  * TRANSPARENT  - meldet genau, was geaendert wurde und was NICHT gefunden wurde
  * --dry-run    - zeigt nur an, was passieren wuerde

Aufruf (aus dem Repo-Wurzelverzeichnis):
    py -3.12 patch_bot_integration.py                 # patchen
    py -3.12 patch_bot_integration.py --dry-run       # nur pruefen
    py -3.12 patch_bot_integration.py --repo <pfad>   # anderes Repo

Danach:
    npm run build
"""
import argparse
import shutil
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Die neue Datei: src/server/bot/BotLauncher.ts
# ─────────────────────────────────────────────────────────────────────────────
BOT_LAUNCHER_TS = r"""import {spawn, spawnSync} from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

/**
 * Startet den Python-Bot (tm_mcts_mp.py --join) als eigenstaendigen Subprozess und haengt ihn
 * an einen bereits erstellten Spieler an.
 *
 * Warum Subprozess (und kein Daemon)?  Bei jedem Spielstart wird ein FRISCHER Python-Prozess
 * gestartet -> er laedt automatisch die aktuelle tm_bot.py / tm_mcts_mp.py. Die Bot-Dateien
 * sind damit jederzeit austauschbar, ohne Server-Neustart und ohne Rebuild.
 *
 * ABLAGE DER BOT-DATEIEN: Standard ist der Ordner `bot/` IM REPO-WURZELVERZEICHNIS:
 *
 *     <repo>/bot/tm_mcts_mp.py
 *     <repo>/bot/tm_bot.py
 *     <repo>/bot/card_db.json
 *
 * Damit laeuft es lokal (Windows) und auf dem Remote-Server (Linux) identisch, ohne absolute
 * Pfade. Voraussetzung auf dem Server: Python 3 + `pip install requests`.
 *
 * Konfiguration (alles optional ausser BOT_ENABLED):
 *   BOT_ENABLED    '1' schaltet das Feature frei
 *   BOT_PYTHON     Python-Kommando        (Default: Windows 'py -3.12', sonst 'python3')
 *   BOT_DIR        Bot-Verzeichnis        (Default: '<cwd>/bot')
 *   BOT_SCRIPT     Startskript            (Default: 'tm_mcts_mp.py')
 *   BOT_ARGS       zusaetzliche Argumente (Default: '--no-mcts')
 *   BOT_SERVER_URL URL zum Server         (Default: http://localhost:$PORT)
 *   BOT_LOG_DIR    Logverzeichnis         (Default: BOT_DIR)
 */

export class BotLaunchError extends Error {}

function config() {
  const defaultPython = process.platform === 'win32' ? 'py -3.12' : 'python3';
  const python = (process.env.BOT_PYTHON ?? defaultPython).trim().split(/\s+/);
  const dir = process.env.BOT_DIR ?? path.join(process.cwd(), 'bot');
  return {
    enabled: process.env.BOT_ENABLED === '1',
    cmd: python[0],
    cmdArgs: python.slice(1),
    dir: dir,
    script: process.env.BOT_SCRIPT ?? 'tm_mcts_mp.py',
    extraArgs: (process.env.BOT_ARGS ?? '--no-mcts').trim().split(/\s+/).filter((s) => s.length > 0),
    logDir: process.env.BOT_LOG_DIR ?? dir,
  };
}

/**
 * Prueft VOR der Spielerstellung, ob der Bot startbar ist. Wirft BotLaunchError mit einer
 * klaren Meldung - die Route reicht sie als HTTP 400 an das Frontend durch, damit der Nutzer
 * nicht mit einer stummen Partie dasteht, in der der Gegner nie zieht.
 */
export function assertBotLaunchable(): void {
  const c = config();
  if (!c.enabled) {
    throw new BotLaunchError(
      'Bot-Gegner ist auf diesem Server nicht aktiviert (Umgebungsvariable BOT_ENABLED=1 setzen).');
  }
  const scriptPath = path.join(c.dir, c.script);
  if (!fs.existsSync(scriptPath)) {
    throw new BotLaunchError(
      `Bot-Skript nicht gefunden: ${scriptPath}. ` +
      `Bot-Dateien (tm_mcts_mp.py, tm_bot.py, card_db.json) nach '${c.dir}' legen ` +
      `oder BOT_DIR auf das richtige Verzeichnis setzen.`);
  }
  const probe = spawnSync(c.cmd, [...c.cmdArgs, '--version'], {timeout: 10_000});
  if (probe.error !== undefined || probe.status !== 0) {
    const reason = probe.error?.message ?? `Exit-Code ${probe.status}`;
    throw new BotLaunchError(
      `Python konnte nicht gestartet werden ('${[c.cmd, ...c.cmdArgs].join(' ')}'): ${reason}. ` +
      `BOT_PYTHON pruefen (Linux meist 'python3', Windows 'py -3.12').`);
  }
  const dep = spawnSync(c.cmd, [...c.cmdArgs, '-c', 'import requests'], {timeout: 15_000});
  if (dep.status !== 0) {
    throw new BotLaunchError(
      `Python-Paket 'requests' fehlt (fuer '${[c.cmd, ...c.cmdArgs].join(' ')}'). ` +
      `Installieren mit: ${c.cmd} -m pip install requests`);
  }
}

/**
 * Startet den Bot fuer die uebergebene Spieler-ID. Der Prozess laeuft unabhaengig weiter und
 * beendet sich selbst, wenn die Partie vorbei ist. stdout/stderr landen in einer Logdatei.
 */
export function launchBot(gameId: string, botPlayerId: string, baseUrl: string): void {
  const c = config();
  const args = [
    ...c.cmdArgs,
    c.script,
    '--join',
    '--player-id', botPlayerId,
    '--url', baseUrl,
    ...c.extraArgs,
  ];

  const logFile = path.join(c.logDir, `bot_${gameId}.log`);
  let out: number | 'ignore' = 'ignore';
  try {
    out = fs.openSync(logFile, 'a');
  } catch (err) {
    console.warn(`[Bot] Logdatei ${logFile} nicht schreibbar: ${String(err)}`);
  }

  const child = spawn(c.cmd, args, {
    cwd: c.dir,
    detached: true,
    stdio: ['ignore', out, out],
  });

  child.on('error', (err) => {
    console.error(`[Bot] Start fehlgeschlagen (game=${gameId}, player=${botPlayerId}): ${err.message}`);
  });
  child.unref();

  console.log(
    `[Bot] gestartet: game=${gameId} player=${botPlayerId} ` +
    `cmd='${c.cmd} ${args.join(' ')}' cwd=${c.dir} log=${logFile}`);
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Die Patches: (Datei, Beschreibung, Marker-schon-gepatcht, [(anker, ersatz), ...])
# Ein Patch wird nur angewandt, wenn der Marker fehlt UND jeder Anker GENAU EINMAL vorkommt.
# ─────────────────────────────────────────────────────────────────────────────
PATCHES = [
    (
        'src/common/game/NewGameConfig.ts',
        'Feld botOpponent + Konstante BOT_PLAYER_NAME',
        'BOT_PLAYER_NAME',
        [
            (
                "export type BoardNameType = BoardName | RandomBoardOption;",
                """export type BoardNameType = BoardName | RandomBoardOption;

/**
 * Name des vom Bot gesteuerten Spielers. Der Server identifiziert den Bot-Spieler UEBER DIESEN
 * NAMEN und nicht ueber den Index: das Frontend MISCHT die Spielerliste, wenn
 * `randomFirstPlayer` aktiv ist -> ein Index waere mal der Bot und mal der Mensch.
 */
export const BOT_PLAYER_NAME = 'Bot';""",
            ),
            (
                """  customCeos: Array<CardName>;
  startingCeos: number;
  startingPreludes: number;
}""",
                """  customCeos: Array<CardName>;
  startingCeos: number;
  startingPreludes: number;

  /**
   * Nur bei genau 2 Spielern: der zweite Spieler wird von einem Python-Bot gesteuert.
   * Der Server startet ihn nach der Spielerstellung (src/server/bot/BotLauncher.ts).
   */
  botOpponent?: boolean;
}""",
            ),
        ],
    ),
    (
        'src/server/routes/ApiCreateGame.ts',
        'Bot-Vorabpruefung + Bot-Start + Fehlerdurchreichung',
        'BotLauncher',
        [
            (
                "import {NewGameConfig} from '../../common/game/NewGameConfig';",
                "import {BOT_PLAYER_NAME, NewGameConfig} from '../../common/game/NewGameConfig';\n"
                "import {assertBotLaunchable, launchBot, BotLaunchError} from '../bot/BotLauncher';",
            ),
            (
                """          const gameReq = JSON.parse(body) as NewGameConfig;
          const gameId = safeCast(generateRandomId('g'), isGameId);""",
                """          const gameReq = JSON.parse(body) as NewGameConfig;

          // Bot-Gegner: VOR der Spielerstellung pruefen, ob der Bot startbar ist. Schlaegt das
          // fehl, entsteht KEIN Spiel und der Nutzer bekommt eine klare Meldung - statt einer
          // stummen Partie, in der der Gegner nie zieht.
          if (gameReq.botOpponent === true) {
            if (gameReq.players.length !== 2) {
              throw new BotLaunchError('Ein Bot-Gegner ist derzeit nur in 2-Spieler-Partien moeglich.');
            }
            assertBotLaunchable();
          }

          const gameId = safeCast(generateRandomId('g'), isGameId);""",
            ),
            (
                """          ctx.gameLoader.add(game);
          responses.writeJson(res, ctx, Server.getSimpleGameModel(game));
        } catch (error) {
          responses.internalServerError(req, res, error);
        }""",
                """          ctx.gameLoader.add(game);

          // Bot-Gegner: Python-Bot an den Bot-Spieler anhaengen. Der Prozess laeuft
          // eigenstaendig weiter und beendet sich beim Spielende selbst.
          if (gameReq.botOpponent === true) {
            // Ueber den NAMEN suchen: das Frontend mischt die Spielerliste bei
            // randomFirstPlayer -> ein Index waere mal der Bot, mal der Mensch.
            const botPlayer = players.find((p) => p.name === BOT_PLAYER_NAME);
            if (botPlayer === undefined) {
              throw new BotLaunchError(
                `Bot-Spieler ('${BOT_PLAYER_NAME}') nicht in der Spielerliste gefunden.`);
            }
            const baseUrl = process.env.BOT_SERVER_URL ?? `http://localhost:${process.env.PORT ?? 8080}`;
            launchBot(gameId, botPlayer.id, baseUrl);
          }

          responses.writeJson(res, ctx, Server.getSimpleGameModel(game));
        } catch (error) {
          // Bot-Startfehler als klare 400-Meldung durchreichen (nicht als generischer 500).
          if (error instanceof BotLaunchError) {
            responses.badRequest(req, res, error.message);
            resolve();
            return;
          }
          responses.internalServerError(req, res, error);
        }""",
            ),
        ],
    ),
    (
        'src/client/components/create/CreateGameForm.vue',
        'Checkbox "Opponent is a bot" + Bot-Platz-Kennzeichnung',
        'botOpponent',
        [
            # a) Checkbox nach dem Muster von "Random first player": direkt im Column, OHNE
            #    eigene 'create-game-page-column-row'. Eine solche Row ist ein ZWEISPALTEN-
            #    Layout (zwei Checkboxen nebeneinander) - eine Row mit nur einer Checkbox
            #    aendert die Spaltenhoehe und verschiebt die Buttons darunter.
            (
                """                            <input type="checkbox" v-model="randomFirstPlayer" id="randomFirstPlayer-checkbox">
                            <label for="randomFirstPlayer-checkbox">
                                <span v-i18n>Random first player</span>
                            </label>""",
                """                            <input type="checkbox" v-if="playersCount === 2" v-model="botOpponent" id="bot-opponent-checkbox">
                            <label v-if="playersCount === 2" for="bot-opponent-checkbox">
                                <span v-i18n>Opponent is a bot</span>
                            </label>

                            <input type="checkbox" v-model="randomFirstPlayer" id="randomFirstPlayer-checkbox">
                            <label for="randomFirstPlayer-checkbox">
                                <span v-i18n>Random first player</span>
                            </label>""",
            ),
            # b) Namensfeld des Bot-Platzes sperren (Anker toleriert '/>' und '>' am Ende)
            (
                """v-model="newPlayer.name" """,
                """v-model="newPlayer.name" :disabled="isBotSeat(index)" """,
            ),
            # c) Import der Konstante
            (
                "import {BoardNameType, NewGameConfig, NewPlayerModel} from '@/common/game/NewGameConfig';",
                "import {BOT_PLAYER_NAME, BoardNameType, NewGameConfig, NewPlayerModel} from '@/common/game/NewGameConfig';",
            ),
            # d) FormModel-Typ
            (
                """type FormModel = {
  preludeToggled: boolean;
  uploading: boolean;
};""",
                """type FormModel = {
  preludeToggled: boolean;
  uploading: boolean;
  /** Nur bei genau 2 Spielern: der zweite Spieler wird vom Python-Bot gesteuert. */
  botOpponent: boolean;
};""",
            ),
            # e) data()
            (
                """      ...defaultCreateGameModel(),
      preludeToggled: false,
      uploading: false,
    };""",
                """      ...defaultCreateGameModel(),
      preludeToggled: false,
      uploading: false,
      botOpponent: false,
    };""",
            ),
            # f) Hilfsmethode isBotSeat
            (
                """    getPlayers(): Array<NewPlayerModel> {
      return this.players.slice(0, this.playersCount);
    },""",
                """    getPlayers(): Array<NewPlayerModel> {
      return this.players.slice(0, this.playersCount);
    },
    /** Ist dieser Platz der Bot-Platz? (2 Spieler + Bot-Haken -> immer der ZWEITE Spieler) */
    isBotSeat(index: number): boolean {
      return this.playersCount === 2 && this.botOpponent === true && index === 1;
    },""",
            ),
            # g) Bot-Spieler VOR dem Shuffle benennen
            (
                """    async serializeSettings() {
      let players = this.players.slice(0, this.playersCount);

      if (this.randomFirstPlayer) {""",
                """    async serializeSettings() {
      let players = this.players.slice(0, this.playersCount);

      // Bot-Spieler eindeutig benennen, BEVOR die Liste gemischt wird. Der Server findet ihn
      // dann ueber den Namen (der Index waere nach dem Shuffle unbrauchbar).
      if (this.playersCount === 2 && this.botOpponent) {
        players = players.map((p, i) => (i === 1 ? {...p, name: BOT_PLAYER_NAME} : p));
      }

      if (this.randomFirstPlayer) {""",
            ),
            # h) botOpponent mitschicken
            (
                """      const dataToSend: NewGameConfig = {
        players,""",
                """      const dataToSend: NewGameConfig = {
        players,
        // Bot-Gegner nur in 2-Spieler-Partien (der Server lehnt alles andere ab).
        botOpponent: this.playersCount === 2 ? this.botOpponent : false,""",
            ),
        ],
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser(description='Patcht die Bot-Integration in ein terraforming-mars-Repo.')
    ap.add_argument('--repo', default='.', help='Repo-Wurzelverzeichnis (Default: aktuelles)')
    ap.add_argument('--dry-run', action='store_true', help='nur pruefen, nichts schreiben')
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    if not (repo / 'package.json').exists():
        print(f'FEHLER: {repo} sieht nicht nach dem Repo-Wurzelverzeichnis aus (keine package.json).')
        return 1

    print(f'Repo: {repo}')
    print(f'Modus: {"DRY-RUN (nichts wird geschrieben)" if args.dry_run else "PATCHEN"}\n')

    ok, skipped, failed = 0, 0, 0

    # ── 1. Neue Datei: BotLauncher.ts ────────────────────────────────────────
    launcher = repo / 'src' / 'server' / 'bot' / 'BotLauncher.ts'
    if launcher.exists() and launcher.read_text(encoding='utf-8') == BOT_LAUNCHER_TS:
        print(f'= {launcher.relative_to(repo)} (bereits aktuell)')
        skipped += 1
    else:
        print(f'+ {launcher.relative_to(repo)} ({"anlegen" if not launcher.exists() else "aktualisieren"})')
        if not args.dry_run:
            launcher.parent.mkdir(parents=True, exist_ok=True)
            if launcher.exists():
                shutil.copyfile(launcher, str(launcher) + '.bak')
            launcher.write_text(BOT_LAUNCHER_TS, encoding='utf-8')
        ok += 1

    # ── 2. Patches ───────────────────────────────────────────────────────────
    for rel, desc, marker, edits in PATCHES:
        path = repo / rel
        if not path.exists():
            print(f'! {rel}: DATEI FEHLT -> uebersprungen')
            failed += 1
            continue

        src = path.read_text(encoding='utf-8')
        if marker in src:
            print(f'= {rel} (bereits gepatcht)')
            skipped += 1
            continue

        # Alle Anker pruefen, BEVOR etwas geaendert wird
        missing = [a for a, _ in edits if src.count(a) != 1]
        if missing:
            print(f'! {rel}: {len(missing)} Anker nicht eindeutig gefunden -> NICHT gepatcht')
            for a in missing:
                n = src.count(a)
                head = a.strip().splitlines()[0][:70]
                print(f'     {n}x  "{head}..."')
            failed += 1
            continue

        for anchor, repl in edits:
            src = src.replace(anchor, repl, 1)

        print(f'+ {rel}: {desc}')
        if not args.dry_run:
            shutil.copyfile(path, str(path) + '.bak')
            path.write_text(src, encoding='utf-8')
        ok += 1

    # ── 3. bot/-Ordner ───────────────────────────────────────────────────────
    botdir = repo / 'bot'
    # tm_mcts_mp.py importiert tm_bot UND tm_mcts -> beide muessen mit.
    needed = ['tm_mcts_mp.py', 'tm_bot.py', 'tm_mcts.py', 'card_db.json']
    have = [f for f in needed if (botdir / f).exists()]
    print()
    if len(have) == len(needed):
        print(f'= bot/ vollstaendig ({", ".join(needed)})')
        # Sicherheitsnetz: importieren die Bot-Dateien noch weitere lokale Module?
        import re as _re
        known = {p.stem for p in botdir.glob('*.py')}
        missing_mods = set()
        for py in botdir.glob('*.py'):
            for m in _re.finditer(r'^(?:from|import)\s+([a-zA-Z_][\w]*)',
                                  py.read_text(encoding='utf-8', errors='replace'), _re.M):
                mod = m.group(1)
                if mod.startswith(('tm_', 'eval', 'card_')) and mod not in known:
                    missing_mods.add(mod)
        if missing_mods:
            print(f'!   ABER: diese lokalen Module fehlen noch: '
                  f'{", ".join(sorted(m + ".py" for m in missing_mods))}')
    else:
        missing_bot = [f for f in needed if f not in have]
        print(f'! bot/ unvollstaendig - es fehlen: {", ".join(missing_bot)}')
        print(f'     Diese Dateien nach {botdir} kopieren (der Server startet den Bot von dort).')
        if not args.dry_run:
            botdir.mkdir(exist_ok=True)

    # ── Zusammenfassung ──────────────────────────────────────────────────────
    print(f'\n{"-" * 60}')
    print(f'geaendert: {ok} | uebersprungen: {skipped} | FEHLGESCHLAGEN: {failed}')
    if failed:
        print('\nBei fehlgeschlagenen Dateien haben sich die Anker-Stellen geaendert (anderer')
        print('Repo-Stand). Betroffene Datei hochladen -> die Anker werden angepasst.')
        return 2
    if not args.dry_run and ok:
        print('\nBackups liegen als *.bak neben den Originalen.')
        print('\nNaechste Schritte:')
        print('  1. Bot-Dateien nach bot/ kopieren (falls oben gemeldet)')
        print('  2. npm run build')
        print('  3. Umgebungsvariablen setzen und Server starten:')
        print('       BOT_ENABLED=1')
        print('       BOT_SERVER_URL=http://localhost:9000   (euer Port)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
