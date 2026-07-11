import {spawn, spawnSync} from 'child_process';
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
