# <a name="README"> Terraforming Mars Open-source

<div align="center">
  <img src="https://raw.githubusercontent.com/bafolts/terraforming-mars/main/assets/expansion_icons/expansion_icon_corporateEra.png">
  <img src="https://raw.githubusercontent.com/bafolts/terraforming-mars/main/assets/expansion_icons/expansion_icon_venus.png">
  <img src="https://raw.githubusercontent.com/bafolts/terraforming-mars/main/assets/expansion_icons/expansion_icon_colonies.png">
  <img src="https://raw.githubusercontent.com/bafolts/terraforming-mars/main/assets/expansion_icons/expansion_icon_turmoil.png">
  <img src="https://raw.githubusercontent.com/bafolts/terraforming-mars/main/assets/expansion_icons/expansion_icon_prelude.png">
</div>
<div align="center">
  <img src="https://raw.githubusercontent.com/bafolts/terraforming-mars/main/assets/expansion_icons/expansion_icon_ares.png">
  <img src="https://raw.githubusercontent.com/bafolts/terraforming-mars/main/assets/expansion_icons/expansion_icon_community.png">
  <img src="https://raw.githubusercontent.com/bafolts/terraforming-mars/main/assets/expansion_icons/expansion_icon_promo.png">
  <img src="https://raw.githubusercontent.com/bafolts/terraforming-mars/main/assets/expansion_icons/expansion_icon_agendas.png">
  <img src="https://raw.githubusercontent.com/bafolts/terraforming-mars/main/assets/expansion_icons/expansion_icon_themoon.png">
  <img src="https://raw.githubusercontent.com/bafolts/terraforming-mars/main/assets/expansion_icons/expansion_icon_pathfinders.png">
  <img src="https://raw.githubusercontent.com/bafolts/terraforming-mars/main/assets/expansion_icons/expansion_icon_escapeVelocity.png">
</div>

This is an open-source online implementation of the great board game Terraforming mars. **It is not affiliated
with FryxGames, Asmodee Digital or Steam in any way.**

**Note**: This project has no affiliation with "Rebalanced Mars", whose authors have refused to open-source their code.
We believe this is both a violation of our GPL3 license, and also of the spirit of collaboration that this project tries
to foster. Note that any new features you see on this repo made available on that server are without our permission.

**Buy The Board Game**

The board game is great and this repository highly recommends [purchasing it](https://www.amazon.com/Stronghold-Games-6005SG-Terraforming-Board/dp/B01GSYA4K2) for personal use.

## ⬤ I want to join the community!
[Join us over on Discord!](https://discord.gg/afeyggbN6Y).

## ⬤ I want to play!
There's a instance available at https://terraforming-mars.herokuapp.com/. It's generally reliable, but read more below.

There's also this excellent
[YouTube playlist](https://youtube.com/playlist?list=PLCGE78n9vCqhhmRe9YCrRh2GLNMPB_3j1) focused on tutorials custom for this app.

NOTE: This site is restarted daily. A multiplayer game will remain available for 15 days, after which it will be flushed from the database.
Unfinished solo games are flushed after one day. We continue to make stability and scalability improvements in step with growth and popularity,
but to make sure your game remains, we highly recommended to host your own web server.

## ⬤ I want to play against a bot!
You can! This fork adds a heuristic bot opponent written in Python. Everything the bot needs is
already in this repository under `bot/` (`tm_mcts_mp.py`, `tm_bot.py`, `tm_mcts.py`,
`card_db.json`) — there is nothing else to download.

The server starts the bot as a separate Python process whenever you create a game with
"Opponent is a bot", so you need a working Python installation next to Node.

### Requirements

| | |
|---|---|
| Node | 22.x (same as upstream) |
| Python | **3.10 or newer** |
| Python packages | `requests` |

Python 3.10 is a hard minimum: the bot uses `X \| Y` type annotations that are evaluated on
import, so 3.8 and 3.9 fail immediately. Note that some distributions still ship an older
`python3` — check with `python3 --version` before you start.

### Installing Node and Python on a fresh Linux server

The distribution packages are usually too old — Node 22 is required, and `apt install npm` pulls in
an older Node. On Debian or Ubuntu, as root:

```bash
apt update
apt install -y curl ca-certificates git
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt install -y nodejs
node -v          # must print v22.x

python3 --version                     # must be 3.10 or newer
apt install -y python3-requests       # the bot's only dependency
```

`requests` is installed from the distribution on purpose: recent Ubuntu and Debian releases ship
`python3` without `pip`, and even with `pip` installed a system-wide `pip install` is refused
(`externally-managed-environment`, PEP 668). If you prefer pip, create a virtualenv and point
`BOT_PYTHON` at its interpreter. If `python3` itself is older than 3.10 (Ubuntu 20.04 ships 3.8),
install a newer interpreter and set `BOT_PYTHON` accordingly.

A quick smoke test that covers the Python version, `requests` and the bot files in one go:

```bash
cd bot && python3 tm_mcts_mp.py --help && cd ..
```

### Setup

```bash
npm install

# the database driver (better-sqlite3) is an OPTIONAL dependency: if no prebuilt binary
# matches your platform, npm install still succeeds and the server only fails later.
node -e "require('better-sqlite3'); console.log('sqlite ok')"
# on error: apt install -y build-essential python3   and run npm install again

npm run build

# install the bot's only dependency
python3 -m pip install requests

# enable the bot — create .env only if you do not have one yet,
# an existing .env must NOT be overwritten (it holds your PORT, TLS paths, ...)
[ -f .env ] || cp .env.sample .env
echo "BOT_ENABLED=1" >> .env

npm run start
```

If you already run this server and have a `.env`, only add the `BOT_ENABLED=1` line — do not
replace the file.

Then open the server, create a **2 player** game, tick **"Opponent is a bot"** in the options and
start the game. The checkbox only appears for 2 player games; the bot is not available for solo or
for 3+ players.

If something is missing, the server refuses to create the game and shows the reason (wrong Python
command, `requests` not installed, bot script not found), so you will not end up in a silent game
whose opponent never moves.

### Configuration

Only `BOT_ENABLED` has to be set. Everything else has a default and is only needed if your setup
differs:

| Variable | Default | Purpose |
|---|---|---|
| `BOT_ENABLED` | *(unset — bot disabled)* | `1` enables the bot opponent |
| `BOT_PYTHON` | `python3`, on Windows `py -3.12` | command used to start Python |
| `BOT_DIR` | `<repo>/bot` | where the bot files live |
| `BOT_SCRIPT` | `tm_mcts_mp.py` | entry point |
| `BOT_ARGS` | `--no-mcts` | extra arguments for the bot |
| `BOT_SERVER_URL` | `http://localhost:$PORT` | how the bot reaches this server |
| `BOT_LOG_DIR` | same as `BOT_DIR` | where per-game bot logs are written |

Two notes on `BOT_PYTHON`: if your `python3` is older than 3.10, point this at a newer interpreter
(an absolute path or a virtualenv works, e.g. `BOT_PYTHON=/opt/python3.12/bin/python3`). And if you
run the server through a process manager, remember that it does not inherit your shell's PATH.

**TLS:** if you set `KEY_PATH` and `CERT_PATH`, the server speaks HTTPS only and the default
`http://localhost:$PORT` will not work. Either terminate TLS in a reverse proxy and keep the
application on plain HTTP (recommended), or set `BOT_SERVER_URL` to the HTTPS URL — the bot
verifies certificates, so a self-signed one will be rejected.

### How it works

Creating a game with a bot opponent spawns one detached Python process that joins the game through
its own player id and exits by itself when the game ends. One process per game, so several
concurrent games mean several Python processes.

Because the process is started fresh each time, you can replace the files in `bot/` without
restarting the server or rebuilding.

### If the bot does not move

Look at `bot/bot_<gameId>.log` — every bot process writes its output there. The most common cause
is a Python version below 3.10: the pre-flight check verifies that Python starts and that
`requests` is importable, but not the version, so a too-old interpreter passes the check and only
fails later on import.

These log files are appended per game and are not rotated, which is worth knowing if you run a
public server.

## ⬤ I want to learn how to play
There are far too many good tutorials online. [Here are the rulebooks, though.](https://github.com/terraforming-mars/terraforming-mars/wiki/Rulebooks)

## ⬤ I want to run a copy of the server locally
Check out our [Local setup wiki page](https://github.com/bafolts/terraforming-mars/wiki/Local-Setup)

Honestly, it's really simple.

## ⬤ I want to run a copy of the server on Heroku
Check out our [Heroku setup wiki page](https://github.com/bafolts/terraforming-mars/wiki/Heroku-Setup)

(As of 2022-11-28, Heroku no longer has a free tier. However, it is still our recommended way to deploy,
as they're the clearest instructions.)

## ⬤ I want to run a copy of the server on Docker
Check out our [Docker setup wiki page](https://github.com/bafolts/terraforming-mars/wiki/Docker-Setup)

(Warning, this is not aggressively supported, though some people are on the Discord.)

## ⬤ I want to run a copy on a YunoHost server
[![Install Terraforming Mars with YunoHost](https://install-app.yunohost.org/install-with-yunohost.svg)](https://install-app.yunohost.org/?app=terraforming-mars)

The code for the Yunohost Terraforming-Mars package is in this [GitHub repo](https://github.com/YunoHost-Apps/terraforming-mars_ynh)

(Warning, this is not specifically supported.)

## ⬤ I want to report a bug or feature request
Add it to our [issues tab](https://github.com/bafolts/terraforming-mars/issues/new).

## ⬤ I want to contribute to development
See [contribution guide](https://github.com/terraforming-mars/terraforming-mars/blob/main/CONTRIBUTING.md) and [local development setup](https://github.com/terraforming-mars/terraforming-mars/wiki/Local-Setup).

## ⬤ I want to win!
Me too, pal. Me too.

## ✨ Contributors ✨

Thanks goes to these wonderful people:

<table border="0">
  <tdata>
    <tr>
      <td><img src="https://avatars1.githubusercontent.com/u/2707843?v=3" width="50px;" alt=""/></td>
      <td><a href="https://github.com/bafolts"><b>Brian Folts</b></a>: All the things</td>
    </tr>
    <tr>
       <td><img src="https://avatars1.githubusercontent.com/u/56086992?v=3" width="50px;" alt=""/></td>
       <td><a href="https://github.com/vincentneko"><b>Vincent Moreau</b></a>: Venus, Prelude, Hellas & Elysium, Colonies, Turmoil</td>
    </tr>
    <tr>
      <td><img src="https://avatars2.githubusercontent.com/u/394311?v=3" width="50px;" alt=""/></td>
      <td><a href="https://github.com/alrusdi"><b>alrusdi</b></a>: Front End, internationalization</td>
    </tr>
    <tr>
      <td><img src="https://avatars3.githubusercontent.com/u/6917565?s=460&v=4" width="50px;" alt=""/></td>
      <td><a href="https://github.com/ssimeonoff"><b>Simeon Simeonov</b></a>: UX, cards and Colonies design</td>
    </tr>
    <tr>
      <td><img src="https://avatars0.githubusercontent.com/u/806950?v=3" width="50px;" alt=""/></td>
      <td><b><a href="https://github.com/pierrehilbert">Pierre Hilbert</b></a>: Turmoil and helps with the things</td>
    </tr>
    <tr>
      <td><img src="https://avatars1.githubusercontent.com/u/2408094?s=460&v=4" width="50px;" alt=""/></td>
      <td><b><a href="https://github.com/nwai90">nwai90</b></a>: Community and Political Agendas fan-made expansions, and helps with the things</td>
    </tr>
    <tr>
      <td><img src="https://avatars1.githubusercontent.com/u/10995145?s=460&v=4" width="50px;" alt=""/></td>
      <td><b><a href="https://github.com/pocc">Pocc</b></a>: He did that one thing one time</td>
    </tr>
    <tr>
      <td><img src="https://avatars1.githubusercontent.com/u/413481?s=460&v=4" width="50px;" alt=""/></td>
      <td><b><a href="https://github.com/kberg">Robert Konigsberg</b></a>: Fan expansions: Ares, The Moon, Pathfinders, Underworld. Prelude 2. Infrastructure cleanup, code reviews, two opinions too many.</a> </td>
    </tr>
    <tr>
      <td><img src="https://avatars.githubusercontent.com/u/836179?s=460&v=4" width="50px;" alt=""/></td>
      <td><a href="https://github.com/chosta"><b>chosta</b></a>: Front end and back end</a> </td>
    </tr>
    <tr>
      <td><img src="https://avatars.githubusercontent.com/u/5318258?s=460&v=4" width="50px;" alt=""/><br />
      <td><a href="https://github.com/Lynesth"><b>Lynesth</b></a>: Help with the things</a> </td>
    </tr>
    <tr>
      <td><img src="https://avatars.githubusercontent.com/u/15874357?s=460&v=4" width="50px;" alt=""/><br />
      <td><a href="https://github.com/derornos"><b>푸른이(derornos)</b></a>: 한국어화 옮긴이(Korean translator)<br>&emsp;<a href="mailto:derornos@gmail.com">메일(email): derornos@gmail.com</a> / <a href="https://open.kakao.com/me/derornos">카카오톡(KakaoTalk, Messenger): link</a></td>
    </tr>
    <tr>
      <td><img src="https://avatars.githubusercontent.com/u/105346182?s=460&v=4" width="50px;" alt=""/><br />
      <td><a href="https://github.com/Borbarad2"><b>Borbarad</b></a>: Translation</a> </td>
    </tr>
    <tr>
      <td><img src="https://avatars.githubusercontent.com/u/2050250?s=460&v=4" width="50px;" alt=""/><br />
      <td><a href="https://github.com/d-little"><b>d-little</b></a>: CEOs</a> </td>
    </tr>
  </tdata>
</table>


## LICENSE

GPLv3

Russian Prototype font: https://fonts-online.ru/fonts/prototype-rus-daymarius (copyright 2001, free for personal use)
Polish Prototype font: https://www.gry-planszowe.pl/viewtopic.php?p=1489006#p1489006 (copyright 2001, free for personal use)
Baord Game Icons: http://www.kenney.nl/  (Creative Commons Zero, CC0)
