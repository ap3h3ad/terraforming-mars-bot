// Copyright © 2020 Jan Keromnes.
// The following code is covered by the MIT license.

const minimist = require('minimist');
const path = require('path');
const request = require('./lib/request');

const { CardFinder } = require('./terraforming-mars/build/src/CardFinder');
const { PlayerInputTypes } = require('./terraforming-mars/build/src/PlayerInputTypes');

const usage = `Usage: node play-bot PLAYER_LINK`;
const argv = minimist(process.argv.slice(2));

if (argv.help || argv._.length !== 1) {
  console.log(usage);
  process.exit();
}

const playerUrl = new URL(argv._[0]);
const serverUrl = playerUrl.origin;
const playerId = playerUrl.searchParams.get('id');
const cardFinder = new CardFinder();

(async () => {
  // Load bot script
  const bot = require('./' + path.join('.', argv.bot || 'bots/random'));

  // Initial research phase
  let game = await request('GET', `${serverUrl}/api/player?id=${playerId}`);
  logGameState(game);
  const availableCorporations = game.waitingFor.options[0].cards;
  const availableCards = game.waitingFor.options[1].cards;
  let move = await bot.playInitialResearchPhase(game, availableCorporations, availableCards);
  console.log('Bot plays:', move);
  game = await request('POST', `${serverUrl}/player/input?id=${playerId}`, move);

  while (game.phase !== 'end') {
    annotateWaitingFor(game.waitingFor);
    logGameState(game);
    move = await bot.play(game, game.waitingFor);
    console.log('Bot plays:', move);
    game = await request('POST', `${serverUrl}/player/input?id=${playerId}`, move);
  }

  console.log('Game ended!');
  logGameState(game);
  logGameScore(game);
})();

// Add useful extra information
function annotateWaitingFor(waitingFor) {
  // Annotate expected player input type (e.g. inputType '2' means playerInputType 'SELECT_AMOUNT')
  const playerInputType = PlayerInputTypes[waitingFor.inputType];
  if (!playerInputType) {
    throw new Error(`Unsupported player input type ${waitingFor.inputType}! Supported types: ${JSON.stringify(PlayerInputTypes, null, 2)}`);
  }
  waitingFor.playerInputType = playerInputType;
  // Annotate any missing card information (e.g. tags)
  for (const card of (waitingFor.cards || [])) {
    const projectCard = cardFinder.getProjectCardByName(card.name);
    if (!projectCard) continue;
    if (!('tags' in card)) {
      card.tags = projectCard.tags;
    }
  }
  // Recursively annotate nested waitingFor options
  for (const option of (waitingFor.options || [])) {
    annotateWaitingFor(option);
  }
}

function logGameState(game) {
  console.log(`Game state (${game.players.length}p): gen=${game.generation}, temp=${game.temperature}, oxy=${game.oxygenLevel}, oceans=${game.oceans}, phase=${game.phase}`);
}

function logGameScore(game) {
  console.log('Final scores:\n' + game.players.map(p => `  - ${p.name} (${p.color}): ${p.victoryPointsBreakdown.total} points`).join('\n'));
}