# Splendor Rules Spec

Authoritative rules specification for the Splendor-playing agent. This is the
source of truth the game engine must implement. Card and noble data live in
separate CSVs alongside this file:

- `env/splendor_cards.csv` - all 90 development cards
(columns: `Level, Color, PV, Black, Blue, Green, Red, White`)
- `env/splendor_nobles.csv` - all 10 noble tiles
(columns: `Name, PV, Black, Blue, Green, Red, White`)

Both CSVs use the same color column order: Black, Blue, Green, Red, White.
The engine should load them once at startup and validate the checks in
section 11 below.

## 1. Colors and Notation

Five gem colors plus gold (wild):

- White = Diamond
- Blue  = Sapphire
- Green = Emerald
- Red   = Ruby
- Black = Onyx
- Gold  = Wild (jokers, only obtained via Reserve)

Any "gem color" reference in these rules excludes gold unless stated
otherwise. Gold is always tracked and limited separately.

## 2. Components

- 40 gem tokens total of the five gem colors: 7 each of White, Blue, Green,
Red, Black
- 5 Gold (wild) tokens
- 90 development cards: 40 Level 1, 30 Level 2, 20 Level 3
(8 / 6 / 4 per bonus color per level)
- 10 Noble tiles, each worth 3 prestige points

## 3. Setup

1. Shuffle each of the three decks separately; place face-down in a column
  in order Level 1 (bottom), Level 2, Level 3 (top).
2. Reveal 4 cards from each deck face-up beside its deck, forming a
  3-row by 4-column grid of visible cards.
3. Shuffle the 10 noble tiles; reveal `num_players + 1` of them
  (3 / 4 / 5 for 2 / 3 / 4 players). Return the rest to the box for the
   remainder of the game.
4. Place gem tokens in piles by color. Token supply per color by player
  count:

  | Players | White | Blue | Green | Red | Black | Gold |
  | ------- | ----- | ---- | ----- | --- | ----- | ---- |
  | 2       | 4     | 4    | 4     | 4   | 4     | 5    |
  | 3       | 5     | 5    | 5     | 5   | 5     | 5    |
  | 4       | 7     | 7    | 7     | 7   | 7     | 5    |

5. Youngest player goes first. In the engine, the starting seat is chosen
  uniformly at random from the active seats.

## 4. Gem Tokens vs Card Bonuses

- **Tokens** (including gold) are spent and returned to the supply to pay
costs.
- Each owned development card grants a **permanent gem bonus** of 1 in its
bonus color (shown on the card). Bonuses act as a one-for-one discount
on costs and are never spent.
- Purchasing power in color c = tokens of color c + number of owned cards
with bonus c. Gold does not count toward any specific color's purchasing
power but may substitute during payment (section 5.D).

## 5. Turn Structure

On your turn you must perform exactly one of the four actions A-D below.
After the action you may be required to discard tokens (section 6) and
then check for noble visits (section 7). Then play passes to the next
active seat in clockwise order.

### A. Take 3 different gem tokens

- Take exactly 1 token each from 3 different non-gold piles.
- If fewer than 3 non-empty non-gold piles exist, take 1 token from each
available non-empty non-gold pile (0, 1, or 2 tokens total).
- You may not take gold with this action.

### B. Take 2 tokens of the same non-gold color

- Legal only if the target pile has at least 4 tokens remaining before
you take.
- You may not take gold with this action.

### C. Reserve a development card

- Choose one of:
  1. Any face-up card from the grid, OR
  2. The top card of any of the three decks (a blind reserve). A blind
    reserve is placed face-down in your personal reserve and is hidden
     from opponents (other than its tier).
- Additionally, if there is at least 1 gold token in the supply, take
exactly 1 gold token.
- You may hold at most 3 reserved cards at any time. Reserving while you
already hold 3 is illegal.
- After a face-up grid card is reserved, immediately refill the vacated
grid slot by drawing the top card of the corresponding deck, if the
deck is non-empty. If the deck is empty, the slot remains empty until
the game ends.

### D. Purchase a development card

- Target a card that is either face-up in the grid or in your own
reserve.
- Compute the payment required: for each color c, payment(c) =
max(0, cost(c) - bonus_count(c)), where bonus_count(c) is the number
of cards you own with bonus color c.
- Pay by returning tokens to the supply. You must pay payment(c) in
color c, but any shortfall in a color may be covered by gold tokens,
one gold token per missing token of any single color.
- The engine must resolve payment deterministically using the minimum
number of gold tokens necessary. Given the required payment vector
p = (p_W, p_B, p_G, p_R, p_K) and your holdings of colored tokens
t = (t_W, t_B, t_G, t_R, t_K) and gold tokens t_gold, the payment is
legal iff sum over c of max(0, p_c - t_c) <= t_gold. Spend
min(p_c, t_c) in each color and max(0, p_c - t_c) in gold.
- All spent tokens (colored and gold) return to the supply.
- Place the purchased card face-up in your tableau; its bonus is
immediate and permanent and applies to all future costs.
- If the purchased card was from the grid, immediately refill that slot
from the top of the corresponding deck, if the deck is non-empty.

## 6. End-of-Turn: Token Limit

If after your action you hold more than 10 tokens total (colored + gold),
you must return tokens to the supply until you have exactly 10 total.
You choose which colors (including gold) to return. This is a mandatory
sub-phase, not an optional one.

## 7. End-of-Turn: Nobles

After your action and any required discard, check each still-available
noble tile. A noble visits you iff for every color c, your
bonus_count(c) is at least that noble's requirement in color c (tokens
and gold do not count). Rules:

- If you qualify for multiple nobles in the same turn, you take exactly
one of them (engine: either prompt an action in a `noble_choice` phase
or break ties deterministically, see section 8 on phases). The others
remain available to you on future turns if you still qualify.
- A noble tile visits at most one player in the entire game; once
claimed, it is removed from availability.
- Claiming a noble grants its PV (always 3) immediately.

## 8. Turn Phases (Engine-Level)

The engine uses a `phase` field to keep the action space small:

- `phase = 0` (main): the acting player picks one of the four main
actions A-D.
- `phase = 1` (discard): same player discards one token at a time
(5 gem + 1 gold = 6 possible discard actions) until total holdings
are exactly 10. Entered automatically after a `take` or `reserve`
action if holdings exceed 10.
- `phase = 2` (noble choice): same player chooses one of the
qualifying nobles. Entered only when 2 or more nobles qualify at
once. Skipped entirely when exactly 1 qualifies (automatic claim)
or when 0 qualify.

The acting seat does not change until `phase` returns to 0 and the
current player's turn is fully resolved.

## 9. Game End

- When any player's prestige total reaches or exceeds 15 at the end of
their turn, mark "last round triggered" with that seat as the final
actor. Continue play until every remaining active seat in the current
round has taken one more turn, so all players have had the same number
of turns.
- The player with the most prestige points wins.
- Tiebreak: among tied players, the one with the fewest owned
development cards wins.
- Further tie: the game is a shared win (engine should record all tied
seats as winners for reward purposes).

## 10. Edge Cases the Engine Must Handle

- Deck empty: the corresponding "reserve blind" action and any grid
refill are no-ops. A grid slot left empty after a refill failure is
flagged non-selectable until end of game.
- Gold pile empty: a successful reserve does NOT grant a gold token in
that case; the reserve itself still succeeds.
- Reserve while holding 3 reserved cards: illegal, must be masked out.
- "Take 3 different" with only 1 or 2 non-empty non-gold piles: take
what is available (may be 0, 1, or 2 tokens).
- "Take 2 same" against a pile with <= 3 tokens: illegal, masked.
- Purchase payment with gold: see section 5.D for deterministic
resolution; the agent does not choose how to apportion gold.
- 15-point trigger and end-of-round: the final round continues through
all seats including the triggering seat's prior seats in the rotation
so that each seat has played the same number of turns. Multiple seats
may cross 15 in the same final round.

## 11. Data Validation Checks

On startup the engine must load the two CSVs and assert:

- Exactly 90 cards with per-level counts of 40 / 30 / 20.
- Exactly 8 / 6 / 4 cards per bonus color in Level 1 / 2 / 3.
- All card costs non-negative and fit in int8; all card PVs in
{0, 1, 2, 3, 4, 5}.
- Exactly 10 nobles, all with PV 3.
- Every noble requirement is either a 4+4 pair (sum 8, two nonzero
colors) or a 3+3+3 triple (sum 9, three nonzero colors).
- No two nobles share the same color-requirement signature.
- Token supply constants for 2 / 3 / 4 players are (4, 5, 7) per gem
color and 5 gold in all cases.

Any failure of these checks is a fatal startup error; training cannot
proceed until the data is fixed.

## 12. Cross-Check Sources

Before training begins, the implementer must spot-check the following
against the physical rulebook or a second authoritative source and
record the outcome in `runs/journal.md`:

- All 10 noble entries in `splendor_nobles.csv`.
- At least 5 randomly sampled cards from `splendor_cards.csv`, including
at least one from each level.
- The token supply table in section 3.
- The end-of-round trigger at 15 points and the tiebreaker by fewest
cards.

Primary sources:

- Space Cowboys (publisher): [https://www.spacecowboys-games.com/game/splendor/](https://www.spacecowboys-games.com/game/splendor/)
- BoardGameGeek rulebook PDFs:
[https://boardgamegeek.com/filepage/280430/splendor-refresh-rulebook-en](https://boardgamegeek.com/filepage/280430/splendor-refresh-rulebook-en)
- Card CSV source: [https://github.com/bouk/splendimax/blob/master/Splendor%20Cards.csv](https://github.com/bouk/splendimax/blob/master/Splendor%20Cards.csv)

