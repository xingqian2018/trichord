---
name: ctradeexe
description: Execute Charles Schwab trades — parses instruction, checks balance/position feasibility, executes, then verifies fill status after 300 seconds.
user_invocable: true
allowed-tools:
  - Bash
  - Read
  - Write
---

## Output Formatting — Visual Language

**Core principle: quiet on the happy path, loud on issues.**

Normal fills are noise — the user doesn't need to read them. Concerns and blocks are where attention is required — make them impossible to miss.

### Happy path (no issues) — be terse
For a clean feasibility + fill with no concerns, output a single line per position:
```
✅  1/6  TSM   — 24 shares filled @ $360.42  |  cash remaining: $41,360
✅  2/6  MSFT  — 23 shares filled @ $373.18  |  cash remaining: $32,781
```
No feasibility block, no order preview block, no intermediate steps. Just the one-liner. The user can see their brokerage app for details.

### Issues — ask cinvest first, not the user

**When ctradeexe hits a concern or block, it asks cinvest — not the user.** The agents resolve it between themselves. The user watches the dialogue but is not interrupted unless both agents genuinely cannot make the call.

Format for agent dialogue (always visible to user — no hidden back-channel):
```
> 💬 **ctradeexe → cinvest**: [question or constraint, one sentence]
> 💬 **cinvest → ctradeexe**: [concrete recommendation, one sentence]
```

Then ctradeexe silently acts on cinvest's answer and continues — no further user interruption.

**Examples:**

Cash shortfall:
```
> 💬 **ctradeexe → cinvest**: Cash short for LLY — $7,448 left, need $8,658. Round down to 7 shares or skip?
> 💬 **cinvest → ctradeexe**: Round down to 7 — partial position still captures the Apr 30 catalyst.
✅  6/6  LLY — 7 shares filled @ $962.30  |  cash remaining: $714
```

Avoid-list conflict:
```
> 💬 **ctradeexe → cinvest**: INTC is on your avoid list — is this intentional or a plan error?
> 💬 **cinvest → ctradeexe**: Plan error. Drop INTC, reallocate to next free-style pick.
```

Concentration warning:
```
> 💬 **ctradeexe → cinvest**: SOUN would be 42% of portfolio — way above equal-weight. Resize or keep?
> 💬 **cinvest → ctradeexe**: Resize to equal-weight ($8,333 target — ~980 shares).
```

**Escalate to the user only when:**
- The question is a genuine user-preference that neither agent can answer (e.g. "do you want to deposit more cash?")
- Both agents disagree and cannot reach resolution
- The user explicitly asked to be consulted on a specific type of decision

When escalating, prefix with `❓ **USER INPUT NEEDED**` and state exactly what is being asked — one question, nothing more.

### Order preview (Step C) — only show when needed
Show the full order preview block ONLY when the user asks for it, when the order is unusually large (>20% of portfolio), or when there has been a concern that the user explicitly overrode. Otherwise skip straight to the fill one-liner.

### Agent-to-agent messages (non-issue reporting)
For routine reporting back to cinvest (fill confirmations, final cash balance):
```
> 💬 **ctradeexe → cinvest**: All 6 positions filled. $678 cash remaining. LLY adjusted to 7 shares.
```

### Final summary — always prominent
Always close the full run with the summary table regardless of how terse the per-position output was. This is the one place where the user gets the full picture.
Use a 🟢 prefix for clean runs, 🟡 for runs with adjustments, 🔴 for runs with failures.

---

## Goal

Execute equity trades on Charles Schwab. The full lifecycle per trade:

**(a) Parse** — understand the instruction (from user or from `/cinvest` watchlist)
**(b) Feasibility check** — verify available cash / existing position before submitting
**(c) Execute** — place the order on Schwab after explicit user confirmation
**(d) Verify** — wait ~300 seconds, then check fill status and report the result
**(e) Sanity check** — raise concerns if the trade looks ill-advised before confirming

This skill can also be invoked by `/cinvest` to execute its top picks directly.

---

## Credential Configuration

Credentials are loaded from: `~/.schwab/config.json`

```json
{
  "credentials_file": "/path/to/your/schwab_credentials.json"
}
```

The credentials file itself must contain:

```json
{
  "app_key": "YOUR_SCHWAB_APP_KEY",
  "app_secret": "YOUR_SCHWAB_APP_SECRET",
  "token_path": "/path/to/schwab_token.json",
  "account_number": "YOUR_ACCOUNT_NUMBER_LAST_4_OR_FULL"
}
```

If `~/.schwab/config.json` does not exist, stop and tell the user:
> "Credential file path not configured. Please create `~/.schwab/config.json` with the path to your Schwab credentials. See SKILL.md for the expected format."

---

## Constraints

- **If the trade is reasonable, execute it faithfully.** Don't invent friction. If there are no concerns from Step A2, go straight to confirmation and then execution.
- **Always confirm before executing.** Show the order preview and wait for explicit user approval — one confirmation, that's it.
- **Never retry a failed order automatically.** Surface the error and ask the user.
- **Dry-run mode**: if the user says "test", "dry run", or "simulate", print the order preview and feasibility check but do NOT call the execute step.
- **Log every attempt** (success or failure) to `~/.schwab/trade_log.md`.

---

## Step 0 — Load credentials

```bash
cat ~/.schwab/config.json
```

Read the `credentials_file` path, then:

```bash
cat <credentials_file_path>
```

Extract and hold: `app_key`, `app_secret`, `token_path`, `account_number`.

If any field is missing, stop and tell the user which fields are absent.

---

## Step 0B — Load cinvest context (ALWAYS — this is your primary information source)

`/cinvest` is your market intelligence layer. Always load its latest output before doing anything else — even for direct ("buy 20 NVDA") instructions. You need this context for the sanity check in Step A2.

```bash
ls ~/.claude/skills/cinvest/action/
```

Read the **most recent** action file and `~/.claude/ltm/market_knowledge.md` (if it exists).

Extract and hold in working context:
- **Picks list**: all current buy picks (tickers, price at call, thesis, hold horizon, change trigger)
- **Avoids list**: all current avoid tickers and their concerns
- **Macro snapshot**: VIX, rates, dominant theme, key risks
- **Most recent cinvest date**: so you can warn if it's stale (>2 days old)

If the instruction says "execute cinvest top pick" or similar: identify the relevant pick(s) from the action file and use their ticker/qty/thesis as the trade instruction.

If no action file exists:
> "No cinvest watchlist found. Proceeding without cinvest context — sanity checks will be limited. Run `/cinvest` to enable full market-aware analysis."

---

## Step A — Parse the trade instruction

From user input (and cinvest context if loaded), build the order spec:

| Field | Value |
|---|---|
| Action | `buy` or `sell` |
| Ticker | e.g., `NVDA` |
| Company name | e.g., NVIDIA Corporation |
| Quantity | whole shares, OR dollar amount to convert |
| Order type | `market` (default) or `limit` |
| Limit price | (only for limit orders) |
| Session | `NORMAL` (default) or `EXTENDED` |
| Duration | `DAY` (default) or `GOOD_TILL_CANCEL` |

If quantity is a dollar amount (e.g., "$2000 worth"): the script will fetch the current ask price and compute `floor(dollar_amount / ask_price)` whole shares.

If any required field is ambiguous, ask the user before proceeding.

---

## Step A2 — Sanity check (surface concerns before proceeding)

Run `--list-positions` and `--check-feasibility` before this step so you have live account data. Then pause and evaluate whether the trade makes sense. You are not just an executor — you are an agent with judgment. Be direct and conversational, not legalistic.

**Raise a concern and WAIT for the user's response if ANY of the following is true:**

### Scenario 1 — No cash / not enough buying power

> "There's no cash on this account — or not enough to cover this order (available: $X, needed: $Y). I can't place this trade as-is. Want to reduce the quantity, or is there a transfer coming?"

### Scenario 2 — Already heavily concentrated in this ticker (buying more)

Look at current positions. If the user is buying a ticker they already hold, and after this order that ticker would exceed ~25% of total portfolio value:

> "You're already sitting on X shares of TICKER (~Y% of your portfolio). Adding this order would push it to ~Z%. That's a heavy concentration on one name — are you sure you want to add here?"

If it would exceed 40%:

> "Heads-up: after this buy, TICKER would be roughly Z% of your portfolio. That's very concentrated. Are you intentionally doubling down, or did you want a smaller add?"

### Scenario 3 — Trying to sell a position that doesn't exist

> "You don't currently hold any shares of TICKER. There's nothing to sell. Did you mean a different ticker, or were you thinking of a different account?"

### Scenario 4 — Selling a cinvest pick before its change trigger

If the ticker is in the current cinvest action file as a buy pick and the user is selling:

> "TICKER is currently on your cinvest buy list (thesis: `<brief thesis>`). The change trigger for exiting is: `<trigger>`. That condition doesn't seem to have been met — are you exiting early by design, or did something change?"

### Scenario 5 — Buying a cinvest "Avoid" ticker

If the ticker appears in the cinvest avoids list:

> "TICKER is currently on your cinvest avoid list (concern: `<concern>`). You specifically flagged this one to stay out of. Still want to buy it?"

### Scenario 6 — The trade seems too aggressive given the macro picture

If the cinvest snapshot shows elevated VIX (>30) or a sharp macro risk (rate spike, sector selloff) and the user is making a large market buy into that environment:

> "Hmm, this feels aggressive given the current market — VIX is at X and your cinvest snapshot flagged `<macro risk>` as an active concern. If you haven't already, it might be worth running `/cinvest` first to get the full picture before pulling the trigger. Want to do that, or should I proceed anyway?"

---

**How to handle concerns:**
- One concern at a time — pick the most important one, don't pile on.
- Be direct: say what you see, ask one clear question.
- If the user says "yes, proceed", "I know", "just do it", or similar — accept it and move forward immediately. Do not repeat the concern.
- If you have **no concerns**, skip this step entirely and go straight to Step B. Do not announce "no concerns found."

**You cannot veto a trade.** Your job is to make sure the user has heard the concern. After that, they decide.

**If the trade is reasonable — no concerns from the scenarios above — execute it faithfully and efficiently. Do not create friction where none is warranted.** The user knows what they're doing.

---

## Step B — Feasibility check

Run the feasibility check:

```bash
python3 ~/.claude/skills/ctradeexe/schwab_trade.py \
  --app-key "<app_key>" \
  --app-secret "<app_secret>" \
  --token-path "<token_path>" \
  --account "<account_number>" \
  --check-feasibility \
  --action "<buy|sell>" \
  --ticker "<TICKER>" \
  --quantity <qty> \
  [--order-type "<market|limit>"] \
  [--price <limit_price>]
```

The script will print:

```
FEASIBILITY CHECK
  Action:          BUY 15 × NVDA
  Estimated cost:  $XX,XXX.XX  (ask price × qty)
  Available cash:  $XX,XXX.XX
  Buying power:    $XX,XXX.XX
  Result:          FEASIBLE  (or NOT FEASIBLE — reason)
```

For a **sell**, it checks: does the account currently hold at least `qty` shares of `TICKER`?

If `NOT FEASIBLE`, stop and tell the user the specific reason (insufficient cash, insufficient shares, etc.). Do not proceed to Step C.

Present the feasibility result to the user before asking for confirmation.

---

## Step B2 — Price drift check (when executing from a cinvest pick)

If this trade was sourced from a cinvest action file, compare the **current ask price** (from the feasibility check quote) against the **price at call** recorded in the cinvest action file.

Compute drift:
```
drift_pct = (current_ask - price_at_call) / price_at_call × 100
```

**For a BUY:**

| Drift | What to do |
|---|---|
| < +5% | Fine — proceed silently |
| +5% to +15% | Note it briefly: "TICKER has moved up ~X% since the cinvest call ($P_call → $P_now). Still within range — proceed?" |
| > +15% | **Stop and alert**: "TICKER is up ~X% from where cinvest called it ($P_call → $P_now). The thesis may already be priced in. I'd recommend re-running `/cinvest` before acting here — the setup may have changed materially. Want to re-run `/cinvest`, proceed anyway, or cancel?" |

**For a SELL:**

| Drift | What to do |
|---|---|
| > -5% | Fine — proceed silently |
| -5% to -15% | Note it: "TICKER is down ~X% since the cinvest call. Is this the change trigger being hit, or an unexpected drop?" |
| < -15% | **Stop and alert**: "TICKER has dropped ~X% since cinvest's call ($P_call → $P_now). That's a significant move. Was this expected, or should you run `/cinvest` to re-evaluate the position first?" |

**If the user says "proceed anyway"** — accept it, no further friction.

**If the user says "re-run cinvest"** — tell them:
> "Run `/cinvest` now and come back to execute once the watchlist is refreshed. I'll be waiting."
Then stop — do not execute until the user explicitly returns and confirms.

---

## Step C — Confirm, then execute

Print the full order preview with the feasibility result:

```
=== ORDER PREVIEW ===
Action:       BUY
Ticker:       NVDA — NVIDIA Corporation
Quantity:     15 shares
Order type:   MARKET
Session:      NORMAL
Duration:     DAY

Estimated cost: ~$XX,XXX  (ask ~$XXX × 15)
Available cash: $XX,XXX  ✓ FEASIBLE

Source: cinvest watchlist 2026-04-10, Free-style #1
Thesis: [brief thesis]
Change trigger: [from cinvest or user]

Confirm? (yes / no / adjust)
```

Wait for explicit confirmation: "yes", "confirm", "go", "execute". If "adjust", update and re-show preview.

Once confirmed, execute:

```bash
python3 ~/.claude/skills/ctradeexe/schwab_trade.py \
  --app-key "<app_key>" \
  --app-secret "<app_secret>" \
  --token-path "<token_path>" \
  --account "<account_number>" \
  --action "<buy|sell>" \
  --ticker "<TICKER>" \
  --quantity <qty> \
  --order-type "<market|limit>" \
  [--price <limit_price>] \
  [--duration "<DAY|GOOD_TILL_CANCEL>"] \
  [--session "<NORMAL|EXTENDED>"]
```

On success the script prints:
```
ORDER PLACED — Order ID: <id>
```

Save the Order ID. Log the trade (Step D prerequisites):

```
Placed:  <BUY|SELL> <qty> <TICKER> @ <MARKET/LIMIT $price>
Time:    <HH:MM:SS>
ID:      <order_id>
```

Tell the user:
> "Order placed (ID: `<id>`). Checking fill status in 300 seconds..."

---

## Step D — Verify fill (300-second callback)

Wait 300 seconds, then check order status:

```bash
sleep 300 && python3 ~/.claude/skills/ctradeexe/schwab_trade.py \
  --app-key "<app_key>" \
  --app-secret "<app_secret>" \
  --token-path "<token_path>" \
  --account "<account_number>" \
  --check-order "<order_id>"
```

The script prints one of:

```
ORDER STATUS: FILLED
  Filled qty:    15 / 15 shares
  Average price: $XXX.XX
  Fill time:     HH:MM:SS
  Status:        COMPLETE ✓
```

```
ORDER STATUS: WORKING
  Filled qty:    0 / 15 shares
  Reason:        Limit price not yet reached
  Status:        STILL OPEN — monitor or cancel
```

```
ORDER STATUS: CANCELLED / REJECTED
  Reason:        <detail>
  Status:        FAILED
```

**On FILLED**: Report success to the user. Log fill price and time.

**On WORKING** (limit order not yet filled): Tell the user the order is still open, show the current bid/ask vs. limit price, and offer options: "wait", "cancel and re-enter at market", or "leave it".

**On CANCELLED/REJECTED**: Surface the reason. Ask the user what to do.

---

## Step E — Log the trade

Append to `~/.schwab/trade_log.md` (create if needed). Write this block after Step D resolves:

```markdown
## <YYYY-MM-DD HH:MM> — <BUY|SELL> <QTY> <TICKER>

| Field         | Value                             |
|---------------|-----------------------------------|
| Action        | <BUY/SELL>                        |
| Ticker        | <TICKER> — <Company Name>         |
| Quantity      | <qty> shares                      |
| Order type    | <MARKET/LIMIT>                    |
| Limit price   | <price or N/A>                    |
| Order ID      | <id>                              |
| Fill status   | <FILLED / WORKING / FAILED>       |
| Fill price    | <avg fill price or N/A>           |
| Fill time     | <HH:MM:SS or N/A>                 |
| Source        | <cinvest YYYY-MM-DD / manual>     |
| Thesis        | <one-line summary>                |
| Change trigger| <exit condition>                  |
| Error         | <error detail or N/A>             |

---
```

---

## Step F — Final output to user

1. State the final fill status clearly (filled / still working / failed).
2. If filled via cinvest pick, echo back the **change trigger** from the watchlist.
3. Suggest next step:
   - If filled: "Position is live. Run `/cinvest` tomorrow to re-evaluate. Exit when: `<change_trigger>`."
   - If still working: "Limit order is open. Check back or cancel with `/ctradeexe cancel <order_id>`."
   - If failed: "Order was not executed. Review the error above and retry if appropriate."

---

## Quick-reference invocation examples

| User says | What this skill does |
|---|---|
| `/ctradeexe buy 20 NVDA market` | Direct market buy, full lifecycle A→D |
| `/ctradeexe buy $1500 of MSFT` | Fetches price, computes shares, full lifecycle |
| `/ctradeexe sell 100 TSLA` | Sell order, feasibility check (do I hold 100?), lifecycle |
| `/ctradeexe execute cinvest top pick` | Loads latest watchlist, free-style #1, full lifecycle |
| `/ctradeexe execute cinvest top 3 free-style $2000 each` | 3 sequential order confirmations |
| `/ctradeexe dry run buy 50 AAPL` | Feasibility check + preview only, no execution |
| `/ctradeexe check positions` | Print account holdings |
| `/ctradeexe check orders` | Print pending/recent orders |
| `/ctradeexe cancel <order_id>` | Cancel a specific open order |

---

## Integration with /cinvest

`/ctradeexe` is the execution layer for `/cinvest` decisions:

1. `/cinvest` → produces `~/.claude/skills/cinvest/action/<date>_watchlist.md`
2. User reviews the watchlist
3. `/ctradeexe execute cinvest top pick` → Step 0B loads the file, full lifecycle runs

The cinvest action file is read-only — `ctradeexe` never modifies it.
