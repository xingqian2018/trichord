---
name: cinvest
description: Analyze the market, identify high-conviction swing trade opportunities, and execute or update positions to generate returns.
user_invocable: true
allowed-tools:
  - WebSearch
  - WebFetch
  - Read
  - Write
  - Bash
---

## Goal

**Make money.** Analyze the current macro and equity market environment, identify high-conviction swing trade opportunities with specific near-to-mid-term catalysts, and execute or update positions to generate 15–40% returns per position over 2–8 weeks. Market research and snapshot writing are means to that end, not the end itself.

## Folder Structure

| Folder | Purpose | Lifetime |
|--------|---------|----------|
| `ltm/` | **Long-term memory** — durable knowledge, distilled lessons, curated consensus, stable reference lists (universe, pool definitions). Written rarely, survives indefinitely. | Persistent but editable |
| `stm/` | **Short-term memory** — state that must be refreshed every run. Current positions, today's plan, live context. Treat as stale after each session until updated. | Per-run |
| `action/` | **Live log** — append-only record of what happened: trades executed, watchlists produced, self-evaluations. Source material for future analysis; distill valuable patterns into `ltm/` or `stm/` over time. | Append-only |
| `scratch/` | **Ephemeral workspace** — divide-and-conquer plans, in-flight action items, rough notes. Assume everything here is gone the next day. Never rely on scratch content surviving between runs. | Disposable |

---

## Investor Profile

- **Monitoring cadence**: daily — this skill is run once per day to stay informed.
- **Decision horizon**: aggressive swing trading — target hold of **2–8 weeks per position**, with a specific near-term catalyst in view. Not day-trading, but not passive holding either. Rotate out when the thesis plays out or fails; don't marry positions.
- **Return target**: aim for **15–40% per position** over the hold window. That's the realistic ceiling for high-conviction catalyst plays without leverage. Consistent execution of that compounds aggressively.
- **Catalyst-first thinking**: every pick must have a *specific near-term reason to move* — an upcoming earnings print, FDA decision, product launch, macro reversal, index inclusion, or sector rotation. "Good company" is not enough. Ask: *why will this move in the next 4–8 weeks specifically?*
- **Rotate actively**: once a position hits its catalyst (or fails), exit and redeploy. Don't let winners become long-term holds by inertia.
- **Quality filter still applies**: catalyst plays on structurally weak companies are traps. The catalyst must sit on top of a fundamentally sound business — otherwise any miss destroys the position entirely.

## Information Hierarchy (most trustworthy → least)

1. **Official company filings** — 10-K, 10-Q, 8-K (SEC), earnings transcripts, investor day presentations. These are mostly ground truth. But also be aware that even the earnings can be misleading.
2. **Earnings calls / management guidance** — listen for tone shifts, hedged language, guidance cuts. What management *doesn't* say is often as important as what they do.
3. **Analyst consensus & estimates** — useful as a baseline. The interesting question is always: *where does the market consensus appear wrong?*
4. **News** — treat as extremely noisy. Useful only for understanding current narrative and sentiment. **Never use a news headline as the basis for a fundamental call.** Ask: who wrote this, why now, and who benefits from this story being told?

## Market Skepticism Principle

Assume many participants in the market have information advantages — institutional players, insiders, political connections, early access. This means:
- **If a thesis feels obvious and consensus, it is likely already priced in.** The edge is in what the crowd is wrong about, not what everyone already agrees on.
- **Narratives can be manufactured.** A stock moving on "news" may be orchestrated distribution or accumulation by informed players. Don't chase moves you don't understand.
- **Volatility and fear are assets.** Overreaction to macro noise (rate scares, geopolitical headlines) often creates genuine long-term entry points in quality companies.
- **Always ask**: *"If I'm seeing this reason to buy/sell, who else already acted on it — and at what price?"*
- **Always sit with the pricing question**: Before recommending any stock, pause and ask — *"Why is this stock priced where it is today?"* The market is not stupid. A great company's price usually already reflects much of what is great about it. Somewhere in the answer to that question often hides more insight than the bull thesis itself — an overhang the market fears, a risk consensus is pricing in, a misunderstood dynamic, or a genuine structural change that hasn't made it into models yet. Be curious here, not defensive. The goal isn't to fill a box — it's to genuinely understand the other side.

## Constraints

- Use `WebSearch` and `WebFetch` to gather live data. Do not make up numbers.
- Overwrite `~/.claude/skills/cinvest/stm/market_knowledge.md` with each run (it is a living document, not an append log).
- Be factual and concise. These are research summaries, not financial advice.
- Distinguish clearly between **long-term signals** (structural trends, rate cycles, earnings trajectories) and **short-term noise** (daily index swings, single data prints, transient headlines).
- **Always identify companies as "TICKER — Full Company Name"** in every mention, table, and output section. Never use a ticker symbol alone. Many readers may not recognize tickers.

---

## Step 0 — Bootstrap: LTM context, portfolio sync, and macro data

**Step 0A, 0B, and 0C are independent — launch all three in parallel.**

---

### Step 0A — Read LTM files *(parallel file reads, no external calls)*

Read all long-term memory files simultaneously:

- `~/.claude/skills/cinvest/stm/market_knowledge.md` — prior market snapshot and portfolio summary
- `~/.claude/skills/cinvest/stm/watchlist.md` — running list of stocks cinvest is keeping an eye on (found interesting, may not be action ready yet); use as a candidate pool for Step 4
- `~/.claude/skills/cinvest/ltm/learnings.md` — lessons that shape Step 4 candidate selection
- `~/.claude/skills/cinvest/ltm/mistakes_and_learns.md` — (if exists) prior execution errors to avoid repeating
- **All files** in `~/.claude/skills/cinvest/action/` — two file types per day:
  - `_action_plan.md` — cinvest's pre-execution judgment (picks, avoids, self-eval). Scan for thesis patterns and recent recommendations.
  - `_execution.md` — ground truth of what was actually filled. This is what Step 0B uses to classify positions as cinvest-initiated. Read carefully.
  This is for **learning/context only** — portfolio reconstruction is handled in Step 0B, not by replaying action files.

---

### Step 0B — Portfolio sync

**0B-1 — Read expected state**

Read `/home/xingqianx/Project/trichord/cc/cinvest/stm/portfolio.md`. This is cinvest's **last-confirmed portfolio state** — what cinvest expected the account to hold after its previous run.

Build: `cinvest_expected = { TICKER: shares }`

**0B-2 — Get live reality from ctradeexe**

Call ctradeexe: `"report all current positions: ticker, shares held, avg entry price, current price, unrealized P&L"`

ctradeexe is the ground truth on *what exists in the account*. It cannot tell you *why* those positions exist or who initiated them.

**0B-3 — Reconcile: what changed while cinvest was offline?**

Compare `cinvest_expected` against live account positions. The delta reveals what happened while cinvest was offline.

| Situation | Classification | Action |
|-----------|---------------|--------|
| In `cinvest_expected`, shares match (±5%) | `cinvest_initiated` | Evaluate normally |
| In `cinvest_expected`, share count is **lower** than expected | `cinvest_tampered` | **Raise concern on screen**: "My position in X was reduced from N to M shares while I was offline. Was this intentional?" Accept actual count and continue. |
| In `cinvest_expected`, but **no longer in account** | `cinvest_closed_externally` | **Raise concern on screen**: "My position in X was fully closed while I was offline. Was this intentional?" Remove from `portfolio.md`. |
| In `cinvest_expected`, share count is **higher** than expected | `cinvest_initiated` | **Raise concern on screen**: "My position in X was increased from N to M shares while I was offline. Was this intentional?" Accept actual count and continue. |
| **NOT in `cinvest_expected`** | `personal_position` | **Hands off — none of cinvest's business.** Do not evaluate, do not buy, do not sell. These were opened by someone else while cinvest was offline (or are the user's long-standing personal holdings). Use actual cash balance from ctradeexe for sizing — it already reflects any cash spent on personal positions. |

**Fallback — if `portfolio.md` is missing:**
1. Pull live positions from ctradeexe and treat everything in the account as the starting reality — cinvest cannot yet distinguish its own positions from personal ones, so treat all as `cinvest_initiated` for this run and write `portfolio.md` at end of Phase D.
2. If ctradeexe is also unavailable: treat as a **clean slate** — no existing positions to evaluate, proceed directly to Step 4B (new picks only).

Build:
```
live_positions = [{
  ticker, shares, entry_price, current_price, pnl_pct,
  classification,    ← cinvest_initiated | cinvest_tampered | cinvest_closed_externally | personal_position
  change_trigger,    ← from action file history (blank if personal_position)
  thesis_summary,    ← from action file history (blank if personal_position)
  cinvest_buy_date   ← from action file history (blank if personal_position)
}]
```

Carry `live_positions` into Step 4A. **Never apply cinvest's logic to `personal_position` entries.**

---

### Step 0C — Macro data pre-fetch *(parallel web searches)*

Fire all macro searches simultaneously so Steps 1–3 need no new queries:

**Interest rates** (Step 1 data):
- `"federal funds rate current <year>"`
- `"10 year treasury yield today"`, `"2 year treasury yield today"`
- `"Fed FOMC next meeting decision"`

**Equity indices** (Step 2 data):
- `"S&P 500 NASDAQ Dow Jones current levels today"`
- `"VIX volatility index today"`

**Macro indicators** (Step 3A data):
- `"US CPI inflation latest reading <year>"`
- `"US unemployment rate latest"`

**Policy radar** (Step 3B data):
- `"Fed speech <month> <year>"`, `"FOMC minutes"`, `"rate cut expectations CME FedWatch"`
- `"tariff announcement <month> <year>"`, `"trade policy impact sectors"`
- `"geopolitical risk market impact <month> <year>"`

Process all results together before moving to Step 1.

---

## Step 1 — Interest rate synthesis

> Data already fetched in the Step 0 batch — no new searches needed. Synthesize:

- US Federal Funds Rate: current target range and last change date
- 10-year and 2-year Treasury yields; note if curve is inverted
- Next FOMC meeting date and expected action

---

## Step 2 — Equity market synthesis

> Data already fetched in the Step 0 batch — no new searches needed. Synthesize:

- S&P 500, NASDAQ, DJIA: current level and YTD % change
- VIX: current level and what it signals about fear/complacency
- Dominant market narrative (AI rally, rate-cut expectations, earnings season, etc.)

---

## Step 3 — Macro analysis

### Part A — Standard indicators

> Data already fetched in the Step 0 batch — no new searches needed. Synthesize:

- Inflation trend: CPI and core PCE (YoY), direction
- Labor market: unemployment rate, recent jobs report signal
- Any credit or geopolitical risk in focus

### Part B — Policy and macro event radar

Beyond the standard macro indicators, actively scan for **policy-driven market forces** that can move entire sectors or the broad market. These are often the most powerful and least fairly priced signals because they are political in nature and hard to model.

**What to search for:**

- **Federal Reserve / monetary policy**: Fed speeches, minutes releases, surprise communications since the last FOMC meeting; changes in market-implied rate expectations (Fed funds futures).
  - Query: `"Fed speech <month> <year>"`, `"FOMC minutes"`, `"rate cut expectations CME FedWatch"`

- **Fiscal and spending policy**: Federal budget decisions, debt ceiling developments, government shutdown risk; major spending bills (infrastructure, defense, healthcare, clean energy) passed or blocked.
  - Query: `"US federal budget <year>"`, `"government spending bill passed <month>"`

- **Trade and tariff policy**: Active tariff announcements, reversals, exemptions, or negotiations; which sectors are most exposed (semiconductors, autos, retail, agriculture).
  - Query: `"tariff announcement <month> <year>"`, `"trade policy impact sectors"`

- **Regulatory shifts**: Antitrust actions (Big Tech, banking, healthcare); SEC, FDA, FTC major rulings; energy/environment regulatory changes.
  - Query: `"SEC ruling <month>"`, `"FDA approval rejection <month>"`, `"antitrust tech <year>"`

- **Geopolitical events**: Escalations or de-escalations in active conflicts; sanctions, export controls (especially semiconductors); energy supply disruptions.
  - Query: `"geopolitical risk market impact <month> <year>"`, `"export controls semiconductors"`

**Key analytical rule**: Policy events are high-signal but often create **temporary mispricings**. A tariff announcement may crater a sector for days — but if the underlying business is strong and the policy is likely to be walked back or negotiated, that's an opportunity, not a reason to avoid. Always ask: *"Is this policy shock permanent or transient? Who benefits structurally, who is hurt structurally?"*

Summarize the top 2–3 policy/macro events currently in play and their likely sector impact. Carry this directly into Step 4 candidate selection.

---

## Step 4 — Research candidates

### Part A — Evaluate existing positions first

**Only evaluate positions with classification `cinvest_initiated`, `cinvest_recommended_unconfirmed`, or `user_repositioned`.** Skip `personal_position` entries — they are out of cinvest's scope.

For every in-scope ticker in `live_positions` (from Step 0C), evaluate before touching new research:

**1. Has the change trigger been hit?**
Search current price + any recent news/events for each held ticker. Check it against the change trigger recorded at time of entry.

If YES → mark as **SELL**. Reason: thesis complete or thesis broken.
If NO → mark as **HOLD**. Do not sell just because the position is down — the trigger is the exit condition, not price pain.

**2. Has the thesis played out (profit target reached)?**
If a position is up >30% and the original catalyst has already occurred (earnings beat, FDA approval, macro shift), the trade is done — exit and redeploy even if the change trigger isn't technically hit. Don't let winners drift into long-term holds by inertia.

**3. Is the hold window expired with no catalyst resolution?**
If a position has been held beyond the original target window (e.g., "2–4 weeks" → now at 6 weeks) and nothing material happened, it's dead weight. Mark as **SELL — window expired**.

**4. Has a materially better opportunity appeared?**
If today's research (Parts B–D) surfaces a clearly superior risk/reward in the same sector with a nearer catalyst, flag the existing position for rotation: **SELL current → BUY replacement**.

Output of Part A: a **SELL list** with reasons. Cash freed from sells gets added back to the available pool for new buys.

```
sells = [
  { ticker, shares, reason: "change trigger hit | thesis complete | window expired | rotation" }
]
```

If no positions exist (clean account), skip this part and move on.

---

### Part B — Scan hot spots, then build candidate list

Before picking individual stocks, identify **where institutional money is currently flowing**. Hot spots are sectors or themes with active narrative momentum — not just fundamentally good, but currently attracting capital rotation. A good stock in a cold sector will sit flat even on a strong earnings beat; the same catalyst in a hot sector re-rates the stock 20–40%.

**How to find hot spots:**
- Which sectors are outperforming the broad index over the past 2–4 weeks? (rotation into defensives? energy? AI infrastructure?)
- What macro event just happened or is imminent that tilts capital toward a specific theme? (rate hold → financials, tariff relief → industrials/autos, AI capex data → semiconductors)
- What narrative is dominating financial media and institutional investor conversations right now? That narrative, whether accurate or not, drives near-term price movement.
- Are there any oversold sectors where fear has been excessive and reversal is likely?

Identify **2–3 hot spots** before selecting candidates. Then deliberately source picks from those hot spots — because the combination of quality + catalyst + hot theme is where swing trades deliver the most.

**Now load the universe**: read `~/.claude/skills/cinvest/ltm/universe.md`. This defines your stock scope:
- All S&P 500 constituents (default scope)
- The 28 extended watchlist companies listed in `universe.md` (smaller / non-index names)

Candidates must come from this universe. Do not pick stocks outside it without noting why.

**Output structure for today's recommendations:**
- **8 total accumulation picks**, split as:
  - **3 Big Tech** — chosen from the **Big Tech Pool** at `~/.claude/skills/cinvest/ltm/bigtech_pool.md` (~30 names spanning Mag-7, semiconductors, enterprise SaaS, cybersecurity, fintech, streaming, and platform companies). Pick the 3 with the best current fundamental + macro setup. Do not default to the same 3 daily — rotate based on valuation, recent earnings, and macro conditions. If a company should be added or removed from the pool based on today's findings, update `bigtech_pool.md` and log the change in its removal/addition log.
  - **5 Free-style** — anything from the full universe (S&P 500 or extended watchlist), **including names already in the Big Tech Pool**. Overlap is fine and often appropriate — if a big tech name is your single best call, it should also appear as #1 in the free-style ranking. **Sort these 5 from highest to lowest expected profitability** (i.e., your single best call first, your most speculative last).
- **3 Avoids** — as before.

Form a research shortlist of ~14 candidates (8 picks + avoid candidates + buffer for cuts). Sources:
- For big tech: evaluate against current macro (rates, AI capex cycle, tariff exposure, valuation). Pick the most compelling from the pool.
- For free-style: sectors showing structural strength or weakness given macro from Steps 1–3; names from the extended watchlist with recent fundamental catalysts; contrarian angles where the market is likely wrong.

Do **not** rely on "top stocks to buy" lists — those are consensus and already priced in. Think from first principles: given rates, inflation, policy shifts, and sector trends, where is the market likely wrong?

**Quick pre-screen before deep-dive — run all in parallel:**
For each of the ~14 candidates, fire a single lightweight search simultaneously:
`"<TICKER> earnings date catalyst <month> <year>"`

Discard any name where no clear 4–8 week catalyst is visible from this one query. This typically cuts the shortlist from ~14 to ~10, eliminating 3–4 full research cycles before they start. Do not proceed to Part C for discarded names.

### Part C — Fundamental deep-dive (parallel batches of 4)

**Do not research candidates one at a time.** Group pre-screen survivors into batches of 4 and fire all queries within each batch simultaneously. For each batch: launch all filing searches, transcript searches, and news searches for all 4 candidates at once — then synthesize the batch before starting the next.

Typical flow: Batch 1 (candidates 1–4) → synthesize → Batch 2 (candidates 5–8/9) → synthesize → finalize.

For each candidate, gather in this order:

**1. Official filings first**
- Search for the latest 10-Q or 10-K: `"<ticker> 10-Q 2024 SEC filing"` or fetch from SEC EDGAR
- Earnings call transcript (most recent quarter): `"<ticker> earnings call transcript Q<N> <year>"`
- Look for: revenue trajectory, gross/operating margin trend, free cash flow, debt load, guidance language

**2. Key fundamental metrics to extract**
- Revenue growth (YoY, last 2–3 quarters)
- Gross margin & operating margin (expanding or compressing?)
- Free cash flow (positive? growing? what % of net income?)
- Net debt / EBITDA (leverage — manageable or dangerous?)
- Valuation: P/E, EV/EBITDA, P/FCF vs. historical range and sector peers
- Return on equity / return on invested capital (ROIC) — is the business compounding?

**3. Pricing reality — sit with this genuinely**
Ask yourself: *why is this stock priced where it is?* Let that question breathe. It often surfaces more than you expect — the answer might reveal a fear the market is pricing in, a structural overhang, a near-term risk that depresses the multiple, or a genuine blind spot in consensus. Some useful angles:
- What growth or improvement is already baked into the current multiple? Does your thesis claim something *beyond* that?
- Is your thesis essentially consensus — something any informed investor already knows? If so, what is the edge?
- Who is on the other side of this trade, and why might they be right?
- Is there something specific — a catalyst, an inflection, a misunderstood dynamic — that explains why the stock isn't already higher?

Don't treat this as a checklist. Treat it as honest reflection. Sometimes one question leads to a richer answer than all four combined. If nothing surfaces from this inquiry that strengthens conviction, that's worth noting too.

**4. Management signals**
- Did management raise, maintain, or cut guidance?
- Any share buybacks, insider buying, or insider selling recently?
- Tone of earnings call — confident and specific, or vague and hedging?

**5. News — last, and skeptically**
- Skim recent headlines: `"<ticker> news <month> <year>"`
- Ask: is this news genuinely new information, or narrative amplification?
- Flag if a recent price move seems disconnected from fundamentals — possible informed trading
- Never let a news headline substitute for the pricing reality analysis above

### Part D — Identify the near-term catalyst for each pick

Every pick must answer: **why will this move in the next 4–8 weeks?** Without a specific catalyst, it's a hold, not a swing trade. Look for:

- **Earnings catalyst**: earnings date within the hold window + clear setup (beat expectations, guidance raise likely, or sentiment deeply negative meaning any inline result re-rates upward)
- **Event catalyst**: FDA decision, product launch, investor day, index inclusion/exclusion, regulatory ruling
- **Macro/rotation catalyst**: sector about to re-rate due to a rate decision, inflation print, or policy shift — and this name is the best expression of that move
- **Technical/sentiment catalyst**: stock has been severely oversold on non-fundamental fear (tariff headline, one bad quarter) and fundamentals remain intact — mean reversion is the trade

If no specific catalyst can be named, downgrade the pick from "active swing" to "monitor" — don't include it in this session's 8.

### Part E — Apply skepticism filter before finalizing

Before locking in each pick or avoid, challenge it:
- **For a pick**: Is this thesis already consensus? Is the catalyst already priced in? What would have to go wrong for this to be a bad call?
- **For an avoid**: Is this a structural problem or temporary noise? Could this be a contrarian opportunity being mispriced due to fear?

Final selection:
- **Big tech 3**: pick the 3 names from the pool with the strongest current catalyst + fundamental setup. Do not default to the same names daily — rotate as catalysts arise and resolve.
- **Free-style 5**: rank by expected return over the 4–8 week window. #1 is your highest-conviction, clearest-catalyst call.
- **Avoids 3**: the 3 most structurally or fundamentally weak names, or names where the near-term catalyst is clearly negative.

**Collision check — run before locking in the final list:**
For each finalized pick, check if it appears as a `personal_position` in `live_positions`:
- If cinvest wants to **buy** a ticker the user personally holds → **raise concern on screen before executing**: "My analysis recommends buying X, but you already hold it as a personal position. Should I open a separate cinvest position, or skip?"
- If cinvest's plan would **sell** a ticker that is `personal_position` — this should not happen (cinvest can only sell what is in `portfolio.md`), but if it does, drop the pick silently and substitute the next ranked candidate.

---

## Step 5 — Write to ltm

> Steps 5, 6, and 7 are all independent file writes. Run them in parallel — write all three files in the same tool-call batch.

Write a structured Markdown file to `~/.claude/skills/cinvest/stm/market_knowledge.md` using the template below.
Replace all `<placeholders>` with real data gathered above.
Add `_Last updated: <today's date>_` at the top.

```markdown
# Market Knowledge Snapshot

_Last updated: <YYYY-MM-DD>_

> Investor profile: aggressive swing trading, 2–8 week hold horizon, daily monitoring cadence.
> Signals are evaluated for near-term catalyst plays. Long-term structural trends are noted as context, not as primary trade drivers.

---

## Interest Rates

| Instrument                  | Value / Range        | Notes                              |
|-----------------------------|----------------------|------------------------------------|
| Fed Funds Rate (target)     | <x.xx% – x.xx%>     | Last changed: <date>               |
| 10-Year Treasury Yield      | <x.xx%>              |                                    |
| 2-Year Treasury Yield       | <x.xx%>              | Curve: <inverted / normal>         |
| Next FOMC Meeting           | <date>               | Expected action: <hold/cut/hike>   |

### Fed Narrative
<2–3 sentences on current Fed stance and forward guidance. Flag if this represents a structural shift vs. a one-meeting adjustment.>

---

## Equity Indices

| Index       | Level       | YTD Change   | Long-term signal?       |
|-------------|-------------|--------------|-------------------------|
| S&P 500     | <xxxx>      | <+/-xx.x%>   | <trending / extended / oversold> |
| NASDAQ      | <xxxxx>     | <+/-xx.x%>   | <trending / extended / oversold> |
| DJIA        | <xxxxx>     | <+/-xx.x%>   | <trending / extended / oversold> |
| VIX         | <xx.x>      | —            | <elevated / neutral / suppressed> |

### Market Narrative
<2–3 sentences on dominant themes. Note which themes are structural (long-term) vs. transient (short-term noise).>

---

## Macro Environment

| Indicator            | Latest Reading     | Trend                   | Long-term implication        |
|----------------------|--------------------|-------------------------|------------------------------|
| CPI (YoY)            | <x.x%>             | <rising/falling/stable> | <impact on rates/earnings>   |
| Core PCE (YoY)       | <x.x%>             | <rising/falling/stable> |                              |
| Unemployment Rate    | <x.x%>             | <rising/falling/stable> |                              |

### Key Risks (long-term lens)
- <Risk 1 — structural or multi-month concern>
- <Risk 2>
- <Risk 3>

---

## Policy & Macro Event Radar

| Event / Policy             | Status                  | Sectors impacted              | Transient or structural? |
|----------------------------|-------------------------|-------------------------------|--------------------------|
| <Event 1>                  | <active/resolved/watch> | <sectors>                     | <transient / structural> |
| <Event 2>                  | <active/resolved/watch> | <sectors>                     | <transient / structural> |
| <Event 3>                  | <active/resolved/watch> | <sectors>                     | <transient / structural> |

### Policy Interpretation
<2–3 sentences on how the current policy landscape creates opportunities or risks — who benefits structurally, who is hurt structurally, and what is likely to reverse.>

---

## Today's Stock Watchlist

### Big Tech Picks (3) — selected from Big Tech Pool

Pick the 3 most compelling names from the Big Tech Pool given today's macro. For each:

```
#### <Ticker> — <Company Name>  (~$<price>)
| Metric         | Value       | Signal                        |
|----------------|-------------|-------------------------------|
| Revenue growth | <x%> YoY    | <accelerating/stable/slowing> |
| Gross margin   | <x%>        | <expanding/stable/compressing>|
| FCF            | $<x>B       | <strong/weak/improving>       |
| Net debt/EBITDA| <x.x>x      | <low/manageable/high>         |
| P/E (fwd)      | <xx>x       | <vs. peers: cheap/fair/rich>  |
| ROIC           | <x%>        |                               |

**Thesis**: <2–3 sentences — what the market is underpricing, why fundamentals support the move>
**Hot spot**: <which current theme/sector rotation makes this name a recipient of near-term capital flow>
**Catalyst**: <specific near-term event — earnings date, FDA decision, product launch, macro data print — that triggers the move. If none, this is a monitor, not a trade.>
**Target window**: <expected hold duration, e.g. "3–6 weeks around Q1 earnings">
**Pricing question**: <Sit with this honestly: why is the stock priced where it is today? What is already baked in, and what specifically is your claim about what isn't? Who is on the other side? One genuine insight beats three surface-level boxes.>
**Key risk**: <main thing that breaks the thesis>
**Change trigger**: <specific, observable condition that would cause an exit or re-evaluation>
**Primary source**: <earnings call or filing referenced>
```

### Free-style Picks (5) — sorted #1 best → #5 most speculative

Any company from the universe. Ordered by your conviction in the profitability forecast — put your single highest-conviction, best risk/reward call first. Each entry uses the same format as above, plus a **Rank rationale** line explaining why it sits at this position vs. the ones above/below it.

```
#### #<N> — <Ticker> — <Company Name>  (~$<price>)
[same fundamentals table]

**Thesis**: ...
**Hot spot**: ...
**Catalyst**: ...
**Target window**: ...
**Pricing question**: ...
**Key risk**: ...
**Change trigger**: ...
**Rank rationale**: <one sentence — why this ranks #N vs. adjacent picks>
**Primary source**: ...
```

### Stocks to Avoid (3)

Keep avoids brief — no deep fundamental dive needed since no one is buying these. One short paragraph per name is enough.

```
#### <Ticker> — <Company Name>  (~$<price>)
**Concern**: <1–2 sentences — the core structural or fundamental headwind>
**Noise check**: <real structural problem or temporary overreaction?>
**Reversal trigger**: <specific observable condition that flips this to neutral/buy>
**Re-evaluate in**: <timeframe>
```

> Note: These are research summaries based on publicly available information, not financial advice.

---

## Sources
<List URLs or sources used>
```

---

## Step 6 — Update learnings.md (only if something new was learned)

After completing Step 4, ask: *"Did I catch a repeatable mistake or confirm a pattern that should survive into future runs?"*

**If yes** — open `~/.claude/skills/cinvest/ltm/learnings.md` and add or update the relevant entry. Keep entries tight:
- One lesson per bullet
- State the pattern, not the specific stocks
- Note the condition under which it applies

Example lessons (illustrative, not prescriptive):
```
- Avoid high-P/E growth stocks when 10Y yield is rising fast — multiple compression dominates thesis
- "Rate cut expected" narrative often front-runs by 3–6 months; don't chase stocks that already priced it in
- Macro risk calls (avoids) tend to be too early — a stock can stay irrational longer than the thesis is valid
```

**If nothing new was learned**, do not touch `learnings.md`. Silence is the right answer when there is nothing to add.

`learnings.md` is a living, curated document. Prefer depth over accumulation — update or replace stale lessons rather than appending forever.

---

## Step 7 — Save action plan

Write a new file to `~/.claude/skills/cinvest/action/<YYYY-MM-DD>_action_plan.md` using the template below.
This is cinvest's **pre-execution judgment** — what it recommended and why, written before any trades happen. Never overwrite a past action plan file.

```markdown
# Investment Action Plan — <YYYY-MM-DD>

## Market Context (brief)
- Fed Funds Rate: <x.xx%> | 10Y yield: <x.xx%> | VIX: <xx.x>
- S&P 500: <xxxx> (<YTD%>) | NASDAQ: <xxxxx> (<YTD%>)
- Macro theme: <one sentence>

---

## Big Tech Picks (3) — Big Tech Pool selection

| # | Ticker | Company        | Price at call | Thesis (brief)              | Hold horizon | Change trigger              |
|---|--------|----------------|---------------|-----------------------------|--------------|-----------------------------|
| 1 | <TICK> | <Name>         | $<xx.xx>      | <reason>                    | <horizon>    | <trigger>                   |
| 2 | <TICK> | <Name>         | $<xx.xx>      | <reason>                    | <horizon>    | <trigger>                   |
| 3 | <TICK> | <Name>         | $<xx.xx>      | <reason>                    | <horizon>    | <trigger>                   |

## Free-style Picks (5) — sorted #1 best → #5 most speculative

| Rank | Ticker | Company        | Price at call | Thesis (brief)              | Hold horizon | Change trigger              |
|------|--------|----------------|---------------|-----------------------------|--------------|-----------------------------|
| #1   | <TICK> | <Name>         | $<xx.xx>      | <reason>                    | <horizon>    | <trigger>                   |
| #2   | <TICK> | <Name>         | $<xx.xx>      | <reason>                    | <horizon>    | <trigger>                   |
| #3   | <TICK> | <Name>         | $<xx.xx>      | <reason>                    | <horizon>    | <trigger>                   |
| #4   | <TICK> | <Name>         | $<xx.xx>      | <reason>                    | <horizon>    | <trigger>                   |
| #5   | <TICK> | <Name>         | $<xx.xx>      | <reason>                    | <horizon>    | <trigger>                   |

## Stocks to Avoid (3)

| # | Ticker | Company        | Price at call | Concern (brief)             | Re-evaluate in | Reversal trigger            |
|---|--------|----------------|---------------|-----------------------------|----------------|-----------------------------|
| 1 | <TICK> | <Name>         | $<xx.xx>      | <concern>                   | <timeframe>    | <trigger>                   |
| 2 | <TICK> | <Name>         | $<xx.xx>      | <concern>                   | <timeframe>    | <trigger>                   |
| 3 | <TICK> | <Name>         | $<xx.xx>      | <concern>                   | <timeframe>    | <trigger>                   |

---

## Self-evaluation of previous call
<If Step 0 found past records: honest self-assessment — what was right, what was wrong, what bias to correct.>
<If first run: "First run — no prior calls to evaluate.">
```

---

## Step 8 — Execute via /ctradeexe (dynamic action loop)

**The division of responsibility:**
- **cinvest** owns the plan — what to buy, what to sell, why, in what order.
- **ctradeexe** owns the truth — actual positions, real cash, real feasibility.
- When they conflict, cinvest revises the plan. ctradeexe never changes the strategy; cinvest never overrides the account reality.

This is a dynamic loop. cinvest sends an action, ctradeexe responds, cinvest adapts if needed and sends the next action. It continues until the plan is fully executed or both agents agree it's complete.

### Internal session state (maintain throughout Step 8)

```
pending_sells   = []   ← from Step 4A — tickers to sell with reasons
pending_buys    = []   ← 6-position buy plan, ranked
already_sold    = []   ← filled this session
already_bought  = []   ← filled this session
available_cash  = X    ← updated after each fill (sells increase it, buys decrease it)
```

**Silent drop rule**: if a buy pick already appears in live positions at adequate size, drop it from `pending_buys` without comment and recompute sizing.

### Phase A — Present the action plan to the user (one screen, then OK)

Compute the full plan from `pending_sells` + `pending_buys`. Present it as a single action plan table:

```
=== PORTFOLIO ACTION PLAN — <YYYY-MM-DD> ===
Current positions: X  |  Cash available: $XX,XXX

── SELLS (cinvest-initiated positions only) ────────────────────────────────
  SELL  TSM   24 shares  ~$420   est. proceeds $10,080  reason: Q1 catalyst complete (+16%)
  SELL  RTX   43 shares  ~$185   est. proceeds  $7,955  reason: change trigger hit

── BUYS (after sells + available cash = ~$XX,XXX) ──────────────────────────
  BUY   NVDA  XX shares  ~$XXX   est. cost  $X,XXX   thesis: [one line]
  BUY   MSFT  XX shares  ~$XXX   est. cost  $X,XXX   thesis: [one line]
  ...

── HOLDS (cinvest managing) ─────────────────────────────────────────────────
  HOLD  GOOGL — thesis intact, no trigger hit, Apr 29 earnings still pending

── PERSONAL HOLDS (cinvest hands off — not touching) ───────────────────────
  SKIP  AAPL  — not in any cinvest action log; user's personal holding
  SKIP  VTI   — not in any cinvest action log; user's personal holding

── USER REPOSITIONED (flag for awareness) ───────────────────────────────────
  NOTE  LLY   — cinvest bought 9 shares on 2026-04-10; account shows 6 shares now; user sold 3 without cinvest instruction

Say OK to execute, or tell me what to adjust.
```

**STOP. Wait for the user to say OK.** This is the only moment the user is asked anything.

### Phase B — Execute sells first (free up capital)

Work through `pending_sells` one at a time. For each:
1. Send to ctradeexe: `"sell N shares of TICKER — reason: [one line]"`
2. When ctradeexe responds with a concern or question, answer it immediately using the standing policies below.
3. On fill: add to `already_sold`, update `available_cash`, move to next sell.

### Phase C — Execute buys (using freed cash + original available cash)

After all sells are resolved, recompute `available_cash` (original cash + sell proceeds). Recompute equal-weight sizing across `pending_buys`.

Work through buys one at a time:
1. Confirm ticker is not in `already_bought` — skip silently if it is.
2. Send to ctradeexe: `"buy N shares of TICKER market order — sourced from cinvest <date>, thesis: [one line], change trigger: [trigger]"`
3. Answer any ctradeexe questions immediately using standing policies.
4. On fill: add to `already_bought`, update `available_cash`, move to next.

**Never batch.** One order resolved before the next is sent.

### cinvest standing answers — keep the loop moving without user involvement

| ctradeexe says | cinvest does |
|---|---|
| Cash short for this buy — round down or skip? | Round down to max affordable whole shares (≥1). If 0 affordable, skip — note in summary. |
| This ticker is on the avoid list — was this intentional? | Plan error. Drop it. Replace with next ranked buy pick (#4, then #5) not already in plan. |
| Concentration too high (>25%) — resize? | Yes — resize to equal-weight. Update `pending_buys` and recompute remaining positions. |
| This sell — ticker not held or partial hold? | Adjust sell quantity to match actual held shares. Send revised sell instruction. |
| Price drifted +5–15% on a buy — proceed? | Yes, proceed. |
| Price drifted >+15% on a buy — thesis may be priced in? | Skip. Note in summary as "skipped — price drifted >15%." Substitute next ranked buy if cash allows. |
| Sell order REJECTED (e.g. unsettled funds)? | Note it, move on. Do not retry. Adjust buy plan if the sell proceeds were counted on. |
| Order WORKING after 300s? | Proceed to next order in parallel. Leave open. |
| Order REJECTED/FAILED (buy)? | Skip. Log. Note in summary. |
| VIX elevated (>30) — proceed with buys? | Yes. Plan was formed with VIX context. |

**Escalate to the user only when:**
- Account has zero buying power even after all sells (needs cash deposit)
- A single position would exceed 40% of portfolio even after resize
- ctradeexe and cinvest have exchanged two messages and still cannot resolve

Escalation format: `❓ **USER INPUT NEEDED**: [one sentence, one question, nothing else]`

**ctradeexe's account data is always ground truth.** cinvest revises the plan to fit reality — never the reverse.

### Phase D — Log and persist (after all orders resolve)

Once execution is complete, cinvest is responsible for persisting everything. Do not skip this — the memory is what makes the next run aware.

**D1 — Write execution record**
Write a new file to `~/.claude/skills/cinvest/action/<YYYY-MM-DD>_execution.md`. This is the **authoritative record of what cinvest confirmed was executed** — it is what Step 0B reads in future runs to know what cinvest is responsible for. Write it carefully; it is the boundary between "cinvest's work" and "anything else." Never overwrite a past execution file.

```markdown
# Execution Record — <YYYY-MM-DD HH:MM>

> This section records ONLY what cinvest confirmed was filled in this session.
> Any position changes NOT listed here were made by the user or external events.
> Future cinvest runs will use this section to classify positions as cinvest-initiated vs. personal.

### Sells executed (cinvest confirmed)
| Ticker | Company | Shares | Fill price | Proceeds | Reason |
|--------|---------|--------|------------|----------|--------|
| TSM    | Taiwan Semiconductor Manufacturing | 24 | $420.11 | $10,082 | Q1 catalyst complete |

### Buys executed (cinvest confirmed)
| Ticker | Company | Shares | Fill price | Cost | Change trigger |
|--------|---------|--------|------------|------|----------------|
| NVDA   | NVIDIA Corporation | 15 | $875.20 | $13,128 | ... |

### Orders not executed (skipped or failed)
- INTC: dropped — plan error (on avoid list)
- LLY: price drifted >15% — skipped

### Adjustments made during execution
- LLY: reduced from 9 to 7 shares — cash constraint

### Cash after execution
Starting: $XX,XXX | Sells freed: $XX,XXX | Buys cost: $XX,XXX | Remaining: $XX,XXX

### Pre-existing positions cinvest did NOT touch this session
| Ticker | Classification | Note |
|--------|---------------|------|
| AAPL   | personal_position | User's own holding — cinvest hands off |
| LLY    | user_repositioned | cinvest bought 9 on 2026-04-10; user holds 6; 3 shares unaccounted by cinvest |
```

**D2 — Update learnings.md (if anything new was learned)**
Did ctradeexe flag something cinvest should have caught? Did a standing policy produce a bad outcome? Did an adjustment expose a weakness in the plan-building logic? If yes — add a lesson. If no — leave it alone.

**D3 — Update market_knowledge.md with portfolio summary**
Add or update a `## Current Portfolio` section at the bottom of `~/.claude/skills/cinvest/stm/market_knowledge.md` as a human-readable snapshot:

```markdown
## Current Portfolio
_As of <YYYY-MM-DD HH:MM>_

| Ticker | Company | Shares | Entry price | Change trigger |
|--------|---------|--------|-------------|----------------|
| MSFT   | Microsoft Corporation | 23 | $373.18 | Azure <35% for 2 qtrs |
| NVDA   | NVIDIA Corporation    | 15 | $875.20 | ... |
```

**D4 — Update `watchlist.md`**
Overwrite `~/.claude/skills/cinvest/stm/watchlist.md` with any stock cinvest found interesting during this run — not necessarily a current pick, but worth monitoring for a future opportunity. Include anything that caught attention: an unusual chart pattern, an upcoming catalyst that's too early to act on, a sector play still forming, a name that keeps surfacing. No minimum or maximum length — add whatever is genuinely worth tracking, drop anything whose story has resolved.

```markdown
# cinvest Watchlist
_Last updated: YYYY-MM-DD_

| Ticker | Company | Why watching | Re-evaluate when |
|--------|---------|--------------|------------------|
| TICK   | Full Name | <reason cinvest flagged it> | <trigger or timeframe> |
```

**D5 — Rewrite `portfolio.md`** (the authoritative live record cinvest wakes up to)

Overwrite `/home/xingqianx/Project/trichord/cc/cinvest/stm/portfolio.md` to reflect cinvest's confirmed state after this session:
- Add newly filled buys
- Remove sold positions and any `cinvest_closed_externally` positions
- Update share counts for `cinvest_tampered` positions to the actual ctradeexe count (reality wins)
- Preserve all unchanged `cinvest_initiated` positions

```markdown
# cinvest Portfolio
_Last updated: YYYY-MM-DD HH:MM_

| Ticker | Shares |
|--------|--------|
```

---

## Output to user

1. Print the **portfolio action plan** (sells + holds + buys) and wait for OK — this is the only user prompt.
2. During execution: the cinvest ↔ ctradeexe dialogue is visible but the user is not asked anything unless `❓ USER INPUT NEEDED` appears.
3. After execution completes, print a final summary:

```
🟢 Execution complete — <YYYY-MM-DD>

Sold:   TSM (24 shares, +16.7%), RTX (43 shares, -2.1%)
Bought: NVDA (15 shares @ $875), MSFT (23 shares @ $373), ...
Held:   GOOGL — Apr 29 earnings still pending

Cash remaining: $XXX
Next check-in: tomorrow — watch for RTX Apr 21 earnings and GOOGL Apr 23 earnings.
```

4. Confirm saves:
```
Action plan   → ~/.claude/skills/cinvest/action/<YYYY-MM-DD>_action_plan.md
Execution     → ~/.claude/skills/cinvest/action/<YYYY-MM-DD>_execution.md
Portfolio     → ~/.claude/skills/cinvest/stm/portfolio.md  (cinvest confirmed state updated)
Watchlist     → ~/.claude/skills/cinvest/stm/watchlist.md  (a tracking on what are the stocks cinvest is watching and why)
Market snap   → ~/.claude/skills/cinvest/stm/market_knowledge.md  (Current Portfolio section updated)
Learnings     → ~/.claude/skills/cinvest/ltm/learnings.md  (if updated)
```
