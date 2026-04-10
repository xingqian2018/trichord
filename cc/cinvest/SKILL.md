---
name: cinvest
description: Study the current stock and interest rate market, then persist key findings and analysis to ~/.claude/ltm/market_knowledge.md for future reference.
user_invocable: true
---

## Goal

Research the current macro and equity market environment — interest rates, major indices, sector trends, and notable events — then write a structured snapshot to `~/.claude/ltm/market_knowledge.md`.

## Investor Profile

- **Monitoring cadence**: daily — this skill is run once per day to stay informed.
- **Decision horizon**: aggressive swing trading — target hold of **2–8 weeks per position**, with a specific near-term catalyst in view. Not day-trading, but not passive holding either. Rotate out when the thesis plays out or fails; don't marry positions.
- **Return target**: aim for **15–40% per position** over the hold window. That's the realistic ceiling for high-conviction catalyst plays without leverage. Consistent execution of that compounds aggressively.
- **Catalyst-first thinking**: every pick must have a *specific near-term reason to move* — an upcoming earnings print, FDA decision, product launch, macro reversal, index inclusion, or sector rotation. "Good company" is not enough. Ask: *why will this move in the next 4–8 weeks specifically?*
- **Rotate actively**: once a position hits its catalyst (or fails), exit and redeploy. Don't let winners become long-term holds by inertia.
- **Quality filter still applies**: catalyst plays on structurally weak companies are traps. The catalyst must sit on top of a fundamentally sound business — otherwise any miss destroys the position entirely.

## Information Hierarchy (most trustworthy → least)

1. **Official company filings** — 10-K, 10-Q, 8-K (SEC), earnings transcripts, investor day presentations. These are the ground truth. Always prioritize.
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
- Overwrite `~/.claude/ltm/market_knowledge.md` with each run (it is a living document, not an append log).
- Be factual and concise. These are research summaries, not financial advice.
- Distinguish clearly between **long-term signals** (structural trends, rate cycles, earnings trajectories) and **short-term noise** (daily index swings, single data prints, transient headlines).
- **Always identify companies as "TICKER — Full Company Name"** in every mention, table, and output section. Never use a ticker symbol alone. Many readers may not recognize tickers.

---

## Step 0 — Load distilled learnings and spot-check recent call

This step has two parts. Keep it fast — the goal is to *apply* knowledge, not re-litigate history.

### Part A — Read learnings.md (always)

Read `~/.claude/skills/cinvest/learnings.md` if it exists.
This file contains distilled, reusable lessons from past runs — biases to avoid, patterns that worked, conditions where judgment was off. Load these into your working context. They directly shape Step 4.

If the file does not exist yet, skip Part A.

### Part B — Spot-check the most recent action file (lightweight)

```
ls ~/.claude/skills/cinvest/action/
```

If action files exist, read only the **most recent** one. For each stock listed:
- Search its current price (one query per stock is enough).
- Note whether the call aged well or poorly.

**Do not re-evaluate every past call.** Only ask: *"Is there a new pattern here worth distilling into learnings.md?"* The bar is high — a one-off miss is noise, a repeated bias is signal.

If nothing new is learnable, move on immediately.

If the file does not exist, note "First run — no history." and move on.

---

## Step 1 — Gather interest rate data

Search for the current state of interest rates:

- US Federal Funds Rate (current target range, last change date)
- US 10-year Treasury yield
- US 2-year Treasury yield (note if the yield curve is inverted)
- Any recent or upcoming Fed meeting decisions / forward guidance

Suggested queries:
- `"federal funds rate current 2025"`
- `"10 year treasury yield today"`
- `"Fed FOMC next meeting decision"`

---

## Step 2 — Gather equity market data

Search for the current state of major stock indices and sentiment:

- S&P 500 (level, YTD % change)
- NASDAQ Composite (level, YTD % change)
- Dow Jones Industrial Average (level, YTD % change)
- VIX (volatility index — fear gauge)
- Any dominant market narrative (e.g., AI rally, rate-cut expectations, earnings season)

Suggested queries:
- `"S&P 500 NASDAQ Dow Jones current levels today"`
- `"VIX volatility index today"`
- `"stock market outlook 2025 key themes"`

---

## Step 3 — Identify macro themes and risks

Search for 2–3 major macro forces currently driving markets:

- Inflation trend (CPI / PCE latest readings)
- Labor market (unemployment rate, recent jobs report)
- Any geopolitical or credit risk events in focus

Suggested queries:
- `"US CPI inflation latest reading 2025"`
- `"US unemployment rate latest"`
- `"macro risks equity market 2025"`

---

## Step 3B — Policy and macro event radar

Beyond the standard macro indicators, actively scan for **policy-driven market forces** that can move entire sectors or the broad market. These are often the most powerful and least fairly priced signals because they are political in nature and hard to model.

**What to search for:**

1. **Federal Reserve / monetary policy**
   - Any Fed speeches, minutes releases, or surprise communications since the last FOMC meeting
   - Changes in market-implied rate expectations (Fed funds futures)
   - Query: `"Fed speech <month> <year>"`, `"FOMC minutes"`, `"rate cut expectations CME FedWatch"`

2. **Fiscal and spending policy**
   - Federal budget decisions, debt ceiling developments, government shutdown risk
   - Any major spending bills (infrastructure, defense, healthcare, clean energy) passed or blocked
   - Query: `"US federal budget <year>"`, `"government spending bill passed <month>"`

3. **Trade and tariff policy**
   - Active tariff announcements, reversals, exemptions, or negotiations
   - Which sectors/companies are most exposed (semiconductors, autos, retail, agriculture)
   - Query: `"tariff announcement <month> <year>"`, `"trade policy impact sectors"`

4. **Regulatory shifts**
   - Antitrust actions (especially Big Tech, banking, healthcare)
   - SEC, FDA, FTC major rulings or new rules
   - Energy/environment regulatory changes (EPA, IRA implementation)
   - Query: `"SEC ruling <month>"`, `"FDA approval rejection <month>"`, `"antitrust tech <year>"`

5. **Geopolitical events with market impact**
   - Escalations or de-escalations in active conflicts
   - Sanctions, export controls (especially semiconductors)
   - Energy supply disruptions
   - Query: `"geopolitical risk market impact <month> <year>"`, `"export controls semiconductors"`

**Key analytical rule**: Policy events are high-signal but often create **temporary mispricings**. A tariff announcement may crater a sector for days — but if the underlying business is strong and the policy is likely to be walked back or negotiated, that's an opportunity, not a reason to avoid. Always ask: *"Is this policy shock permanent or transient? Who benefits structurally, who is hurt structurally?"*

Summarize the top 2–3 policy/macro events currently in play and their likely sector impact. Carry this directly into Step 4 candidate selection.

---

## Step 4 — Identify and deeply research candidates

### Part A — Scan hot spots first, then build candidates

Before picking individual stocks, identify **where institutional money is currently flowing**. Hot spots are sectors or themes with active narrative momentum — not just fundamentally good, but currently attracting capital rotation. A good stock in a cold sector will sit flat even on a strong earnings beat; the same catalyst in a hot sector re-rates the stock 20–40%.

**How to find hot spots:**
- Which sectors are outperforming the broad index over the past 2–4 weeks? (rotation into defensives? energy? AI infrastructure?)
- What macro event just happened or is imminent that tilts capital toward a specific theme? (rate hold → financials, tariff relief → industrials/autos, AI capex data → semiconductors)
- What narrative is dominating financial media and institutional investor conversations right now? That narrative, whether accurate or not, drives near-term price movement.
- Are there any oversold sectors where fear has been excessive and reversal is likely?

Identify **2–3 hot spots** before selecting candidates. Then deliberately source picks from those hot spots — because the combination of quality + catalyst + hot theme is where swing trades deliver the most.

**Now load the universe**: read `~/.claude/skills/cinvest/universe.md`. This defines your stock scope:
- All S&P 500 constituents (default scope)
- The 28 extended watchlist companies listed in `universe.md` (smaller / non-index names)

Candidates must come from this universe. Do not pick stocks outside it without noting why.

**Output structure for today's recommendations:**
- **8 total accumulation picks**, split as:
  - **3 Big Tech** — chosen from the **Big Tech Pool** at `~/.claude/ltm/bigtech_pool.md` (~30 names spanning Mag-7, semiconductors, enterprise SaaS, cybersecurity, fintech, streaming, and platform companies). Pick the 3 with the best current fundamental + macro setup. Do not default to the same 3 daily — rotate based on valuation, recent earnings, and macro conditions. If a company should be added or removed from the pool based on today's findings, update `bigtech_pool.md` and log the change in its removal/addition log.
  - **5 Free-style** — anything from the full universe (S&P 500 or extended watchlist), **including names already in the Big Tech Pool**. Overlap is fine and often appropriate — if a big tech name is your single best call, it should also appear as #1 in the free-style ranking. **Sort these 5 from highest to lowest expected profitability** (i.e., your single best call first, your most speculative last).
- **3 Avoids** — as before.

Form a research shortlist of ~14 candidates (8 picks + some avoid candidates + some that get cut). Sources:
- For big tech: evaluate each Mag-7 against current macro (rates, AI capex cycle, tariff exposure, valuation). Pick the 3 most compelling.
- For free-style: sectors showing structural strength or weakness given macro from Steps 1–3 and policy from Step 3B; names from the extended watchlist with recent fundamental catalysts; contrarian angles where the market is likely wrong.

Do **not** rely on "top stocks to buy" lists — those are consensus and already priced in. Think from first principles: given rates, inflation, policy shifts, and sector trends, where is the market likely wrong?

### Part B — Fundamental deep-dive (for each candidate)

For each candidate, research in this order:

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

### Part C — Identify the near-term catalyst for each pick

Every pick must answer: **why will this move in the next 4–8 weeks?** Without a specific catalyst, it's a hold, not a swing trade. Look for:

- **Earnings catalyst**: earnings date within the hold window + clear setup (beat expectations, guidance raise likely, or sentiment deeply negative meaning any inline result re-rates upward)
- **Event catalyst**: FDA decision, product launch, investor day, index inclusion/exclusion, regulatory ruling
- **Macro/rotation catalyst**: sector about to re-rate due to a rate decision, inflation print, or policy shift — and this name is the best expression of that move
- **Technical/sentiment catalyst**: stock has been severely oversold on non-fundamental fear (tariff headline, one bad quarter) and fundamentals remain intact — mean reversion is the trade

If no specific catalyst can be named, downgrade the pick from "active swing" to "monitor" — don't include it in this session's 8.

### Part D — Apply skepticism filter before finalizing

Before locking in each pick or avoid, challenge it:
- **For a pick**: Is this thesis already consensus? Is the catalyst already priced in? What would have to go wrong for this to be a bad call?
- **For an avoid**: Is this a structural problem or temporary noise? Could this be a contrarian opportunity being mispriced due to fear?

Final selection:
- **Big tech 3**: pick the 3 names from the pool with the strongest current catalyst + fundamental setup. Do not default to the same names daily — rotate as catalysts arise and resolve.
- **Free-style 5**: rank by expected return over the 4–8 week window. #1 is your highest-conviction, clearest-catalyst call.
- **Avoids 3**: the 3 most structurally or fundamentally weak names, or names where the near-term catalyst is clearly negative.

---

## Step 5 — Write to ltm

Write a structured Markdown file to `~/.claude/ltm/market_knowledge.md` using the template below.
Replace all `<placeholders>` with real data gathered above.
Add `_Last updated: <today's date>_` at the top.

```markdown
# Market Knowledge Snapshot

_Last updated: <YYYY-MM-DD>_

> Investor profile: long-term horizon, daily monitoring cadence.
> Signals are evaluated for multi-month to multi-year relevance. Daily noise is noted but deprioritized.

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

### Big Tech Picks (3) — Magnificent 7 selection

Pick the 3 most compelling Mag-7 names given today's macro. For each:

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

**If yes** — open `~/.claude/skills/cinvest/learnings.md` and add or update the relevant entry. Keep entries tight:
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

## Step 7 — Save action record

Write a new file to `~/.claude/skills/cinvest/action/<YYYY-MM-DD>_watchlist.md` using the template below.
This file is **immutable raw evidence** — never overwrite a past action file. It exists so future runs can spot-check prices, nothing more.

```markdown
# Investment Watchlist — <YYYY-MM-DD>

## Market Context (brief)
- Fed Funds Rate: <x.xx%> | 10Y yield: <x.xx%> | VIX: <xx.x>
- S&P 500: <xxxx> (<YTD%>) | NASDAQ: <xxxxx> (<YTD%>)
- Macro theme: <one sentence>

---

## Big Tech Picks (3) — Mag-7 selection

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

## Output to user

1. If Step 0 found past records, print the **Self-evaluation** section first so the user sees how the previous call aged.
2. Print the full **Today's Watchlist** using the exact table format from Step 7 — including **all columns**: Ticker, Company, Price at call, Thesis (brief), **Hold horizon**, and **Change trigger**. Do NOT abbreviate or drop columns. Print both the Big Tech table and the Free-style table in full.
3. Print the Avoids table (Ticker, Company, Price at call, Concern (brief), Re-evaluate in, Reversal trigger).
4. Confirm with:

```
Action saved → ~/.claude/skills/cinvest/action/<YYYY-MM-DD>_watchlist.md
Full snapshot → ~/.claude/ltm/market_knowledge.md
```
