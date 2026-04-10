#!/usr/bin/env python3
"""
schwab_trade.py — Charles Schwab trade executor for /ctradeexe skill

Dependencies:
    pip install schwab-py

Usage:
    # Check feasibility before placing an order
    python3 schwab_trade.py --app-key KEY --app-secret SECRET --token-path /path/token.json \
        --account XXXX --check-feasibility --action buy --ticker RTX --quantity 15

    # Place a market buy order
    python3 schwab_trade.py --app-key KEY ... --action buy --ticker RTX --quantity 15

    # Place a limit buy order
    python3 schwab_trade.py --app-key KEY ... --action buy --ticker MSFT --quantity 10 \
        --order-type limit --price 370.00

    # Check fill status (Step D callback — run after sleep 300)
    python3 schwab_trade.py --app-key KEY ... --check-order <order_id>

    # Cancel an open order
    python3 schwab_trade.py --app-key KEY ... --cancel-order <order_id>

    # List current positions
    python3 schwab_trade.py --app-key KEY ... --list-positions

    # List recent/pending orders
    python3 schwab_trade.py --app-key KEY ... --list-orders
"""

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal


def parse_args():
    p = argparse.ArgumentParser(description="Schwab trade executor")

    p.add_argument("--app-key",    required=True,  help="Schwab app key (client ID)")
    p.add_argument("--app-secret", required=True,  help="Schwab app secret")
    p.add_argument("--token-path", required=True,  help="Path to OAuth token JSON file")
    p.add_argument("--account",    required=True,  help="Account number (last 4 or full)")

    p.add_argument("--action",        choices=["buy", "sell"])
    p.add_argument("--ticker",        help="Equity ticker symbol")
    p.add_argument("--quantity",      type=int)
    p.add_argument("--dollar-amount", type=float,  help="Dollar amount to convert to shares")
    p.add_argument("--order-type",    choices=["market", "limit"], default="market")
    p.add_argument("--price",         type=float,  help="Limit price")
    p.add_argument("--duration",      choices=["DAY", "GOOD_TILL_CANCEL"], default="DAY")
    p.add_argument("--session",       choices=["NORMAL", "EXTENDED", "SEAMLESS"], default="NORMAL")

    p.add_argument("--check-feasibility", action="store_true")
    p.add_argument("--check-order",  metavar="ORDER_ID")
    p.add_argument("--cancel-order", metavar="ORDER_ID")
    p.add_argument("--list-positions", action="store_true")
    p.add_argument("--list-orders",    action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def build_client(app_key, app_secret, token_path):
    try:
        import schwab
    except ImportError:
        print("ERROR: schwab-py not installed. Run: pip install schwab-py", file=sys.stderr)
        sys.exit(1)
    try:
        client = schwab.auth.client_from_token_file(token_path, app_key, app_secret)
        return client, schwab
    except FileNotFoundError:
        print(f"ERROR: Token file not found at: {token_path}", file=sys.stderr)
        print("Run the initial OAuth flow to create it:", file=sys.stderr)
        print(f"  python3 -c \"import schwab; schwab.auth.client_from_login_flow("
              f"'{app_key}', '{app_secret}', 'https://127.0.0.1', '{token_path}')\"", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Authentication failed — {e}", file=sys.stderr)
        sys.exit(1)


def resolve_account_hash(client, account_number):
    resp = client.get_account_numbers()
    if not resp.ok:
        print(f"ERROR: Could not fetch account numbers — {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)
    accounts = resp.json()
    for entry in accounts:
        acct = entry.get("accountNumber", "")
        if acct == account_number or acct.endswith(str(account_number)):
            return entry["hashValue"]
    available = [e.get("accountNumber", "?")[-4:] for e in accounts]
    print(f"ERROR: Account '{account_number}' not found. Available (last 4): {available}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Quote helper
# ---------------------------------------------------------------------------

def get_quote(client, ticker):
    resp = client.get_quote(ticker)
    if not resp.ok:
        return None
    data = resp.json()
    quote = data.get(ticker, {}).get("quote", {})
    return {
        "ask":  quote.get("askPrice",  quote.get("lastPrice", 0)),
        "bid":  quote.get("bidPrice",  quote.get("lastPrice", 0)),
        "last": quote.get("lastPrice", 0),
    }


# ---------------------------------------------------------------------------
# Feasibility check
# ---------------------------------------------------------------------------

def check_feasibility(client, account_hash, action, ticker, quantity, dollar_amount, order_type, price):
    resp = client.get_account(account_hash, fields=[client.Account.Fields.POSITIONS])
    if not resp.ok:
        print(f"ERROR: Could not fetch account data — {resp.status_code}", file=sys.stderr)
        sys.exit(1)

    acct_data = resp.json().get("securitiesAccount", {})
    balances  = acct_data.get("currentBalances", {})
    cash = balances.get("cashBalance", balances.get("availableFunds", 0))
    bp   = balances.get("buyingPower", balances.get("availableFundsNonMarginableTrade", cash))

    if dollar_amount and not quantity:
        q = get_quote(client, ticker)
        if not q or q["ask"] <= 0:
            print(f"ERROR: Could not fetch quote for {ticker}.", file=sys.stderr)
            sys.exit(1)
        quantity = math.floor(dollar_amount / q["ask"])
        print(f"  ${dollar_amount:.2f} / ${q['ask']:.2f} = {quantity} shares")

    q   = get_quote(client, ticker)
    ask = q["ask"] if q else 0
    cost_per       = price if (order_type == "limit" and price) else ask
    estimated_cost = cost_per * (quantity or 0)

    print("FEASIBILITY CHECK")
    print(f"  Action:          {action.upper()} {quantity} × {ticker.upper()}")
    print(f"  Estimated cost:  ${estimated_cost:>12,.2f}  (@{'limit' if order_type=='limit' else 'ask'} ${cost_per:.2f})")
    print(f"  Available cash:  ${cash:>12,.2f}")
    print(f"  Buying power:    ${bp:>12,.2f}")

    if action == "buy":
        if bp >= estimated_cost:
            print(f"  Result:          FEASIBLE ✓")
        else:
            shortfall = estimated_cost - bp
            print(f"  Result:          NOT FEASIBLE — short by ${shortfall:,.2f}")
            sys.exit(2)
    elif action == "sell":
        positions = acct_data.get("positions", [])
        held = 0
        for pos in positions:
            if pos.get("instrument", {}).get("symbol", "").upper() == ticker.upper():
                held = pos.get("longQuantity", 0)
                break
        print(f"  Shares held:     {held:.0f}")
        if held >= (quantity or 0):
            print(f"  Result:          FEASIBLE ✓")
        else:
            print(f"  Result:          NOT FEASIBLE — hold {held:.0f} shares, need {quantity}")
            sys.exit(2)

    return quantity


# ---------------------------------------------------------------------------
# Place order
# ---------------------------------------------------------------------------

def build_and_place_order(client, schwab, account_hash, action, ticker, quantity,
                           order_type, price, duration, session):
    from schwab.orders.equities import (
        equity_buy_market, equity_buy_limit,
        equity_sell_market, equity_sell_limit,
    )
    from schwab.orders.common import Duration, Session

    ticker   = ticker.upper()
    dur_enum = Duration[duration]
    ses_enum = Session[session]

    if   action == "buy"  and order_type == "market": order = equity_buy_market(ticker, quantity)
    elif action == "buy"  and order_type == "limit":
        if not price: print("ERROR: --price required for limit orders.", file=sys.stderr); sys.exit(1)
        order = equity_buy_limit(ticker, quantity, Decimal(str(price)))
    elif action == "sell" and order_type == "market": order = equity_sell_market(ticker, quantity)
    elif action == "sell" and order_type == "limit":
        if not price: print("ERROR: --price required for limit orders.", file=sys.stderr); sys.exit(1)
        order = equity_sell_limit(ticker, quantity, Decimal(str(price)))
    else:
        print(f"ERROR: Unsupported: {action}/{order_type}", file=sys.stderr); sys.exit(1)

    order = order.set_duration(dur_enum).set_session(ses_enum)
    resp  = client.place_order(account_hash, order)

    if resp.ok:
        location = resp.headers.get("Location", "")
        order_id = location.rstrip("/").split("/")[-1] if location else "unknown"
        print(f"ORDER PLACED — Order ID: {order_id}")
        return order_id
    else:
        print(f"ERROR: Order rejected — HTTP {resp.status_code}", file=sys.stderr)
        try:    print(f"  Detail: {json.dumps(resp.json(), indent=2)}", file=sys.stderr)
        except: print(f"  Body: {resp.text}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Check order status
# ---------------------------------------------------------------------------

def check_order_status(client, account_hash, order_id):
    resp = client.get_order(order_id, account_hash)
    if not resp.ok:
        print(f"ERROR: Could not fetch order {order_id} — {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)

    o      = resp.json()
    status = o.get("status", "UNKNOWN")
    filled = o.get("filledQuantity", 0)
    total  = o.get("quantity", 0)
    avg_px = o.get("orderActivityCollection", [{}])[0].get("executionLegs", [{}])[0].get("price", None)
    close_t= o.get("closeTime", o.get("enteredTime", ""))[:19].replace("T", " ")

    print(f"ORDER STATUS: {status}")
    print(f"  Filled qty:    {filled:.0f} / {total:.0f} shares")

    if status == "FILLED":
        if avg_px: print(f"  Average price: ${float(avg_px):.2f}")
        print(f"  Fill time:     {close_t}")
        print(f"  Status:        COMPLETE ✓")
    elif status in ("WORKING", "QUEUED", "ACCEPTED", "PENDING_ACTIVATION"):
        legs   = o.get("orderLegCollection", [{}])
        ticker = legs[0].get("instrument", {}).get("symbol", "?") if legs else "?"
        q      = get_quote(client, ticker)
        if q: print(f"  Current bid/ask: ${q['bid']:.2f} / ${q['ask']:.2f}")
        lim = o.get("price")
        if lim: print(f"  Limit price:   ${float(lim):.2f}")
        print(f"  Status:        STILL OPEN — monitor or cancel")
    elif status in ("CANCELLED", "REPLACED", "REJECTED", "EXPIRED"):
        reason = o.get("cancelMessage", o.get("statusDescription", "no detail"))
        print(f"  Reason:        {reason}")
        print(f"  Status:        FAILED / CLOSED")
    else:
        print(f"  Raw status:    {status}")
        print(f"  Status:        UNKNOWN — check Schwab.com")


# ---------------------------------------------------------------------------
# Cancel order
# ---------------------------------------------------------------------------

def cancel_order(client, account_hash, order_id):
    resp = client.cancel_order(order_id, account_hash)
    if resp.ok:
        print(f"ORDER CANCELLED — Order ID: {order_id}")
    else:
        try:    print(f"CANCEL FAILED — {resp.json()}", file=sys.stderr)
        except: print(f"CANCEL FAILED — {resp.text}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# List positions
# ---------------------------------------------------------------------------

def list_positions(client, account_hash):
    resp = client.get_account(account_hash, fields=[client.Account.Fields.POSITIONS])
    if not resp.ok:
        print(f"ERROR: {resp.status_code}", file=sys.stderr); sys.exit(1)

    acct      = resp.json().get("securitiesAccount", {})
    bal       = acct.get("currentBalances", {})
    cash      = bal.get("cashBalance", 0)
    bp        = bal.get("buyingPower", cash)
    positions = acct.get("positions", [])
    total_mkt = cash + sum(p.get("marketValue", 0) for p in positions)

    print(f"Cash balance:  ${cash:>12,.2f}")
    print(f"Buying power:  ${bp:>12,.2f}")
    print()
    if not positions:
        print("No open positions."); return

    print(f"{'Ticker':<8} {'Qty':>8} {'Avg Cost':>12} {'Mkt Value':>12} {'P&L':>12} {'P&L %':>8} {'% Port':>8}")
    print("-" * 75)
    for pos in sorted(positions, key=lambda x: x.get("marketValue", 0), reverse=True):
        instr   = pos.get("instrument", {})
        ticker  = instr.get("symbol", "?")
        qty     = pos.get("longQuantity", 0) - pos.get("shortQuantity", 0)
        avg     = pos.get("averagePrice", 0)
        mkt_val = pos.get("marketValue", 0)
        cost    = avg * abs(qty)
        pnl     = mkt_val - cost
        pnl_pct = (pnl / cost * 100) if cost > 0 else 0
        port_pct= (mkt_val / total_mkt * 100) if total_mkt > 0 else 0
        print(f"{ticker:<8} {qty:>8.0f} {avg:>12.2f} {mkt_val:>12.2f} {pnl:>+12.2f} {pnl_pct:>+7.1f}% {port_pct:>7.1f}%")


# ---------------------------------------------------------------------------
# List orders
# ---------------------------------------------------------------------------

def list_orders(client, account_hash):
    from_date = datetime.now(timezone.utc) - timedelta(days=30)
    resp = client.get_orders_for_account(account_hash, from_entered_datetime=from_date)
    if not resp.ok:
        print(f"ERROR: {resp.status_code}", file=sys.stderr); sys.exit(1)

    orders = resp.json()
    if not orders:
        print("No orders in the last 30 days."); return

    print(f"{'Date':<12} {'Status':<18} {'Act':<5} {'Ticker':<8} {'Qty':>6} {'Type':<8} {'Price':>10} {'Order ID'}")
    print("-" * 85)
    for o in sorted(orders, key=lambda x: x.get("enteredTime", ""), reverse=True)[:25]:
        entered = o.get("enteredTime", "")[:10]
        status  = o.get("status", "?")[:17]
        legs    = o.get("orderLegCollection", [{}])
        leg     = legs[0] if legs else {}
        act     = leg.get("instruction", "?")[:4]
        ticker  = leg.get("instrument", {}).get("symbol", "?")
        qty     = leg.get("quantity", 0)
        otype   = o.get("orderType", "?")[:7]
        price   = o.get("price", o.get("stopPrice", "MKT"))
        oid     = str(o.get("orderId", "?"))
        print(f"{entered:<12} {status:<18} {act:<5} {ticker:<8} {qty:>6.0f} {otype:<8} {str(price):>10} {oid}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    client, schwab = build_client(args.app_key, args.app_secret, args.token_path)
    account_hash   = resolve_account_hash(client, args.account)

    if args.list_positions:
        list_positions(client, account_hash); return
    if args.list_orders:
        list_orders(client, account_hash); return
    if args.check_order:
        check_order_status(client, account_hash, args.check_order); return
    if args.cancel_order:
        cancel_order(client, account_hash, args.cancel_order); return

    if args.check_feasibility:
        if not args.action or not args.ticker:
            print("ERROR: --action and --ticker required.", file=sys.stderr); sys.exit(1)
        if not args.quantity and not args.dollar_amount:
            print("ERROR: --quantity or --dollar-amount required.", file=sys.stderr); sys.exit(1)
        check_feasibility(client, account_hash, args.action, args.ticker,
                          args.quantity, args.dollar_amount, args.order_type, args.price)
        return

    if not args.action or not args.ticker:
        print("ERROR: --action and --ticker required.", file=sys.stderr); sys.exit(1)

    qty = args.quantity
    if not qty:
        if args.dollar_amount:
            q = get_quote(client, args.ticker)
            if not q or q["ask"] <= 0:
                print(f"ERROR: Could not fetch quote for {args.ticker}.", file=sys.stderr); sys.exit(1)
            qty = math.floor(args.dollar_amount / q["ask"])
            print(f"  ${args.dollar_amount:.2f} / ${q['ask']:.2f} = {qty} shares")
        else:
            print("ERROR: --quantity or --dollar-amount required.", file=sys.stderr); sys.exit(1)

    if qty <= 0:
        print("ERROR: Computed quantity is 0.", file=sys.stderr); sys.exit(1)

    build_and_place_order(client, schwab, account_hash, args.action, args.ticker, qty,
                          args.order_type, args.price, args.duration, args.session)


if __name__ == "__main__":
    main()
