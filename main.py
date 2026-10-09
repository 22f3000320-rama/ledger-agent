import os, json, operator, requests
from datetime import datetime
from zoneinfo import ZoneInfo
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

EXPORT_URL = os.environ["EXPORT_URL"]
client = OpenAI(api_key=os.environ["AIPIPE_TOKEN"], base_url="https://aipipe.org/openai/v1")

IST = ZoneInfo("Asia/Kolkata")
RATES = {"USD": 1, "EUR": 1.02, "INR": 0.01188}

def parse(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

def load():
    text = requests.get(EXPORT_URL, timeout=60).text
    rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    latest = {}
    for r in rows:
        k = r["id"]
        if k not in latest or parse(r["updated_at"]) > parse(latest[k]["updated_at"]):
            latest[k] = r
    out = []
    for r in latest.values():
        d = parse(r["created_at"]).astimezone(IST)
        out.append({**r, "date": d.strftime("%Y-%m-%d"), "month": d.strftime("%Y-%m"),
                    "usd": r["amount"] * RATES[r["currency"]]})
    return out

clean = load()

SYSTEM = """You convert questions about an order ledger into a JSON query plan. Reply ONLY with JSON.

Each order has these fields:
- id, customer (like "C1198"), region (Central/North/West/South/East),
- product (Air Fryer, Blender, Grinder, Juicer, Kettle, Mixer, Rice Cooker, Toaster),
- qty, status (paid / refunded / void),
- date (YYYY-MM-DD, India time), month (YYYY-MM),
- usd (order value in USD).

Plan format:
{
 "status": "paid" | "refunded" | "void" | "all",
 "filters": [{"field": "...", "op": "==|>|<|>=|<=", "value": ...}],
 "metric": "sum|count|avg|max|min|count_distinct",
 "field": "usd | qty | customer | product | region | null",
 "group_by": null | "product" | "region" | "customer" | "month",
 "order": "desc" | "asc",
 "limit": 1
}

Rules:
- Revenue questions use status "paid". Refund questions use status "refunded". Only use "all" if the question clearly asks about all orders.
- "revenue", "sales", "total" means metric "sum" with field "usd".
- "how many orders" means metric "count".
- "how many customers/products" means metric "count_distinct" with field "customer" or "product".
- For "which product/region/customer has the most/highest/best", use group_by, metric "sum" with field "usd", order "desc", limit 1.
- For "lowest/least", use order "asc".
- Dates: "March 2026" means filter month == "2026-03". A specific day uses field "date".
- Use region, product and customer values exactly as spelled in the list above.
- Without group_by, leave order and limit as null.
"""

def make_plan(q):
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        temperature=0,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": q}])
    return json.loads(r.choices[0].message.content)

OPS = {"==": lambda a, b: str(a).lower() == str(b).lower(),
       ">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le}

def run(plan):
    status = plan.get("status") or "paid"
    rows = clean if status == "all" else [r for r in clean if r["status"] == status]
    for f in plan.get("filters") or []:
        rows = [r for r in rows if OPS[f["op"]](r[f["field"]], f["value"])]

    m = plan["metric"]
    fld = plan.get("field")
    if fld in (None, "null", ""):
        fld = None
    g = plan.get("group_by")
    if g in ("null", ""):
        g = None

    def agg(rs):
        if m == "count": return len(rs)
        if m == "count_distinct": return len({r[fld] for r in rs})
        vals = [r[fld] for r in rs]
        if not vals: return 0
        if m == "sum": return round(sum(vals), 2)
        if m == "avg": return round(sum(vals) / len(vals), 2)
        if m == "max": return round(max(vals), 2)
        if m == "min": return round(min(vals), 2)

    if g:
        groups = {}
        for r in rows:
            groups.setdefault(r[g], []).append(r)
        ranked = sorted(((k, agg(v)) for k, v in groups.items()),
                        key=lambda x: x[1], reverse=(plan.get("order") != "asc"))
        lim = plan.get("limit") or 1
        return ranked[0][0] if lim == 1 else [k for k, _ in ranked[:lim]]
    return agg(rows)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def health():
    return {"status": "ok", "orders": len(clean)}

@app.post("/")
def answer(body: dict):
    try:
        return {"answer": run(make_plan(body["question"]))}
    except Exception as e:
        return {"answer": None, "error": str(e)}
