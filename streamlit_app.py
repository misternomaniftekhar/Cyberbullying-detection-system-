"""Cyberbullying Detection: Agentic LLM Judge v2 (single-file Streamlit app).
Needs only: streamlit, python-dotenv, requests, pydantic  +  GROQ_API_KEY (or GEMINI_API_KEY) in Secrets."""
import hashlib, json, os, re, sqlite3
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="Cyberbullying Detection", layout="wide")
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass

# ---------------- Demo data (20 threads, last message = target) ----------------
THREADS = json.loads(r'''[{"id": "t01", "expected": "Safe", "messages": [{"id": "t01_m1", "text": "Yaar kal ka assignment submit kar diya?"}, {"id": "t01_m2", "text": "Haan bhai, raat ko hi kar diya"}, {"id": "t01_m3", "text": "Shukar hai, mujhe abhi karna hai"}]}, {"id": "t02", "expected": "Safe", "messages": [{"id": "t02_m1", "text": "Are you coming to the match tomorrow?"}, {"id": "t02_m2", "text": "Yes, 5 pm at the ground"}, {"id": "t02_m3", "text": "Great, I'll bring the cricket kit"}]}, {"id": "t03", "expected": "Safe", "messages": [{"id": "t03_m1", "text": "Aaj ki lecture bohat boring thi"}, {"id": "t03_m2", "text": "Haan yaar, sir ki slides bakwas thi"}, {"id": "t03_m3", "text": "Next time notes share kar dena"}]}, {"id": "t04", "expected": "Safe", "messages": [{"id": "t04_m1", "text": "Bhai tum bohat lazy ho, kabhi time pe nahi aate"}, {"id": "t04_m2", "text": "Haha sahi kaha, alarm hi nahi bajta"}, {"id": "t04_m3", "text": "Chalo aaj chai meri taraf se"}]}, {"id": "t05", "expected": "Safe", "messages": [{"id": "t05_m1", "text": "Dinner mein kya banayein?"}, {"id": "t05_m2", "text": "Biryani bana lete hain"}, {"id": "t05_m3", "text": "Perfect, main raita bana deta hoon"}]}, {"id": "t06", "expected": "Safe", "messages": [{"id": "t06_m1", "text": "I think your code has a bug in line 40"}, {"id": "t06_m2", "text": "Oh thanks, let me check"}, {"id": "t06_m3", "text": "Yeah, the loop index is off by one"}]}, {"id": "t07", "expected": "Safe", "messages": [{"id": "t07_m1", "text": "Mubarak ho! Result aa gaya"}, {"id": "t07_m2", "text": "Shukriya, 3.8 CGPA aya hai"}, {"id": "t07_m3", "text": "Wah, party to banti hai!"}]}, {"id": "t08", "expected": "Safe", "messages": [{"id": "t08_m1", "text": "Ye government ki policy bilkul bakwaas hai"}, {"id": "t08_m2", "text": "Sahi, fuel prices phir barh gayi"}, {"id": "t08_m3", "text": "Haan, public ko relief milna chahiye"}]}, {"id": "t09", "expected": "Safe", "messages": [{"id": "t09_m1", "text": "Kal ki meeting ka agenda bhejo please"}, {"id": "t09_m2", "text": "Sending now on email"}, {"id": "t09_m3", "text": "Thanks, got it"}]}, {"id": "t10", "expected": "Safe", "messages": [{"id": "t10_m1", "text": "Mera phone ka battery bohat kharab ho gaya"}, {"id": "t10_m2", "text": "Naya le lo, sale chal rahi hai"}, {"id": "t10_m3", "text": "Haan soch raha hoon, thanks yaar"}]}, {"id": "t11", "expected": "Harassment", "messages": [{"id": "t11_m1", "text": "Tumhari presentation dekhi?"}, {"id": "t11_m2", "text": "Haan, poori bakwas thi"}, {"id": "t11_m3", "text": "Tum jaise bewakoof se aur kya umeed thi"}]}, {"id": "t12", "expected": "Harassment", "messages": [{"id": "t12_m1", "text": "Look at her new profile picture"}, {"id": "t12_m2", "text": "Ugh so fake and ugly"}, {"id": "t12_m3", "text": "Nobody likes you here, just leave the group"}]}, {"id": "t13", "expected": "Harassment", "messages": [{"id": "t13_m1", "text": "Is ladke ko dekho, kitna mota hai"}, {"id": "t13_m2", "text": "Haha moti bhains lagta hai"}, {"id": "t13_m3", "text": "Gym jaane ki bajaye khana band kar, mota!"}]}, {"id": "t14", "expected": "Harassment", "messages": [{"id": "t14_m1", "text": "Tum phir se galat answer de rahe ho"}, {"id": "t14_m2", "text": "Tum hamesha fail hote ho"}, {"id": "t14_m3", "text": "Tum ek loser ho, koi tumhe dost nahi banata"}]}, {"id": "t15", "expected": "Harassment", "messages": [{"id": "t15_m1", "text": "Why does she even post pictures"}, {"id": "t15_m2", "text": "Because she thinks she is pretty lol"}, {"id": "t15_m3", "text": "You're disgusting and pathetic, delete your account"}]}, {"id": "t16", "expected": "Severe Abuse", "messages": [{"id": "t16_m1", "text": "Tune mere khilaf complaint kyun ki?"}, {"id": "t16_m2", "text": "Mujhe tumse masla hai"}, {"id": "t16_m3", "text": "Main tujhe jaan se maar dunga, ghar se nikal ke dikha"}]}, {"id": "t17", "expected": "Severe Abuse", "messages": [{"id": "t17_m1", "text": "I know where you live"}, {"id": "t17_m2", "text": "You better watch your back"}, {"id": "t17_m3", "text": "I will find you and hurt you badly, you won't survive"}]}, {"id": "t18", "expected": "Severe Abuse", "messages": [{"id": "t18_m1", "text": "Ye log hamare mulk ke nahi hain"}, {"id": "t18_m2", "text": "Inki wajah se sab kharab hai"}, {"id": "t18_m3", "text": "Ye kaum gandi hai, inhein maar do sab ko"}]}, {"id": "t19", "expected": "Severe Abuse", "messages": [{"id": "t19_m1", "text": "Send me your photos or I'll leak your chats"}, {"id": "t19_m2", "text": "Please stop, I'm begging you"}, {"id": "t19_m3", "text": "Do it or I'll send everything to your family and ruin you"}]}, {"id": "t20", "expected": "Severe Abuse", "messages": [{"id": "t20_m1", "text": "Tum kisi kaam ke nahi ho"}, {"id": "t20_m2", "text": "Tumhari wajah se sab pareshan hain"}, {"id": "t20_m3", "text": "Tum mar kyun nahi jaate, khudkushi kar lo sab khush honge"}]}]''')

# ---------------- Context builder (k=2) + mock OCR ----------------
def extract_text_from_image(image_path=None):
    return ""  # Future work: real OCR

def build_context(msgs, k=2):
    idx = len(msgs) - 1
    prev = msgs[max(0, idx - k):idx]
    lines = [f"[Prev Message {i + 1}] {m['text']}" for i, m in enumerate(prev)]
    lines.append(f"[TARGET MESSAGE] {msgs[idx]['text']}")
    return "\n".join(lines)

# ---------------- LLM judge ----------------
CACHE_PATH = "llm_cache.json"
LABEL_TO_ACTION = {"Safe": "allow", "Harassment": "flag", "Severe Abuse": "escalate"}
SYSTEM_PROMPT = """You are a content-safety moderator for a chat platform used by Pakistani users.
Messages are in Roman Urdu, English, Urdu, or a mix (code-mixed). Spelling varies
(bakwas/bakwaas/baqwas). Judge ONLY the line marked [TARGET MESSAGE]; earlier lines are context.

Labels:
- "Safe": normal chat, banter between friends, criticism of ideas, non-abusive frustration.
- "Harassment": insults, name-calling, mocking, shaming, repeated targeting, exclusion, mild abuse.
- "Severe Abuse": threats of violence or harm, hate speech (religion, ethnicity, gender, sect),
  sexual harassment, doxxing, encouraging self-harm.

Rules:
- Use context: friendly teasing between friends is Safe; the same words aimed at a victim after
  earlier hostility are Harassment.
- Do not flag profanity that is not directed at a person.
- Reply with ONLY a JSON object, no markdown, in exactly this shape:
{"label": "Safe|Harassment|Severe Abuse", "confidence": 0.0-1.0, "intent": "short phrase",
 "target": "who is targeted or none", "rationale": "1-2 sentences in English",
 "action": "allow|flag|escalate"}
"""

def _load_cache():
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_cache(c):
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(c, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

DEPRECATED_GROQ = {"openai/gpt-oss-120b", "whisper-large-v3", "whisper-large-v3-turbo"}


def _call_groq(ctx):
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    if model in DEPRECATED_GROQ:  # retired by Groq; auto-switch
        model = "openai/gpt-oss-20b"
    body = {"model": model, "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ctx}]}
    headers = {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}
    url = "https://api.groq.com/openai/v1/chat/completions"
    r = requests.post(url, headers=headers, json=body, timeout=60)
    if r.status_code == 400:  # retry without JSON mode; output is parsed leniently anyway
        body.pop("response_format")
        r = requests.post(url, headers=headers, json=body, timeout=60)
    if not r.ok:
        raise RuntimeError(f"Groq {r.status_code} (model={model}): {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]


def _call_gemini(ctx):
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": os.environ["GEMINI_API_KEY"]},
        json={"systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
              "contents": [{"role": "user", "parts": [{"text": ctx}]}],
              "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}},
        timeout=60)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]

def _parse(raw):
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    d = json.loads(m.group(0) if m else raw)
    label = d.get("label", "Safe")
    if label not in LABEL_TO_ACTION:
        label = "Safe"
    return {"label": label, "confidence": float(d.get("confidence", 0.5)),
            "intent": str(d.get("intent", "")), "target": str(d.get("target", "")),
            "rationale": str(d.get("rationale", "")), "action": LABEL_TO_ACTION[label]}

def judge(ctx):
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    key = hashlib.sha256(f"{provider}|{ctx}".encode("utf-8")).hexdigest()
    cache = _load_cache()
    if key in cache:
        return cache[key]
    raw = _call_groq(ctx) if provider == "groq" else _call_gemini(ctx)
    try:
        res = _parse(raw)
    except Exception:
        res = {"label": "Safe", "confidence": 0.0, "intent": "parse_error", "target": "none",
               "rationale": f"Unparseable LLM output: {raw[:200]}", "action": "flag"}
    cache[key] = res
    _save_cache(cache)
    return res

# ---------------- SQLite (predictions + decisions) ----------------
DB_PATH = "moderation.db"

def db():
    c = sqlite3.connect(DB_PATH)
    c.executescript("""
    CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id TEXT, context TEXT,
        label TEXT, confidence REAL, intent TEXT, target TEXT, rationale TEXT, action TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS decisions (id INTEGER PRIMARY KEY AUTOINCREMENT, prediction_id INTEGER,
        reviewer_decision TEXT, decided_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);""")
    return c

def run_thread(t):
    ctx = build_context(t["messages"])
    try:
        res = judge(ctx)
    except KeyError as e:
        st.error(f"Missing API key {e}. Add GROQ_API_KEY in Streamlit Secrets."); st.stop()
    except Exception as e:
        st.error(f"LLM call failed: {e}"); st.stop()
    with db() as c:
        cur = c.execute("INSERT INTO predictions (thread_id,context,label,confidence,intent,target,rationale,action)"
                        " VALUES (?,?,?,?,?,?,?,?)",
                        (t["id"], ctx, res["label"], res["confidence"], res["intent"], res["target"], res["rationale"], res["action"]))
        res = dict(res, prediction_id=cur.lastrowid)
    return res

def get_decision(pid):
    with db() as c:
        row = c.execute("SELECT reviewer_decision FROM decisions WHERE prediction_id=? ORDER BY id DESC LIMIT 1", (pid,)).fetchone()
    return row[0] if row else None

def save_decision(pid, d):
    with db() as c:
        c.execute("INSERT INTO decisions (prediction_id, reviewer_decision) VALUES (?,?)", (pid, d))

# ---------------- Dashboard v2 ----------------
import pandas as pd

LABELS = ["Safe", "Harassment", "Severe Abuse"]
COLOR = {"Safe": "#16a34a", "Harassment": "#ea580c", "Severe Abuse": "#dc2626"}
ICON = {"Safe": "🟢", "Harassment": "🟠", "Severe Abuse": "🔴"}

st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}
.badge {display:inline-block;padding:3px 12px;border-radius:999px;color:#fff;font-weight:600;font-size:0.85rem;}
.card {border:1px solid rgba(128,128,128,.3);border-left-width:6px;border-radius:10px;padding:14px 16px;margin:8px 0;}
.msg {background:rgba(128,128,128,.12);border-radius:8px;padding:8px 12px;margin:4px 0;}
.target {border:2px solid #6366f1;}
</style>""", unsafe_allow_html=True)

st.session_state.setdefault("results", {})
st.session_state.setdefault("selected", THREADS[0]["id"])
R = st.session_state.results


def badge(label):
    return f'<span class="badge" style="background:{COLOR[label]}">{label}</span>'


def show_result(res, key):
    st.markdown(f'<div class="card" style="border-left-color:{COLOR[res["label"]]}">'
                f'{badge(res["label"])} &nbsp; <b>Action:</b> {res["action"].upper()}</div>', unsafe_allow_html=True)
    st.progress(min(max(res["confidence"], 0.0), 1.0), text=f"Confidence {res['confidence']:.0%}")
    c1, c2 = st.columns(2)
    c1.markdown(f"**Intent**  \n{res['intent'] or '-'}")
    c2.markdown(f"**Target**  \n{res['target'] or '-'}")
    st.info(f"**Rationale:** {res['rationale']}")
    with st.expander("Raw JSON"):
        st.json({k: res[k] for k in ("label", "confidence", "intent", "target", "rationale", "action")})
    dec = get_decision(res["prediction_id"])
    if dec:
        st.success(f"Reviewer decision: {dec}")
    else:
        a, d = st.columns(2)
        if a.button("✅ Approve", key=f"a_{key}", width="stretch"):
            save_decision(res["prediction_id"], "approve"); st.rerun()
        if d.button("❌ Dismiss", key=f"d_{key}", width="stretch"):
            save_decision(res["prediction_id"], "dismiss"); st.rerun()


def analyze_all():
    bar = st.progress(0.0, text="Running LLM Judge...")
    for i, t in enumerate(THREADS):
        if t["id"] not in R:
            R[t["id"]] = run_thread(t)
        bar.progress((i + 1) / len(THREADS), text=f"Analyzed {i + 1}/{len(THREADS)}")
    bar.empty()


# ----- Sidebar -----
with st.sidebar:
    st.header("🛡️ Control Panel")
    st.caption(f"Provider: **{os.getenv('LLM_PROVIDER', 'groq')}**  \nModel: **{os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b')}**")
    if st.button("▶ Analyze all threads", type="primary", width="stretch"):
        analyze_all()
    if st.button("↺ Clear results (UI only)", width="stretch"):
        R.clear(); st.rerun()
    st.divider()
    flt = st.multiselect("Filter feed", LABELS + ["Not analyzed"], default=LABELS + ["Not analyzed"])
    st.divider()
    st.caption("Zero-training LLM Judge · Roman Urdu + English · k=2 context window")

st.title("🛡️ Cyberbullying Detection System")
st.caption("Agentic LLM Judge: no training data, works on day 0")

# ----- KPI row -----
done = [R[t["id"]] for t in THREADS if t["id"] in R]
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Threads", len(THREADS))
k2.metric("Analyzed", len(done))
k3.metric("Safe", sum(r["label"] == "Safe" for r in done))
k4.metric("Flagged", sum(r["action"] == "flag" for r in done))
k5.metric("Escalated", sum(r["action"] == "escalate" for r in done))

tab_review, tab_live, tab_eval, tab_hist = st.tabs(["📋 Thread Review", "⚡ Live Analyzer", "📊 Evaluation", "🗂️ History"])

# ----- Tab 1: Thread review -----
with tab_review:
    left, right = st.columns([1, 1.5])
    with left:
        st.subheader("Thread Feed")
        shown = 0
        for t in THREADS:
            res = R.get(t["id"])
            tag = res["label"] if res else "Not analyzed"
            if tag not in flt:
                continue
            shown += 1
            icon = ICON.get(tag, "⚪")
            if st.button(f"{icon} {t['id']} · {t['messages'][-1]['text'][:50]}", key=f"b_{t['id']}", width="stretch"):
                st.session_state.selected = t["id"]
        if not shown:
            st.caption("No threads match the filter.")
    with right:
        st.subheader("Inspector")
        t = next(x for x in THREADS if x["id"] == st.session_state.selected)
        st.markdown(f"**Thread {t['id']}**")
        for m in t["messages"][:-1]:
            st.markdown(f'<div class="msg">💬 {m["text"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="msg target">🎯 <b>{t["messages"][-1]["text"]}</b></div>', unsafe_allow_html=True)
        if t["id"] not in R:
            if st.button("🔍 Analyze this thread", type="primary"):
                with st.spinner("Running LLM Judge..."):
                    R[t["id"]] = run_thread(t)
                st.rerun()
        else:
            show_result(R[t["id"]], t["id"])

# ----- Tab 2: Live analyzer -----
with tab_live:
    st.subheader("Try your own message")
    st.caption("Type any Roman Urdu / English message. Add earlier messages for context (one per line, optional).")
    ctx_in = st.text_area("Previous messages (optional, max 2 used)", height=80, placeholder="Tum kal kahan the?\\nTumhe kya matlab")
    msg_in = st.text_input("Target message", placeholder="Tum bohat bewakoof ho")
    if st.button("Analyze message", type="primary", disabled=not msg_in.strip()):
        prev = [x.strip() for x in ctx_in.splitlines() if x.strip()]
        live = {"id": "live", "messages": [{"text": x} for x in prev] + [{"text": msg_in.strip()}]}
        with st.spinner("Running LLM Judge..."):
            st.session_state.live = run_thread(live)
    if "live" in st.session_state:
        show_result(st.session_state.live, "live")

# ----- Tab 3: Evaluation -----
with tab_eval:
    st.subheader("Accuracy on 20 demo threads")
    if st.button("Run evaluation"):
        analyze_all()
    rows = [{"thread": t["id"], "expected": t["expected"], "predicted": R[t["id"]]["label"],
             "confidence": R[t["id"]]["confidence"]} for t in THREADS if t["id"] in R]
    if rows:
        df = pd.DataFrame(rows)
        df["correct"] = df.expected == df.predicted
        a1, a2 = st.columns(2)
        a1.metric("Accuracy", f"{df.correct.mean():.0%}", f"{int(df.correct.sum())}/{len(df)}")
        sev = df[df.expected != "Safe"]
        a2.metric("Harm detection recall", f"{(sev.predicted != 'Safe').mean():.0%}" if len(sev) else "-")
        st.markdown("**Confusion matrix** (rows = expected, columns = predicted)")
        cm = pd.crosstab(df.expected, df.predicted).reindex(index=LABELS, columns=LABELS, fill_value=0)
        st.dataframe(cm, width="stretch")
        prec = {}
        for l in LABELS:
            p = (df.predicted == l).sum(); tp = ((df.predicted == l) & (df.expected == l)).sum()
            r = (df.expected == l).sum()
            prec[l] = {"precision": tp / p if p else 0, "recall": tp / r if r else 0}
        st.bar_chart(pd.DataFrame(prec).T)
        miss = df[~df.correct]
        if len(miss):
            st.markdown("**Misclassified**")
            st.dataframe(miss, width="stretch", hide_index=True)
    else:
        st.caption("Click **Run evaluation** (or Analyze all in the sidebar).")

# ----- Tab 4: History -----
with tab_hist:
    st.subheader("Moderation log (SQLite)")
    with db() as c:
        hist = pd.read_sql_query(
            "SELECT p.id, p.thread_id, p.label, p.action, p.confidence, p.rationale, "
            "COALESCE(d.reviewer_decision,'pending') AS reviewer, p.created_at "
            "FROM predictions p LEFT JOIN decisions d ON d.prediction_id=p.id ORDER BY p.id DESC", c)
    if len(hist):
        st.dataframe(hist, width="stretch", hide_index=True)
        st.download_button("⬇ Download CSV", hist.to_csv(index=False).encode("utf-8"),
                           "moderation_log.csv", "text/csv")
    else:
        st.caption("No predictions logged yet.")
