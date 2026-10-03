"""Cyberbullying Detection: Agentic LLM Judge (single-file Streamlit app).
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

def _call_groq(ctx):
    r = requests.post("https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
        json={"model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"), "temperature": 0,
              "response_format": {"type": "json_object"},
              "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ctx}]},
        timeout=60)
    r.raise_for_status()
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

# ---------------- Dashboard ----------------
st.session_state.setdefault("results", {})
st.session_state.setdefault("selected", THREADS[0]["id"])
ICON = {"Safe": "🟢", "Harassment": "🟠", "Severe Abuse": "🔴"}

st.title("Cyberbullying Detection: Agentic LLM Judge")
left, right = st.columns([1, 1.4])

with left:
    st.subheader("Thread Feed")
    if st.button("Analyze all threads"):
        with st.spinner("Running LLM Judge..."):
            for t in THREADS:
                if t["id"] not in st.session_state.results:
                    st.session_state.results[t["id"]] = run_thread(t)
    for t in THREADS:
        res = st.session_state.results.get(t["id"])
        icon = ICON.get(res["label"], "⚪") if res else "⚪"
        if st.button(f"{icon} {t['id']}: {t['messages'][-1]['text'][:55]}", key=f"b_{t['id']}", use_container_width=True):
            st.session_state.selected = t["id"]

with right:
    st.subheader("Inspector")
    t = next(x for x in THREADS if x["id"] == st.session_state.selected)
    for m in t["messages"][:-1]:
        st.markdown(f"> {m['text']}")
    st.markdown(f"**TARGET:** {t['messages'][-1]['text']}")
    if t["id"] not in st.session_state.results:
        if st.button("Analyze this thread", type="primary"):
            with st.spinner("Running LLM Judge..."):
                st.session_state.results[t["id"]] = run_thread(t)
            st.rerun()
    else:
        res = st.session_state.results[t["id"]]
        c1, c2, c3 = st.columns(3)
        c1.metric("Label", res["label"]); c2.metric("Confidence", f"{res['confidence']:.2f}"); c3.metric("Action", res["action"])
        st.markdown(f"**Intent:** {res['intent']}  \n**Target:** {res['target']}")
        st.info(f"**Rationale:** {res['rationale']}")
        st.json({k: res[k] for k in ("label", "confidence", "intent", "target", "rationale", "action")})
        existing = get_decision(res["prediction_id"])
        if existing:
            st.success(f"Reviewer decision: {existing}")
        else:
            a, d = st.columns(2)
            if a.button("✅ Approve", use_container_width=True):
                save_decision(res["prediction_id"], "approve"); st.rerun()
            if d.button("❌ Dismiss", use_container_width=True):
                save_decision(res["prediction_id"], "dismiss"); st.rerun()

    st.divider()
    if st.button("Run accuracy on 20 demo threads"):
        with st.spinner("Evaluating..."):
            ok = 0
            for x in THREADS:
                r = st.session_state.results.get(x["id"]) or run_thread(x)
                st.session_state.results[x["id"]] = r
                ok += r["label"] == x["expected"]
        st.success(f"Accuracy: {ok}/{len(THREADS)} = {ok / len(THREADS):.0%}")
