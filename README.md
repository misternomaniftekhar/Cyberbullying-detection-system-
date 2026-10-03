# Cyberbullying Detection System: Agentic LLM Approach (Zero-Training MVP)

Flow: `Thread -> Context Builder (k=2) -> LLM Judge -> Policy Action -> SQLite -> Streamlit`

## Setup
```
pip install -r requirements.txt
cp .env.example .env   # add GROQ_API_KEY or GEMINI_API_KEY, set LLM_PROVIDER
```
## Run
```
python evaluate.py              # accuracy on 20 demo threads (results cached in llm_cache.json)
streamlit run src/ui/app.py     # dashboard
```
Run both from the repo root.

## Structure
- `src/data_pipeline/thread_builder.py`: Pydantic `Message`/`Thread` + loader (M1)
- `src/ocr_context/`: `context_builder.py`, mock `ocr_engine.py` (M2)
- `src/agent_db/llm_judge.py`: prompt, Groq/Gemini call, JSON parse, cache (M3 core)
- `src/agent_db/db_manager.py`, `workflow.py`: SQLite + pipeline function (M4)
- `src/ui/app.py`: dashboard (M5)
- `data/demo_threads.json`: 10 Safe / 5 Harassment / 5 Severe Abuse
- `database/schema.sql`: `predictions`, `decisions`

## Future work
Real OCR (EasyOCR) for image text, RAG over moderation policy, larger evaluation set, CSV export.

## Deploy: GitHub + Streamlit Cloud
1. Push this repo to GitHub (`.env` is git-ignored).
2. share.streamlit.io -> New app -> select repo, branch `main`, main file `streamlit_app.py`.
3. App settings -> Secrets -> paste contents of `.streamlit/secrets.toml.example` with your real key.
4. Optional: run `python evaluate.py` locally first and commit `llm_cache.json` so the cloud demo needs no API calls.

## Quick deploy (single file)
`streamlit_app.py` in the repo root is a standalone, self-contained version of the whole system
(demo threads, context builder, LLM judge, SQLite, dashboard). Streamlit Cloud main file: `streamlit_app.py`.
The `src/` modules are the same logic split into the proposal's module structure.
