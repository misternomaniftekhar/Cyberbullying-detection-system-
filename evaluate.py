"""Accuracy on the 20 demo threads. Run: python evaluate.py
If Streamlit Cloud is (mis)configured to run this file, it launches the dashboard instead."""
import os, sys, runpy

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

try:
    from streamlit.runtime import exists as _in_streamlit
    IN_STREAMLIT = _in_streamlit()
except Exception:
    IN_STREAMLIT = False

if IN_STREAMLIT:
    runpy.run_path(os.path.join(ROOT, "src", "ui", "app.py"), run_name="__main__")
else:
    from collections import Counter
    from src.data_pipeline.thread_builder import load_threads
    from src.ocr_context.context_builder import build_context
    from src.agent_db.llm_judge import judge

    threads = load_threads()
    correct, confusion = 0, Counter()
    for t in threads:
        pred = judge(build_context(t))["label"]
        ok = pred == t.expected
        correct += ok
        confusion[(t.expected, pred)] += 1
        print(f"{t.id}  expected={t.expected:<13} predicted={pred:<13} {'OK' if ok else 'MISS'}")
    print(f"\nAccuracy: {correct}/{len(threads)} = {correct / len(threads):.0%}")
    for (e, p), n in sorted(confusion.items()):
        print(f"  {e} -> {p}: {n}")
