# -*- coding: utf-8 -*-
# =============================================================================
# FlexiConc-Ready Setup Script (Colab friendly)
# - Installs requirements
# - Optionally sets up BERT-KPE
# - Runs Streamlit app (app_flexiconc.py)
# =============================================================================
import os, sys, subprocess

REQS = [
    "torch>=1.9.0",
    "transformers>=4.18.0",
    "sentence-transformers>=2.2.0",
    "spacy>=3.4.0",
    "coreferee>=1.4.0",
    "nltk>=3.7",
    "PyPDF2>=3.0.0",
    "pandas>=1.4.0",
    "numpy>=1.21.0",
    "streamlit>=1.25.0",
    "matplotlib>=3.5.0",
    "plotly>=5.10.0",
    "networkx>=2.8.0",
    "scikit-learn>=1.0.0",
    "openpyxl>=3.0.0",
    "pyngrok>=7.0.0",
    # optional for vectors
    "faiss-cpu; platform_system!='Darwin'",
]

def _run(cmd, **kw):
    print("•", " ".join(cmd))
    return subprocess.run(cmd, check=True, **kw)

def install_requirements():
    print("📦 Installing requirements...")
    for r in REQS:
        try:
            _run([sys.executable, "-m", "pip", "install", r], capture_output=True, text=True)
            print("  ✅", r)
        except subprocess.CalledProcessError as e:
            print("  ⚠️ Failed:", r, "–", e)

def setup_spacy():
    print("🔤 Installing spaCy models...")
    try:
        _run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    except Exception as e:
        print("  ⚠️ spaCy model issue:", e)
    try:
        _run([sys.executable, "-m", "coreferee", "install", "en"])
    except Exception as e:
        print("  ⚠️ coreferee install issue:", e)

def setup_bert_kpe(clone=True):
    if not clone:
        return
    try:
        if not os.path.exists("/content/BERT-KPE"):
            print("🔄 Cloning BERT-KPE...")
            _run(["git", "clone", "https://github.com/thunlp/BERT-KPE.git"], cwd="/content")
            print("  ✅ BERT-KPE ready")
        else:
            print("  ℹ️ BERT-KPE already exists")
    except Exception as e:
        print("  ⚠️ BERT-KPE clone issue:", e)

def main():
    print("🚀 FlexiConc-ready setup")
    install_requirements()
    setup_spacy()
    setup_bert_kpe(clone=True)
    print("\n✅ Setup complete.")
    print("To LAUNCH with ngrok (recommended for production):")
    print("  export NGROK_AUTHTOKEN='<your-token>'")
    print("\nColab fallback (no ngrok):")
    print("  python launch_flexiconc.py --app app_flexiconc.py --port 8501 --no-ngrok --colab-iframe")
    print("   streamlit run app_flexiconc.py --server.port=8501 --server.headless=true")

if __name__ == "__main__":
    main()

"""
Created on Sat Aug 16 17:31:39 2025

@author: niran
"""

