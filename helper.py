# =============================================================================
# helper.py - Shared functions for SDG Analysis System
# =============================================================================
import json
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, BertTokenizerFast, BertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
import PyPDF2
import spacy
from fastcoref import FCoref
import nltk
import matplotlib.pyplot as plt
from IPython.display import HTML, display
import ipywidgets as widgets
from nltk.tokenize import sent_tokenize
from pathlib import Path
from difflib import SequenceMatcher
from typing import List, Dict, Any, Iterable, Sequence
from spacy.lang.en.stop_words import STOP_WORDS
import networkx as nx




# =============================================================================
# Model Initialization (Shared)
# =============================================================================

# Load spaCy + coreferee
nlp = spacy.load("en_core_web_sm")
# Send everything to CUDA when available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BERT classification model
BERT_CHECKPOINT = "sadickam/sdg-classification-bert"
config = AutoConfig.from_pretrained(BERT_CHECKPOINT)
config.architectures = ["BertForSequenceClassification"]

bert_tokenizer = BertTokenizerFast.from_pretrained(
    BERT_CHECKPOINT, use_fast=True)
bert_model = BertForSequenceClassification.from_pretrained(
    BERT_CHECKPOINT,
    config=config,
    trust_remote_code=True
)
bert_model.to(device).eval()

def encode_sdg_hidden(texts: Sequence[str] | Iterable[str] | str,
                      *,
                      batch_size: int = 16,
                      max_length: int = 512) -> np.ndarray:
    """Return CLS hidden states from the SDG-BERT classifier."""
    arr_texts = _ensure_text_list(texts)
    if not arr_texts:
        hidden = int(getattr(bert_model.config, "hidden_size", 768))
        return np.zeros((0, hidden), dtype=np.float32)
    reps = []
    for start in range(0, len(arr_texts), int(batch_size)):
        batch = arr_texts[start:start + int(batch_size)]
        enc = bert_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = bert_model(**enc, output_hidden_states=True, return_dict=True)
            cls = outputs.hidden_states[-1][:, 0, :]
        reps.append(cls.detach().cpu().numpy())
    if not reps:
        hidden = int(getattr(bert_model.config, "hidden_size", 768))
        return np.zeros((0, hidden), dtype=np.float32)
    return np.concatenate(reps, axis=0).astype(np.float32)

# Load MPNet similarity model
SIM_CHECKPOINT = "sentence-transformers/paraphrase-mpnet-base-v2"
sim_tokenizer = AutoTokenizer.from_pretrained(SIM_CHECKPOINT)
sim_model = SentenceTransformer(SIM_CHECKPOINT)
sim_model.to(device).eval()

def _ensure_text_list(texts: Sequence[str] | Iterable[str] | str | None) -> List[str]:
    if texts is None:
        return []
    if isinstance(texts, str):
        return [texts]
    try:
        iterable = list(texts)
    except TypeError:
        return [str(texts)]
    out: List[str] = []
    for item in iterable:
        out.append(item if isinstance(item, str) else str(item))
    return out

def encode_mpnet(texts: Sequence[str] | Iterable[str] | str,
                 *,
                 batch_size: int = 32,
                 normalize: bool = False) -> np.ndarray:
    """Encode sentences using the shared MPNet SentenceTransformer."""
    arr_texts = _ensure_text_list(texts)
    if not arr_texts:
        dim = getattr(sim_model, "get_sentence_embedding_dimension", lambda: 768)()
        return np.zeros((0, dim), dtype=np.float32)
    embs = sim_model.encode(
        arr_texts,
        batch_size=int(batch_size),
        convert_to_numpy=True,
        normalize_embeddings=bool(normalize),
        show_progress_bar=False,
    )
    return np.asarray(embs, dtype=np.float32)

def encode_scico(texts: Sequence[str] | Iterable[str] | str,
                 *,
                 batch_size: int = 32) -> np.ndarray:
    """SciCo sentence embeddings (MPNet with normalization)."""
    return encode_mpnet(texts, batch_size=batch_size, normalize=True)

FASTCOREF_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_fastcoref = FCoref(device=FASTCOREF_DEVICE)  # loads default English model

# SDG target descriptions
SDG_GOALS = [
    "No Poverty", "Zero Hunger", "Good Health and Well-being", "Quality Education",
    "Gender Equality", "Clean Water and Sanitation", "Affordable and Clean Energy",
    "Decent Work and Economic Growth", "Industry, Innovation and Infrastructure",
    "Reduced Inequality", "Sustainable Cities and Communities",
    "Responsible Consumption and Production", "Climate Action", "Life Below Water",
    "Life on Land", "Peace, Justice and Strong Institutions"
]

SDG_TARGETS = json.load(open('/content/drive/MyDrive/ENLENS/sdg_targets.json', 'r'))

# =============================================================================
# Data Classes
# =============================================================================
class Interval:
    __slots__ = ("start", "end", "data")
    def __init__(self, start, end, data):
        self.start, self.end, self.data = start, end, data
        
class IntervalTree:
    """Balanced-lookup-ish tree: store intervals, query by point."""
    def __init__(self, intervals=None):
        self._ivs = sorted(intervals or [], key=lambda iv: iv.start)
        self._starts = [iv.start for iv in self._ivs]
        # Precompute max_end prefix for pruning (simple variant)
        self._max_ends = []
        cur = float("-inf")
        for iv in self._ivs:
            cur = max(cur, iv.end)
            self._max_ends.append(cur)
    @property
    def nodes(self):
        return self._ivs

    def __repr__(self):
        spans = [(n.start, n.end, n.data) for n in self.nodes]
        return f"<IntervalTree spans={spans}>"
    
    def at(self, point):
        """Return all intervals that contain `point`."""
        res = []
        # binary search left boundary
        import bisect
        idx = bisect.bisect_right(self._starts, point)
        # scan left while intervals still can cover `point`
        i = idx - 1
        while i >= 0 and self._ivs[i].start <= point:
            if self._ivs[i].end > point:
                res.append(self._ivs[i].data)
            i -= 1
        # scan right while start <= point (rarely needed but safe)
        j = idx
        n = len(self._ivs)
        while j < n and self._ivs[j].start <= point:
            if self._ivs[j].end > point:
                res.append(self._ivs[j].data)
            j += 1
        return res

# =============================================================================
# PDF Extraction Functions
# =============================================================================

def safe_mention_text(m):
    if isinstance(m, str):
        return m
    if isinstance(m, dict):
        return m.get("text") or m.get("span_text") or ""
    if isinstance(m, (list, tuple)):
        return m[2] if len(m) > 2 else ""
    return str(m)


def extract_text_from_pdf_robust(pdf_path):
    """Robust text extraction: handles PDFs and plain .txt files."""
    from pathlib import Path
    import mimetypes
    import PyPDF2

    p = Path(pdf_path)
    print(f"📄 Extracting text from: {p}")
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    # 1) Accept plain text directly
    mtype, _ = mimetypes.guess_type(str(p))
    if str(p).lower().endswith(".txt") or (mtype and mtype.startswith("text/")):
        return p.read_text(encoding="utf-8", errors="ignore")

    # 2) Parse PDF
    text = ""
    pages_extracted = 0
    try:
        with p.open("rb") as fh:
            reader = PyPDF2.PdfReader(fh)

            # Try decrypt if needed (empty password)
            if getattr(reader, "is_encrypted", False):
                print("🔒 Encrypted PDF: attempting blank-password decrypt")
                try:
                    reader.decrypt("")
                except Exception:
                    pass  # continue; some readers still allow reading

            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    t = page.extract_text() or ""
                    if t.strip():
                        text += t + "\n\n"
                        pages_extracted += 1
                except Exception as e:
                    print(f"  ⚠️ Page {page_num}: {e}")

        print(f"✅ Extracted {pages_extracted}/{len(reader.pages)} pages, {len(text)} chars")
        return text.strip()
    except Exception as e:
        raise Exception(f"PDF extraction failed: {e}")
        
def preprocess_pdf_text(text, max_length=None, return_paragraphs=False):
    """Clean and preprocess extracted PDF text"""
    print("🧹 Preprocessing PDF text…")

    # Merge hyphenated line-breaks
    text = re.sub(r'-\s*\n\s*', '', text)

    # Temporarily mark true paragraph breaks
    text = re.sub(r'\n\s*\n+', '<PARA>', text)

    # Collapse any other newlines into spaces
    text = re.sub(r'\n+', ' ', text)

    # Restore paragraph breaks
    text = text.replace('<PARA>', '\n\n')

    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text).strip()

    # Split into paragraphs and filter
    paras = text.split('\n\n')
    cleaned_paras = []
    for para in paras:
        p = para.strip()
        if len(p) < 20 or p.isdigit():
            continue
        cleaned_paras.append(p)

    # Reconstruct full text
    cleaned_text = '\n\n'.join(cleaned_paras)

    # Truncate if needed
    if max_length and len(cleaned_text) > max_length:
        snippet = cleaned_text[:max_length]
        cut = snippet.rfind('.')
        cleaned_text = (snippet[:cut+1] if cut >
                        max_length * 0.8 else snippet).strip()
        print(f"⚠️  Text truncated to {len(cleaned_text)} characters")
        cleaned_paras = cleaned_text.split('\n\n')

    print(
        f"✅ Text preprocessing complete: {len(cleaned_text)} characters, {len(cleaned_paras)} paragraphs")
    return cleaned_paras if return_paragraphs else cleaned_text

# =============================================================================
# Sentence Filtering Functions
# =============================================================================

def is_citation_or_reference(sentence):
    """Enhanced citation and reference detection"""
    citation_patterns = [
        r'\([^)]*\d{4}[^)]*\)', r'\[\d+\]', r'\b[A-Z][a-z]+\s+et\s+al\.',
        r'^\s*\d+\.', r'^\s*[A-Z]\.', r'doi\.org|DOI:|http://|https://|www\.',
        r'^\s*Fig\.|^\s*Figure|^\s*Table|^\s*Eq\.',
        r'See\s+(Fig|Figure|Table|Section|Appendix)',
        r'^\s*References?\s*$', r'^\s*Bibliography\s*$'
    ]
    return any(re.search(pattern, sentence, re.IGNORECASE) for pattern in citation_patterns)


def has_meaningful_content(sentence):
    """Check if sentence has meaningful content"""
    cleaned = re.sub(r'\([^)]*\d{4}[^)]*\)', '', sentence)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    words = cleaned.split()
    meaningful_words = [w for w in words if len(w) > 2 and not w.isdigit()]

    if len(meaningful_words) < 5:
        return False

    common_verbs = [
        'is', 'are', 'was', 'were', 'has', 'have', 'had', 'can', 'could', 'will', 'would',
        'show', 'shows', 'indicate', 'suggest', 'demonstrate', 'reveal', 'provide', 'present'
    ]

    has_verb = any(verb in cleaned.lower() for verb in common_verbs)
    ends_properly = sentence.strip().endswith(('.', '!', '?', ';'))

    return has_verb and ends_properly


def extract_and_filter_sentences(text):
    """Extract and filter sentences from text"""
    print("✂️  Extracting and filtering sentences...")

    raw_sentences = sent_tokenize(text)
    filtered_sentences = []

    for sentence in raw_sentences:
        sentence = re.sub(r'\s+', ' ', sentence).strip()

        if (len(sentence) < 20 or
            is_citation_or_reference(sentence) or
            not has_meaningful_content(sentence) or
                len(sentence.split()) < 6):
            continue

        filtered_sentences.append(sentence)

    print(
        f"✅ Filtered {len(filtered_sentences)} valid sentences from {len(raw_sentences)} raw")
    return filtered_sentences


def prep_text(text):
    """Text preprocessing for BERT"""
    clean_sents = []
    sent_tokens = sent_tokenize(str(text))
    for sent_token in sent_tokens:
        word_tokens = [str(word_token).strip().lower()
                       for word_token in sent_token.split()]
        clean_sents.append(' '.join(word_tokens))
    joined = ' '.join(clean_sents).strip(' ')
    joined = re.sub(r'`', "", joined)
    joined = re.sub(r'"', "", joined)
    return joined

# =============================================================================
# Classification Functions
# =============================================================================


def classify_sentence_bert(sentence):
    """Classify sentence using BERT model"""
    inputs = bert_tokenizer(sentence, return_tensors="pt",
                            truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = bert_model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze(0)
        label = int(torch.argmax(probs))
        conf = float(probs[label])
    return label, conf


def classify_sentence_similarity(sentence,
                                 sdg_targets: Dict[str,str],
                                 threshold: float = 0.6):
    """
    sdg_targets: e.g. {
      '1.1': 'By 2030, eradicate extreme poverty for all people…',
      '1.2': 'By 2030, reduce at least by half the proportion…',
      …
    }
    Returns (goal_level:int, target_code:str, score:float) or (None, None, score)
    """
    # 1. turn dict into two parallel lists
    codes = list(sdg_targets.keys())                 # ['1.1', '1.2', …]
    texts = [ sdg_targets[c] for c in codes ]        # corresponding descriptions

    # 2. embed your sentence + targets
    sent_emb    = sim_model.encode(sentence,   convert_to_tensor=True, device=device)
    target_embs = sim_model.encode(texts,      convert_to_tensor=True, device=device)

    # 3. compute cosine similarities
    sims     = util.cos_sim(sent_emb, target_embs)[0]  # 169‑dim vector
    best_idx = int(sims.argmax())                      # index in codes/texts
    best_score = float(sims[best_idx])

    # 4. thresholding
    if best_score < threshold:
        return None, None, best_score

    # 5. pull out the code and level
    selected_code  = codes[best_idx]                   # e.g. '1.3'
    goal_level     = int(selected_code.split('.')[0])  # e.g. 1

    return goal_level, selected_code, best_score


def determine_dual_consensus(bert_label, bert_conf, sim_label, sim_conf,
                             agree_thresh=0.1, disagree_thresh=0.2, min_conf=0.5):
    """Determine consensus between BERT and similarity approaches"""
    if sim_label is None or sim_label < 0:
        return 'bert_only' if bert_conf >= min_conf else 'mistrust'
    if bert_conf < min_conf and sim_conf < min_conf:
        return 'mistrust'
    if bert_label == sim_label and bert_conf >= min_conf and sim_conf >= min_conf:
        return 'agreement'
    diff = abs(bert_conf - sim_conf)
    if diff >= disagree_thresh:
        return 'bert_only' if bert_conf > sim_conf else 'similarity_only'
    if bert_label != sim_label and diff >= agree_thresh:
        return 'disagreement'
    if bert_conf >= min_conf:
        return 'bert_only'
    if sim_conf >= min_conf:
        return 'similarity_only'
    return 'disagreement'

# =============================================================================
# Coreference Analysis Functions
# =============================================================================

def normalize_mention_string(s: str) -> str:
    s = re.sub(r'\s+', ' ', s.strip().lower())
    s = re.sub(r'[^a-z0-9 \-_/]', '', s)
    return s


def expand_to_full_phrase(text, char_start, char_end):
    """Expand span to full phrase using dependency parsing"""
    doc = nlp(text)
    target_token = None

    for token in doc:
        if token.idx <= char_start < token.idx + len(token.text):
            target_token = token
            break

    if target_token is None:
        return text[char_start:char_end], (char_start, char_end)

    # Get the subtree
    subtree = list(target_token.subtree)
    start = subtree[0].idx
    end = subtree[-1].idx + len(subtree[-1].text)

    # Try to extend for modifiers
    sent = target_token.sent
    i = subtree[-1].i + 1
    while i < len(doc) and doc[i].idx < sent.end_char:
        if doc[i].dep_ in ("prep", "amod", "advmod", "compound", "pobj"):
            end = doc[i].idx + len(doc[i].text)
        else:
            break
        i += 1

    return text[start:end], (start, end)

def _fc_as_char_clusters(coref_res, text):
    """
    Normalize FastCoref outputs to: List[List[(start_char, end_char)]]
    Works with CorefResult from fastcoref>=2.x and older list-like outputs.
    """
    # Case A: we got a CorefResult
    if hasattr(coref_res, "get_clusters"):
        # Try to get spans directly (not strings)
        try:
            clusters = coref_res.get_clusters(as_strings=False)
        except TypeError:
            clusters = coref_res.get_clusters()  # some versions ignore the kwarg

        # Validate that we have [(s,e)] ints; if not, fall back to strings → locate
        if isinstance(clusters, (list, tuple)) and clusters:
            m0 = clusters[0][0] if clusters[0] else None
            if isinstance(m0, (list, tuple)) and len(m0) == 2 and all(isinstance(x, int) for x in m0):
                return clusters

        # Fallback: strings → approximate char positions (greedy left→right)
        try:
            str_clusters = coref_res.get_clusters(as_strings=True)
            out = []
            for cl in str_clusters:
                spans = []
                cursor = 0
                for s in cl:
                    i = text.find(s, cursor)
                    if i < 0:  # fallback: first occurrence
                        i = text.find(s)
                    if i >= 0:
                        spans.append((i, i + len(s)))
                        cursor = i + len(s)
                if spans:
                    out.append(spans)
            return out
        except Exception:
            return []

    # Case B: old API returning plain lists already
    if isinstance(coref_res, (list, tuple)):
        return [list(map(lambda p: (int(p[0]), int(p[1])), cl or [])) for cl in coref_res]

    return []

def _fastcoref_in_windows(full_text: str, k_sentences=3, stride=2):
    doc = nlp(full_text)
    sents = list(doc.sents)

    windows = []
    i = 0
    while i < len(sents):
        win = sents[i:i+k_sentences]
        if not win: break
        start = win[0].start_char
        end   = win[-1].end_char
        windows.append((start, end, full_text[start:end]))
        i += stride

    clusters_abs = []
    for (start, end, chunk) in windows:
        res = _fastcoref.predict([chunk])          # list[CorefResult]
        coref_res = res[0]
        clust = _fc_as_char_clusters(coref_res, chunk)
        clusters_abs.extend([[(start + s, start + e) for (s, e) in c] for c in clust])

    return clusters_abs


def analyze_full_text_coreferences(full_text: str):
    """
    Coreference via FastCoref, returning:
    {"chains": [{"chain_id": int, "representative": str, "mentions": [
        {"text": str, "start_char": int, "end_char": int}
    ]}, ...]}
    """
    if not full_text or not full_text.strip():
        return {"chains": []}

    try:
        # Whole-document mode:
        res = _fastcoref.predict([full_text])   # list[CorefResult]
        clusters = _fc_as_char_clusters(res[0], full_text)

        # If you prefer local-only coref, comment the two lines above and use:
        # clusters = _fastcoref_in_windows(full_text, k_sentences=3, stride=2)

    except Exception as e:
        return {"chains": [], "error": f"fastcoref failed: {e}"}

    chains_data = []
    for cid, cluster in enumerate(clusters or []):
        if not cluster: 
            continue
        mentions = [{"text": full_text[s:e], "start_char": s, "end_char": e}
                    for (s, e) in cluster if 0 <= s < e <= len(full_text)]
        if not mentions:
            continue
        representative = max(mentions, key=lambda m: len(m["text"]))["text"]
        chains_data.append({
            "chain_id": cid,
            "representative": representative,
            "mentions": mentions
        })

    return {"chains": chains_data}

def merge_and_sort_spans(*span_lists):
    """
    Given any number of span‐importance lists (each a list of dicts
    with 'start_char', 'end_char', 'importance'), dedupe on span coords
    and return a single list sorted by descending importance.
    """
    merged = {}
    for lst in span_lists:
        for span in lst:
            key = (span['start_char'], span['end_char'])
            # keep the highest‐importance if duplicates occur
            if key not in merged or span['importance'] > merged[key]['importance']:
                merged[key] = span
    return sorted(merged.values(), key=lambda s: -s['importance'])


def normalize_span_for_chaining(sentence, local_start, local_end):
    """Generate candidate spans for coreference matching"""
    doc = nlp(sentence)
    span_text = sentence[local_start:local_end]
    span_doc = nlp(span_text)
    candidates = []

    # COMPOUND‐NOUN AWARENESS
    for tok in span_doc:
        if tok.pos_ == "NOUN" and tok.dep_ == "compound":
            for nc in doc.noun_chunks:
                if nc.start <= tok.i < nc.end:
                    # full compound phrase
                    candidates.append((nc.text, nc.start_char, nc.end_char))
                    # head noun of that compound
                    head = tok.head
                    candidates.append(
                        (head.text, head.idx, head.idx + len(head.text)))
                    break

    # Rule 1: If span lacks any NOUN, expand to nearest noun phrase
    if not any(t.pos_ == "NOUN" for t in span_doc):
        anchor = next((t for t in doc if t.idx == local_start), None)
        if anchor:
            left = anchor
            while left.i > 0 and left.pos_ != "NOUN":
                left = doc[left.i - 1]
            for nc in doc.noun_chunks:
                if nc.start <= left.i < nc.end:
                    candidates.append((nc.text, nc.start_char, nc.end_char))
                    break

    # Rule 2: If span has no subject/object or root not noun, include subj & obj heads
    root = next((tok for tok in span_doc if tok.dep_ == 'ROOT'), None)
    if root:
        sent = root.sent
        subj = next((t for t in sent if t.dep_ == 'nsubj'), None)
        obj = next((t for t in sent if t.dep_ in ('dobj', 'obj')), None)
        if root.pos_ != 'NOUN' or not ({t.dep_ for t in span_doc} & {'nsubj', 'dobj', 'obj'}):
            if subj:
                candidates.append(
                    (subj.text, subj.idx, subj.idx + len(subj.text)))
            if obj:
                candidates.append(
                    (obj.text,  obj.idx,  obj.idx + len(obj.text)))

    # Rule 3: If span is mostly stopwords, strip them and retry
    words = span_text.split()
    if words and all(w.lower() in STOP_WORDS for w in words[:2] + words[-2:]):
        stripped = ' '.join([w for w in words if w.lower() not in STOP_WORDS])
        pos = sentence.find(stripped)
        if stripped:
            # use spaCy to get exact char offset inside sentence
            stripped_doc = nlp(stripped)
            if stripped_doc:
                s_off = stripped_doc[0].idx
                e_off = s_off + len(stripped)
                candidates.append((stripped, s_off, e_off))
    # Rule 4 & 5: For any noun_chunks inside the span, add full chunk + each noun token
    for nc in span_doc.noun_chunks:
        # full chunk
        s_abs = local_start + nc.start_char
        e_abs = local_start + nc.end_char
        candidates.append((nc.text, s_abs, e_abs))
        # each noun in the chunk
        for tok in nc:
            if tok.pos_ == 'NOUN':
                # token.idx is offset in 'sentence'
                s_off = tok.idx
                e_off = s_off + len(tok.text)
                candidates.append((tok.text, s_off, e_off))
    # Always include the original span
    candidates.append((span_text, local_start, local_end))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for text, s, e in candidates:
        key = (text, s, e)
        if key not in seen:
            seen.add(key)
            unique.append((text, s, e))
    return unique

def find_span_coreferences_fixed(full_text, span_text, span_coords):
    """Find coreferences for a span using fixed API"""
    doc = nlp(full_text)

    if not hasattr(doc._, 'coref_chains'):
        return {"chain_found": False, "error": "Coreferee not loaded", "related_mentions": []}

    # Find target tokens
    target_tokens = []
    for token in doc:
        if (token.idx >= span_coords[0] and token.idx < span_coords[1]) or \
           (token.idx < span_coords[0] and token.idx + len(token.text) > span_coords[0]):
            target_tokens.append(token)

    if not target_tokens:
        return {"chain_found": False, "message": "No tokens found", "related_mentions": []}

    # Check document-level chains
    for chain_idx, chain in enumerate(doc._.coref_chains):
        target_in_chain = False

        for target_token in target_tokens:
            for mention in chain:
                try:
                    if hasattr(mention, 'token_indexes'):
                        token_indices = mention.token_indexes
                    else:
                        token_indices = list(mention)

                    if target_token.i in token_indices:
                        target_in_chain = True
                        break
                except:
                    continue

            if target_in_chain:
                break

        if target_in_chain:
            # Collect related mentions
            related_mentions = []

            for mention in chain:
                try:
                    if hasattr(mention, 'token_indexes'):
                        token_indices = mention.token_indexes
                    else:
                        token_indices = list(mention)

                    mention_tokens = [doc[i] for i in token_indices if 0 <= i < len(doc)]

                    if not mention_tokens:
                        continue

                    mention_start = mention_tokens[0].idx
                    mention_end = mention_tokens[-1].idx + len(mention_tokens[-1].text)

                    # Skip if overlaps with original span
                    if not (mention_start <= span_coords[0] < mention_end or
                           span_coords[0] <= mention_start < span_coords[1]):

                        mention_text = " ".join([t.text for t in mention_tokens])
                        sent = mention_tokens[0].sent
                        sent_idx = list(doc.sents).index(sent)

                        related_mentions.append({
                            "text": mention_text,
                            "start_char": mention_start,
                            "end_char": mention_end,
                            "sentence_idx": sent_idx,
                            "sentence_text": sent.text,
                            "token_indices": token_indices
                        })

                except Exception as e:
                    continue

            # Try to resolve representative
            try:
                resolved = doc._.coref_chains.resolve(target_tokens[0])
                representative = " ".join([t.text for t in resolved]) if resolved else span_text
            except:
                representative = span_text

            return {
                "chain_found": True,
                "chain_id": chain_idx,
                "representative": representative,
                "related_mentions": related_mentions,
                "total_mentions": len(related_mentions) + 1
            }

    return {"chain_found": False, "message": "No coreference chain found", "related_mentions": []}

def extract_sentences_from_coreference_chain(full_text, span_text, span_coords):
    """Extract all sentences containing mentions from the same coreference chain"""
    coref_result = find_span_coreferences_fixed(full_text, span_text, span_coords)

    if not coref_result["chain_found"]:
        return {
            "found": False,
            "error": coref_result.get("message", "No chain found"),
            "sentences": []
        }

    # Collect sentences from all related mentions
    sentences_with_mentions = []
    sentence_indices = set()

    # Add original sentence
    doc = nlp(full_text)
    for token in doc:
        if span_coords[0] <= token.idx < span_coords[1]:
            sent = token.sent
            sent_idx = list(doc.sents).index(sent)
            if sent_idx not in sentence_indices:
                sentence_indices.add(sent_idx)
                sentences_with_mentions.append({
                    "sentence_idx": sent_idx,
                    "sentence_text": sent.text,
                    "mention_text": span_text,
                    "is_target_sentence": True
                })
            break

    # Add sentences from related mentions
    for mention in coref_result["related_mentions"]:
        sent_idx = mention["sentence_idx"]
        if sent_idx not in sentence_indices:
            sentence_indices.add(sent_idx)
            sentences_with_mentions.append({
                "sentence_idx": sent_idx,
                "sentence_text": mention["sentence_text"],
                "mention_text": mention["text"],
                "is_target_sentence": False
            })

    # Sort by sentence order
    sentences_with_mentions.sort(key=lambda x: x["sentence_idx"])

    return {
        "found": True,
        "target_span": span_text,
        "representative": coref_result["representative"],
        "num_sentences": len(sentences_with_mentions),
        "sentences": sentences_with_mentions
    }

def extract_sentences_from_coreference_chain(full_text, span_text, span_coords):
    """Extract all sentences containing mentions from the same coreference chain"""
    coref_result = find_span_coreferences_fixed(full_text, span_text, span_coords)

    if not coref_result["chain_found"]:
        return {
            "found": False,
            "error": coref_result.get("message", "No chain found"),
            "sentences": []
        }

    # Collect sentences from all related mentions
    sentences_with_mentions = []
    sentence_indices = set()

    # Add original sentence
    doc = nlp(full_text)
    for token in doc:
        if span_coords[0] <= token.idx < span_coords[1]:
            sent = token.sent
            sent_idx = list(doc.sents).index(sent)
            if sent_idx not in sentence_indices:
                sentence_indices.add(sent_idx)
                sentences_with_mentions.append({
                    "sentence_idx": sent_idx,
                    "sentence_text": sent.text,
                    "mention_text": span_text,
                    "is_target_sentence": True
                })
            break

    # Add sentences from related mentions
    for mention in coref_result["related_mentions"]:
        sent_idx = mention["sentence_idx"]
        if sent_idx not in sentence_indices:
            sentence_indices.add(sent_idx)
            sentences_with_mentions.append({
                "sentence_idx": sent_idx,
                "sentence_text": mention["sentence_text"],
                "mention_text": mention["text"],
                "is_target_sentence": False
            })

    # Sort by sentence order
    sentences_with_mentions.sort(key=lambda x: x["sentence_idx"])

    return {
        "found": True,
        "target_span": span_text,
        "representative": coref_result["representative"],
        "num_sentences": len(sentences_with_mentions),
        "sentences": sentences_with_mentions
    }


# =============================================================================
# Visualization Functions
# =============================================================================
# File: helper.py

import string
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# reuse a single SentenceTransformer instance
_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
PRONOUNS = {'it','its','they','them','their','itself'}
def _mention_tuple(m):
    # returns (start, end, text)
    if isinstance(m, dict):
        return m['start_char'], m['end_char'], m['text']
    return m  # already a tuple

# helper.py (top of the clustering section)
import unicodedata, string, re

def _norm_text(x):
    """Robust text normalizer for clustering/coref."""
    # 1) extract text if dict-like
    if isinstance(x, dict):
        x = x.get('text', '')
    elif isinstance(x, (list, tuple)):
        x = ' '.join(map(str, x))
    else:
        x = str(x)

    # 2) unicode & whitespace normalization
    x = unicodedata.normalize('NFKC', x)
    x = re.sub(r'\s+', ' ', x)

    # 3) strip punctuation/space and lowercase
    x = x.strip(string.punctuation + string.whitespace).lower()
    return x

def list_and_filter_coref_clusters_from_kpe(sentence_analyses):
    """
    Build clusters from each keyphrase/span coreference_analysis.
    Keeps only chains that link across ≥2 sentences & have ≥1 non-pronoun mention.
    Expects sentence_analyses = [
      {
        'sentence_id': int,
        'sentence_text': str,
        'keyphrase_analysis' or 'span_analysis': List[{
            'phrase' or 'span': str,
            'expanded_phrase': str,
            'coreference_analysis': {
               'chain_found': bool,
               'chain_id': int,
               …
            }
        }]
      }, …
    ]
    Returns list of {chain_id, mentions, sentences}.
    """
    raw = {}
    for sa in sentence_analyses:
        sent_text = sa['sentence_text']
        for kp in sa.get('keyphrase_analysis', sa.get('span_analysis', [])):
            cr = kp['coreference_analysis']
            if not cr.get('chain_found'): 
                continue
            cid = cr['chain_id']
            mention = kp.get('expanded_phrase', kp.get('phrase', kp.get('span')))
            raw_txt = safe_mention_text(mention)
            clean   = raw_txt.strip().strip(string.punctuation).lower()

            if (len(clean)<2
                or clean in PRONOUNS
                or not any(c.isalpha() for c in clean)):
                continue
            entry = raw.setdefault(cid, {'mentions': set(), 'sentences': set()})
            entry['mentions'].add(clean)
            entry['sentences'].add(sent_text)

    clusters = []
    for cid, data in raw.items():
        if len(data['sentences']) < 2:
            continue
        clusters.append({
            'chain_id':  cid,
            'mentions':  sorted(data['mentions']),
            'sentences': list(data['sentences'])
        })
    return clusters


def build_cluster_graphs(clusters_dict):
    """
    Given clusters_dict: { chain_id: {'mentions':…, 'sentences':[…]}, … }
    returns { chain_id: networkx.Graph } where nodes are sentences,
    edges exist when cosine-sim > 0.1 in the SentenceTransformer embedding space.
    """
    graphs = {}
    for cid, data in clusters_dict.items():
        sents = data['sentences']
        if len(sents) < 2:
            continue
        emb = _model.encode(sents, convert_to_numpy=True)
        sim = cosine_similarity(emb)

        G = nx.Graph()
        for i, text in enumerate(sents):
            G.add_node(i, text=text)
        for i in range(len(sents)):
            for j in range(i+1, len(sents)):
                if sim[i, j] > 0.1:
                    G.add_edge(i, j, weight=float(sim[i, j]))
        graphs[cid] = G
    return graphs


def render_cluster_graph(cluster_id, clusters, cluster_graphs, focus_idx=None):
    """
    Display cluster graph using Plotly for interactivity
    """
    import plotly.graph_objects as go
    import networkx as nx
    from IPython.display import display
    
    G_full = cluster_graphs[cluster_id]
    info = next(c for c in clusters if c['chain_id'] == cluster_id)
    sentences, mentions = info['sentences'], info['mentions']
    
    G = G_full
    if focus_idx is not None and focus_idx in G_full:
        nodes = [focus_idx] + list(G_full.neighbors(focus_idx))
        G = G_full.subgraph(nodes)
    
    # Get layout
    pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        # Full sentence as hover text
        sent = sentences[node] if node < len(sentences) else f"Node {node}"
        # Truncate for display
        display_text = sent[:80] + "..." if len(sent) > 80 else sent
        node_text.append(f"[{node}] {display_text}")
        # Highlight focus node
        node_colors.append('red' if node == focus_idx else 'lightblue')
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=node_text,
        text=[str(i) for i in G.nodes()],
        textposition="middle center",
        marker=dict(
            color=node_colors,
            size=20,
            line=dict(color='darkblue', width=2)
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f"Cluster {cluster_id} — mentions: {', '.join(mentions)}",
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=500
                    ))
    
    display(fig)
    
    print("\nSentences in this cluster:")
    for idx, s in enumerate(sentences):
        tag = " ← FOCUS" if idx == focus_idx else ""
        print(f" [{idx}] {s[:100]}...{tag}")

# ========================================
# ANALYSIS OF YOUR render_cluster_graph FUNCTION
# ========================================

# ✅ WHAT'S CORRECT:
# - fig.show() is the right way to display plotly figures
# - Basic structure and plotly syntax are correct
# - Layout and traces look properly configured

# ❌ POTENTIAL ISSUES:

# Issue 1: No error handling
# Issue 2: Mentions data type assumption
# Issue 3: Node-sentence index mismatch
# Issue 4: No empty graph check
# Issue 5: Cluster lookup could fail

# ========================================
# IMPROVED VERSION WITH DEBUGGING
# ========================================

def render_cluster_graph_fixed(cluster_id, clusters, cluster_graphs, focus_idx=None):
    """
    Display cluster graph using Plotly for interactivity - IMPROVED VERSION
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        import networkx as nx
        from plotly.offline import init_notebook_mode
        
        init_notebook_mode(connected=True)
        # Option A: Jupyter notebook renderer
        #pio.renderers.default = "notebook"

        # Option B: If in JupyterLab
        # pio.renderers.default = "jupyterlab"

        # Option C: Browser renderer (opens in new tab)
        # pio.renderers.default = "browser"
        
        # Option D: Colab renderer (if using Google Colab)
        pio.renderers.default = "colab"
        
        print(f"🔍 DEBUG: Rendering graph for cluster {cluster_id}")
        
        # Check if cluster exists in graphs
        if cluster_id not in cluster_graphs:
            available_ids = list(cluster_graphs.keys())
            print(f"❌ Cluster {cluster_id} not found in graphs!")
            print(f"💡 Available cluster IDs: {available_ids}")
            return
        
        G_full = cluster_graphs[cluster_id]
        print(f"🔍 DEBUG: Graph has {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges")
        
        # Check if graph is empty
        if G_full.number_of_nodes() == 0:
            print(f"❌ Graph for cluster {cluster_id} is empty!")
            return
        
        # Find cluster info
        info = None
        for c in clusters:
            if c.get('chain_id') == cluster_id:
                info = c
                break
        
        if info is None:
            print(f"❌ Cluster info not found for ID {cluster_id}")
            print(f"💡 Available cluster info IDs: {[c.get('chain_id') for c in clusters]}")
            return
        
        sentences = info.get('sentences', [])
        mentions = info.get('mentions', [])
        
        print(f"🔍 DEBUG: Cluster has {len(sentences)} sentences, {len(mentions)} mentions")
        
        # Handle focus subgraph
        G = G_full
        if focus_idx is not None and focus_idx in G_full:
            nodes = [focus_idx] + list(G_full.neighbors(focus_idx))
            G = G_full.subgraph(nodes)
            print(f"🔍 DEBUG: Using subgraph focused on node {focus_idx}: {G.number_of_nodes()} nodes")
        
        # Get layout
        pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
        
        # Create edge trace
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='Connections'
        )
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        hover_text = []
        node_colors = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Safe sentence access
            if node < len(sentences):
                sent = sentences[node]
                # Truncate for display
                display_text = sent[:60] + "..." if len(sent) > 60 else sent
                hover_text.append(f"<b>Sentence {node}</b><br>{sent[:200]}{'...' if len(sent) > 200 else ''}")
            else:
                display_text = f"Node {node}"
                hover_text.append(f"Node {node} (no sentence data)")
            
            node_text.append(str(node))
            
            # Color coding
            if node == focus_idx:
                node_colors.append('red')
            elif node < len(sentences):
                node_colors.append('lightblue')
            else:
                node_colors.append('lightgray')
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            hovertext=hover_text,
            text=node_text,
            textposition="middle center",
            marker=dict(
                color=node_colors,
                size=25,
                line=dict(color='darkblue', width=2)
            ),
            name='Sentences'
        )
        
        # Safe mentions handling
        mention_texts = []
        for m in mentions:
            if isinstance(m, dict):
                mention_texts.append(m.get('text', str(m)))
            elif isinstance(m, str):
                mention_texts.append(m)
            else:
                mention_texts.append(str(m))
        
        # Limit mentions in title
        mention_display = ', '.join(mention_texts[:3])
        if len(mention_texts) > 3:
            mention_display += f' (+{len(mention_texts)-3} more)'
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])
        
        fig.update_layout(
            title=dict(
                text=f"<b>Cluster {cluster_id}</b><br><sub>Mentions: {mention_display}</sub>",
                font=dict(size=14),
                x=0.5
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=40, l=5, r=5, t=80),
            annotations=[
                dict(
                    text=f"📊 {G.number_of_nodes()} sentences • {G.number_of_edges()} connections",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.05,
                    xanchor='center', yanchor='top',
                    font=dict(color="gray", size=10)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500,
            plot_bgcolor='white'
        )
        
        print("📊 Displaying plotly graph...")
        
        # DISPLAY THE FIGURE - This should work!
        fig.show(renderer="colab")
        
        # Alternative display methods for debugging
        # Uncomment these if fig.show() doesn't work:
        
        # Method 1: IPython display
        # from IPython.display import display
        #display(fig)
        
        # Method 2: Return figure for manual display
        # return fig
        
        # Print sentence details
        print(f"\n📝 Sentences in cluster {cluster_id}:")
        print("-" * 60)
        for idx, s in enumerate(sentences):
            tag = " ← FOCUS" if idx == focus_idx else ""
            status = "🟢" if idx < G.number_of_nodes() else "⚪"
            print(f" {status} [{idx}] {s[:80]}{'...' if len(s) > 80 else ''}{tag}")
        
        return fig  # Return figure in case manual display is needed
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Install plotly with: pip install plotly")
        return None
    except Exception as e:
        print(f"❌ Error rendering graph: {e}")
        import traceback
        traceback.print_exc()
        return None


# ========================================
# DIAGNOSTIC VERSION - EXTRA DEBUGGING
# ========================================

def render_cluster_graph_debug(cluster_id, clusters, cluster_graphs, focus_idx=None):
    """
    DIAGNOSTIC VERSION with extensive debugging
    """
    print("🔍" + "="*50)
    print(f"🔍 DEBUGGING render_cluster_graph for cluster {cluster_id}")
    print("🔍" + "="*50)
    
    # Step 1: Check inputs
    print(f"📊 Input cluster_id: {cluster_id} (type: {type(cluster_id)})")
    print(f"📊 Available cluster_graphs keys: {list(cluster_graphs.keys())}")
    print(f"📊 Number of clusters in list: {len(clusters)}")
    
    # Step 2: Check cluster existence
    if cluster_id not in cluster_graphs:
        print(f"❌ PROBLEM: Cluster {cluster_id} not in cluster_graphs!")
        return None
    
    # Step 3: Check graph
    G = cluster_graphs[cluster_id]
    print(f"📊 Graph type: {type(G)}")
    print(f"📊 Graph nodes: {G.number_of_nodes()}")
    print(f"📊 Graph edges: {G.number_of_edges()}")
    print(f"📊 Node list: {list(G.nodes())}")
    
    # Step 4: Check cluster info
    cluster_info = None
    for c in clusters:
        if c.get('chain_id') == cluster_id:
            cluster_info = c
            break
    
    if cluster_info:
        print(f"📊 Cluster info found: {list(cluster_info.keys())}")
        sentences = cluster_info.get('sentences', [])
        print(f"📊 Number of sentences: {len(sentences)}")
        for i, s in enumerate(sentences[:3]):
            print(f"   S{i}: {s[:50]}...")
    else:
        print(f"❌ PROBLEM: No cluster info found for {cluster_id}")
        return None
    
    # Step 5: Try plotting
    try:
        import plotly.graph_objects as go
        print("✅ Plotly imported successfully")
        
        # Simple test plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3], mode='markers'))
        fig.update_layout(title="Test Plot")
        
        print("📊 Showing test plot...")
        fig.show()
        
        print("✅ Test plot worked! Now trying real graph...")
        
        # Now try the real graph
        return render_cluster_graph_fixed(cluster_id, clusters, cluster_graphs, focus_idx)
        
    except Exception as e:
        print(f"❌ Plotly error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ========================================
# USAGE EXAMPLES
# ========================================

# Replace your current function call with:
# render_cluster_graph_fixed(cid, clusters_list, graphs_nx)

# Or for debugging:
# render_cluster_graph_debug(cid, clusters_list, graphs_nx)

# If fig.show() still doesn't work, try manual display:
# fig = render_cluster_graph_fixed(cid, clusters_list, graphs_nx)
# if fig:
#     from IPython.display import display
#     display(fig)

def print_explanation_summary(pipeline_results):
    """Print summary statistics and consensus distribution"""
    counts = {}

    # Determine if this is keyphrase or span analysis
    sample_result = pipeline_results["results"][0] if pipeline_results["results"] else None
    is_keyphrase = sample_result and "keyphrase_analysis" in sample_result

    if is_keyphrase:
        keyphrase_counts = {"with_coreference": 0, "without_coreference": 0}

        for r in pipeline_results["results"]:
            c = r["consensus"]
            counts[c] = counts.get(c, 0) + 1

            # Count keyphrases with/without coreference
            for kp in r["keyphrase_analysis"]:
                if kp["coreference_analysis"]["chain_found"]:
                    keyphrase_counts["with_coreference"] += 1
                else:
                    keyphrase_counts["without_coreference"] += 1

        analysis_type = "Keyphrase"
        analysis_counts = keyphrase_counts

    else:
        span_counts = {"with_coreference": 0, "without_coreference": 0}

        for r in pipeline_results["results"]:
            c = r["consensus"]
            counts[c] = counts.get(c, 0) + 1

            # Count spans with/without coreference
            for span in r["span_analysis"]:
                if span["coreference_analysis"]["chain_found"]:
                    span_counts["with_coreference"] += 1
                else:
                    span_counts["without_coreference"] += 1

        analysis_type = "Span"
        analysis_counts = span_counts

    # Plot consensus distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Consensus distribution
    labels = list(counts.keys())
    vals = list(counts.values())
    ax1.bar(labels, vals)
    ax1.set_title("Classification Consensus Distribution")
    ax1.set_ylabel("Number of Sentences")
    ax1.tick_params(axis='x', rotation=45)

    # Analysis-coreference distribution
    analysis_labels = list(analysis_counts.keys())
    analysis_vals = list(analysis_counts.values())
    ax2.bar(analysis_labels, analysis_vals, color=['green', 'orange'])
    ax2.set_title(f"{analysis_type} Coreference Analysis")
    ax2.set_ylabel(f"Number of {analysis_type}s")
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    total_sentences = len(pipeline_results["results"])
    total_chains = len(pipeline_results["coref_chains"])

    print(f"\n📊 PIPELINE SUMMARY ({analysis_type} Analysis)")
    print("=" * 50)
    print(f"📄 Total sentences processed: {total_sentences}")
    print(f"🔗 Coreference chains found: {total_chains}")

    if is_keyphrase:
        kp_summary = pipeline_results["keyphrase_summary"]
        print(
            f"🔑 Sentences with keyphrases: {kp_summary['total_sentences_with_keyphrases']}")
        print(
            f"🔑 Total keyphrases extracted: {kp_summary['total_keyphrases']}")
        print(
            f"✅ Keyphrases with coreferences: {analysis_counts['with_coreference']}")
        print(
            f"❌ Keyphrases without coreferences: {analysis_counts['without_coreference']}")
    else:
        sp_summary = pipeline_results["span_summary"]
        print(
            f"🎯 Sentences with spans: {sp_summary['total_sentences_with_spans']}")
        print(f"🎯 Total spans extracted: {sp_summary['total_spans']}")
        print(
            f"✅ Spans with coreferences: {analysis_counts['with_coreference']}")
        print(
            f"❌ Spans without coreferences: {analysis_counts['without_coreference']}")


def analyze_and_display(idx, pipeline_results):
    """Display detailed analysis for a specific sentence"""
    r = pipeline_results["results"][idx]
    sent = r["sentence"]
    cons = r["consensus"]

    print(f"📝 Sentence [{idx}]: {sent}")
    print(f"🏷️  Consensus: {cons}")
    print(
        f"🤖 BERT: Label {r['primary_result']['label']}, Confidence {r['primary_result']['confidence']:.3f}")
    print(
        f"🔍 Similarity: Label {r['secondary_result']['label']}, Confidence {r['secondary_result']['confidence']:.3f}")
    print("-" * 80)

    # Check if this is keyphrase or span analysis
    if "keyphrase_analysis" in r:
        # Keyphrase analysis
        if r["keyphrase_analysis"]:
            print(f"🔑 KEYPHRASE ANALYSIS:")
            print("-" * 40)

            for i, kp in enumerate(r["keyphrase_analysis"], 1):
                print(f"\n🎯 Keyphrase {i}: '{kp['phrase']}'")

                if kp['expanded_phrase'] != kp['phrase']:
                    print(f"   🔄 Expanded: '{kp['expanded_phrase']}'")

                coref = kp["coreference_analysis"]
                if coref["chain_found"]:
                    print(f"   🔗 Coreference Chain Found!")
                    print(f"      Representative: {coref['representative']}")
                    print(
                        f"      Related mentions: {len(coref['related_mentions'])}")

                    if coref['related_mentions']:
                        print(f"      Examples:")
                        for mention in coref['related_mentions'][:3]:
                            print(f"         • '{mention['text']}'")
                else:
                    print(f"   ❌ No coreference chain found")
        else:
            print("🔑 No keyphrases extracted for this sentence")

    elif "span_analysis" in r:
        # Span analysis
        if r["span_analysis"]:
            print(f"🎯 SPAN ANALYSIS:")
            print("-" * 40)

            for i, span_info in enumerate(r["span_analysis"], 1):
                span = span_info["span"]
                print(f"\n🎯 Span {i}: '{span['text']}'")
                print(f"   📊 Importance: {span['importance']:.3f}")

                if span_info['expanded_phrase']['text'] != span['text']:
                    print(
                        f"   🔄 Expanded: '{span_info['expanded_phrase']['text']}'")

                coref = span_info["coreference_analysis"]
                if coref["chain_found"]:
                    print(f"   🔗 Coreference Chain Found!")
                    print(f"      Representative: {coref['representative']}")
                    print(
                        f"      Related mentions: {len(coref['related_mentions'])}")

                    if coref['related_mentions']:
                        print(f"      Examples:")
                        for mention in coref['related_mentions'][:3]:
                            print(f"         • '{mention['text']}'")
                else:
                    print(f"   ❌ No coreference chain found")
        else:
            print("🎯 No important spans found for this sentence")

    print()


def create_selector(pipeline_results):
    """Create interactive sentence selector widget"""
    opts = [("Select a sentence...", None)]

    # Determine analysis type
    sample_result = pipeline_results["results"][0] if pipeline_results["results"] else None
    is_keyphrase = sample_result and "keyphrase_analysis" in sample_result

    for i, r in enumerate(pipeline_results["results"]):
        txt = r["sentence"][:60].replace("\n", " ") + "..."
        cons = r['consensus']
        bert_conf = r['primary_result']['confidence']

        if is_keyphrase:
            num_items = len(r['keyphrase_analysis'])
            num_coref = sum(
                1 for kp in r['keyphrase_analysis'] if kp['coreference_analysis']['chain_found'])
            label = f"{i}: {cons} | BERT={bert_conf:.2f} | KP={num_items}({num_coref}) | {txt}"
        else:
            num_items = len(r['span_analysis'])
            num_coref = sum(
                1 for span in r['span_analysis'] if span['coreference_analysis']['chain_found'])
            label = f"{i}: {cons} | BERT={bert_conf:.2f} | SP={num_items}({num_coref}) | {txt}"

        opts.append((label, i))

    dd = widgets.Dropdown(
        options=opts,
        description='Sentence:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='1000px')
    )

    out = widgets.Output()

    def on_change(change):
        out.clear_output()
        if change["new"] is not None:
            with out:
                analyze_and_display(change["new"], pipeline_results)

    dd.observe(on_change, names="value")
    

    return dd, out

# -----------------------------------------------------------------------------
# Cluster dashboard
# -----------------------------------------------------------------------------
import ipywidgets as widgets
from IPython.display import display, clear_output
from helper import _mention_tuple  # if you added it as suggested


def prepare_clusters(production_output):
    sas    = production_output['sentence_analyses']
    chains = production_output['coreference_analysis']['chains']
    full_text = production_output.get('full_text', "")

    # Build interval tree over sentences once
    sent_tree = IntervalTree([
        Interval(sa['doc_start'], sa['doc_end'], sa['sentence_id'])
        for sa in sas
    ])
    sent_by_id = {sa['sentence_id']: sa for sa in sas}

    def sid_for(pos):
        hits = sent_tree.at(pos)
        return hits[0] if hits else -1

    clusters_dict = {}

    for ch in chains:
        cid = ch['chain_id']
        mentions = []
        sent_ids_coref = set()

        for m in ch['mentions']:
            if isinstance(m, dict):
                s = m.get('start_char', m.get('start'))
                e = m.get('end_char',   m.get('end'))
                txt = m.get('text')
            else:
                # assume tuple/list (start, end, text)
                s, e, txt = m

            if txt is None and full_text and s is not None and e is not None:
                txt = full_text[s:e]

            sid = sid_for(s) if s is not None else -1
            if sid >= 0:
                sent_ids_coref.add(sid)

            mentions.append({'start': s, 'end': e, 'text': txt, 'sent_id': sid})

        # Sentences collected from span_analysis projection
        span_sids = {
            sa['sentence_id']
            for sa in sas
            for sp in sa.get('span_analysis', [])
            if sp.get('coreference_analysis') and
               sp['coreference_analysis'].get('chain_id') == cid
        }

        all_sids = sorted(sent_ids_coref | span_sids)

        clusters_dict[cid] = {
            'mentions': mentions,
            'sent_ids': all_sids,
            'sentences': [sent_by_id[sid]['sentence_text'] for sid in all_sids],
        }

    clusters = [
        {
            'chain_id': cid,
            'mentions': [m['text'] for m in data['mentions']],
            'sentences': data['sentences']
        }
        for cid, data in clusters_dict.items()
    ]

    graphs = build_cluster_graphs({
        cid: {'sentences': data['sentences']}
        for cid, data in clusters_dict.items()
        if len(data['sentences']) > 1
    })

    return clusters, clusters_dict, graphs


def create_cluster_selector(pipeline_results):
    """
    Returns (widget, output) pair to browse coref clusters and visualize their graphs.
    """
    clusters, clusters_dict, graphs = prepare_clusters(pipeline_results)
    if not clusters:
        box = widgets.HTML("<b>No multi-sentence clusters found.</b>")
        return box, widgets.Output()

    # dropdown of clusters
    opts = [("Select a cluster…", None)]
    for c in clusters:
        cid = c['chain_id']
        mention_sample = ", ".join(c['mentions'][:3])
        opts.append((f"#{cid}  |  {mention_sample}", cid))

    dd = widgets.Dropdown(options=opts, description='Cluster:',
                          style={'description_width': 'initial'},
                          layout=widgets.Layout(width='600px'))
    out = widgets.Output()

    def on_change(change):
        if change['new'] is None:
            return
        with out:
            clear_output(wait=True)
            cid = change['new']
            render_cluster_graph(cid, clusters, graphs)
    dd.observe(on_change, names='value')
    return dd, out

def create_full_dashboard(pipeline_results):
    """
    High-level convenience: sentence viewer + cluster viewer in tabs.
    Returns a top-level widget to display.
    """
    sent_sel, sent_out = create_selector(pipeline_results)
    clus_sel, clus_out = create_cluster_selector(pipeline_results)

    tab = widgets.Tab(children=[widgets.VBox([sent_sel, sent_out]),
                                widgets.VBox([clus_sel, clus_out])])
    tab.set_title(0, 'Sentences')
    tab.set_title(1, 'Coref Clusters')
    return tab
