import io
import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from datetime import datetime

import pandas as pd
import streamlit as st

try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    import fitz
except Exception:
    fitz = None

try:
    from nltk.stem import WordNetLemmatizer
except Exception:
    WordNetLemmatizer = None


BASE_LEXICON_FILE = "base_lexicon.json"
VOCAB_CORPUS_FILE = "base_vocab_corpus.json"
EXAM_DB_FILE = "past_exams_db.json"

BASE_STATUSES = {"core_verified", "exam_format", "watch_known"}
STRICT_EXCLUDED_STATUS = "strict_excluded"
NOISE_STATUS = "proper_noun_or_noise"
EXCLUDED_STATUSES = {STRICT_EXCLUDED_STATUS, NOISE_STATUS}

STATUS_LABELS = {
    "pending": "未判定",
    "core_verified": "基礎語",
    "exam_format": "試験語",
    "watch_known": "注意多義語",
    "learning_target": "単語帳対象",
    "strict_excluded": "完全除外",
    "proper_noun_or_noise": "固有名詞・ノイズ",
}

STATUS_OPTIONS = list(STATUS_LABELS.keys())

STRICT_EXCLUDED_WORDS = {
    "a", "an", "the", "and", "but", "or", "for", "nor", "on", "at", "to",
    "from", "by", "in", "of", "with", "as", "if", "then", "than", "so",
    "that", "this", "these", "those", "it", "its", "i", "you", "he", "she",
    "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "their", "our", "who", "whom", "whose", "which", "what", "when", "where",
    "why", "how", "is", "are", "am", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "can", "will", "would",
    "could", "should", "may", "might", "must", "not", "no", "yes", "very",
    "too", "all", "any", "some", "et", "al", "st", "pp", "vol", "ed", "ii",
    "iii", "iv", "vi", "vii", "viii", "ix", "x", "don", "doesn", "didn",
    "isn", "aren", "wasn", "weren", "hasn", "haven", "hadn", "won", "wouldn",
    "shouldn", "couldn", "ll", "ve", "re", "t", "s", "m", "d",
}

EXAM_FORMAT_WORDS = {
    "question", "answer", "option", "choose", "choice", "correct", "incorrect",
    "following", "passage", "paragraph", "statement", "blank", "section",
    "page", "line", "text", "chart", "graph", "figure", "table", "article",
    "conversation", "dialogue", "speaker", "interview", "email", "notice",
}

WATCH_KNOWN_WORDS = {
    "account", "address", "approach", "article", "bear", "charge", "claim",
    "class", "course", "current", "deal", "degree", "draw", "drive", "field",
    "figure", "fine", "form", "issue", "line", "matter", "mind", "object",
    "order", "passage", "plant", "present", "right", "scale", "sense", "state",
    "subject", "term", "view", "work",
}

PROPER_NOUN_OR_NOISE_WORDS = {
    "aston", "azuma", "ben", "berger", "cindy", "danny", "farnsworth", "harry",
    "hibari", "jessica", "kawanaka", "ken", "leon", "maki", "mccurdy", "michael",
    "mitsuki", "nhl", "patrick", "rca", "ryan", "sabine", "sakura", "sarah",
    "takuya", "tokyo", "tq", "uk", "york",
}

LEMMA_OVERRIDES = {
    "children": "child",
    "people": "person",
    "men": "man",
    "women": "woman",
    "teeth": "tooth",
    "feet": "foot",
    "mice": "mouse",
    "made": "make",
    "making": "make",
    "taken": "take",
    "took": "take",
    "went": "go",
    "gone": "go",
    "found": "find",
    "felt": "feel",
    "thought": "think",
    "brought": "bring",
    "bought": "buy",
    "said": "say",
    "told": "tell",
    "wrote": "write",
    "written": "write",
    "chose": "choose",
    "chosen": "choose",
}

NO_VERB_LEMMA_WORDS = {
    "clothes", "clothing", "news", "series", "species", "means", "physics",
    "economics", "mathematics", "politics", "headquarters",
}

wordnet_lemmatizer = WordNetLemmatizer() if WordNetLemmatizer else None


def normalize_word(value):
    word = str(value).strip().lower()
    word = re.sub(r"[^a-z'-]", "", word)
    word = word.strip("-'")
    return word


def split_meanings(value):
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    parts = re.split(r"[;\n；]+", text)
    return [p.strip() for p in parts if p.strip()]


def meanings_to_text(value):
    return "；".join(split_meanings(value))


def load_json_file(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json_file(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def empty_corpus():
    return {"version": 1, "documents": {}}


def load_corpus():
    corpus = load_json_file(VOCAB_CORPUS_FILE, empty_corpus())
    if not isinstance(corpus, dict):
        return empty_corpus()
    if "documents" not in corpus or not isinstance(corpus["documents"], dict):
        corpus["documents"] = {}
    corpus.setdefault("version", 1)
    return corpus


def file_digest(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def make_document_id(file_name, digest):
    base = normalize_word(os.path.splitext(file_name)[0]).replace("'", "")
    if not base:
        base = "pdf"
    return f"{base}_{digest}"


def get_secret_api_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return ""


def get_environment_status(api_key=""):
    checks = []
    checks.append(
        {
            "item": "PyMuPDF",
            "status": "OK" if fitz is not None else "NG",
            "detail": "文字情報のあるPDFを高速に抽出できます。" if fitz is not None else "PDF高速抽出が使えません。Gemini抽出を使ってください。",
        }
    )
    checks.append(
        {
            "item": "Gemini",
            "status": "OK" if genai is not None and bool(api_key) else "注意",
            "detail": "OCR寄りのPDFや文字情報が薄いPDFを抽出できます。" if genai is not None and bool(api_key) else "APIキーがない場合、Gemini抽出は使えません。",
        }
    )

    lemma_status = "OK"
    lemma_detail = "WordNetで簡易的な原形化ができます。"
    if wordnet_lemmatizer is None:
        lemma_status = "注意"
        lemma_detail = "nltk がないため、原形化は最低限の補正だけで行います。"
    else:
        try:
            wordnet_lemmatizer.lemmatize("studies", pos="n")
        except Exception:
            lemma_status = "注意"
            lemma_detail = "WordNet辞書が未準備のため、原形化は最低限の補正だけで行います。"

    checks.append({"item": "原形化", "status": lemma_status, "detail": lemma_detail})
    return checks


def render_environment_status(api_key):
    checks = get_environment_status(api_key)
    ok_count = sum(1 for c in checks if c["status"] == "OK")
    with st.expander(f"環境チェック: {ok_count}/{len(checks)} OK", expanded=ok_count < len(checks)):
        st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)
        if ok_count < len(checks):
            st.caption("注意が出ても、別の抽出方法や最低限の補正で作業を続けられるようにしています。")


def extract_text_with_pymupdf(uploaded_file):
    if fitz is None:
        raise RuntimeError("PyMuPDF が使えません。requirements.txt の PyMuPDF を確認してください。")

    pdf_bytes = uploaded_file.getvalue()
    text_parts = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text("text"))
    return "\n".join(text_parts).strip()


def extract_text_with_gemini(uploaded_file, api_key):
    if genai is None:
        raise RuntimeError("google-generativeai が使えません。requirements.txt を確認してください。")
    if not api_key:
        raise RuntimeError("Gemini API Key が必要です。")

    genai.configure(api_key=api_key)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        g_file = genai.upload_file(tmp_path)
        model = genai.GenerativeModel(model_name="gemini-2.5-pro")
        prompt = (
            "このPDFファイルは英語の試験問題です。英語本文・設問・選択肢に含まれる英語だけを"
            "できるだけ漏れなく書き起こしてください。日本語の説明や挨拶は不要です。"
        )
        res = model.generate_content([g_file, prompt])
        return str(res.text or "").strip()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def extract_text_from_pdf(uploaded_file, method, api_key):
    if method == "Gemini":
        return extract_text_with_gemini(uploaded_file, api_key), "Gemini"

    if method == "PyMuPDF":
        return extract_text_with_pymupdf(uploaded_file), "PyMuPDF"

    try:
        local_text = extract_text_with_pymupdf(uploaded_file)
    except Exception:
        return extract_text_with_gemini(uploaded_file, api_key), "Gemini fallback"

    # Text-layer PDFs are usually enough. If extraction is tiny, use Gemini as an OCR-like fallback.
    if len(local_text) >= 500:
        return local_text, "PyMuPDF"
    return extract_text_with_gemini(uploaded_file, api_key), "Gemini fallback"


def read_csv_flexibly(uploaded_file):
    raw = uploaded_file.getvalue()
    last_error = None
    for encoding in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except Exception as e:
            last_error = e
    raise last_error


def detect_word_and_count_columns(df):
    normalized_columns = {str(c).strip().lower(): c for c in df.columns}
    word_candidates = ["単語", "word", "words", "lemma", "base_form", "語"]
    count_candidates = ["出現回数", "回数", "count", "frequency", "freq", "total"]

    word_col = next((normalized_columns[c] for c in word_candidates if c in normalized_columns), None)
    count_col = next((normalized_columns[c] for c in count_candidates if c in normalized_columns), None)

    if word_col is None:
        object_cols = [c for c in df.columns if df[c].dtype == "object"]
        word_col = object_cols[0] if object_cols else df.columns[0]

    if count_col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != word_col]
        count_col = numeric_cols[0] if numeric_cols else None

    return word_col, count_col


def lemmatize_word(word):
    clean_word = normalize_word(word)
    if not clean_word:
        return ""
    if clean_word in LEMMA_OVERRIDES:
        return LEMMA_OVERRIDES[clean_word]
    if clean_word in NO_VERB_LEMMA_WORDS:
        return clean_word
    if wordnet_lemmatizer is None:
        return clean_word

    try:
        noun_lemma = wordnet_lemmatizer.lemmatize(clean_word, pos="n")
        if noun_lemma != clean_word:
            return noun_lemma
        verb_lemma = wordnet_lemmatizer.lemmatize(clean_word, pos="v")
        return verb_lemma
    except Exception:
        return clean_word


def extract_words_from_text(text):
    raw_words = re.findall(r"\b[a-zA-Z][a-zA-Z'-]*\b", text)
    words = []
    for raw_word in raw_words:
        pieces = re.split(r"['-]", raw_word.lower())
        for piece in pieces:
            lemma = lemmatize_word(piece)
            if not lemma or len(lemma) <= 1:
                continue
            words.append(lemma)
    return words


def count_words_from_text(text):
    counter = defaultdict(int)
    for word in extract_words_from_text(text):
        counter[word] += 1
    return dict(counter)


def add_frequency(counter, sources, word, count, source_label):
    clean_word = lemmatize_word(word)
    if not clean_word or len(clean_word) <= 1:
        return
    try:
        clean_count = int(count)
    except Exception:
        clean_count = 1
    if clean_count <= 0:
        return
    counter[clean_word] += clean_count
    sources[clean_word].add(source_label)


def collect_from_csv_files(uploaded_files, counter, sources):
    for uploaded_file in uploaded_files:
        df = read_csv_flexibly(uploaded_file)
        word_col, count_col = detect_word_and_count_columns(df)
        source_label = uploaded_file.name
        for _, row in df.iterrows():
            count = row[count_col] if count_col else 1
            add_frequency(counter, sources, row[word_col], count, source_label)


def collect_from_exam_db_node(node, counter, sources, path_parts):
    if isinstance(node, dict):
        frequencies = node.get("frequencies")
        if isinstance(frequencies, dict):
            source_label = " / ".join(path_parts[-5:]) or EXAM_DB_FILE
            for word, count in frequencies.items():
                add_frequency(counter, sources, word, count, source_label)
            return

        for key, value in node.items():
            collect_from_exam_db_node(value, counter, sources, path_parts + [str(key)])


def collect_from_corpus(corpus, counter, sources, selected_doc_ids=None):
    docs = corpus.get("documents", {}) if isinstance(corpus, dict) else {}
    selected = None if selected_doc_ids is None else set(selected_doc_ids)
    for doc_id, doc in docs.items():
        if selected is not None and doc_id not in selected:
            continue
        frequencies = doc.get("frequencies", {})
        if not isinstance(frequencies, dict):
            continue
        source_label = doc.get("label") or doc.get("file_name") or doc_id
        for word, count in frequencies.items():
            add_frequency(counter, sources, word, count, source_label)


def register_pdf_documents(corpus, uploaded_pdfs, label_prefix, extraction_method, api_key, skip_duplicates=True):
    results = []
    documents = corpus.setdefault("documents", {})

    for uploaded_pdf in uploaded_pdfs:
        pdf_bytes = uploaded_pdf.getvalue()
        digest = file_digest(pdf_bytes)
        doc_id = make_document_id(uploaded_pdf.name, digest)

        if skip_duplicates and doc_id in documents:
            results.append(
                {
                    "file_name": uploaded_pdf.name,
                    "status": "skipped",
                    "message": "登録済みのためスキップ",
                    "doc_id": doc_id,
                }
            )
            continue

        text, actual_method = extract_text_from_pdf(uploaded_pdf, extraction_method, api_key)
        frequencies = count_words_from_text(text)
        label_base = label_prefix.strip() if label_prefix else "過去問PDF"
        label = f"{label_base} / {os.path.splitext(uploaded_pdf.name)[0]}"

        documents[doc_id] = {
            "label": label,
            "file_name": uploaded_pdf.name,
            "file_hash": digest,
            "source_type": "pdf",
            "extraction_method": actual_method,
            "registered_at": datetime.now().isoformat(timespec="seconds"),
            "text_char_count": len(text),
            "total_words": int(sum(frequencies.values())),
            "unique_words": len(frequencies),
            "frequencies": frequencies,
        }
        results.append(
            {
                "file_name": uploaded_pdf.name,
                "status": "registered",
                "message": f"{len(frequencies):,}種類 / {sum(frequencies.values()):,}語",
                "doc_id": doc_id,
            }
        )

    return results


def corpus_to_dataframe(corpus):
    rows = []
    for doc_id, doc in sorted(corpus.get("documents", {}).items(), key=lambda item: item[1].get("registered_at", "")):
        rows.append(
            {
                "doc_id": doc_id,
                "label": doc.get("label", doc_id),
                "file_name": doc.get("file_name", ""),
                "method": doc.get("extraction_method", ""),
                "total_words": doc.get("total_words", 0),
                "unique_words": doc.get("unique_words", 0),
                "text_chars": doc.get("text_char_count", 0),
                "registered_at": doc.get("registered_at", ""),
            }
        )
    return pd.DataFrame(rows)


def remove_documents_from_corpus(corpus, doc_ids):
    documents = corpus.setdefault("documents", {})
    removed = 0
    for doc_id in doc_ids:
        if doc_id in documents:
            del documents[doc_id]
            removed += 1
    return removed


def lexicon_to_dataframe(lexicon, statuses=None):
    rows = []
    status_set = set(statuses) if statuses else None
    for word, entry in sorted(lexicon.items()):
        status = entry.get("status", "pending")
        if status_set and status not in status_set:
            continue
        rows.append(
            {
                "word": word,
                "status": status,
                "status_label": STATUS_LABELS.get(status, status),
                "meanings": meanings_to_text(entry.get("meanings", [])),
                "note": entry.get("note", ""),
                "last_seen_count": entry.get("last_seen_count", ""),
                "last_seen_rank": entry.get("last_seen_rank", ""),
                "coverage_policy": entry.get("coverage_policy", coverage_policy_for_status(status)),
            }
        )
    return pd.DataFrame(rows)


def build_readiness_rows(candidates, lexicon, target_count):
    candidate_count = 0 if candidates.empty else len(candidates)
    base_count = count_base_words(lexicon)
    excluded_count = sum(1 for entry in lexicon.values() if entry.get("status") in EXCLUDED_STATUSES)
    pending_count = 0 if candidates.empty else int((candidates["status"] == "pending").sum())
    missing_meanings = sum(
        1
        for entry in lexicon.values()
        if entry.get("status") in BASE_STATUSES and not split_meanings(entry.get("meanings", []))
    )
    target_count = int(target_count)

    if candidate_count >= target_count:
        candidate_status = "OK"
        candidate_detail = f"{candidate_count:,}語の候補があります。"
    else:
        candidate_status = "不足"
        candidate_detail = f"候補は {candidate_count:,}語です。目標まであと {target_count - candidate_count:,}語ほど必要です。"

    if base_count >= target_count:
        base_status = "OK"
        base_detail = f"基礎語は {base_count:,}語あります。"
    else:
        base_status = "作業中"
        base_detail = f"基礎語は {base_count:,}語です。目標まであと {target_count - base_count:,}語です。"

    if missing_meanings == 0:
        meaning_status = "OK" if base_count else "作業前"
        meaning_detail = "基礎語の意味メモは空ではありません。" if base_count else "基礎語を選ぶと確認できます。"
    else:
        meaning_status = "要確認"
        meaning_detail = f"意味メモが空の基礎語が {missing_meanings:,}語あります。"

    return [
        {"確認項目": "候補量", "状態": candidate_status, "詳細": candidate_detail},
        {"確認項目": "基礎語3000", "状態": base_status, "詳細": base_detail},
        {"確認項目": "意味メモ", "状態": meaning_status, "詳細": meaning_detail},
        {"確認項目": "未分類語", "状態": "参考", "詳細": f"未分類 {pending_count:,}語 / 除外・ノイズ指定 {excluded_count:,}語"},
    ]


def render_readiness_panel(candidates, lexicon, target_count):
    rows = build_readiness_rows(candidates, lexicon, target_count)
    blocking = [row for row in rows if row["状態"] in {"不足", "要確認"}]
    title = "3000語作成チェック"
    if not blocking:
        title += ": 良好"
    with st.expander(title, expanded=bool(blocking)):
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if blocking:
            st.caption("この表示は作業の目安です。候補量を増やすにはPDFを追加し、意味メモは基礎語の完成度確認に使います。")


def run_self_check(api_key=""):
    rows = []
    sample_text = "The students studied clothes and companies. Maki used it in the test."
    expected_words = {"student", "study", "clothes", "company", "use"}

    try:
        words = set(extract_words_from_text(sample_text))
        missing = sorted(expected_words - words)
        rows.append(
            {
                "確認項目": "単語抽出",
                "状態": "OK" if not missing else "注意",
                "詳細": "サンプル英文から想定語を抽出できました。" if not missing else f"不足: {', '.join(missing)}",
            }
        )
    except Exception as e:
        rows.append({"確認項目": "単語抽出", "状態": "NG", "詳細": str(e)})

    for check in get_environment_status(api_key):
        rows.append({"確認項目": check["item"], "状態": check["status"], "詳細": check["detail"]})

    try:
        temp_corpus = empty_corpus()
        temp_corpus["documents"]["sample"] = {
            "label": "sample",
            "frequencies": {"student": 2, "study": 1, "clothes": 1},
        }
        sample_candidates = build_candidates([], False, corpus=temp_corpus, include_corpus=True, selected_doc_ids=["sample"])
        rows.append(
            {
                "確認項目": "候補化",
                "状態": "OK" if len(sample_candidates) == 3 else "注意",
                "詳細": f"サンプル候補 {len(sample_candidates)}語を作成しました。",
            }
        )
    except Exception as e:
        rows.append({"確認項目": "候補化", "状態": "NG", "詳細": str(e)})

    return rows


def render_self_check(api_key):
    with st.expander("セルフチェック"):
        st.write("PDFを追加する前に、この画面の基本処理が動くか確認できます。")
        if st.button("セルフチェックを実行", use_container_width=True):
            rows = run_self_check(api_key)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def build_candidates(uploaded_files, include_exam_db, corpus=None, include_corpus=True, selected_doc_ids=None):
    counter = defaultdict(int)
    sources = defaultdict(set)

    if uploaded_files:
        collect_from_csv_files(uploaded_files, counter, sources)

    if include_exam_db:
        exam_db = load_json_file(EXAM_DB_FILE, {})
        collect_from_exam_db_node(exam_db, counter, sources, [])

    if include_corpus and corpus:
        collect_from_corpus(corpus, counter, sources, selected_doc_ids=selected_doc_ids)

    rows = []
    total_count = sum(counter.values())
    running_count = 0
    for rank, (word, count) in enumerate(sorted(counter.items(), key=lambda x: (-x[1], x[0])), start=1):
        running_count += count
        source_list = sorted(sources[word])
        rows.append(
            {
                "rank": rank,
                "word": word,
                "count": count,
                "token_share": (count / total_count * 100) if total_count else 0,
                "cumulative_coverage": (running_count / total_count * 100) if total_count else 0,
                "source_count": len(source_list),
                "sources": " / ".join(source_list[:3]) + (" ..." if len(source_list) > 3 else ""),
            }
        )

    return pd.DataFrame(rows)


def default_status_for_word(word, lexicon):
    if word in lexicon:
        return lexicon[word].get("status", "pending")
    if word in STRICT_EXCLUDED_WORDS:
        return "strict_excluded"
    if word in PROPER_NOUN_OR_NOISE_WORDS:
        return NOISE_STATUS
    return "pending"


def coverage_policy_for_status(status):
    if status in BASE_STATUSES:
        return "auto_covered"
    if status in EXCLUDED_STATUSES:
        return "excluded"
    if status == "learning_target":
        return "learning_target"
    return "pending"


def add_lexicon_columns(df, lexicon):
    if df.empty:
        return df

    enriched = df.copy()
    enriched["status"] = enriched["word"].map(lambda w: default_status_for_word(w, lexicon))
    enriched["status_label"] = enriched["status"].map(STATUS_LABELS)
    enriched["meanings"] = enriched["word"].map(lambda w: meanings_to_text(lexicon.get(w, {}).get("meanings", [])))
    enriched["note"] = enriched["word"].map(lambda w: lexicon.get(w, {}).get("note", ""))
    enriched["coverage_policy"] = enriched["status"].map(coverage_policy_for_status)
    return enriched


def count_base_words(lexicon):
    return sum(1 for entry in lexicon.values() if entry.get("status") in BASE_STATUSES)


def count_status(lexicon, status):
    return sum(1 for entry in lexicon.values() if entry.get("status") == status)


def update_lexicon_from_rows(lexicon, rows):
    changed = 0
    for _, row in rows.iterrows():
        word = normalize_word(row["word"])
        status = str(row.get("status", "pending")).strip()
        meanings = split_meanings(row.get("meanings", ""))
        note = str(row.get("note", "") or "").strip()

        if status not in STATUS_OPTIONS:
            status = "pending"

        if status == "pending" and not meanings and not note and word in lexicon:
            del lexicon[word]
            changed += 1
            continue

        if status == "pending" and not meanings and not note:
            continue

        old = lexicon.get(word, {})
        new_entry = {
            "status": status,
            "coverage_policy": coverage_policy_for_status(status),
            "meanings": meanings,
            "note": note,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "source": old.get("source", "base_vocab_manager"),
        }

        if "count" in row:
            try:
                new_entry["last_seen_count"] = int(row["count"])
            except Exception:
                pass
        if "rank" in row:
            try:
                new_entry["last_seen_rank"] = int(row["rank"])
            except Exception:
                pass

        if old != new_entry:
            lexicon[word] = new_entry
            changed += 1

    return changed


def auto_mark_words(lexicon, candidate_df, status, limit=None, word_set=None, allow_override=False):
    changed = 0
    if candidate_df.empty:
        return changed

    target_df = candidate_df
    if word_set is not None:
        target_df = target_df[target_df["word"].isin(word_set)]
    if limit is not None:
        target_df = target_df.head(limit)

    for _, row in target_df.iterrows():
        word = row["word"]
        current_status = lexicon.get(word, {}).get("status", "pending")
        if not allow_override and current_status not in {"pending", status}:
            continue
        lexicon[word] = {
            **lexicon.get(word, {}),
            "status": status,
            "coverage_policy": coverage_policy_for_status(status),
            "meanings": split_meanings(lexicon.get(word, {}).get("meanings", [])),
            "note": lexicon.get(word, {}).get("note", ""),
            "last_seen_count": int(row["count"]),
            "last_seen_rank": int(row["rank"]),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "source": lexicon.get(word, {}).get("source", "base_vocab_manager"),
        }
        changed += 1
    return changed


def filtered_candidates(df, status_filter, min_count, min_source_count, search_text, max_rows):
    result = df.copy()
    if status_filter != "すべて":
        reverse_labels = {label: status for status, label in STATUS_LABELS.items()}
        status = reverse_labels.get(status_filter, status_filter)
        result = result[result["status"] == status]
    if min_count:
        result = result[result["count"] >= min_count]
    if min_source_count:
        result = result[result["source_count"] >= min_source_count]
    if search_text:
        result = result[result["word"].str.contains(search_text.lower(), case=False, regex=False)]
    return result.head(max_rows)


def main():
    st.set_page_config(page_title="基礎語3000 選定プラットフォーム", page_icon="📚", layout="wide")
    st.title("基礎語3000 選定プラットフォーム")
    st.caption("過去問PDF・CSV・登録済みDBから候補語を集め、AIに回さない基礎語3000を作るための作業台です。")
    with st.expander("基本の作業順"):
        st.markdown(
            """
1. 左側で過去問PDFを登録する。
2. 必要ならCSVや past_exams_db.json も読み込む。
3. 完全除外、固有名詞・ノイズ、試験語、注意多義語を先に分類する。
4. 上位語を基礎語として仮分類し、変な語を手で直す。
5. `base_lexicon.json` に保存する。
            """.strip()
        )

    if "base_lexicon_work" not in st.session_state:
        st.session_state.base_lexicon_work = load_json_file(BASE_LEXICON_FILE, {})
    if "base_vocab_corpus" not in st.session_state:
        st.session_state.base_vocab_corpus = load_corpus()
    if "last_pdf_results" not in st.session_state:
        st.session_state.last_pdf_results = []

    lexicon = st.session_state.base_lexicon_work
    corpus = st.session_state.base_vocab_corpus
    default_api_key = get_secret_api_key()

    with st.sidebar:
        st.header("候補データ")
        uploaded_files = st.file_uploader(
            "単語CSVを追加",
            type=["csv"],
            accept_multiple_files=True,
            help="列名は「単語」「出現回数」、または word/count 形式に対応します。",
        )

        st.divider()
        st.subheader("PDFを追加")
        uploaded_pdfs = st.file_uploader(
            "過去問PDF",
            type=["pdf"],
            accept_multiple_files=True,
            help="本文は保存せず、抽出した単語頻度だけを作業データに保存します。",
        )
        label_prefix = st.text_input("出典ラベル", value="過去問PDF")
        extraction_label = st.selectbox(
            "本文抽出",
            ["PyMuPDF（速い）", "PyMuPDF→不足時Gemini", "Gemini（OCR向け）"],
            index=1,
        )
        extraction_method = {
            "PyMuPDF（速い）": "PyMuPDF",
            "PyMuPDF→不足時Gemini": "Auto",
            "Gemini（OCR向け）": "Gemini",
        }[extraction_label]
        api_key = default_api_key
        if extraction_method in {"Gemini", "Auto"}:
            if default_api_key:
                st.caption("Gemini API Key は secrets から読み込み済みです。")
            else:
                api_key = st.text_input("Gemini API Key", type="password")
        render_environment_status(api_key)
        render_self_check(api_key)
        skip_duplicates = st.checkbox("同じPDFは重複登録しない", value=True)

        if st.button("PDFを候補データに登録", type="primary", use_container_width=True):
            if not uploaded_pdfs:
                st.warning("登録するPDFを選んでください。")
            else:
                try:
                    results = register_pdf_documents(
                        corpus=corpus,
                        uploaded_pdfs=uploaded_pdfs,
                        label_prefix=label_prefix,
                        extraction_method=extraction_method,
                        api_key=api_key,
                        skip_duplicates=skip_duplicates,
                    )
                    save_json_file(VOCAB_CORPUS_FILE, corpus)
                    st.session_state.last_pdf_results = results
                    registered = sum(1 for r in results if r["status"] == "registered")
                    skipped = sum(1 for r in results if r["status"] == "skipped")
                    st.success(f"PDF登録: {registered}件 / スキップ: {skipped}件")
                except Exception as e:
                    st.error(f"PDF登録中にエラーが出ました: {e}")

        for result in st.session_state.last_pdf_results[-3:]:
            st.caption(f"{result['file_name']}: {result['message']}")

        st.divider()
        st.subheader("読み込むデータ")
        include_exam_db = st.checkbox(
            "past_exams_db.json も読む",
            value=os.path.exists(EXAM_DB_FILE),
            disabled=not os.path.exists(EXAM_DB_FILE),
        )
        include_corpus = st.checkbox(
            "PDFコーパスも読む",
            value=bool(corpus.get("documents")),
            disabled=not bool(corpus.get("documents")),
        )
        corpus_doc_ids = list(corpus.get("documents", {}).keys())
        selected_doc_ids = []
        if include_corpus and corpus_doc_ids:
            selected_doc_ids = st.multiselect(
                "使うPDF",
                options=corpus_doc_ids,
                default=corpus_doc_ids,
                format_func=lambda doc_id: corpus["documents"][doc_id].get("label", doc_id),
            )

        target_count = st.number_input("基礎語の目標数", min_value=100, max_value=10000, value=3000, step=100)
        st.divider()
        if st.button("保存済みJSONを再読み込み", use_container_width=True):
            st.session_state.base_lexicon_work = load_json_file(BASE_LEXICON_FILE, {})
            st.session_state.base_vocab_corpus = load_corpus()
            st.rerun()

    try:
        raw_candidates = build_candidates(
            uploaded_files=uploaded_files,
            include_exam_db=include_exam_db,
            corpus=corpus,
            include_corpus=include_corpus,
            selected_doc_ids=selected_doc_ids,
        )
    except Exception as e:
        st.error(f"候補データの読み込みでエラーが出ました: {e}")
        raw_candidates = pd.DataFrame()

    candidates = add_lexicon_columns(raw_candidates, lexicon)

    base_count = count_base_words(lexicon)
    strict_count = count_status(lexicon, STRICT_EXCLUDED_STATUS)
    noise_count = count_status(lexicon, NOISE_STATUS)
    learning_count = count_status(lexicon, "learning_target")
    remaining = max(int(target_count) - base_count, 0)

    if candidates.empty:
        total_tokens = 0
        covered_tokens = 0
        excluded_tokens = 0
    else:
        total_tokens = int(candidates["count"].sum())
        covered_tokens = int(candidates[candidates["status"].isin(BASE_STATUSES)]["count"].sum())
        excluded_tokens = int(candidates[candidates["status"].isin(EXCLUDED_STATUSES)]["count"].sum())

    ai_target_count = 0 if candidates.empty else len(candidates[~candidates["status"].isin(BASE_STATUSES | EXCLUDED_STATUSES)])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("候補語数", f"{len(candidates):,}")
    col2.metric("基礎語登録", f"{base_count:,}", delta=f"目標まで {remaining:,}")
    col3.metric("除外・ノイズ", f"{strict_count + noise_count:,}")
    col4.metric("AI・単語帳候補", f"{ai_target_count:,}", delta=f"手動指定 {learning_count:,}")

    if total_tokens:
        cov1, cov2 = st.columns(2)
        cov1.progress(min(covered_tokens / total_tokens, 1.0), text=f"基礎語でカバー: {covered_tokens / total_tokens:.1%}")
        cov2.progress(min(excluded_tokens / total_tokens, 1.0), text=f"除外・ノイズ: {excluded_tokens / total_tokens:.1%}")

    render_readiness_panel(candidates, lexicon, target_count)

    tab_pick, tab_base, tab_corpus, tab_save = st.tabs(["候補を選ぶ", "基礎語リスト", "PDF・コーパス", "保存・出力"])

    with tab_pick:
        st.subheader("候補一覧")

        if candidates.empty:
            st.info("PDFを登録するか、CSVをアップロードするか、past_exams_db.json を読むと候補が表示されます。")
        else:
            with st.expander("自動で仮分類する", expanded=True):
                auto_min_source_count = st.number_input(
                    "基礎語仮分類の最低出典数",
                    min_value=1,
                    max_value=1000,
                    value=1,
                    step=1,
                    help="複数年・複数大学のPDFを入れた後は 2 以上にすると、たまたま1回の文章だけに出た語を拾いにくくできます。",
                )
                a1, a2, a3 = st.columns(3)
                if a1.button("AI処理を省く基礎語として仮分類", use_container_width=True):
                    excluded_words = STRICT_EXCLUDED_WORDS | PROPER_NOUN_OR_NOISE_WORDS
                    eligible = candidates[
                        (~candidates["word"].isin(excluded_words))
                        & (candidates["source_count"] >= int(auto_min_source_count))
                    ]
                    changed = auto_mark_words(lexicon, eligible, "core_verified", limit=int(target_count))
                    st.success(f"{changed}語を基礎語として仮登録しました。")
                    st.rerun()
                if a2.button("試験語候補を分類", use_container_width=True):
                    changed = auto_mark_words(lexicon, candidates, "exam_format", word_set=EXAM_FORMAT_WORDS, allow_override=True)
                    st.success(f"{changed}語を試験語として分類しました。")
                    st.rerun()
                if a3.button("注意多義候補を分類", use_container_width=True):
                    changed = auto_mark_words(lexicon, candidates, "watch_known", word_set=WATCH_KNOWN_WORDS, allow_override=True)
                    st.success(f"{changed}語を注意多義語として分類しました。")
                    st.rerun()

                b1, b2, b3 = st.columns(3)
                if b1.button("完全除外語を分類", use_container_width=True):
                    changed = auto_mark_words(lexicon, candidates, "strict_excluded", word_set=STRICT_EXCLUDED_WORDS, allow_override=True)
                    st.success(f"{changed}語を完全除外として分類しました。")
                    st.rerun()
                if b2.button("固有名詞・ノイズ候補を分類", use_container_width=True):
                    changed = auto_mark_words(lexicon, candidates, NOISE_STATUS, word_set=PROPER_NOUN_OR_NOISE_WORDS, allow_override=True)
                    st.success(f"{changed}語を固有名詞・ノイズとして分類しました。")
                    st.rerun()
                if b3.button("低頻度語を単語帳対象にする", use_container_width=True):
                    low_freq = candidates[(candidates["count"] <= 2) & (candidates["status"] == "pending")]
                    changed = auto_mark_words(lexicon, low_freq, "learning_target")
                    st.success(f"{changed}語を単語帳対象として分類しました。")
                    st.rerun()

            f1, f2, f3, f4, f5 = st.columns([1.2, 1, 1, 1, 1])
            status_filter = f1.selectbox("状態", ["すべて"] + list(STATUS_LABELS.values()))
            min_count = f2.number_input("最低回数", min_value=0, max_value=10000, value=0, step=1)
            min_source_count = f3.number_input("最低出典数", min_value=0, max_value=1000, value=0, step=1)
            search_text = f4.text_input("検索")
            max_rows = f5.number_input("表示行数", min_value=50, max_value=10000, value=500, step=50)

            view_df = filtered_candidates(candidates, status_filter, min_count, min_source_count, search_text, int(max_rows))
            edit_columns = [
                "rank",
                "word",
                "count",
                "cumulative_coverage",
                "source_count",
                "status",
                "meanings",
                "note",
                "sources",
            ]
            edited_df = st.data_editor(
                view_df[edit_columns],
                use_container_width=True,
                hide_index=True,
                disabled=["rank", "word", "count", "cumulative_coverage", "source_count", "sources"],
                column_config={
                    "rank": st.column_config.NumberColumn("順位"),
                    "word": st.column_config.TextColumn("単語"),
                    "count": st.column_config.NumberColumn("回数"),
                    "cumulative_coverage": st.column_config.ProgressColumn(
                        "累積カバー",
                        min_value=0,
                        max_value=100,
                        format="%.1f%%",
                    ),
                    "source_count": st.column_config.NumberColumn("出典数"),
                    "status": st.column_config.SelectboxColumn(
                        "分類",
                        options=STATUS_OPTIONS,
                        help="core_verified=基礎語、watch_known=注意多義語、strict_excluded=完全除外",
                    ),
                    "meanings": st.column_config.TextColumn("意味メモ", width="large"),
                    "note": st.column_config.TextColumn("メモ", width="medium"),
                    "sources": st.column_config.TextColumn("出典"),
                },
            )

            if st.button("表示中の編集を作業データに反映", type="primary"):
                changed = update_lexicon_from_rows(lexicon, edited_df)
                st.success(f"{changed}件を作業データに反映しました。")
                st.rerun()

    with tab_base:
        st.subheader("現在の基礎語・試験語・注意多義語")
        base_df = lexicon_to_dataframe(lexicon, statuses=BASE_STATUSES)
        if base_df.empty:
            st.info("まだ基礎語は登録されていません。候補タブから仮登録してください。")
        else:
            st.write(f"登録数: **{len(base_df):,}語**")
            st.dataframe(base_df, use_container_width=True, hide_index=True)

            missing_meanings = base_df[base_df["meanings"].astype(str).str.strip() == ""]
            if not missing_meanings.empty:
                st.warning(f"意味メモが空の基礎語が {len(missing_meanings):,}語あります。3000語を完成させる前に確認してください。")

    with tab_corpus:
        st.subheader("PDFコーパス")
        st.write(f"`{VOCAB_CORPUS_FILE}` には、PDF本文ではなく単語頻度だけを保存します。")
        corpus_df = corpus_to_dataframe(corpus)
        if corpus_df.empty:
            st.info("まだPDFは登録されていません。左側の「PDFを追加」から登録できます。")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("登録PDF", f"{len(corpus_df):,}")
            c2.metric("総語数", f"{int(corpus_df['total_words'].sum()):,}")
            c3.metric("平均種類数", f"{int(corpus_df['unique_words'].mean()):,}")
            st.dataframe(corpus_df.drop(columns=["doc_id"]), use_container_width=True, hide_index=True)

            delete_ids = st.multiselect(
                "削除するPDFデータ",
                options=corpus_df["doc_id"].tolist(),
                format_func=lambda doc_id: corpus["documents"][doc_id].get("label", doc_id),
            )
            if st.button("選択したPDFデータを削除", use_container_width=True):
                removed = remove_documents_from_corpus(corpus, delete_ids)
                save_json_file(VOCAB_CORPUS_FILE, corpus)
                st.success(f"{removed}件を削除しました。")
                st.rerun()

    with tab_save:
        st.subheader("保存・出力")
        st.write(f"`{BASE_LEXICON_FILE}` に保存すると、次回も同じ分類を続きから編集できます。")

        save_col, download_col, corpus_col = st.columns(3)
        if save_col.button("base_lexicon.json に保存", type="primary", use_container_width=True):
            save_json_file(BASE_LEXICON_FILE, lexicon)
            st.success(f"{BASE_LEXICON_FILE} に保存しました。")

        download_col.download_button(
            "JSONをダウンロード",
            data=json.dumps(lexicon, ensure_ascii=False, indent=2),
            file_name=BASE_LEXICON_FILE,
            mime="application/json",
            use_container_width=True,
        )

        corpus_col.download_button(
            "PDFコーパスJSONをダウンロード",
            data=json.dumps(corpus, ensure_ascii=False, indent=2),
            file_name=VOCAB_CORPUS_FILE,
            mime="application/json",
            use_container_width=True,
        )

        if not candidates.empty:
            st.download_button(
                "候補一覧CSVをダウンロード",
                data=candidates.to_csv(index=False).encode("utf-8-sig"),
                file_name="base_vocab_candidates.csv",
                mime="text/csv",
                use_container_width=True,
            )

        base_df = lexicon_to_dataframe(lexicon, statuses=BASE_STATUSES)
        if not base_df.empty:
            st.download_button(
                "基礎語リストCSVをダウンロード",
                data=base_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="base_vocab_selected.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with st.expander("分類の意味"):
            st.markdown(
                """
- **基礎語**: AIに毎回意味付けさせず、登録済み意味でカバー済みにする語。
- **試験語**: question, option など、試験形式として頻出する語。
- **注意多義語**: 基礎語だが、文脈で意味がずれると読解に効く語。
- **単語帳対象**: 基礎語には入れず、生徒の単語帳やAI解析の対象にする語。
- **完全除外**: 冠詞・代名詞・助動詞の破片など、語彙カバー率の分母から外す語。
- **固有名詞・ノイズ**: 人名・略語・本文固有の記号など、教材語として扱わない語。
                """.strip()
            )


if __name__ == "__main__":
    main()
