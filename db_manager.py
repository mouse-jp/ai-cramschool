import streamlit as st
import pandas as pd
import google.generativeai as genai
import tempfile
import re
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

LOCAL_NLTK_DATA = os.path.join(os.path.dirname(__file__), ".nltk_data")
os.makedirs(LOCAL_NLTK_DATA, exist_ok=True)
os.environ["NLTK_DATA"] = LOCAL_NLTK_DATA

# --- 追加: NLTKのセットアップ ---
import nltk
from nltk.stem import WordNetLemmatizer

if LOCAL_NLTK_DATA not in nltk.data.path:
    nltk.data.path.insert(0, LOCAL_NLTK_DATA)

try:
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet', download_dir=LOCAL_NLTK_DATA, quiet=True)

lemmatizer = WordNetLemmatizer()
# --------------------------------

# --- 設定 ---
DB_FILE = "past_exams_db.json"
MY_DATA_FILE = "my_data.json"
BASE_LEXICON_FILE = "base_lexicon.json"
IDIOM_LEXICON_FILE = "idiom_lexicon.json"
WORD_MEANING_LEXICON_FILE = "word_meaning_lexicon.json"
BASE_VOCAB_STATUSES = {"core_verified", "exam_format", "watch_known"}
EXCLUDED_BASE_VOCAB_STATUSES = {"strict_excluded", "proper_noun_or_noise"}

gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
try:
    gemini_api_key = gemini_api_key or st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
except Exception:
    pass

if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
else:
    st.warning("Gemini APIキーが見つかりません。PDF抽出・熟語/文法AI解析を使う場合は secrets または環境変数に設定してください。")

# --- 修正: ストップワードの大幅拡充 ---
STOP_WORDS = {
    # 既存の基本単語
    "a", "an", "the", "and", "but", "or", "for", "nor", "on", "at", "to", "from", "by", "is", "are", "was", "were", "am", "be", "been", "being", "in", "of", "with", "as", "it", "this", "that", "these", "those", "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "their", "our", "its", "which", "who", "whom", "whose", "have", "has", "had", "do", "does", "did", "can", "will", "would", "could", "should", "not", "no", "if", "then", "than", "so", "very", "too", "all", "any", "some",
    # 追加のノイズ（ローマ数字、論文の略語、短縮形の破片など）
    "et", "al", "st", "pp", "vol", "ed", "ii", "iii", "iv", "vi", "vii", "viii", "ix", "x",
    "don", "doesn", "didn", "isn", "aren", "wasn", "weren", "hasn", "haven", "hadn",
    "won", "wouldn", "shouldn", "couldn", "ll", "ve", "re", "t", "s", "m", "d"
}
def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    return {}

def save_db(db):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def load_json_file(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    return default

def load_base_lexicon():
    return load_json_file(BASE_LEXICON_FILE, {})

def normalize_vocab_word(word):
    word = str(word).strip().lower()
    lemma = lemmatizer.lemmatize(word, pos="v")
    lemma = lemmatizer.lemmatize(lemma, pos="n")
    return lemma

def split_meanings(value):
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        text = re.sub(r"\s+", " ", value.strip())
        if not text:
            return []
        if re.search(r"[①②③④⑤⑥⑦⑧⑨⑩]", text):
            parts = re.split(r"\s*(?=[①②③④⑤⑥⑦⑧⑨⑩])", text)
            return [p.strip(" ;；") for p in parts if p.strip(" ;；")]
        return [v.strip() for v in re.split(r"[；;]", text) if v.strip()]
    return []

def get_frequency_strong_words(lexicon):
    return {
        normalize_vocab_word(word)
        for word, entry in lexicon.items()
        if entry.get("status") in BASE_VOCAB_STATUSES
    }

def get_frequency_excluded_words(lexicon):
    return {
        normalize_vocab_word(word)
        for word, entry in lexicon.items()
        if entry.get("status") in EXCLUDED_BASE_VOCAB_STATUSES
    }

def build_meaning_registry(lexicon):
    registry = {}
    for word, entry in lexicon.items():
        meanings = split_meanings(entry.get("meanings", []))
        if meanings:
            registry[normalize_vocab_word(word)] = "；".join(meanings)

    my_data = load_json_file(MY_DATA_FILE, {})
    for item in my_data.get("vocabulary", []):
        word = normalize_vocab_word(item.get("title", ""))
        meaning = str(item.get("content", "")).strip()
        if word and meaning:
            registry[word] = meaning

    for book in my_data.get("vocab_books", []):
        for item in book.get("enriched_vocab", []):
            word = normalize_vocab_word(item.get("word", ""))
            meaning = "；".join(split_meanings(item.get("meanings", "")))
            if word and meaning:
                registry[word] = meaning
    return registry

def summarize_words_without_frequency_strong(freqs):
    lexicon = load_base_lexicon()
    strong_words = get_frequency_strong_words(lexicon)
    excluded_words = get_frequency_excluded_words(lexicon)
    meaning_registry = build_meaning_registry(lexicon)

    normalized_counts = Counter()
    for word, count in freqs.items():
        normalized = normalize_vocab_word(word)
        if not normalized:
            continue
        normalized_counts[normalized] += int(count)

    strong_tokens = sum(count for word, count in normalized_counts.items() if word in strong_words)
    excluded_tokens = sum(count for word, count in normalized_counts.items() if word in excluded_words)
    remaining = {
        word: count
        for word, count in normalized_counts.items()
        if word not in strong_words and word not in excluded_words
    }
    known_meaning_words = {word for word in remaining if word in meaning_registry}
    missing_meaning_words = set(remaining) - known_meaning_words

    return {
        "strong_words": strong_words,
        "remaining": dict(sorted(remaining.items(), key=lambda item: item[1], reverse=True)),
        "strong_tokens": strong_tokens,
        "excluded_tokens": excluded_tokens,
        "remaining_tokens": sum(remaining.values()),
        "known_meaning_words": known_meaning_words,
        "missing_meaning_words": missing_meaning_words,
        "meaning_registry": meaning_registry,
    }

def extract_text_with_pymupdf(uploaded_file):
    try:
        import fitz
    except Exception:
        return ""

    doc = None
    try:
        doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
        pages = []
        for page in doc:
            pages.append(page.get_text("text"))
        return "\n".join(pages).strip()
    except Exception:
        return ""
    finally:
        if doc is not None:
            doc.close()


def is_text_extraction_good_enough(text):
    text = str(text or "")
    english_chars = len(re.findall(r"[A-Za-z]", text))
    words = len(re.findall(r"\b[A-Za-z]{2,}\b", text))
    return english_chars >= 500 and words >= 80


def filter_english_exam_text(text):
    """Keep English passages, questions, options, and word lists; drop Japanese-only instructions."""
    kept_lines = []
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line:
            if kept_lines and kept_lines[-1] != "":
                kept_lines.append("")
            continue
        if re.search(r"[A-Za-z]", line):
            kept_lines.append(line)
    return "\n".join(kept_lines).strip()


def wait_for_gemini_file(g_file, timeout_seconds=60):
    start = time.time()
    while True:
        state = getattr(g_file, "state", None)
        state_name = getattr(state, "name", str(state)).upper() if state is not None else ""
        if "PROCESSING" not in state_name:
            if "FAILED" in state_name:
                raise RuntimeError("Gemini側でPDFファイル処理に失敗しました。")
            return g_file
        if time.time() - start > timeout_seconds:
            raise TimeoutError(f"GeminiのPDF準備が{timeout_seconds}秒を超えました。")
        time.sleep(2)
        g_file = genai.get_file(g_file.name)


def make_pdf_page_chunks(uploaded_file, pages_per_request=2):
    pdf_bytes = uploaded_file.getvalue()
    try:
        import fitz
    except Exception:
        return [("全ページ", pdf_bytes)]

    chunks = []
    doc = None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = doc.page_count
        batch_size = max(1, int(pages_per_request))
        for start_page in range(0, page_count, batch_size):
            end_page = min(start_page + batch_size - 1, page_count - 1)
            out_doc = fitz.open()
            out_doc.insert_pdf(doc, from_page=start_page, to_page=end_page)
            chunk_bytes = out_doc.tobytes()
            out_doc.close()
            if start_page == end_page:
                label = f"{start_page + 1}ページ"
            else:
                label = f"{start_page + 1}-{end_page + 1}ページ"
            chunks.append((label, chunk_bytes))
        return chunks or [("全ページ", pdf_bytes)]
    except Exception:
        return [("全ページ", pdf_bytes)]
    finally:
        if doc is not None:
            doc.close()


def extract_pdf_bytes_with_gemini(pdf_bytes, prompt, model_name, timeout_seconds=180):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    g_file = None
    try:
        g_file = genai.upload_file(tmp_path)
        g_file = wait_for_gemini_file(g_file, timeout_seconds=min(60, int(timeout_seconds)))
        model = genai.GenerativeModel(model_name=model_name)
        res = model.generate_content([g_file, prompt], request_options={"timeout": int(timeout_seconds)})
        return str(getattr(res, "text", "") or "").strip()
    finally:
        if g_file is not None:
            try:
                genai.delete_file(g_file.name)
            except Exception:
                pass
        os.remove(tmp_path)


def extract_pdf_bytes_with_retry(pdf_bytes, prompt, model_name, timeout_seconds=180, attempts=2):
    last_error = None
    for attempt in range(max(1, int(attempts))):
        try:
            return extract_pdf_bytes_with_gemini(
                pdf_bytes,
                prompt,
                model_name,
                timeout_seconds=timeout_seconds,
            )
        except Exception as e:
            last_error = e
            if attempt + 1 < attempts:
                time.sleep(3)
    raise last_error


def extract_text_with_gemini(
    uploaded_file,
    model_name="gemini-2.5-pro",
    timeout_seconds=180,
    allow_fallback=False,
    pages_per_request=4,
    max_workers=2,
):
    prompt = """
このPDFファイルは英語の入試問題です。
英語だけを抽出してください。

【残すもの】
- 英語本文
- 英語の設問文、空所補充の英文、選択肢
- 並び替え問題の英語語群
- 英語の見出しや番号

【捨てるもの】
- 日本語の指示文、注意書き、解答用紙に関する説明
- 日本語の見出し、注釈、和訳、解説
- ページ番号、著作権表示、余計な挨拶、要約

英語と日本語が混ざっている行は、可能なら英語部分だけを残してください。
抽出した英語テキストだけを返してください。
"""
    page_chunks = make_pdf_page_chunks(uploaded_file, pages_per_request=pages_per_request)
    worker_count = max(1, min(int(max_workers), len(page_chunks)))
    progress = st.progress(0, text=f"PDFをAIで読み取り中... 0/{len(page_chunks)}")

    def read_one_chunk(index, label, pdf_bytes):
        candidates = (
            get_gemini_model_chain(model_name)
            if allow_fallback
            else [normalize_gemini_model_name(model_name)]
        )
        last_error = None
        for candidate in candidates:
            try:
                text = extract_pdf_bytes_with_retry(
                    pdf_bytes,
                    prompt,
                    candidate,
                    timeout_seconds=timeout_seconds,
                    attempts=2,
                )
                if text:
                    return index, label, text, None
                last_error = RuntimeError(f"{candidate} が空の応答を返しました。")
            except Exception as e:
                last_error = e
        return index, label, "", last_error

    results = [""] * len(page_chunks)
    errors = []
    done_count = 0

    if worker_count == 1:
        for index, (label, pdf_bytes) in enumerate(page_chunks):
            result_index, result_label, text, error = read_one_chunk(index, label, pdf_bytes)
            results[result_index] = text
            if error:
                errors.append(f"{result_label}: {error}")
            done_count += 1
            progress.progress(done_count / len(page_chunks), text=f"PDFをAIで読み取り中... {done_count}/{len(page_chunks)}")
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(read_one_chunk, index, label, pdf_bytes)
                for index, (label, pdf_bytes) in enumerate(page_chunks)
            ]
            for future in as_completed(futures):
                result_index, result_label, text, error = future.result()
                results[result_index] = text
                if error:
                    errors.append(f"{result_label}: {error}")
                done_count += 1
                progress.progress(done_count / len(page_chunks), text=f"PDFをAIで読み取り中... {done_count}/{len(page_chunks)}")

    progress.empty()
    extracted_text = filter_english_exam_text("\n\n".join(text for text in results if text.strip()).strip())
    if errors:
        st.warning(f"PDF読み取りの一部で失敗しました。成功したページだけ表示します。失敗: {len(errors)}件")
        with st.expander("失敗したページを見る", expanded=False):
            for error in errors[:20]:
                st.write(error)
    if not extracted_text:
        raise RuntimeError("PDF読み取り結果が空でした。ページ分割を1にするか、待ち時間を長くして再試行してください。")
    return extracted_text

def extract_words_from_text(text):
    # アポストロフィを除外し、純粋なアルファベットのみを抽出
    # （例: student's -> student と s に分かれ、s は後で消える）
    raw_words = re.findall(r"\b[a-z]+\b", text.lower())
    
    cleaned_words = []
    for w in raw_words:
        # ③ ノイズの言葉を除外（1文字以下の単語とストップワードを無視）
        if len(w) <= 1 or w in STOP_WORDS:
            continue
            
        # ① すべて原形にする（動詞として変換を試し、変わらなければ名詞として試す）
        lemma = lemmatizer.lemmatize(w, pos='v')
        lemma = lemmatizer.lemmatize(lemma, pos='n')
        
        cleaned_words.append(lemma)
        
    # ② 戻り値のリストを返す（この後、呼び出し元の word_counts = Counter(new_words) で同じ原形が自動で統合されます）
    return cleaned_words


def flatten_tags(tags_data):
    """
    AIが文字列、リスト、あるいは多重リスト（例: [["関係詞"]]）のいずれを返してきても、
    完全に平坦な（1次元の）文字列リストに展開して返す強靭なフィルター。
    """
    flat_list = []
    if isinstance(tags_data, str):
        flat_list.append(tags_data)
    elif isinstance(tags_data, list):
        for item in tags_data:
            flat_list.extend(flatten_tags(item))
    return flat_list

def infer_question_type(q):
    """
    文法問題を「選択問題」か「整序問題」かに分類する。
    既存DBに question_type がない問題にも対応するための保険。
    """
    question = str(q.get("question", ""))
    options = q.get("options", [])
    answer = q.get("answer", "")

    # すでに分類済みならそれを使う
    if q.get("question_type") in ["multiple_choice", "ordering"]:
        return q["question_type"]

    # answer が options の中にあるなら、普通の選択問題
    if isinstance(options, list) and answer in options:
        return "multiple_choice"

    # ①②③... が question にあるなら、整序問題の可能性が高い
    if any(mark in question for mark in ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨"]):
        return "ordering"

    # 選択肢が多く、answer が完成英文っぽいなら整序問題
    if isinstance(options, list) and len(options) >= 5 and isinstance(answer, str) and " " in answer:
        return "ordering"

    # 不明な場合は今まで通り選択問題扱い
    return "multiple_choice"


def normalize_answer_text(s):
    """
    整序問題の採点用に、大小文字・句読点・余分な空白をならす。
    """
    s = str(s).strip().lower()
    s = re.sub(r"[?.!,，。！？]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def extract_idioms_with_gemini(text):
    # --- (sys_prompt の定義はそのまま) ---
    sys_prompt = """AIを使って長文から熟語・イディオムを抽出し、JSONで返す関数"""
    sys_prompt = """
    あなたは大学受験英語の傾向分析と対策に命を懸けているプロの予備校講師であり、精密な構文解析スキャナーです。
    提供された過去問のテキスト（長文、四択問題、整序問題など）を【1文ずつ舐めるように精査】し、大学受験レベル(B1〜B2以上)で重要な熟語・語法を【絶対に一つも漏らさず】抽出し、JSON形式で出力してください。

    【抽出対象の4カテゴリ】（※これらに該当するものは全て拾うこと）
    1. 句動詞・動詞の語法（例: let down, divide into, associate with, arise from, depend on）
    2. 形容詞＋前置詞のコロケーション（例: particular about, familiar with, independent of, aware of）
    3. 名詞＋前置詞のコロケーション（例: lack of, key to, responsibility for, demand for）
    4. 定型イディオム・構文・フレーズ（例: on the other hand, in spite of, not only A but also B）

    【最重要：設問・選択肢への特殊スキャンルール】
    ・空所補充問題（例: 文中に `(    ) about` とあり、選択肢に `particular` がある場合）は、空所と選択肢を脳内で結合し、「particular about」として抽出すること。正解・不正解は問わず、受験で狙われる重要な組み合わせは全て抽出せよ。
    ・並び替え問題でバラバラになっている単語群も、意味を成す熟語として結合して抽出すること。
    
    【長文に対するルール】
    ・長文の意味や内容は一切理解しなくてよい。「どんな熟語・語法パーツが使われているか」という機械的な視点でのみテキストをスキャンすること。

    【無視・除外ルール】
    ・「設問の日本語の指示文」や「グラフや図表に関する記述（Q2など）」はノイズなので完全に無視すること。
    ・「global warming」のような単なる複合名詞は除外すること。

    【出力ルール】
    ・過去形や進行形、選択肢でバラバラな状態であっても、"base_form"は必ず【原形・基本形】に統一。
    ・"quotes_in_text" には、テキスト内で実際に使われていた形（または空所と選択肢の組み合わせ）をそのまま引用。

    【JSONフォーマット】
    {
      "idioms": [
        {
          "base_form": "particular about",
          "quotes_in_text": ["(    ) about / particular (選択肢より)"],
          "count": 1
        },
        {
          "base_form": "let down",
          "quotes_in_text": ["let me (    ) / down (選択肢より)"],
          "count": 1
        }
      ]
    }
    """
    
    model = genai.GenerativeModel(model_name="gemini-2.5-pro", system_instruction=sys_prompt, generation_config={"response_mime_type": "application/json"})
    try:
        res = model.generate_content(f"以下のテキストから熟語を抽出してください:\n\n{text}")
        parsed = json.loads(res.text)
        
        # 🛡️ AIがリストを直返ししてきた場合の強靭化処理
        if isinstance(parsed, list):
            return {"idioms": parsed}
        return parsed
        
    except Exception as e:
        st.error(f"熟語の抽出中にエラーが発生しました: {e}")
        return {"idioms": []}
    

GEMINI_MODEL_ALIASES = {
    "3-flash": "gemini-2.5-flash",
    "gemini-3-flash": "gemini-2.5-flash",
    "gemini-3.5-flash": "gemini-2.5-flash",
    "3-flash-lite": "gemini-2.5-flash-lite",
    "gemini-3-flash-lite": "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite": "gemini-2.5-flash-lite",
    "gemini-flash-lite": "gemini-2.5-flash-lite",
}


def normalize_gemini_model_name(model_name):
    model_name = str(model_name or "").strip()
    return GEMINI_MODEL_ALIASES.get(model_name, model_name or "gemini-2.5-flash-lite")


def get_gemini_model_chain(model_name):
    primary = normalize_gemini_model_name(model_name)
    chain = [primary]
    for fallback in ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"]:
        if fallback not in chain:
            chain.append(fallback)
    return chain


def parse_json_response(text):
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def call_gemini_json(prompt, sys_prompt, model_name="gemini-2.5-flash-lite", max_output_tokens=3500, timeout_seconds=60):
    last_error = None
    for candidate in get_gemini_model_chain(model_name):
        try:
            model = genai.GenerativeModel(
                model_name=candidate,
                system_instruction=sys_prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "max_output_tokens": max_output_tokens,
                },
            )
            res = model.generate_content(prompt, request_options={"timeout": timeout_seconds})
            return parse_json_response(res.text)
        except Exception as e:
            last_error = e
    raise last_error


def normalize_idiom_base(value):
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def is_valid_idiom_base(base_form):
    """Keep multi-word idioms/collocations only; single vocabulary words belong in word analysis."""
    words = re.findall(r"[a-z]+", normalize_idiom_base(base_form))
    return len(words) >= 2


def normalize_meaning_key(value):
    key = re.sub(r"\s+", " ", str(value or "").strip())
    return key or "未分類"


IDIOM_MEANING_FAMILIES = [
    {
        "id": "idea_create",
        "canonical": "考え・解決策を思いつく、考案する",
        "keywords": ["思いつ", "考案", "考え出", "提案", "案を出", "解決策"],
    },
    {
        "id": "spread",
        "canonical": "徐々に浸透する、普及する",
        "keywords": ["浸透", "普及", "広ま", "行き渡"],
    },
    {
        "id": "draw_extract",
        "canonical": "〜を引き出す、吸い出す",
        "keywords": ["引き出", "吸い出", "吸収", "取り出"],
    },
    {
        "id": "transition",
        "canonical": "AからBへ移行する、変化する",
        "keywords": ["移行", "移り変", "変化", "移る", "変わる"],
    },
    {
        "id": "origin",
        "canonical": "〜に由来する、〜から来る",
        "keywords": ["由来", "から来", "起源", "源"],
    },
    {
        "id": "use",
        "canonical": "〜を利用する、活用する",
        "keywords": ["利用", "活用"],
    },
    {
        "id": "exploit",
        "canonical": "人の弱みなどにつけこむ、悪用する",
        "keywords": ["つけこ", "弱み", "悪用"],
    },
    {
        "id": "past_habit",
        "canonical": "以前は〜だった、かつて〜していた",
        "keywords": ["以前", "かつて", "昔は", "今は違う"],
    },
    {
        "id": "accustomed",
        "canonical": "〜に慣れている",
        "keywords": ["慣れて", "慣れる"],
    },
    {
        "id": "execute",
        "canonical": "実行する、実施する",
        "keywords": ["実行", "実施", "遂行", "行う"],
    },
]


def simplify_meaning_text(value):
    text = normalize_meaning_key(value)
    text = re.sub(r"[（）()「」『』【】\[\]〈〉]", "", text)
    text = re.sub(r"[〜~・、。，．;；:：/／\s]", "", text)
    text = re.sub(r"(など|こと|もの|人の|Aの|Bの|AからBへ|AからBの)", "", text)
    return text.lower()


def meaning_family_ids(*values):
    text = "".join(simplify_meaning_text(value) for value in values)
    family_ids = set()
    for family in IDIOM_MEANING_FAMILIES:
        if any(simplify_meaning_text(keyword) in text for keyword in family["keywords"]):
            family_ids.add(family["id"])
    return family_ids


def char_bigrams(text):
    text = simplify_meaning_text(text)
    if len(text) <= 1:
        return {text} if text else set()
    return {text[i:i + 2] for i in range(len(text) - 1)}


def meaning_similarity(a, b):
    a_bigrams = char_bigrams(a)
    b_bigrams = char_bigrams(b)
    if not a_bigrams or not b_bigrams:
        return 0.0
    return len(a_bigrams & b_bigrams) / len(a_bigrams | b_bigrams)


def should_merge_idiom_meanings(existing_key, existing_ja, new_key, new_ja):
    existing_text = f"{existing_key} {existing_ja}"
    new_text = f"{new_key} {new_ja}"
    existing_simple = simplify_meaning_text(existing_text)
    new_simple = simplify_meaning_text(new_text)

    if existing_simple and existing_simple == new_simple:
        return True

    existing_families = meaning_family_ids(existing_text)
    new_families = meaning_family_ids(new_text)
    common_families = existing_families & new_families
    if common_families:
        return True

    if min(len(existing_simple), len(new_simple)) >= 6:
        if existing_simple in new_simple or new_simple in existing_simple:
            return True
        if meaning_similarity(existing_text, new_text) >= 0.62:
            return True

    return False


def canonical_idiom_meaning_label(existing_key, existing_ja, new_key, new_ja):
    existing_text = f"{existing_key} {existing_ja}"
    new_text = f"{new_key} {new_ja}"
    common_families = meaning_family_ids(existing_text) & meaning_family_ids(new_text)
    for family in IDIOM_MEANING_FAMILIES:
        if family["id"] in common_families:
            return family["canonical"]

    existing_ja = normalize_meaning_key(existing_ja or existing_key)
    new_ja = normalize_meaning_key(new_ja or new_key)
    return existing_ja if len(existing_ja) >= len(new_ja) else new_ja


def slug_fragment(value, default="meaning"):
    slug = re.sub(r"[^a-z0-9]+", "_", normalize_idiom_base(value)).strip("_")
    return (slug[:48].strip("_") or default)


def is_meaning_id(value):
    return bool(re.match(r"^(idm|wd)_[a-z0-9_]+_\d{3}$", str(value or "")))


def next_meaning_id(kind, base_form, meanings):
    prefix = "idm" if kind == "idiom" else "wd"
    stem = slug_fragment(base_form, default=prefix)
    base_id = f"{prefix}_{stem}_"
    max_num = 0
    for existing_id in meanings.keys():
        match = re.match(rf"^{re.escape(base_id)}(\d+)$", str(existing_id))
        if match:
            max_num = max(max_num, int(match.group(1)))
    while True:
        max_num += 1
        candidate = f"{base_id}{max_num:03d}"
        if candidate not in meanings:
            return candidate


def add_unique_alias(entry, alias, limit=20):
    alias = normalize_meaning_key(alias)
    if not alias:
        return
    aliases = entry.setdefault("aliases", [])
    if alias not in aliases and alias != entry.get("meaning_key") and alias != entry.get("meaning_ja"):
        if len(aliases) < limit:
            aliases.append(alias)


def normalize_meaning_entry(meaning_id, data, fallback_key=None):
    if not isinstance(data, dict):
        data = {"meaning_ja": normalize_meaning_key(data or fallback_key)}
    meaning_key = normalize_meaning_key(data.get("meaning_key") or fallback_key or data.get("meaning_ja") or meaning_id)
    meaning_ja = normalize_meaning_key(data.get("meaning_ja") or meaning_key)
    entry = {
        "meaning_id": meaning_id,
        "meaning_key": meaning_key,
        "meaning_ja": meaning_ja,
        "aliases": list(data.get("aliases", [])) if isinstance(data.get("aliases"), list) else [],
        "usage_hints": list(data.get("usage_hints", [])) if isinstance(data.get("usage_hints"), list) else [],
    }
    add_unique_alias(entry, fallback_key)
    add_unique_alias(entry, meaning_key)
    add_unique_alias(entry, meaning_ja)
    return entry


def meaning_entry_pairs(meaning_id, data):
    if not isinstance(data, dict):
        text = normalize_meaning_key(data or meaning_id)
        return [(meaning_id, text), (text, text)]
    pairs = []
    meaning_key = normalize_meaning_key(data.get("meaning_key") or meaning_id)
    meaning_ja = normalize_meaning_key(data.get("meaning_ja") or meaning_key)
    pairs.append((meaning_key, meaning_ja))
    pairs.append((meaning_ja, meaning_ja))
    for alias in data.get("aliases", []):
        alias = normalize_meaning_key(alias)
        if alias:
            pairs.append((alias, meaning_ja))
    return pairs


def find_matching_idiom_meaning_key(meanings, meaning_key, meaning_ja):
    for existing_key, existing_data in meanings.items():
        for existing_label, existing_ja in meaning_entry_pairs(existing_key, existing_data):
            if should_merge_idiom_meanings(existing_label, existing_ja, meaning_key, meaning_ja):
                return existing_key
    return meaning_key


def find_or_create_meaning_id(meanings, kind, base_form, meaning_key, meaning_ja):
    meaning_key = normalize_meaning_key(meaning_key)
    meaning_ja = normalize_meaning_key(meaning_ja or meaning_key)
    matched_id = None
    for existing_id, existing_data in meanings.items():
        for existing_label, existing_ja in meaning_entry_pairs(existing_id, existing_data):
            if should_merge_idiom_meanings(existing_label, existing_ja, meaning_key, meaning_ja):
                matched_id = existing_id
                break
        if matched_id:
            break

    if not matched_id:
        matched_id = next_meaning_id(kind, base_form, meanings)
        meanings[matched_id] = normalize_meaning_entry(matched_id, {
            "meaning_key": meaning_key,
            "meaning_ja": meaning_ja,
            "usage_hints": [],
            "aliases": [],
        })

    entry = normalize_meaning_entry(matched_id, meanings.get(matched_id, {}), fallback_key=meaning_key)
    entry["meaning_ja"] = canonical_idiom_meaning_label(
        entry.get("meaning_key", matched_id),
        entry.get("meaning_ja", meaning_key),
        meaning_key,
        meaning_ja,
    )
    add_unique_alias(entry, meaning_key)
    add_unique_alias(entry, meaning_ja)
    meanings[matched_id] = entry
    return matched_id, entry


def migrate_lexicon_meanings(lex_entry, kind, base_form):
    raw_meanings = lex_entry.get("meanings", {})
    if not isinstance(raw_meanings, dict):
        raw_meanings = {}
    migrated = {}
    for raw_key, raw_data in raw_meanings.items():
        existing_id = None
        if isinstance(raw_data, dict) and is_meaning_id(raw_data.get("meaning_id")):
            existing_id = raw_data.get("meaning_id")
        elif is_meaning_id(raw_key):
            existing_id = raw_key

        if not existing_id:
            existing_id = next_meaning_id(kind, base_form, migrated)

        entry = normalize_meaning_entry(existing_id, raw_data, fallback_key=None if is_meaning_id(raw_key) else raw_key)
        match_id = find_matching_idiom_meaning_key(migrated, entry["meaning_key"], entry["meaning_ja"]) if migrated else existing_id
        if is_meaning_id(match_id):
            existing_id = match_id
        target = migrated.setdefault(existing_id, normalize_meaning_entry(existing_id, entry, fallback_key=entry["meaning_key"]))
        target["meaning_ja"] = canonical_idiom_meaning_label(
            target.get("meaning_key", existing_id),
            target.get("meaning_ja", existing_id),
            entry.get("meaning_key", existing_id),
            entry.get("meaning_ja", existing_id),
        )
        for alias in entry.get("aliases", []):
            add_unique_alias(target, alias)
        for hint in entry.get("usage_hints", []):
            if hint and hint not in target.setdefault("usage_hints", []) and len(target["usage_hints"]) < 6:
                target["usage_hints"].append(hint)
    lex_entry["meanings"] = migrated
    return migrated


def collapse_similar_meaning_counts(meaning_counts):
    collapsed = {}
    for meaning_key, data in meaning_counts.items():
        if not isinstance(data, dict):
            continue
        meaning_key = normalize_meaning_key(meaning_key)
        meaning_ja = normalize_meaning_key(data.get("meaning_ja") or meaning_key)
        target_key = find_matching_idiom_meaning_key(collapsed, meaning_key, meaning_ja)

        if target_key not in collapsed:
            collapsed[target_key] = {
                "meaning_ja": meaning_ja,
                "count": 0,
                "usage_hints": [],
            }

        bucket = collapsed[target_key]
        bucket["meaning_ja"] = canonical_idiom_meaning_label(
            target_key,
            bucket.get("meaning_ja", target_key),
            meaning_key,
            meaning_ja,
        )
        bucket["count"] = safe_count(bucket.get("count"), 0) + safe_count(data.get("count"), 0)
        for hint in data.get("usage_hints", []):
            if hint and hint not in bucket.setdefault("usage_hints", []) and len(bucket["usage_hints"]) < 5:
                bucket["usage_hints"].append(hint)
    return collapsed


def meaning_counts_use_ids(meaning_counts):
    if not isinstance(meaning_counts, dict):
        return False
    for key, data in meaning_counts.items():
        if is_meaning_id(key):
            return True
        if isinstance(data, dict) and is_meaning_id(data.get("meaning_id")):
            return True
    return False


def normalize_count_key_from_occurrence(occurrence):
    meaning_id = occurrence.get("meaning_id") if isinstance(occurrence, dict) else ""
    if is_meaning_id(meaning_id):
        return meaning_id
    return normalize_meaning_key(occurrence.get("meaning_key") if isinstance(occurrence, dict) else "")


def safe_count(value, default=1):
    try:
        return max(0, int(value))
    except Exception:
        return default


def split_text_for_ai(text, chunk_chars=4500):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    text = re.sub(r"([.!?])(?=[A-Z0-9(])", r"\1 ", text)
    text = re.sub(r"(\))(?=[A-Z])", r"\1 ", text)
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""

    def append_piece(piece):
        piece = piece.strip()
        while len(piece) > chunk_chars:
            split_at = max(
                piece.rfind(" ", 0, chunk_chars),
                piece.rfind(",", 0, chunk_chars),
                piece.rfind(";", 0, chunk_chars),
            )
            if split_at < max(200, chunk_chars // 3):
                split_at = chunk_chars
            chunks.append(piece[:split_at].strip())
            piece = piece[split_at:].strip()
        if piece:
            chunks.append(piece)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > chunk_chars:
            if current:
                chunks.append(current)
                current = ""
            append_piece(sentence)
            continue
        if len(current) + len(sentence) + 1 > chunk_chars and current:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def load_idiom_lexicon():
    data = load_json_file(IDIOM_LEXICON_FILE, {"version": 2, "idioms": {}})
    if not isinstance(data, dict):
        data = {"version": 2, "idioms": {}}
    data["version"] = max(2, safe_count(data.get("version"), 2))
    data.setdefault("idioms", {})
    return data


def save_idiom_lexicon(lexicon):
    with open(IDIOM_LEXICON_FILE, "w", encoding="utf-8") as f:
        json.dump(lexicon, f, ensure_ascii=False, indent=2)


def normalize_idiom_items_with_lexicon(raw_items, lexicon):
    normalized_items = []
    lexicon["version"] = 2
    idiom_dict = lexicon.setdefault("idioms", {})

    for item in raw_items:
        if not isinstance(item, dict):
            continue

        base_form = normalize_idiom_base(item.get("base_form"))
        if not base_form or not is_valid_idiom_base(base_form):
            continue

        occurrences = item.get("occurrences")
        if not isinstance(occurrences, list) or not occurrences:
            occurrences = [{
                "meaning_key": item.get("meaning_key") or item.get("meaning_ja") or "未分類",
                "meaning_ja": item.get("meaning_ja") or item.get("meaning_key") or "意味未分類",
                "usage_hint": item.get("usage_hint", ""),
                "surface_form": item.get("surface_form", base_form),
                "count": item.get("count", 1),
            }]

        lex_entry = idiom_dict.setdefault(base_form, {"base_form": base_form, "meanings": {}})
        lex_entry["base_form"] = base_form
        meanings = migrate_lexicon_meanings(lex_entry, "idiom", base_form)
        normalized_occurrences = []

        for occurrence in occurrences:
            if not isinstance(occurrence, dict):
                continue

            meaning_ja = normalize_meaning_key(occurrence.get("meaning_ja") or occurrence.get("meaning_key"))
            meaning_key = normalize_meaning_key(occurrence.get("meaning_key") or meaning_ja)
            count = safe_count(occurrence.get("count"), 1)
            usage_hint = str(occurrence.get("usage_hint", "")).strip()
            surface_form = normalize_idiom_base(occurrence.get("surface_form") or base_form)

            meaning_id, meaning_entry = find_or_create_meaning_id(
                meanings,
                "idiom",
                base_form,
                meaning_key,
                meaning_ja,
            )
            if usage_hint and usage_hint not in meaning_entry.setdefault("usage_hints", []) and len(meaning_entry["usage_hints"]) < 6:
                meaning_entry["usage_hints"].append(usage_hint)

            normalized_occurrences.append({
                "meaning_id": meaning_id,
                "meaning_key": meaning_key,
                "meaning_ja": meaning_entry.get("meaning_ja", meaning_ja),
                "usage_hint": usage_hint,
                "surface_form": surface_form,
                "count": count,
            })

        if normalized_occurrences:
            normalized_items.append({
                "base_form": base_form,
                "count": sum(occ["count"] for occ in normalized_occurrences),
                "occurrences": normalized_occurrences,
            })

    return normalized_items, lexicon


def merge_idiom_items(existing_idioms, idiom_items):
    merged = existing_idioms if isinstance(existing_idioms, dict) else {}

    for item in idiom_items:
        base_form = normalize_idiom_base(item.get("base_form"))
        if not base_form or not is_valid_idiom_base(base_form):
            continue

        entry = merged.setdefault(base_form, {"count": 0, "meaning_counts": {}})
        entry["count"] = safe_count(entry.get("count"), 0) + safe_count(item.get("count"), 0)
        if not meaning_counts_use_ids(entry.get("meaning_counts", {})):
            entry["meaning_counts"] = collapse_similar_meaning_counts(entry.get("meaning_counts", {}))
        meaning_counts = entry.setdefault("meaning_counts", {})

        for occurrence in item.get("occurrences", []):
            meaning_id = occurrence.get("meaning_id")
            meaning_key = normalize_meaning_key(occurrence.get("meaning_key") or occurrence.get("meaning_ja"))
            meaning_ja = normalize_meaning_key(occurrence.get("meaning_ja") or meaning_key)
            count = safe_count(occurrence.get("count"), 1)
            usage_hint = str(occurrence.get("usage_hint", "")).strip()

            count_key = meaning_id if is_meaning_id(meaning_id) else meaning_key
            if not is_meaning_id(count_key):
                count_key = find_matching_idiom_meaning_key(meaning_counts, meaning_key, meaning_ja)
            bucket = meaning_counts.setdefault(count_key, {
                "meaning_id": count_key if is_meaning_id(count_key) else "",
                "meaning_key": meaning_key,
                "meaning_ja": meaning_ja,
                "count": 0,
                "usage_hints": [],
            })
            if is_meaning_id(count_key):
                bucket["meaning_id"] = count_key
            bucket.setdefault("meaning_key", meaning_key)
            bucket["meaning_ja"] = canonical_idiom_meaning_label(
                bucket.get("meaning_key") or count_key,
                bucket.get("meaning_ja", meaning_key),
                occurrence.get("meaning_key") or meaning_ja,
                meaning_ja,
            )
            bucket["count"] = safe_count(bucket.get("count"), 0) + count
            if usage_hint and usage_hint not in bucket.setdefault("usage_hints", []) and len(bucket["usage_hints"]) < 5:
                bucket["usage_hints"].append(usage_hint)

    for entry in merged.values():
        if isinstance(entry, dict) and not meaning_counts_use_ids(entry.get("meaning_counts", {})):
            entry["meaning_counts"] = collapse_similar_meaning_counts(entry.get("meaning_counts", {}))

    return dict(sorted(merged.items(), key=lambda x: safe_count(x[1].get("count"), 0), reverse=True))


def format_idiom_meaning_counts(data):
    meaning_counts = data.get("meaning_counts", {}) if isinstance(data, dict) else {}
    if not meaning_counts:
        return "未解析"
    if not meaning_counts_use_ids(meaning_counts):
        meaning_counts = collapse_similar_meaning_counts(meaning_counts)
    rows = sorted(
        meaning_counts.items(),
        key=lambda item: safe_count(item[1].get("count"), 0),
        reverse=True,
    )
    return " / ".join(
        f"{value.get('meaning_ja', key)}:{safe_count(value.get('count'), 0)}"
        for key, value in rows
    )


def load_word_meaning_lexicon():
    data = load_json_file(WORD_MEANING_LEXICON_FILE, {"version": 2, "words": {}})
    if not isinstance(data, dict):
        data = {"version": 2, "words": {}}
    data["version"] = max(2, safe_count(data.get("version"), 2))
    data.setdefault("words", {})
    return data


def save_word_meaning_lexicon(lexicon):
    with open(WORD_MEANING_LEXICON_FILE, "w", encoding="utf-8") as f:
        json.dump(lexicon, f, ensure_ascii=False, indent=2)


def normalize_word_meaning_items_with_lexicon(raw_items, lexicon):
    normalized_items = []
    lexicon["version"] = 2
    word_dict = lexicon.setdefault("words", {})

    for item in raw_items:
        if not isinstance(item, dict):
            continue

        word = normalize_vocab_word(item.get("word") or item.get("base_form"))
        if not word or len(word) <= 1 or word in STOP_WORDS:
            continue

        occurrences = item.get("occurrences")
        if not isinstance(occurrences, list) or not occurrences:
            occurrences = [{
                "meaning_key": item.get("meaning_key") or item.get("meaning_ja") or "未分類",
                "meaning_ja": item.get("meaning_ja") or item.get("meaning_key") or "意味未分類",
                "usage_hint": item.get("usage_hint", ""),
                "surface_form": item.get("surface_form", word),
                "count": item.get("count", 1),
            }]

        lex_entry = word_dict.setdefault(word, {"word": word, "meanings": {}})
        lex_entry["word"] = word
        meanings = migrate_lexicon_meanings(lex_entry, "word", word)
        normalized_occurrences = []

        for occurrence in occurrences:
            if not isinstance(occurrence, dict):
                continue

            meaning_ja = normalize_meaning_key(occurrence.get("meaning_ja") or occurrence.get("meaning_key"))
            meaning_key = normalize_meaning_key(occurrence.get("meaning_key") or meaning_ja)
            count = safe_count(occurrence.get("count"), 1)
            usage_hint = str(occurrence.get("usage_hint", "")).strip()
            surface_form = str(occurrence.get("surface_form") or word).strip().lower()

            meaning_id, meaning_entry = find_or_create_meaning_id(
                meanings,
                "word",
                word,
                meaning_key,
                meaning_ja,
            )
            if usage_hint and usage_hint not in meaning_entry.setdefault("usage_hints", []) and len(meaning_entry["usage_hints"]) < 6:
                meaning_entry["usage_hints"].append(usage_hint)

            normalized_occurrences.append({
                "meaning_id": meaning_id,
                "meaning_key": meaning_key,
                "meaning_ja": meaning_entry.get("meaning_ja", meaning_ja),
                "usage_hint": usage_hint,
                "surface_form": surface_form,
                "count": count,
            })

        if normalized_occurrences:
            normalized_items.append({
                "word": word,
                "count": sum(occ["count"] for occ in normalized_occurrences),
                "occurrences": normalized_occurrences,
            })

    return normalized_items, lexicon


def merge_word_meaning_items(existing_words, word_items):
    merged = existing_words if isinstance(existing_words, dict) else {}

    for item in word_items:
        word = normalize_vocab_word(item.get("word") or item.get("base_form"))
        if not word or word in STOP_WORDS:
            continue

        entry = merged.setdefault(word, {"count": 0, "meaning_counts": {}})
        entry["count"] = safe_count(entry.get("count"), 0) + safe_count(item.get("count"), 0)
        if not meaning_counts_use_ids(entry.get("meaning_counts", {})):
            entry["meaning_counts"] = collapse_similar_meaning_counts(entry.get("meaning_counts", {}))
        meaning_counts = entry.setdefault("meaning_counts", {})

        for occurrence in item.get("occurrences", []):
            meaning_id = occurrence.get("meaning_id")
            meaning_key = normalize_meaning_key(occurrence.get("meaning_key") or occurrence.get("meaning_ja"))
            meaning_ja = normalize_meaning_key(occurrence.get("meaning_ja") or meaning_key)
            count = safe_count(occurrence.get("count"), 1)
            usage_hint = str(occurrence.get("usage_hint", "")).strip()

            count_key = meaning_id if is_meaning_id(meaning_id) else meaning_key
            if not is_meaning_id(count_key):
                count_key = find_matching_idiom_meaning_key(meaning_counts, meaning_key, meaning_ja)
            bucket = meaning_counts.setdefault(count_key, {
                "meaning_id": count_key if is_meaning_id(count_key) else "",
                "meaning_key": meaning_key,
                "meaning_ja": meaning_ja,
                "count": 0,
                "usage_hints": [],
            })
            if is_meaning_id(count_key):
                bucket["meaning_id"] = count_key
            bucket.setdefault("meaning_key", meaning_key)
            bucket["meaning_ja"] = canonical_idiom_meaning_label(
                bucket.get("meaning_key") or count_key,
                bucket.get("meaning_ja", meaning_key),
                occurrence.get("meaning_key") or meaning_ja,
                meaning_ja,
            )
            bucket["count"] = safe_count(bucket.get("count"), 0) + count
            if usage_hint and usage_hint not in bucket.setdefault("usage_hints", []) and len(bucket["usage_hints"]) < 5:
                bucket["usage_hints"].append(usage_hint)

    for entry in merged.values():
        if isinstance(entry, dict) and not meaning_counts_use_ids(entry.get("meaning_counts", {})):
            entry["meaning_counts"] = collapse_similar_meaning_counts(entry.get("meaning_counts", {}))

    return dict(sorted(merged.items(), key=lambda x: safe_count(x[1].get("count"), 0), reverse=True))


def format_word_meaning_counts(data):
    return format_idiom_meaning_counts(data)


def extract_idiom_chunk_with_gemini(text, model_name):
    sys_prompt = """
あなたは大学受験英語の熟語・語法を分析するAIです。
目的は、熟語の総回数だけでなく、意味ごとの回数を保存することです。
英文の長い引用は保存しません。
JSON以外は出力しないでください。
"""
    prompt = f"""
次の英語テキストから、大学受験で重要な熟語・イディオム・語法コロケーションを抽出してください。

【出力ルール】
- base_form は原形・基本形に統一してください。
- 同じ熟語でも意味が違う場合は occurrences を分けてください。
- meaning_key は「受験生が覚える意味単位」の短く安定した日本語ラベルにしてください。
- meaning_ja は生徒に見せる自然な日本語の意味にしてください。
- usage_hint は本文の長い引用ではなく、日本語で短く文脈を説明してください。細かい文脈差は meaning_key ではなく usage_hint に入れてください。
- surface_form は実際に使われた熟語部分だけにしてください。長い文を入れないでください。
- count は同じ意味・同じ用法の出現数です。
- 設問の選択肢や空所補充で完成する熟語も拾ってください。
- 熟語・構文・コロケーションだけを抽出してください。1語だけの単語は絶対に出力しないでください。
- global warming のような単なる複合名詞は除外してください。

【意味分けの粒度】
- 表現が少し違うだけなら、同じ meaning_key に統合してください。日本語訳の言い換え、目的語の種類、括弧内の補足、方向性の補足だけで意味を分けないでください。
- 分けるのは、受験生が読解で別の意味として覚えないと誤読する場合だけです。
- 例: come up with の「思いつく」「考案する」「考え出す」「提案する」は同じ意味単位として「考え・解決策を思いつく、考案する」にまとめる。
- 例: trickle down の「下位に浸透する」「徐々に浸透する」「普及する」は同じ意味単位として「徐々に浸透する、普及する」にまとめる。
- 例: draw out の「引き出す」「吸い出す」「吸収する」は、同じ文脈なら「〜を引き出す、吸い出す」にまとめる。
- 例: go from A to B の「移行する」「変化する」「移り変わる」は「AからBへ移行する、変化する」にまとめる。
- 例: take advantage of は「利用する、活用する」と「人の弱みにつけこむ、悪用する」を必ず分ける。
- 例: used to do は base_form を "used to"、意味を「以前は〜だった、かつて〜していた」にする。be used to 名詞/V-ing は base_form を "be used to"、意味を「〜に慣れている」にする。両者を混ぜない。

【JSON形式】
{{
  "idioms": [
    {{
      "base_form": "take advantage of",
      "count": 2,
      "occurrences": [
        {{
          "meaning_key": "利用する",
          "meaning_ja": "〜を利用する、活用する",
          "surface_form": "take advantage of",
          "usage_hint": "機会や制度を活用する文脈",
          "count": 1
        }},
        {{
          "meaning_key": "つけこむ",
          "meaning_ja": "人の弱みなどにつけこむ",
          "surface_form": "take advantage of",
          "usage_hint": "相手の弱い立場を悪用する文脈",
          "count": 1
        }}
      ]
    }}
  ]
}}

【テキスト】
{text}
"""
    parsed = call_gemini_json(
        prompt,
        sys_prompt,
        model_name=model_name,
        max_output_tokens=3500,
        timeout_seconds=75,
    )
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return parsed.get("idioms", [])
    return []


def extract_idiom_chunk_with_retry(text, model_name, max_depth=2, depth=0):
    try:
        return extract_idiom_chunk_with_gemini(text, model_name), []
    except Exception as e:
        if depth >= max_depth or len(str(text)) < 1200:
            return [], [str(e)]

    smaller_chunks = split_text_for_ai(text, chunk_chars=max(1200, len(str(text)) // 2))
    if len(smaller_chunks) <= 1:
        mid = len(str(text)) // 2
        split_at = str(text).rfind(" ", 0, mid)
        if split_at < 400:
            split_at = mid
        smaller_chunks = [str(text)[:split_at], str(text)[split_at:]]

    items = []
    errors = []
    for chunk in smaller_chunks:
        chunk_items, chunk_errors = extract_idiom_chunk_with_retry(
            chunk,
            model_name,
            max_depth=max_depth,
            depth=depth + 1,
        )
        items.extend(chunk_items)
        errors.extend(chunk_errors)
    return items, errors


def extract_idioms_with_gemini(text, model_name="gemini-2.5-pro", max_workers=2, chunk_chars=2500):
    chunks = split_text_for_ai(text, chunk_chars=chunk_chars)
    if not chunks:
        return {"idioms": []}

    raw_items = []
    errors = []
    worker_count = max(1, min(int(max_workers), len(chunks)))
    progress = st.progress(0, text=f"熟語を意味別に解析中... 0/{len(chunks)}")
    done_count = 0

    if worker_count == 1:
        for chunk in chunks:
            chunk_items, chunk_errors = extract_idiom_chunk_with_retry(chunk, model_name)
            raw_items.extend(chunk_items)
            errors.extend(chunk_errors)
            done_count += 1
            progress.progress(done_count / len(chunks), text=f"熟語を意味別に解析中... {done_count}/{len(chunks)}")
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(extract_idiom_chunk_with_retry, chunk, model_name) for chunk in chunks]
            for future in as_completed(futures):
                chunk_items, chunk_errors = future.result()
                raw_items.extend(chunk_items)
                errors.extend(chunk_errors)
                done_count += 1
                progress.progress(done_count / len(chunks), text=f"熟語を意味別に解析中... {done_count}/{len(chunks)}")

    progress.empty()

    lexicon = load_idiom_lexicon()
    idiom_items, lexicon = normalize_idiom_items_with_lexicon(raw_items, lexicon)
    save_idiom_lexicon(lexicon)

    if errors:
        st.warning(f"熟語解析の一部で失敗しました。成功分だけ保存します。失敗チャンク: {len(errors)}")

    return {"idioms": idiom_items, "chunks": len(chunks), "errors": errors}


def extract_word_meaning_chunk_with_gemini(text, model_name):
    sys_prompt = """
あなたは大学受験英語の単語の意味を、本文の文脈ごとに分類するAIです。
目的は、単語の総回数だけでなく、意味ごとの回数を保存することです。
英文の長い引用は保存しません。
JSON以外は出力しないでください。
"""
    prompt = f"""
次の英語テキストから、大学受験で読解上重要な英単語を抽出し、文脈上の意味ごとに分類してください。

【抽出対象】
- 名詞・動詞・形容詞・副詞を中心に、読解で意味判断が必要な語をできるだけ広く拾ってください。
- base word は小文字の原形・単数形に統一してください。例: made -> make, studies -> study, better -> good
- 冠詞、前置詞、代名詞、助動詞、be動詞などの機能語は除外してください。
- 人名・地名・大学名・OCR崩れ・記号・数字だけの語は除外してください。

【意味分けの粒度】
- meaning_key は「受験生がその単語を読むときに選ぶ意味単位」の短い日本語ラベルにしてください。
- 日本語訳の言い換えだけなら同じ meaning_key に統合してください。
- 分けるのは、別の意味として覚えないと誤読する場合だけです。
- 例: make は「作る」「AをBにする」「〜させる」「たどり着く・間に合う」を必要に応じて分ける。
- 例: like は「好む」「〜のような」「〜に似ている」を分ける。
- 例: way は「道」「方法・やり方」「点・側面」を分ける。
- 例: people は「人々」と「国民・民族」を必要に応じて分ける。
- 文脈の細かい違いは usage_hint に入れ、意味ラベルを増やしすぎないでください。

【出力ルール】
- meaning_ja は生徒に見せる自然な日本語の意味にしてください。
- usage_hint は本文の長い引用ではなく、日本語で短く文脈を説明してください。
- surface_form は実際に出た語形だけにしてください。
- count は同じ単語・同じ意味の出現数です。

【JSON形式】
{{
  "words": [
    {{
      "word": "make",
      "count": 3,
      "occurrences": [
        {{
          "meaning_key": "作る",
          "meaning_ja": "〜を作る",
          "surface_form": "made",
          "usage_hint": "何かを作成する文脈",
          "count": 1
        }},
        {{
          "meaning_key": "AをBにする",
          "meaning_ja": "AをBの状態にする",
          "surface_form": "makes",
          "usage_hint": "目的語の状態を変える文脈",
          "count": 2
        }}
      ]
    }}
  ]
}}

【テキスト】
{text}
"""
    parsed = call_gemini_json(
        prompt,
        sys_prompt,
        model_name=model_name,
        max_output_tokens=5000,
        timeout_seconds=75,
    )
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return parsed.get("words", [])
    return []


def extract_word_meaning_chunk_with_retry(text, model_name, max_depth=2, depth=0):
    try:
        return extract_word_meaning_chunk_with_gemini(text, model_name), []
    except Exception as e:
        if depth >= max_depth or len(str(text)) < 1200:
            return [], [str(e)]

    smaller_chunks = split_text_for_ai(text, chunk_chars=max(1200, len(str(text)) // 2))
    if len(smaller_chunks) <= 1:
        mid = len(str(text)) // 2
        split_at = str(text).rfind(" ", 0, mid)
        if split_at < 400:
            split_at = mid
        smaller_chunks = [str(text)[:split_at], str(text)[split_at:]]

    items = []
    errors = []
    for chunk in smaller_chunks:
        chunk_items, chunk_errors = extract_word_meaning_chunk_with_retry(
            chunk,
            model_name,
            max_depth=max_depth,
            depth=depth + 1,
        )
        items.extend(chunk_items)
        errors.extend(chunk_errors)
    return items, errors


def extract_word_meanings_with_gemini(text, model_name="gemini-2.5-pro", max_workers=2, chunk_chars=2000):
    chunks = split_text_for_ai(text, chunk_chars=chunk_chars)
    if not chunks:
        return {"words": []}

    raw_items = []
    errors = []
    worker_count = max(1, min(int(max_workers), len(chunks)))
    progress = st.progress(0, text=f"単語を意味別に解析中... 0/{len(chunks)}")
    done_count = 0

    if worker_count == 1:
        for chunk in chunks:
            chunk_items, chunk_errors = extract_word_meaning_chunk_with_retry(chunk, model_name)
            raw_items.extend(chunk_items)
            errors.extend(chunk_errors)
            done_count += 1
            progress.progress(done_count / len(chunks), text=f"単語を意味別に解析中... {done_count}/{len(chunks)}")
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(extract_word_meaning_chunk_with_retry, chunk, model_name) for chunk in chunks]
            for future in as_completed(futures):
                chunk_items, chunk_errors = future.result()
                raw_items.extend(chunk_items)
                errors.extend(chunk_errors)
                done_count += 1
                progress.progress(done_count / len(chunks), text=f"単語を意味別に解析中... {done_count}/{len(chunks)}")

    progress.empty()

    lexicon = load_word_meaning_lexicon()
    word_items, lexicon = normalize_word_meaning_items_with_lexicon(raw_items, lexicon)
    save_word_meaning_lexicon(lexicon)

    if errors:
        st.warning(f"単語の意味別解析の一部で失敗しました。成功分だけ保存します。失敗チャンク: {len(errors)}")

    return {"words": word_items, "chunks": len(chunks), "errors": errors}


def extract_grammar_with_gemini(text):
    sys_prompt = """
    あなたは大学受験英語の文法・語法問題を分析する予備校講師です。
    入力された過去問テキストから、文法・語法問題の出題意図を抽出し、
    全く同じ知識を問う著作権フリーの完全オリジナル問題を作成してください。

    【重要】
    ・過去問の英文を変えつつ、全く同じ文法、語法の知識を問う問題を出題すること。
    ・必ず required_knowledge を付ける。
    ・JSON以外は出力しない。
    ・解説は正解と不正解の根拠をすべて解説し、丁寧に背景知識等(lieの問題であったら例のように網羅的に）も解説すること。
    ・問題形式は必ず question_type で示すこと。

    【question_type】
    ・4択・空所補充問題は "multiple_choice"。
    ・語句整序・並び替え問題は "ordering"。
    ・multiple_choice の answer は options の中の1つと完全一致させること。
    ・ordering の answer は完成英文にし、answer_order に正しい番号順を入れること。

    【JSON出力形式】
    {
      "grammar_questions": [
        {
          "question_type": "multiple_choice",
          "question": "The dog was (      ) on the bed.",
          "options": ["laying", "lying", "lain", "lied"],
          "answer": "lying",
          "required_knowledge": ["自動詞lieと他動詞layの区別", "現在分詞"],
          "explanation": "正解はlying。lieは自動詞として使われ、嘘をつくと横になる（横たわる）の二つの意味がある。lie「嘘をつく」の場合は、lie(原型)-lied(過去形)-lied（過去分詞）-lying(進行形)という活用形をとる。また、lie「横になる」は、lie(原型)-lay(過去形)-lain(過去分詞)-lying(進行形)という活用形になる。layは他動詞として使われ、「横たえる・置く」という意味がある。活用形は、lay-laid-laid-layingとなる。本文は自動詞として使われ現在分詞が必要であるからlyingが正解。layingは他動詞layの現在分詞だから不適。lainは過去分詞であるため不適。liedは「嘘をつく」の過去形・過去分詞であり不適。"
        },
        {
          "question_type": "ordering",
          "question": "次の語句を並べ替えて、自然な英文を完成させなさい。",
          "options": ["what", "the presentation", "so successful", "made", "was", "that", "it"],
          "answer_order": [1, 5, 7, 6, 4, 2, 3],
          "answer": "What was it that made the presentation so successful?",
          "required_knowledge": ["強調構文", "疑問文の語順"],
          "explanation": "正解は What was it that made the presentation so successful?。これは強調構文 it is/was ... that ... を疑問文にした形である。疑問詞 what が文頭に出て、その後は was it that ... の語順になる。made の主語は what で、the presentation so successful が目的語と補語の関係になる。"
        }
      ]
    }
    """

    model = genai.GenerativeModel(
        model_name="gemini-2.5-pro",
        system_instruction=sys_prompt,
        generation_config={"response_mime_type": "application/json"}
    )

    try:
        res = model.generate_content(
            f"以下のテキストから文法・語法の出題意図を抽出し、オリジナル問題を作ってください:\n\n{text}"
        )
        parsed = json.loads(res.text)

        if isinstance(parsed, list):
            return {"grammar_questions": parsed}

        if "grammar_questions" not in parsed:
            return {"grammar_questions": []}

        return parsed

    except Exception as e:
        st.error(f"文法の抽出中にエラーが発生しました: {e}")
        return {"grammar_questions": []}
    
# --- UI用ヘルパー：選択保持 + ソート機能付き ---
def select_or_create(label, options, key_prefix):
    sorted_options = sorted(list(options))
    choices = sorted_options + ["+ 新規作成"]
    
    state_key = f"last_sel_{key_prefix}"
    default_idx = 0
    if state_key in st.session_state and st.session_state[state_key] in choices:
        default_idx = choices.index(st.session_state[state_key])
    
    selected = st.selectbox(label, choices, index=default_idx, key=key_prefix+"_sel")
    st.session_state[state_key] = selected
    
    if selected == "+ 新規作成":
        return st.text_input(f"🆕 新しい{label}を入力", key=key_prefix+"_new")
    return selected

# ==========================================
# UI 画面構築
# ==========================================
st.set_page_config(page_title="過去問DBマネージャー", page_icon="📂", layout="wide")
st.title("📂 過去問データベース管理システム")

tab1, tab2 = st.tabs(["📥 データ登録・追加", "🔍 データベース閲覧・編集"])

db = load_db()

if "draft_text" not in st.session_state:
    st.session_state.draft_text = None

# ------------------------------------------
# タブ1: データ登録・追加
# ------------------------------------------
with tab1:
    input_method = st.radio("データの入力方法", ["PDFをアップロード (AI抽出)", "テキストを直接貼り付け"], horizontal=True)
    
    st.markdown("#### 📂 登録先（階層）の指定")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cat_val = select_or_create("カテゴリ", db.keys(), "c")
        uni_options = db.get(cat_val, {}).keys() if cat_val and cat_val != "+ 新規作成" else []
        uni_val = select_or_create("大学名", uni_options, "u")
        
    with col2:
        fac_options = db.get(cat_val, {}).get(uni_val, {}).keys() if uni_val and uni_val != "+ 新規作成" else []
        fac_val = select_or_create("学部", fac_options, "f")
        
        year_options = [str(y) for y in range(2030, 1990, -1)]
        state_key_y = "last_sel_y"
        default_idx_y = year_options.index("2026") if "2026" in year_options else 0
        if state_key_y in st.session_state and st.session_state[state_key_y] in year_options:
            default_idx_y = year_options.index(st.session_state[state_key_y])
        year_val = st.selectbox("年度", year_options, index=default_idx_y, key="y_sel")
        st.session_state[state_key_y] = year_val
        
    with col3:
        method_options = db.get(cat_val, {}).get(uni_val, {}).get(fac_val, {}).get(year_val, {}).keys() if fac_val and fac_val != "+ 新規作成" else []
        method_val = select_or_create("方式・日程", method_options, "m")

    st.markdown("---")
    
    if input_method == "PDFをアップロード (AI抽出)":
        uploaded_pdf = st.file_uploader("過去問PDFをアップロード", type=["pdf"])
        pdf_col1, pdf_col2 = st.columns([2, 1])
        with pdf_col1:
            pdf_extract_model = st.selectbox(
                "PDF読み取りモデル",
                ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
                index=0,
                key="pdf_extract_model",
                help="見落としを減らしたい場合は Pro のままがおすすめです。軽いモデルは速い一方で読み落としが増える可能性があります。",
            )
        with pdf_col2:
            pdf_extract_timeout = st.number_input(
                "待ち時間上限（秒）",
                min_value=60,
                max_value=600,
                value=180,
                step=30,
                key="pdf_extract_timeout",
            )
        pdf_detail_col1, pdf_detail_col2 = st.columns(2)
        with pdf_detail_col1:
            pdf_pages_per_request = st.number_input(
                "一度に読むページ数",
                min_value=1,
                max_value=5,
                value=4,
                step=1,
                key="pdf_pages_per_request_v2",
                help="4ページずつ読むと速度と安定性のバランスを取りやすいです。失敗時は1〜2に下げてください。",
            )
        with pdf_detail_col2:
            pdf_extract_workers = st.number_input(
                "同時に読む数",
                min_value=1,
                max_value=4,
                value=2,
                step=1,
                key="pdf_extract_workers",
                help="2が安定寄りです。混雑時や失敗時は1に下げてください。",
            )
        pdf_allow_fallback = st.checkbox(
            "失敗したときだけ軽いモデルでも試す（見落としの可能性あり）",
            value=False,
            key="pdf_extract_allow_fallback",
        )
        if uploaded_pdf and st.button("🚀 1. AIでテキストを抽出する"):
            with st.spinner(f"AIがPDFを読み取り中...（{pdf_extract_model} / 最大{int(pdf_extract_timeout)}秒）"):
                try:
                    st.session_state.draft_text = extract_text_with_gemini(
                        uploaded_pdf,
                        model_name=pdf_extract_model,
                        timeout_seconds=int(pdf_extract_timeout),
                        allow_fallback=pdf_allow_fallback,
                        pages_per_request=int(pdf_pages_per_request),
                        max_workers=int(pdf_extract_workers),
                    )
                    if st.session_state.draft_text:
                        st.rerun()
                    else:
                        st.error("AI抽出結果が空でした。モデルや待ち時間を変えて再試行してください。")
                except Exception as e:
                    st.error(f"AI抽出に失敗しました: {e}")
    else:
        pasted_text = st.text_area("テキストを貼り付け", height=100)
        if st.button("📝 1. このテキストで確認へ進む"):
            st.session_state.draft_text = pasted_text
            st.rerun()

    if st.session_state.draft_text:
        edited_text = st.text_area("読み取り結果の確認・編集", value=st.session_state.draft_text, height=250)
        
       # --- 新規追加: 抽出オプションの選択（独立トグル方式） ---
        st.markdown("#### 🎯 抽出オプション")
        st.caption("抽出したい項目をオンにしてください（複数選択可）")
        col_opt1, col_opt2, col_opt3, col_opt4 = st.columns(4)
        with col_opt1:
            ext_words = st.toggle("🔤 単語の抽出 (AI不使用・高速)", value=False)
        with col_opt2:
            ext_word_meanings = st.toggle("🧠 単語の意味別解析 (AI使用)", value=True)
        with col_opt3:
            ext_idioms = st.toggle("🔗 熟語の抽出 (AI使用)", value=True)
        with col_opt4:
            ext_grammar = st.toggle("📖 文法・語法の抽出 (AI使用)", value=True)
        
        word_meaning_model_name = "gemini-2.5-pro"
        word_meaning_workers = 2
        word_meaning_chunk_chars = 2000
        if ext_word_meanings:
            with st.expander("🧠 単語の意味別解析設定", expanded=False):
                st.caption("単語を文脈ごとの意味で数えます。重い処理なので、まずは並列2がおすすめです。")
                col_word_ai1, col_word_ai2, col_word_ai3 = st.columns(3)
                with col_word_ai1:
                    word_meaning_model_name = st.selectbox(
                        "単語意味解析モデル",
                        ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
                        index=0,
                        key="word_meaning_analysis_model",
                    )
                with col_word_ai2:
                    word_meaning_workers = st.number_input(
                        "単語解析の並列数",
                        min_value=1,
                        max_value=6,
                        value=2,
                        step=1,
                        key="word_meaning_analysis_workers",
                    )
                with col_word_ai3:
                    word_meaning_chunk_chars = st.number_input(
                        "単語解析の1回あたり文字数",
                        min_value=1200,
                        max_value=7000,
                        value=2000,
                        step=500,
                        key="word_meaning_analysis_chunk_chars_v2",
                    )
                st.info("単語の意味は word_meaning_lexicon.json に蓄積し、各過去問DBには意味別の回数を保存します。")

        idiom_model_name = "gemini-2.5-pro"
        idiom_workers = 2
        idiom_chunk_chars = 2500
        if ext_idioms:
            with st.expander("🔗 熟語の全解析設定", expanded=False):
                st.caption("意味別カウント用です。まずは並列2がおすすめです。詰まる場合は1、速くしたい場合は3〜4に上げます。")
                col_idiom_ai1, col_idiom_ai2, col_idiom_ai3 = st.columns(3)
                with col_idiom_ai1:
                    idiom_model_name = st.selectbox(
                        "熟語解析モデル",
                        ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
                        index=0,
                        key="idiom_analysis_model",
                    )
                with col_idiom_ai2:
                    idiom_workers = st.number_input(
                        "並列数",
                        min_value=1,
                        max_value=6,
                        value=2,
                        step=1,
                        key="idiom_analysis_workers",
                    )
                with col_idiom_ai3:
                    idiom_chunk_chars = st.number_input(
                        "1回あたりの文字数",
                        min_value=1500,
                        max_value=8000,
                        value=2500,
                        step=500,
                        key="idiom_analysis_chunk_chars_v2",
                    )
                st.info("熟語は共通辞書 idiom_lexicon.json に意味を登録し、各過去問DBには意味別の回数だけを保存します。")

        col_save, col_cancel = st.columns(2)
        
        if col_save.button("💾 2. この内容でデータベースに登録", type="primary"):
            if not (cat_val and uni_val and fac_val and year_val and method_val):
                st.error("階層をすべて入力してください")
            elif not (ext_words or ext_word_meanings or ext_idioms or ext_grammar):
                st.warning("⚠️ 抽出するオプションを少なくとも1つはオンにしてください。")
            else:
                executed_tasks = []
                if ext_words: executed_tasks.append("単語")
                if ext_word_meanings: executed_tasks.append("単語の意味別解析")
                if ext_idioms: executed_tasks.append("熟語")
                if ext_grammar: executed_tasks.append("文法")

                with st.spinner(f"データを抽出・集計中...（{', '.join(executed_tasks)}）"):
                    # 階層の初期化（空の辞書を作る）
                    if cat_val not in db: db[cat_val] = {}
                    if uni_val not in db[cat_val]: db[cat_val][uni_val] = {}
                    if fac_val not in db[cat_val][uni_val]: db[cat_val][uni_val][fac_val] = {}
                    if year_val not in db[cat_val][uni_val][fac_val]: db[cat_val][uni_val][fac_val][year_val] = {}
                    if method_val not in db[cat_val][uni_val][fac_val][year_val]:
                        db[cat_val][uni_val][fac_val][year_val][method_val] = {
                            "total_words": 0, "unique_words": 0, "frequencies": {}, "word_meanings": {}, "idioms": {}, "grammar_questions": [], "grammar_tags": {}
                        }
                    
                    # 更新対象のデータベース階層への参照
                    target_db = db[cat_val][uni_val][fac_val][year_val][method_val]
                    
                    # -----------------------------------------
                    # ルートA: 単語の抽出（Python処理）
                    # -----------------------------------------
                    if ext_words or ext_word_meanings:
                        new_words = extract_words_from_text(edited_text)
                        word_counts = Counter(new_words)
                        
                        existing_freqs = target_db.get("frequencies", {})
                        if not isinstance(existing_freqs, dict):
                            existing_freqs = {}
                            
                        merged_words = Counter(existing_freqs) + word_counts
                        
                        target_db["total_words"] = target_db.get("total_words", 0) + len(new_words)
                        target_db["unique_words"] = len(merged_words)
                        target_db["frequencies"] = dict(merged_words.most_common())

                    # -----------------------------------------
                    # ルートA2: 単語の意味別解析（AI処理）
                    # -----------------------------------------
                    if ext_word_meanings:
                        extracted_word_meaning_data = extract_word_meanings_with_gemini(
                            edited_text,
                            model_name=word_meaning_model_name,
                            max_workers=int(word_meaning_workers),
                            chunk_chars=int(word_meaning_chunk_chars),
                        )
                        target_db["word_meanings"] = merge_word_meaning_items(
                            target_db.get("word_meanings", {}),
                            extracted_word_meaning_data.get("words", []),
                        )
                        save_db(db)
                        st.info(
                            f"単語意味解析: {len(extracted_word_meaning_data.get('words', []))}件 / "
                            f"{extracted_word_meaning_data.get('chunks', 1)}分割"
                        )
                    
                    # -----------------------------------------
                    # ルートB: 熟語の抽出（AI処理）
                    # -----------------------------------------
                    if ext_idioms:
                        extracted_idioms_data = extract_idioms_with_gemini(
                            edited_text,
                            model_name=idiom_model_name,
                            max_workers=int(idiom_workers),
                            chunk_chars=int(idiom_chunk_chars),
                        )
                        merged_idioms = merge_idiom_items(
                            target_db.get("idioms", {}),
                            extracted_idioms_data.get("idioms", []),
                        )
                        
                        for item in []:
                            base_form = item["base_form"]
                            if base_form in merged_idioms:
                                merged_idioms[base_form]["count"] += item["count"]
                                # ▼ 著作権対策：ここで quotes の結合・保存処理を完全に削除（捨てる）
                            else:
                                merged_idioms[base_form] = {"count": item["count"]}
                                # ▼ 著作権対策：ここでも quotes は辞書に入れない（捨てる）

                        target_db["idioms"] = merged_idioms
                        save_db(db)
                        st.info(
                            f"熟語解析: {len(extracted_idioms_data.get('idioms', []))}件 / "
                            f"{extracted_idioms_data.get('chunks', 1)}分割"
                        )

                                       # -----------------------------------------
                    # ルートC: 文法・語法の抽出（AI処理）
                    # -----------------------------------------
                    if ext_grammar:
                        extracted_grammar_data = extract_grammar_with_gemini(edited_text)

                        # デバッグ用：Geminiが何を返したか画面に出す
                        st.write("DEBUG 文法抽出結果:", extracted_grammar_data)

                        grammar_questions = extracted_grammar_data.get("grammar_questions", [])

                        if not grammar_questions:
                            st.warning("⚠️ 文法問題が抽出されませんでした。入力テキストかGeminiの返答を確認してください。")

                        merged_grammar_questions = target_db.get("grammar_questions", [])
                        merged_grammar_tags = Counter(target_db.get("grammar_tags", {}))

                        for q in grammar_questions:
                            if not isinstance(q, dict):
                                continue

                            q.setdefault("question", "")
                            q.setdefault("options", [])
                            q.setdefault("answer", "")
                            q.setdefault("explanation", "")
                            q.setdefault("required_knowledge", [])

                            # 問題形式を必ず付ける
                            q["question_type"] = infer_question_type(q)

                            options = q.get("options", [])
                            answer = q.get("answer", "")

                            # 選択問題の最低限チェック
                            if q["question_type"] == "multiple_choice":
                                if not isinstance(options, list):
                                    continue
                                if len(options) < 2:
                                    continue
                                if answer not in options:
                                    continue

                            # 整序問題の最低限チェック
                            if q["question_type"] == "ordering":
                                if not isinstance(options, list):
                                    continue
                                if len(options) < 3:
                                    continue
                                if not isinstance(answer, str) or not answer.strip():
                                    continue

                                # answer_order がない場合、answer から自動推定を試す
                                if "answer_order" not in q or not isinstance(q.get("answer_order"), list):
                                    normalized_answer = normalize_answer_text(answer)
                                    normalized_options = [normalize_answer_text(x) for x in options]
                                    answer_words = normalized_answer.split()

                                    guessed_order = []
                                    used = set()

                                    for aw in answer_words:
                                        found_idx = None
                                        for i, opt in enumerate(normalized_options):
                                            if i in used:
                                                continue
                                            if opt == aw:
                                                found_idx = i
                                                break

                                        if found_idx is not None:
                                            guessed_order.append(found_idx + 1)
                                            used.add(found_idx)

                                    if len(guessed_order) == len(options):
                                        q["answer_order"] = guessed_order

                            merged_grammar_questions.append(q)

                            raw_tags = q.get("primary_tags", q.get("required_knowledge", []))
                            safe_tags = flatten_tags(raw_tags)

                            for tag in safe_tags:
                                if tag:
                                    merged_grammar_tags[tag] += 1

                        target_db["grammar_questions"] = merged_grammar_questions
                        target_db["grammar_tags"] = dict(merged_grammar_tags)

                    # -----------------------------------------
                    # 最終保存処理
                    # -----------------------------------------
                    save_db(db)
                    st.success("✅ データベースに保存しました。")

                    
# ------------------------------------------
# タブ2: データベース閲覧・編集（復活・統合）
# ------------------------------------------

                    
# ------------------------------------------
# タブ2: データベース閲覧・編集（復活・統合）
# ------------------------------------------
with tab2:
    if not db:
        st.info("DBが空です")
    else:
        col_c, col_u, col_f, col_y, col_m = st.columns(5)
        with col_c: sel_cat = st.selectbox("📁 カテゴリ", sorted(db.keys()), key="v_c")
        with col_u:
            uni_list = sorted(db[sel_cat].keys()) if sel_cat else []
            sel_uni = st.selectbox("🏫 大学名", uni_list, key="v_u") if uni_list else None
        with col_f:
            fac_list = sorted(db[sel_cat][sel_uni].keys()) if sel_uni else []
            sel_fac = st.selectbox("📚 学部", fac_list, key="v_f") if fac_list else None
        with col_y:
            year_list = sorted(db[sel_cat][sel_uni][sel_fac].keys(), reverse=True) if sel_fac else []
            sel_year = st.selectbox("🗓️ 年度", year_list, key="v_y") if year_list else None
        with col_m:
            method_list = sorted(db[sel_cat][sel_uni][sel_fac][sel_year].keys()) if sel_year else []
            sel_method = st.selectbox("📝 方式", method_list, key="v_m") if method_list else None
        
        st.markdown("---")
        
        if sel_method:
            target_data = db[sel_cat][sel_uni][sel_fac][sel_year][sel_method]
            st.write(f"### 📊 {sel_uni} {sel_fac} ({sel_year}) - {sel_method}")
            
            # --- 復活箇所1: 階層の編集・削除 expander ---
            with st.expander("✏️ この階層の名前変更・移動・削除", expanded=False):
                with st.form("edit_hierarchy_form"):
                    e_cat = st.text_input("カテゴリ", value=sel_cat)
                    e_uni = st.text_input("大学名", value=sel_uni)
                    e_fac = st.text_input("学部", value=sel_fac)
                    e_year = st.text_input("年度", value=sel_year)
                    e_method = st.text_input("方式", value=sel_method)
                    
                    col_s, col_d = st.columns(2)
                    if col_s.form_submit_button("💾 変更を保存"):
                        data_to_move = target_data
                        # 古い場所の削除
                        del db[sel_cat][sel_uni][sel_fac][sel_year][sel_method]
                        # 空になった親階層を掃除
                        if not db[sel_cat][sel_uni][sel_fac][sel_year]: del db[sel_cat][sel_uni][sel_fac][sel_year]
                        if not db[sel_cat][sel_uni][sel_fac]: del db[sel_cat][sel_uni][sel_fac]
                        if not db[sel_cat][sel_uni]: del db[sel_cat][sel_uni]
                        if not db[sel_cat]: del db[sel_cat]
                        
                        # 新しい場所の作成
                        if e_cat not in db: db[e_cat] = {}
                        if e_uni not in db[e_cat]: db[e_cat][e_uni] = {}
                        if e_fac not in db[e_cat][e_uni]: db[e_cat][e_uni][e_fac] = {}
                        if e_year not in db[e_cat][e_uni][e_fac]: db[e_cat][e_uni][e_fac][e_year] = {}
                        db[e_cat][e_uni][e_fac][e_year][e_method] = data_to_move
                        save_db(db)
                        st.success("✅ 移動・変更完了")
                        st.rerun()
                        
                    if col_d.form_submit_button("🗑️ この階層ごと削除"):
                        del db[sel_cat][sel_uni][sel_fac][sel_year][sel_method]
                        if not db[sel_cat][sel_uni][sel_fac][sel_year]: del db[sel_cat][sel_uni][sel_fac][sel_year]
                        if not db[sel_cat][sel_uni][sel_fac]: del db[sel_cat][sel_uni][sel_fac]
                        if not db[sel_cat][sel_uni]: del db[sel_cat][sel_uni]
                        if not db[sel_cat]: del db[sel_cat]
                        save_db(db)
                        st.success("🗑️ 階層を削除しました")
                        st.rerun()

            # =========================================================
            # 🎯 データベース管理・削除システム（レイアウト最適化版）
            # =========================================================
            st.markdown("---")
            
            # 1. 一番上: データを全部空にする枠（完全リセット）
            st.markdown("### 🧹 データベース全消去（一括リセット枠）")
            with st.container(border=True):
                st.markdown("⚠️ 選択中の過去問階層にある**すべての単語データおよび熟語データ**を完全に消去します。")
                if st.button("🚨 単語・熟語・文法データをすべて空にする", type="primary", use_container_width=True, key="clear_all_data"):
                    target_data["frequencies"] = {}
                    target_data["word_meanings"] = {}
                    target_data["total_words"] = 0
                    target_data["unique_words"] = 0
                    target_data["idioms"] = {}
                    target_data["grammar_questions"] = []
                    target_data["grammar_tags"] = {}
                    save_db(db)
                    st.warning("すべてのデータを空にしました。")
                    st.rerun()

            st.markdown("---")
            
            # 2. 単語編集欄（指定削除 ＆ 横のすぐ下に全削除）
            st.markdown("### 🏆 頻出単語ランキングと管理")
            
            col_w1, col_w2 = st.columns([3, 1])
            with col_w1:
                del_word = st.text_input("単語の指定削除", placeholder="削除したい単語を入力（例: apple）", key="del_word_input")
            with col_w2:
                # 入力欄のラベル高さを合わせるための空隙
                st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                # 上側：指定削除
                if st.button("選択した単語を削除", use_container_width=True, key="btn_del_word"):
                    if del_word and del_word in target_data.get("frequencies", {}):
                        target_data["total_words"] -= target_data["frequencies"][del_word]
                        del target_data["frequencies"][del_word]
                        target_data.get("word_meanings", {}).pop(del_word, None)
                        target_data["unique_words"] = len(target_data["frequencies"])
                        save_db(db)
                        st.success(f"単語「{del_word}」を削除しました。")
                        st.rerun()
                    else:
                        st.error("その単語は見つかりませんでした。")
                # 横のすぐ下：全削除
                if st.button("⚠️ 単語を全削除", use_container_width=True, key="btn_clear_words"):
                    target_data["frequencies"] = {}
                    target_data["word_meanings"] = {}
                    target_data["total_words"] = 0
                    target_data["unique_words"] = 0
                    save_db(db)
                    st.warning("単語リストをすべて消去しました。")
                    st.rerun()
                    
            frequencies = target_data.get("frequencies", {})
            word_meanings = target_data.get("word_meanings", {})
            if not frequencies and word_meanings:
                frequencies = {
                    word: safe_count(data.get("count"), 0)
                    for word, data in word_meanings.items()
                    if isinstance(data, dict)
                }
            display_total_words = target_data.get("total_words", 0) or sum(safe_count(v, 0) for v in frequencies.values())
            display_unique_words = target_data.get("unique_words", 0) or len(frequencies)
            st.write(f"総語数: **{display_total_words} 語** / 種類: **{display_unique_words} 種類**")
            top_words = [
                {
                    "単語": k,
                    "回数": v,
                    "意味別回数": format_word_meaning_counts(word_meanings.get(k, {})) if word_meanings else "未解析",
                }
                for k, v in list(frequencies.items())[:200]
            ]
            st.dataframe(pd.DataFrame(top_words), use_container_width=True)

            with st.expander("💪 頻度つよつよ単語3000を外した意味カウント", expanded=False):
                strong_summary = summarize_words_without_frequency_strong(target_data.get("frequencies", {}))
                raw_total_tokens = sum(int(v) for v in target_data.get("frequencies", {}).values())
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                col_s1.metric("頻度つよつよでカバー", f"{strong_summary['strong_tokens']:,}語")
                col_s2.metric("3000語除外後", f"{strong_summary['remaining_tokens']:,}語")
                col_s3.metric("意味あり", f"{len(strong_summary['known_meaning_words']):,}種類")
                col_s4.metric("意味なし", f"{len(strong_summary['missing_meaning_words']):,}種類")

                if raw_total_tokens:
                    st.progress(
                        min(strong_summary["strong_tokens"] / raw_total_tokens, 1.0),
                        text=f"頻度つよつよ単語による無条件カバー: {strong_summary['strong_tokens'] / raw_total_tokens:.1%}"
                    )

                remaining_rows = []
                for word, count in strong_summary["remaining"].items():
                    meaning = strong_summary["meaning_registry"].get(word, "")
                    remaining_rows.append({
                        "単語": word,
                        "回数": count,
                        "意味登録": "あり" if meaning else "なし",
                        "意味": meaning,
                    })
                remaining_df = pd.DataFrame(remaining_rows)
                missing_df = remaining_df[remaining_df["意味登録"] == "なし"] if not remaining_df.empty else remaining_df

                dl_col1, dl_col2 = st.columns(2)
                dl_col1.download_button(
                    "📥 3000語除外後の意味カウントCSV",
                    data=remaining_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="meaning_count_without_frequency_strong.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                dl_col2.download_button(
                    "📥 意味なしだけCSV",
                    data=missing_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="missing_meanings_without_frequency_strong.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.dataframe(remaining_df.head(500), use_container_width=True)

            st.markdown("---")
            
            # 3. 熟語編集欄（指定削除 ＆ 横のすぐ下に全削除）
            st.markdown("### 🔗 頻出熟語・イディオムと管理")
            
            if "idioms" in target_data and target_data["idioms"]:
                col_i1, col_i2 = st.columns([3, 1])
                with col_i1:
                    del_idiom = st.text_input("熟語の指定削除", placeholder="削除したい熟語を入力（例: take advantage of）", key="del_idiom_input")
                with col_i2:
                    # ラベル高さを合わせるための空隙
                    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                    # 上側：指定削除
                    if st.button("選択した熟語を削除", use_container_width=True, key="btn_del_idiom"):
                        if del_idiom and del_idiom in target_data["idioms"]:
                            del target_data["idioms"][del_idiom]
                            save_db(db)
                            st.success(f"熟語「{del_idiom}」を削除しました。")
                            st.rerun()
                        else:
                            st.error("その熟語は見つかりませんでした。")
                    # 横のすぐ下：全削除
                    if st.button("⚠️ 熟語を全削除", use_container_width=True, key="btn_clear_idioms"):
                        target_data["idioms"] = {}
                        save_db(db)
                        st.warning("熟語リストをすべて消去しました。")
                        st.rerun()

                st.markdown("#### 🏆 頻出熟語ランキング")
                sorted_idioms = sorted(
                    target_data["idioms"].items(),
                    key=lambda x: safe_count(x[1].get("count"), 0),
                    reverse=True,
                )
                idiom_display = []
                for base_form, data in sorted_idioms:
                    idiom_display.append({
                        "意味別回数": format_idiom_meaning_counts(data),
                        "熟語・構文": base_form,
                        "回数": data["count"],
                        # ▼ データに quotes が残っている場合は表示し、無い場合は「(著作権保護のため非表示)」とする
                        "本文中での使われ方 (Quote)": " / ".join(data.get("quotes", [])) if "quotes" in data else "(著作権保護のため非表示)"
                    })
                st.dataframe(pd.DataFrame(idiom_display), use_container_width=True)
            else:
                st.info("💡 この階層に登録されている熟語データは現在ありません。")

            st.markdown("---")

           # 4. 文法編集欄（指定削除 ＆ 横のすぐ下に全削除）
            st.markdown("### 📖 頻出文法・必須知識タグと管理")
            
            # 修正: grammar_tagsではなく、問題本体(grammar_questions)が存在するかで表示を判定
            if "grammar_questions" in target_data and target_data["grammar_questions"]:
                col_g1, col_g2 = st.columns([3, 1])
                with col_g1:
                    del_grammar = st.text_input("文法タグの指定削除", placeholder="削除したい必須知識タグを入力", key="del_grammar_input")
                with col_g2:
                    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                    if st.button("選択したタグを削除", use_container_width=True, key="btn_del_grammar"):
                        if del_grammar and del_grammar in target_data.get("grammar_tags", {}):
                            del target_data["grammar_tags"][del_grammar]
                            save_db(db)
                            st.success(f"タグ「{del_grammar}」を削除しました。")
                            st.rerun()
                        else:
                            st.error("そのタグは見つかりませんでした。")
                    if st.button("⚠️ 文法データを全削除", use_container_width=True, key="btn_clear_grammar"):
                        target_data["grammar_tags"] = {}
                        target_data["grammar_questions"] = []
                        save_db(db)
                        st.warning("文法データをすべて消去しました。")
                        st.rerun()

                # --- 🤖 自動修復機能：もし grammar_tags が空なら、問題データから再集計して直す ---
                display_tags = target_data.get("grammar_tags", {})
                if not display_tags:
                    temp_counter = Counter()
                    for q in target_data["grammar_questions"]:
                        
                        # 取得したタグデータを安全に平坦化してからカウントする
                        raw_tags = q.get("primary_tags", q.get("required_knowledge", []))
                        safe_tags = flatten_tags(raw_tags)
                        
                        for t in safe_tags:
                            temp_counter[t] += 1
                            
                    display_tags = dict(temp_counter)
                    target_data["grammar_tags"] = display_tags
                    save_db(db) # データベースをここでこっそり修復・保存
                
                # タグランキングの表示
                st.markdown("#### 🏆 必須知識タグ ランキング")
                sorted_tags = sorted(display_tags.items(), key=lambda x: x[1], reverse=True)
                tag_display = [{"必須知識タグ (Required Knowledge)": k, "出現回数": v} for k, v in sorted_tags]
                st.dataframe(pd.DataFrame(tag_display), use_container_width=True)
                
                # 抽出された問題のプレビュー
                with st.expander("👀 抽出された文法問題の一覧を見る", expanded=False):
                    for idx, q in enumerate(target_data["grammar_questions"]):
                        st.markdown(f"**Q{idx+1}. {q.get('question', '')}**")
                        st.markdown(f"- **選択肢:** {', '.join(q.get('options', []))}")
                        st.markdown(f"- **正解:** {q.get('answer', '')}")
                        
                        tags = q.get("primary_tags", q.get("required_knowledge", []))
                        st.markdown(f"- **必須知識タグ:** {', '.join(tags)}")
                        
                        st.markdown(f"- **解説:** {q.get('explanation', '')}")
                        st.divider()
            else:
                st.info("💡 この階層に登録されている文法データは現在ありません。")
