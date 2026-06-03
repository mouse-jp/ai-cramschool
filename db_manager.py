import streamlit as st
import pandas as pd
import google.generativeai as genai
import tempfile
import re
import json
import os
from collections import Counter

# --- 追加: NLTKのセットアップ ---
import nltk
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
# --------------------------------

# --- 設定 ---
DB_FILE = "past_exams_db.json"
MY_DATA_FILE = "my_data.json"
BASE_LEXICON_FILE = "base_lexicon.json"
BASE_VOCAB_STATUSES = {"core_verified", "exam_format", "watch_known"}
EXCLUDED_BASE_VOCAB_STATUSES = {"strict_excluded", "proper_noun_or_noise"}

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("APIキーが見つかりません。.streamlit/secrets.toml を確認してください。")

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
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_db(db):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def load_json_file(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
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

def extract_text_with_gemini(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        g_file = genai.upload_file(tmp_path)
        model = genai.GenerativeModel(model_name="gemini-2.5-pro")
        prompt = "このPDFファイル（英語の試験問題）から、英語の文章をすべて書き起こしてください。余計な挨拶や説明は不要です。"
        res = model.generate_content([g_file, prompt])
        return res.text
    finally:
        os.remove(tmp_path)

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
        if uploaded_pdf and st.button("🚀 1. AIでテキストを抽出する"):
            with st.spinner("AIが解析中..."):
                st.session_state.draft_text = extract_text_with_gemini(uploaded_pdf)
                st.rerun()
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
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        with col_opt1:
            ext_words = st.toggle("🔤 単語の抽出 (AI不使用・高速)", value=True)
        with col_opt2:
            ext_idioms = st.toggle("🔗 熟語の抽出 (AI使用)", value=True)
        with col_opt3:
            ext_grammar = st.toggle("📖 文法・語法の抽出 (AI使用)", value=True)
        
        col_save, col_cancel = st.columns(2)
        
        if col_save.button("💾 2. この内容でデータベースに登録", type="primary"):
            if not (cat_val and uni_val and fac_val and year_val and method_val):
                st.error("階層をすべて入力してください")
            elif not (ext_words or ext_idioms or ext_grammar):
                st.warning("⚠️ 抽出するオプションを少なくとも1つはオンにしてください。")
            else:
                executed_tasks = []
                if ext_words: executed_tasks.append("単語")
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
                            "total_words": 0, "unique_words": 0, "frequencies": {}, "idioms": {}, "grammar_questions": [], "grammar_tags": {}
                        }
                    
                    # 更新対象のデータベース階層への参照
                    target_db = db[cat_val][uni_val][fac_val][year_val][method_val]
                    
                    # -----------------------------------------
                    # ルートA: 単語の抽出（Python処理）
                    # -----------------------------------------
                    if ext_words:
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
                    # ルートB: 熟語の抽出（AI処理）
                    # -----------------------------------------
                    if ext_idioms:
                        extracted_idioms_data = extract_idioms_with_gemini(edited_text)
                        merged_idioms = target_db.get("idioms", {})
                        
                        for item in extracted_idioms_data.get("idioms", []):
                            base_form = item["base_form"]
                            if base_form in merged_idioms:
                                merged_idioms[base_form]["count"] += item["count"]
                                # ▼ 著作権対策：ここで quotes の結合・保存処理を完全に削除（捨てる）
                            else:
                                merged_idioms[base_form] = {"count": item["count"]}
                                # ▼ 著作権対策：ここでも quotes は辞書に入れない（捨てる）
                                
                        target_db["idioms"] = merged_idioms

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
                        target_data["unique_words"] = len(target_data["frequencies"])
                        save_db(db)
                        st.success(f"単語「{del_word}」を削除しました。")
                        st.rerun()
                    else:
                        st.error("その単語は見つかりませんでした。")
                # 横のすぐ下：全削除
                if st.button("⚠️ 単語を全削除", use_container_width=True, key="btn_clear_words"):
                    target_data["frequencies"] = {}
                    target_data["total_words"] = 0
                    target_data["unique_words"] = 0
                    save_db(db)
                    st.warning("単語リストをすべて消去しました。")
                    st.rerun()
                    
            st.write(f"総語数: **{target_data.get('total_words', 0)} 語** / 種類: **{target_data.get('unique_words', 0)} 種類**")
            top_words = [{"単語": k, "回数": v} for k, v in list(target_data.get('frequencies', {}).items())[:200]]
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
                sorted_idioms = sorted(target_data["idioms"].items(), key=lambda x: x[1]["count"], reverse=True)
                idiom_display = []
                for base_form, data in sorted_idioms:
                    idiom_display.append({
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
