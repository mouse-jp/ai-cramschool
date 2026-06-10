import streamlit as st
import google.generativeai as genai
import tempfile
import os
import json
import html
import re
import random
import io
import csv
import streamlit.components.v1 as components
from datetime import datetime
from collections import Counter
from difflib import SequenceMatcher

LOCAL_NLTK_DATA = os.path.join(os.path.dirname(__file__), ".nltk_data")
os.makedirs(LOCAL_NLTK_DATA, exist_ok=True)
os.environ["NLTK_DATA"] = LOCAL_NLTK_DATA

import nltk
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

if LOCAL_NLTK_DATA not in nltk.data.path:
    nltk.data.path.insert(0, LOCAL_NLTK_DATA)

# NLTKの辞書データをダウンロード（初回のみ裏で自動実行されます）
try:
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet', download_dir=LOCAL_NLTK_DATA, quiet=True)

lemmatizer = WordNetLemmatizer()

# --- app.py側で動的に弾くゴミ単語リスト ---
DYNAMIC_STOP_WORDS = {
    "et", "al", "st", "pp", "vol", "ed", 
    "don", "doesn", "didn", "isn", "aren", "wasn", "weren", "hasn", "haven", "hadn", 
    "won", "wouldn", "shouldn", "couldn", "can", "ll", "ve", "re", "t", "s", "m", "d"
}

def process_frequencies(raw_freqs):
    """DBの生データからゴミを除外し、原形に変換して再集計する関数"""
    processed_freqs = defaultdict(int)
    for word, count in raw_freqs.items():
        if word in DYNAMIC_STOP_WORDS:
            continue
        
        # 動詞として原形変換を試す (例: studied -> study)
        lemma = lemmatizer.lemmatize(word, pos='v')
        if lemma == word:
            # 動詞で変わらなければ名詞として試す (例: apples -> apple)
            lemma = lemmatizer.lemmatize(word, pos='n')
            
        # 同じ原形になった単語のカウントを合算する
        processed_freqs[lemma] += count
        
    return dict(processed_freqs)

# --- 1. データ保存・読み込み ---
DATA_FILE = "my_data.json"
EXAM_DB_FILE = "past_exams_db.json" # 裏アプリ(db_manager)で作ったデータ
BASE_LEXICON_FILE = "base_lexicon.json"
BASE_LEXICON_LOG_FILE = "base_lexicon_generation_log.json"
FREQUENCY_STRONG_TITLE = "頻度つよつよ単語"
BASE_VOCAB_STATUSES = {"core_verified", "exam_format", "watch_known"}
EXCLUDED_BASE_VOCAB_STATUSES = {"strict_excluded", "proper_noun_or_noise"}
CIRCLED_NUMBERS = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"]

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    return {"vocabulary": [], "grammar": [], "strategy": [], "meta": []}

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_exam_db():
    if os.path.exists(EXAM_DB_FILE):
        with open(EXAM_DB_FILE, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    return {}

def load_base_lexicon():
    if os.path.exists(BASE_LEXICON_FILE):
        with open(BASE_LEXICON_FILE, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    return {}

def save_base_lexicon(lexicon):
    with open(BASE_LEXICON_FILE, "w", encoding="utf-8") as f:
        json.dump(lexicon, f, ensure_ascii=False, indent=2)

def load_base_lexicon_generation_log():
    if os.path.exists(BASE_LEXICON_LOG_FILE):
        try:
            with open(BASE_LEXICON_LOG_FILE, "r", encoding="utf-8-sig") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def append_base_lexicon_generation_log(entry):
    logs = load_base_lexicon_generation_log()
    logs.append(entry)
    with open(BASE_LEXICON_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs[-200:], f, ensure_ascii=False, indent=2)

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

def meanings_to_text(value):
    return "；".join(split_meanings(value))

def numbered_meanings_to_text(value):
    meanings = split_meanings(value)
    numbered = []
    for index, meaning in enumerate(meanings):
        meaning = str(meaning).strip()
        if not meaning:
            continue
        if re.match(r"^[①②③④⑤⑥⑦⑧⑨⑩]", meaning):
            numbered.append(meaning)
        else:
            prefix = CIRCLED_NUMBERS[index] if index < len(CIRCLED_NUMBERS) else f"{index + 1}."
            numbered.append(f"{prefix} {meaning}")
    return "；".join(numbered)

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

def build_frequency_strong_meaning_report(lexicon):
    rows = []
    for word, entry in lexicon.items():
        if entry.get("status") not in BASE_VOCAB_STATUSES:
            continue
        meanings = split_meanings(entry.get("meanings", []))
        rows.append({
            "順位": entry.get("last_seen_rank", ""),
            "単語": word,
            "出現回数": entry.get("last_seen_count", 0),
            "出典数": entry.get("source_count", 0),
            "意味数": len(meanings),
            "意味": "；".join(meanings),
            "注意": entry.get("alert", ""),
            "生成状態": entry.get("meaning_review_status", ""),
            "意味更新日時": entry.get("meaning_updated_at", ""),
        })
    rows.sort(key=lambda row: (
        row["順位"] if isinstance(row["順位"], int) else 999999,
        row["単語"],
    ))
    missing = [row for row in rows if row["意味数"] == 0]
    thin = [
        row for row in rows
        if row["意味数"] == 1 and (int(row.get("出典数") or 0) >= 8 or int(row.get("出現回数") or 0) >= 20)
    ]
    no_alert_polysemy = [
        row for row in rows
        if row["意味数"] >= 3 and not str(row.get("注意", "")).strip()
    ]
    return {
        "rows": rows,
        "missing": missing,
        "thin": thin,
        "no_alert_polysemy": no_alert_polysemy,
    }

def rows_to_csv(rows):
    output = io.StringIO()
    fieldnames = ["順位", "単語", "出現回数", "出典数", "意味数", "意味", "注意", "生成状態", "意味更新日時"]
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")

def build_frequency_strong_book(lexicon):
    core_items = [
        (word, entry)
        for word, entry in lexicon.items()
        if entry.get("status") in BASE_VOCAB_STATUSES
    ]
    if not core_items:
        return None

    core_items.sort(key=lambda item: (
        item[1].get("last_seen_rank") if isinstance(item[1].get("last_seen_rank"), int) else 999999,
        item[0]
    ))
    excluded_words = [
        word
        for word, entry in lexicon.items()
        if entry.get("status") in EXCLUDED_BASE_VOCAB_STATUSES
    ]
    enriched_vocab = []
    for word, entry in core_items:
        meanings = numbered_meanings_to_text(entry.get("meanings", []))
        if not meanings:
            continue
        enriched_vocab.append({
            "word": word,
            "forms": entry.get("forms", ""),
            "meanings": meanings,
            "chunks": entry.get("chunks", []),
            "context": entry.get("context", "頻度つよつよ単語に固定登録済み。"),
            "alert": entry.get("alert", ""),
        })

    return {
        "title": FREQUENCY_STRONG_TITLE,
        "main_vocab": [word for word, _ in core_items],
        "excluded_vocab": excluded_words,
        "counts": {word: entry.get("last_seen_count", 0) for word, entry in core_items},
        "origins": {word: ["頻度つよつよ単語3000"] for word, _ in core_items},
        "enriched_vocab": enriched_vocab,
        "skipped_vocab": [],
        "fixed_source": BASE_LEXICON_FILE,
    }

def apply_frequency_strong_enrichment(lexicon, enriched_items, skipped_words):
    changed = 0
    for item in enriched_items:
        word = normalize_vocab_word(item.get("word", ""))
        if not word or word not in lexicon:
            continue
        entry = lexicon[word]
        meanings = split_meanings(item.get("meanings", ""))
        if meanings:
            entry["meanings"] = meanings
        for key in ["forms", "chunks", "context", "alert"]:
            if item.get(key):
                entry[key] = item.get(key)
        entry["source"] = "frequency_strong_gemini_enrichment"
        entry["meaning_policy"] = "broad_exam_coverage"
        entry["meaning_review_status"] = "ai_generated"
        entry["meaning_updated_at"] = datetime.now().isoformat(timespec="seconds")
        changed += 1

    for word in skipped_words:
        normalized = normalize_vocab_word(word)
        if normalized in lexicon:
            lexicon[normalized]["note"] = "Gemini meaning generation skipped this word."
    if changed:
        save_base_lexicon(lexicon)
    return changed

def reset_frequency_strong_enrichment(lexicon):
    reset_count = 0
    generated_keys = [
        "forms",
        "chunks",
        "context",
        "alert",
        "meaning_policy",
        "meaning_review_status",
        "meaning_updated_at",
    ]
    for entry in lexicon.values():
        if entry.get("status") not in BASE_VOCAB_STATUSES:
            continue
        entry["meanings"] = []
        for key in generated_keys:
            entry.pop(key, None)
        reset_count += 1
    save_base_lexicon(lexicon)
    return reset_count

def build_frequency_strong_enrich_prompt():
    return """
    あなたは大学受験英語に精通したプロの予備校講師です。
    提供された英単語リストは、すでに「頻度つよつよ単語」として固定採用された基礎語です。
    この3000語はあとから通常の単語解析AIで逐次追加しない前提なので、簡潔さより網羅性を優先してください。
    スキップは絶対にしないでください。入力された全単語を "enriched" に入れてください。
    各単語について、大学入試で読解に必要になり得る意味を、基本義・抽象義・品詞違い・入試頻出の多義語まで漏れなく日本語で列挙してください。
    ただし、専門辞書にしか載らない極端にまれな意味は除き、共通テスト・私大・国公立二次の英文読解で誤読を防ぐための意味に絞ってください。
    "meanings" は文字列ではなく、意味グループごとの配列にしてください。似た意味はまとめてよいですが、別義は分けてください。各項目には [形] [副] [動] [名] など品詞ラベルを必要に応じて付けてください。
    画面側で ①②③ を付けるので、"meanings" の各項目には番号を書かないでください。
    "chunks" には、主要な意味の訳し分けが分かる短い句を2〜5個入れてください。各句の先頭には、必ず対応する意味番号を書いてください。例: "① more people（もっと多くの人々）"、"② more importantly（さらに重要なことに）"。
    文法説明をここに入れすぎないでください。文法ノートに回すべき説明ではなく、単語の意味判別に必要な最小限の注意だけ "alert" に書いてください。
    "alert" には、訳し分け・多義語・不可算/可算・自動詞/他動詞など、入試で誤読しやすい注意点がある場合だけ短く書いてください。
    {
      "enriched": [
        {
          "word": "company",
          "forms": "複数形: companies",
          "meanings": ["[名] 会社、企業", "[名] 仲間、同席", "[名] 一緒にいること、付き合い"],
          "chunks": ["① run a company（会社を経営する）", "② in company with ...（...と一緒に）", "③ enjoy someone's company（人と一緒にいるのを楽しむ）"],
          "context": "ビジネス・社会系の長文で頻出。",
          "alert": "「仲間」「同席」の意味では company を単数・不可算的に読む場面がある。"
        }
      ],
      "skipped": []
    }
    """

def apply_frequency_strong_ai_response(lexicon, target_words, parsed_data):
    target_norms = {normalize_vocab_word(word) for word in target_words}
    enriched_items = parsed_data.get("enriched", [])
    returned_words = {
        normalize_vocab_word(item.get("word", ""))
        for item in enriched_items
        if normalize_vocab_word(item.get("word", ""))
    }
    extra_returned = sorted(returned_words - target_norms)
    valid_enriched_items = [
        item
        for item in enriched_items
        if normalize_vocab_word(item.get("word", "")) in target_norms
    ]
    missing_returned = [
        word
        for word in target_words
        if normalize_vocab_word(word) not in returned_words
    ]
    changed = apply_frequency_strong_enrichment(
        lexicon,
        valid_enriched_items,
        parsed_data.get("skipped", []),
    )
    return {
        "requested": len(target_words),
        "saved": changed,
        "missing": missing_returned,
        "extra": extra_returned,
    }

my_data = load_data()
exam_db = load_exam_db()
base_lexicon = load_base_lexicon()
frequency_strong_words = get_frequency_strong_words(base_lexicon)
frequency_excluded_words = get_frequency_excluded_words(base_lexicon)

# --- 2. 設定とUI ---
st.set_page_config(page_title="自律型AI塾", page_icon="🧭", layout="wide")
st.sidebar.title("設定")

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
if not api_key:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "") or st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        api_key = ""
if not api_key:
    api_key = st.sidebar.text_input("Gemini API Key", type="password")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.sidebar.warning("Gemini API Keyを入力するとAI生成が使えます。")
uploaded_pdf = st.sidebar.file_uploader("問題PDF", type=["pdf"])

if st.sidebar.button("⏹️ リセット"):
    st.session_state.clear()
    st.rerun()

# --- 💾 データ管理 (セーブ＆ロード) ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 💾 セーブ＆ロード")
st.sidebar.caption("※ブラウザを閉じる前に「セーブ」を押してデータを保存してください。次回そのファイルを読み込むと続きから再開できます。")

# 1. ダウンロード（セーブ）ボタン
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        json_str = f.read()
    st.sidebar.download_button(
        label="⬇️ 今のデータを保存 (セーブ)",
        data=json_str,
        file_name="my_learning_data.json",
        mime="application/json",
        use_container_width=True
    )

# 2. アップロード（ロード）ボタン
uploaded_data = st.sidebar.file_uploader("⬆️ 続きから始める (ロード)", type=["json"])
if uploaded_data is not None:
    if st.sidebar.button("🔄 データを復元する", type="primary", use_container_width=True):
        try:
            loaded_json = json.load(uploaded_data)
            save_data(loaded_json)
            st.sidebar.success("復元成功！画面を更新します...")
            st.rerun()
        except Exception as e:
            st.sidebar.error("エラー：正しいファイルを選んでください")
st.sidebar.markdown("---")    

mode = st.sidebar.radio("モード", [
    "📖 志望校別単語帳", 
    "🔗 志望校別熟語帳", 
    "📝 志望校別文法・語法ノート", 
    "📚 長文読解",
    "🏆 過去問演習・合格分析"
])


# --- 3. AI呼び出し関数 ---
GEMINI_MODEL_ALIASES = {
    "gemini-3-flash": "gemini-2.5-flash",
    "gemini-3.5-flash": "gemini-2.5-flash",
    "gemini-3.1-flash-lite": "gemini-2.5-flash-lite",
    "gemini-flash": "gemini-2.5-flash",
    "gemini-flash-lite": "gemini-2.5-flash-lite",
}

def normalize_gemini_model_name(model_name):
    model_name = (model_name or "gemini-2.5-flash").strip()
    return GEMINI_MODEL_ALIASES.get(model_name, model_name)

def get_gemini_model_chain(model_name):
    model_name = normalize_gemini_model_name(model_name)
    if model_name == "gemini-2.5-flash-lite":
        return ["gemini-2.5-flash-lite", "gemini-2.5-flash"]
    if model_name == "gemini-2.5-flash":
        return ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    return [model_name]

def call_ai(prompt, sys_msg, use_pdf=False, is_json=False, model_name="gemini-2.5-pro", max_output_tokens=None, timeout_seconds=60):
    if not api_key:
        raise ValueError("Gemini API Keyが未入力です。左サイドバーにAPIキーを入力してから実行してください。")
    genai.configure(api_key=api_key)

    generation_config = {}
    if is_json:
        generation_config["response_mime_type"] = "application/json"
    if max_output_tokens:
        generation_config["max_output_tokens"] = int(max_output_tokens)

    tmp_path = None
    try:
        content = prompt
        if use_pdf and uploaded_pdf:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.getvalue())
                tmp_path = tmp.name
            g_file = genai.upload_file(tmp_path)
            content = [g_file, prompt]

        last_error = None
        for candidate_model in get_gemini_model_chain(model_name):
            try:
                model = genai.GenerativeModel(
                    model_name=candidate_model,
                    system_instruction=sys_msg,
                    generation_config=generation_config or None,
                )
                res = model.generate_content(content, request_options={"timeout": timeout_seconds})
                return res.text
            except Exception as e:
                last_error = e

        raise last_error
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

def split_reading_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', str(text).replace('\n', ' ')) if s.strip()]

def clean_reading_text(text):
    raw = str(text or "").replace("\ufeff", "").strip()
    if not raw:
        return ""

    raw = re.sub(r"```[a-zA-Z]*|```", "", raw).strip()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    meta_patterns = [
        r"^(title|passage|practice passage|reading passage)\s*[:：]",
        r"^(here(?:'s| is)|below is|the following is|this is)\b.*\b(passage|text|paragraph|essay|warm-?up|exercise)\b",
        r"^(以下|次の英文|今回扱う英文|英文本文|タイトル)\b",
    ]
    kept_lines = []
    for line in lines:
        check = line.strip(" -*#")
        if any(re.match(pattern, check, flags=re.IGNORECASE) for pattern in meta_patterns):
            continue
        kept_lines.append(line)

    cleaned = " ".join(kept_lines) if kept_lines else raw
    leading_meta_sentence = (
        r"^\s*(?:(?:Here(?:'s| is)|Below is|The following is|This is)\b"
        r"[^.!?]*(?:passage|text|paragraph|essay|warm-?up|exercise)[^.!?]*[.!?]\s*)+"
    )
    cleaned = re.sub(leading_meta_sentence, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or raw

def clear_reading_state(prefix="reading"):
    suffixes = [
        "target_text", "sentences", "current_idx", "chat_logs", "chat_sentence_idx",
        "temp_memo", "show_word_selector", "global_memory", "translations", "structures",
        "last_note_title", "last_note_content"
    ]
    for suffix in suffixes:
        key = f"{prefix}_{suffix}"
        if key in st.session_state:
            del st.session_state[key]
    for key in ["temp_memo", "show_word_selector"]:
        if key in st.session_state:
            del st.session_state[key]

def set_reading_text(text, prefix="reading"):
    text = clean_reading_text(text)
    st.session_state[f"{prefix}_target_text"] = text
    st.session_state[f"{prefix}_sentences"] = split_reading_sentences(text)
    st.session_state[f"{prefix}_current_idx"] = 0
    st.session_state[f"{prefix}_chat_sentence_idx"] = -1
    st.session_state[f"{prefix}_temp_memo"] = []
    st.session_state[f"{prefix}_show_word_selector"] = False
    st.session_state[f"{prefix}_global_memory"] = ""
    st.session_state[f"{prefix}_translations"] = {}
    st.session_state[f"{prefix}_structures"] = {}
    st.session_state.temp_memo = []
    st.session_state.show_word_selector = False

def extract_reading_text_from_uploaded_media(uploaded_file, model_name="gemini-2.5-pro"):
    suffix = os.path.splitext(uploaded_file.name or "")[1].lower()
    if suffix not in [".pdf", ".png", ".jpg", ".jpeg", ".webp"]:
        suffix = ".pdf" if uploaded_file.type == "application/pdf" else ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        try:
            g_file = genai.upload_file(tmp_path, mime_type=uploaded_file.type)
        except TypeError:
            g_file = genai.upload_file(tmp_path)
        model = genai.GenerativeModel(model_name=model_name)
        prompt = """
        この資料から英語の長文本文だけを抽出してください。
        PDFでも写真でも、見えている英語本文を自然な段落に直してください。
        設問、選択肢、解答番号、日本語の指示文、ページ番号、注釈、不要なレイアウト情報は除外してください。
        日本語は出力に混ぜないでください。英語本文のみを出力してください。
        前置き、説明、タイトル、箇条書きは不要です。1文字目から英語本文を始めてください。
        OCRで読めない箇所は無理に補完せず、読める範囲を自然につないでください。
        """
        res = model.generate_content([g_file, prompt])
        return res.text
    finally:
        os.remove(tmp_path)

def render_reading_audio_controls(text, key, label="音声で聞く"):
    text = str(text or "").strip()
    if not text:
        return
    dom_id = re.sub(r"[^a-zA-Z0-9_-]", "_", str(key))
    var_id = re.sub(r"[^a-zA-Z0-9_]", "_", dom_id)
    if not re.match(r"^[a-zA-Z_]", var_id):
        var_id = f"tts_{var_id}"
    safe_text = json.dumps(text)
    safe_label = html.escape(label)
    components.html(f"""
    <div class="tts-row">
      <button id="play-{dom_id}" type="button">▶ {safe_label}</button>
      <button id="pause-{dom_id}" type="button" title="一時停止/再開">⏯</button>
      <button id="stop-{dom_id}" type="button" title="停止">■</button>
      <label>速度 <input id="rate-{dom_id}" type="range" min="0.65" max="1.1" step="0.05" value="0.9"></label>
    </div>
    <script>
      const text_{var_id} = {safe_text};
      const play_{var_id} = document.getElementById("play-{dom_id}");
      const pause_{var_id} = document.getElementById("pause-{dom_id}");
      const stop_{var_id} = document.getElementById("stop-{dom_id}");
      const rate_{var_id} = document.getElementById("rate-{dom_id}");
      play_{var_id}.onclick = () => {{
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text_{var_id});
        utterance.lang = "en-US";
        utterance.rate = Number(rate_{var_id}.value || 0.9);
        window.speechSynthesis.speak(utterance);
      }};
      pause_{var_id}.onclick = () => {{
        if (window.speechSynthesis.paused) {{
          window.speechSynthesis.resume();
        }} else {{
          window.speechSynthesis.pause();
        }}
      }};
      stop_{var_id}.onclick = () => window.speechSynthesis.cancel();
    </script>
    <style>
      .tts-row {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: sans-serif;
      }}
      .tts-row button {{
        border: 1px solid #3b4454;
        border-radius: 6px;
        background: #121821;
        color: #f5f7fb;
        padding: 6px 10px;
        cursor: pointer;
        font-weight: 700;
      }}
      .tts-row label {{
        color: #aeb7c5;
        font-size: 12px;
        display: flex;
        align-items: center;
        gap: 5px;
      }}
      .tts-row input {{
        width: 84px;
      }}
    </style>
    """, height=44)

def normalize_dictation_text(text):
    return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", str(text).lower())

def compare_dictation_answer(target, answer):
    target_words = normalize_dictation_text(target)
    answer_words = normalize_dictation_text(answer)
    if not target_words:
        return {"score": 0, "missing": [], "extra": [], "target_count": 0, "answer_count": len(answer_words)}
    score = round(SequenceMatcher(None, target_words, answer_words).ratio() * 100)
    missing = []
    temp_answer = answer_words.copy()
    for word in target_words:
        if word in temp_answer:
            temp_answer.remove(word)
        elif word not in missing:
            missing.append(word)
    extra = []
    temp_target = target_words.copy()
    for word in answer_words:
        if word in temp_target:
            temp_target.remove(word)
        elif word not in extra:
            extra.append(word)
    return {
        "score": score,
        "missing": missing[:10],
        "extra": extra[:10],
        "target_count": len(target_words),
        "answer_count": len(answer_words),
    }

def split_syntax_practice_chunks(sentence):
    text = re.sub(r"\s+", " ", str(sentence)).strip()
    chunks = [
        part.strip(" ,;:")
        for part in re.split(r"\s*,\s*|\s*;\s*|\s+\b(?:and|or|but|so|yet|nor)\b\s+", text, flags=re.IGNORECASE)
        if part.strip(" ,;:")
    ]
    connectors = [
        match.group(0)
        for match in re.finditer(r"\b(and|or|but|so|yet|nor)\b", text, flags=re.IGNORECASE)
    ]
    return chunks[:8], connectors[:6]

def split_syntax_label_tokens(sentence):
    text = re.sub(r"\s+", " ", str(sentence)).strip()
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.,;:!?()]", text)

def render_inline_syntax_label_practice(sentence, idx):
    tokens = split_syntax_label_tokens(sentence)
    if not any(re.match(r"[A-Za-z]", token) for token in tokens):
        st.info("この文ではラベルを振れる英単語が見つかりませんでした。")
        return

    pieces = []
    word_count = 0
    for token_i, token in enumerate(tokens):
        if re.match(r"[A-Za-z]", token):
            word_count += 1
            safe_token = html.escape(token)
            pieces.append(
                f"""
                <button type="button" class="syntax-unit" data-role-index="0" aria-label="{safe_token} の役割を切り替える">
                  <span class="syntax-label"></span>
                  <span class="syntax-word">{safe_token}</span>
                </button>
                """
            )
        else:
            pieces.append(f"<span class='syntax-punct'>{html.escape(token)}</span>")

    dom_id = f"syntax-inline-{idx}"
    height = 170 if word_count <= 14 else 230 if word_count <= 30 else 300
    components.html(
        f"""
        <div id="{dom_id}" class="syntax-inline-panel">
          <div class="syntax-help">
            単語をタップするたびに S→V→O→C→M… と切り替わります。もう一度押すと戻せます。全部埋めなくて大丈夫です。
          </div>
          <div class="syntax-legend">
            S 主語 / V 動詞 / O 目的語 / C 補語 / M 修飾 / 接 接続語 / 句 名詞句など / 節 関係詞節など
          </div>
          <div class="syntax-sentence">
            {''.join(pieces)}
          </div>
        </div>
        <script>
          const roles_{idx} = ["", "S", "V", "O", "C", "M", "接", "句", "節", "?"];
          const root_{idx} = document.getElementById("{dom_id}");
          root_{idx}.querySelectorAll(".syntax-unit").forEach((unit) => {{
            const label = unit.querySelector(".syntax-label");
            const sync = () => {{
              const i = Number(unit.getAttribute("data-role-index")) || 0;
              label.textContent = roles_{idx}[i] || "";
              unit.classList.toggle("selected", i > 0);
            }};
            unit.addEventListener("click", () => {{
              let i = Number(unit.getAttribute("data-role-index")) || 0;
              i = (i + 1) % roles_{idx}.length;
              unit.setAttribute("data-role-index", String(i));
              sync();
            }});
            sync();
          }});
        </script>
        <style>
          .syntax-inline-panel {{
            box-sizing: border-box;
            width: 100%;
            min-height: 100%;
            padding: 10px 10px 12px;
            background: #0f131b;
            border: 1px solid #303846;
            border-radius: 8px;
            color: #f5f7fb;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          }}
          .syntax-help {{
            font-size: 12px;
            color: #d7dde8;
            margin-bottom: 4px;
          }}
          .syntax-legend {{
            font-size: 11px;
            color: #9aa5b6;
            margin-bottom: 10px;
          }}
          .syntax-sentence {{
            display: flex;
            flex-wrap: wrap;
            align-items: flex-end;
            gap: 10px 10px;
            line-height: 1.2;
          }}
          .syntax-unit {{
            display: inline-flex;
            flex-direction: column;
            align-items: center;
            min-width: 30px;
            max-width: 150px;
            padding: 2px 4px 3px;
            border: none;
            border-radius: 6px;
            background: transparent;
            cursor: pointer;
            -webkit-tap-highlight-color: transparent;
          }}
          .syntax-unit:active {{
            background: #1a2230;
          }}
          .syntax-label {{
            height: 17px;
            margin-bottom: 2px;
            color: #43d17a;
            font-size: 13px;
            font-weight: 900;
            line-height: 1;
          }}
          .syntax-word {{
            border-bottom: 2px solid #5ea8ff;
            padding: 0 2px 3px;
            color: #f9fbff;
            font-size: 14px;
            font-weight: 700;
            max-width: 150px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
          }}
          .syntax-unit.selected .syntax-word {{
            border-bottom-color: #43d17a;
          }}
          .syntax-punct {{
            align-self: flex-end;
            color: #aeb7c5;
            font-size: 15px;
            padding-bottom: 3px;
            margin-left: -6px;
          }}
        </style>
        """,
        height=height,
        scrolling=True,
    )

def save_reading_note(title, content, category="構文・文法", source="長文読解", also_grammar=False):
    if "reading_notes" not in my_data:
        my_data["reading_notes"] = []
    note = {
        "title": title,
        "content": content,
        "category": category,
        "source": source,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    my_data["reading_notes"].append(note)
    if also_grammar:
        if "grammar" not in my_data:
            my_data["grammar"] = []
        my_data["grammar"].append({
            "title": title,
            "content": content,
            "source": f"{source} / {category}",
        })
    save_data(my_data)

def render_reading_input_section():
    read_method = st.radio(
        "題材となる長文の準備方法",
        ["📝 自分でテキストを貼り付ける", "📄 PDF・写真から抽出する", "🤖 AIにウォームアップ文を作ってもらう"],
        horizontal=True,
        key="close_read_method",
    )

    if read_method == "📝 自分でテキストを貼り付ける":
        pasted = st.text_area("英語の長文を貼り付けてください", height=150, key="close_read_pasted")
        if st.button("✅ このテキストで精読を始める", type="primary", key="close_read_start_text") and pasted:
            set_reading_text(pasted)
            st.rerun()

    elif read_method == "📄 PDF・写真から抽出する":
        up_media = st.file_uploader(
            "PDFまたは写真をアップロード",
            type=["pdf", "png", "jpg", "jpeg", "webp"],
            key="close_read_media",
        )
        if st.button("🚀 PDF・写真から長文を抽出する", type="primary", key="close_read_extract_media") and up_media:
            with st.spinner("AIが英語の長文本文だけを抽出しています..."):
                extracted_text = extract_reading_text_from_uploaded_media(up_media)
                set_reading_text(extracted_text)
                st.rerun()

    else:
        st.info("ウォームアップ用です。実力を伸ばす本番練習では、過去問PDFか実際の英文を使うのがおすすめです。")
        if st.button("✨ ウォームアップ文を生成する", type="primary", key="close_read_generate_warmup"):
            with st.spinner("AIがウォームアップ用の英文を書いています..."):
                sys_gen_read = """
                あなたは大学受験用の英文作成者です。
                180〜250語の英語長文を1つだけ作成してください。
                出力は英語本文だけにしてください。
                タイトル、前置き、説明、日本語、箇条書き、注釈は禁止です。
                "Here is a passage..." や "This passage..." のような導入文も禁止です。
                1文字目から本文の最初の文を始めてください。
                関係詞、分詞、不定詞、比較、接続、指示語が自然に含まれる英文にしてください。
                """
                text = call_ai(
                    "Create one original passage for reading practice. Output only the passage itself.",
                    sys_gen_read,
                    use_pdf=False,
                    model_name="gemini-2.5-flash-lite",
                    max_output_tokens=650,
                    timeout_seconds=35,
                )
                set_reading_text(text)
                st.rerun()


def render_reading_structure_buttons(current_sentence, idx):
    col_t1, col_t2, col_t3 = st.columns(3)
    if col_t1.button("🈁 訳を見る", use_container_width=True, key=f"show_translation_{idx}"):
        with st.spinner("自然な日本語訳を作成中..."):
            sys_translation = "あなたは大学受験英語の講師です。英文を自然な日本語に訳してください。必要以上の解説はせず、訳だけを出力してください。"
            if "reading_translations" not in st.session_state:
                st.session_state.reading_translations = {}
            st.session_state.reading_translations[idx] = call_ai(
                current_sentence,
                sys_translation,
                model_name="gemini-2.5-flash-lite",
                max_output_tokens=160,
                timeout_seconds=25,
            ).strip()
            st.rerun()
    if col_t2.button("🧩 直感で構造", use_container_width=True, key=f"show_structure_intuitive_{idx}"):
        with st.spinner("構文を分解中..."):
            sys_structure = """
            あなたは大学受験英語の構文解説者です。英文を読むために必要な形だけを、短く説明してください。

            【文体ルール】
            - 「受験生の皆さん、こんにちは」「今日の一文」などの定型あいさつは禁止。
            - 励まし、雑談、長いテーマ紹介、まとめメッセージは禁止。
            - 「超重要」「良問」「どこよりも詳しく」などの誇張表現は禁止。
            - 参考書の欄外メモのように短く書く。
            - 英文全体を再掲しない。必要な語句だけ引用する。
            - 220〜360字以内。
            - 説明は「結論 → 形 → 訳」の順にする。
            - 文法名から説明を始めない。まず「この形はこう読めばよい」という直観的な型に落とし込む。
            - 文法用語は必要な場合だけ、最後に1行で補足する。

            【必ず含める項目】
            1. 結論: この文の大まかな意味を1文
            2. 形: 覚えるべき形を「形 = 意味」で1〜2個
            3. 切り方: 2〜3区切り
            4. 訳: 自然な日本語訳

            【出力形式】
            見出しは「結論」「形」「切り方」「訳」だけ。
            各見出しは1〜2行まで。

            【禁止】
            「今回扱う英文」「ステップ」「まとめ」「講師から」などの長い教材風構成は禁止。
            S/V/O/C の記号をメインにした説明は禁止。
            """
            if "reading_structures" not in st.session_state:
                st.session_state.reading_structures = {}
            st.session_state[f"reading_structure_mode_{idx}"] = "intuitive"
            st.session_state.reading_structures[f"{idx}_intuitive"] = call_ai(
                current_sentence,
                sys_structure,
                model_name="gemini-2.5-flash-lite",
                max_output_tokens=460,
                timeout_seconds=35,
            ).strip()
            st.rerun()
    if col_t3.button("📐 S/V/O/C", use_container_width=True, key=f"show_structure_theory_{idx}"):
        with st.spinner("文型と修飾関係を整理中..."):
            sys_structure_theory = """
            あなたは大学受験英語の構文解説者です。英文をS/V/O/C・句・節ラベルで整理してください。

            【文体】
            - あいさつ、励まし、長い前置きは禁止。
            - 参考書の解答欄のように短く正確に書く。
            - 220〜420字以内。
            - 文法名を使ってよいが、解説は最小限にする。
            - 長い修飾説明は禁止。ラベルを振ってから、必要な注意だけ書く。

            【必ず含める項目】
            1. 文型: 第何文型かを1行
            2. 骨格: S / V / O / C を1行
            3. ラベル: 関係詞節・名詞句・動詞句・前置詞句・分詞句などを最大4つ
            4. 接続: and / or / but が何をつなぐか。なければ「目立つ接続なし」
            5. 訳: 自然な日本語訳を1文

            【出力形式】
            見出しは「文型」「骨格」「ラベル」「接続」「訳」だけ。
            各見出しは最大2行。長い講義にしない。
            """
            if "reading_structures" not in st.session_state:
                st.session_state.reading_structures = {}
            st.session_state[f"reading_structure_mode_{idx}"] = "theory"
            st.session_state.reading_structures[f"{idx}_theory"] = call_ai(
                current_sentence,
                sys_structure_theory,
                model_name="gemini-2.5-flash-lite",
                max_output_tokens=520,
                timeout_seconds=40,
            ).strip()
            st.rerun()

    if idx in st.session_state.get("reading_translations", {}):
        st.success(f"日本語訳: {st.session_state['reading_translations'][idx]}")
    structures = st.session_state.get("reading_structures", {})
    active_structure_mode = st.session_state.get(f"reading_structure_mode_{idx}", "intuitive")
    active_structure_key = f"{idx}_{active_structure_mode}"
    if active_structure_key in structures or idx in structures:
        structure_text = structures.get(active_structure_key, structures.get(idx, ""))
        structure_label = "S/V/O/C整理" if active_structure_mode == "theory" else "直感で読む構造"
        old_long_style = (
            "受験生の皆さん" in structure_text
            or "今日の一文" in structure_text
            or "今回扱う英文" in structure_text
            or len(structure_text) > 850
        )
        st.markdown(f"**🧩 構造解説：{structure_label}**")
        if old_long_style:
            st.info("長い形式の解説が残っています。上の構造ボタンを押すと、短い形式で作り直します。")
        else:
            with st.container(border=True, height=240):
                st.markdown(structure_text)


def render_reading_chat_panel(current_sentence, idx):
    col_chat_title, col_chat_reset = st.columns([3, 1])
    with col_chat_title:
        st.markdown("##### 🤖 和訳チャレンジ＆伴走チャット")
    if col_chat_reset.button("🧹 リセット", use_container_width=True, key=f"reset_reading_chat_{idx}"):
        st.session_state.reading_chat_logs = [
            {"role": "assistant", "content": "まずはこの文を一緒に短くほどきます。和訳でも質問でも大丈夫です。"}
        ]
        st.rerun()
    chat_height = 175 if len(st.session_state.reading_chat_logs) <= 1 else 290
    with st.container(border=True, height=chat_height):
        for msg in st.session_state.reading_chat_logs:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    with st.form(f"reading_chat_form_{idx}", clear_on_submit=True):
        user_req = st.text_area(
            "和訳・質問",
            height=70,
            key=f"reading_chat_input_{idx}",
            label_visibility="collapsed",
            placeholder="和訳を入力、または質問する...",
        )
        sent = st.form_submit_button("送信 ▶", use_container_width=True, type="primary")
    if sent and user_req.strip():
        st.session_state.reading_chat_logs.append({"role": "user", "content": user_req})
        with st.spinner("解説を作成中..."):
            sys_reading = f"""
            あなたは大学受験英語のやさしい読解サポーターです。
            知識を見せるためではなく、生徒が次に自分で読めるようになるためだけに答えてください。

            【これまでの長文全体の対話履歴】
            {st.session_state.get('reading_global_memory', 'まだありません')}

            【現在のターゲット文】
            {current_sentence}

            【指導方針】
            - 「受験生の皆さん」「今日の一文」などの定型あいさつは禁止です。
            - 生徒の回答に対する短い励まし・受け止めは必ず入れてください。例:「その読み方でかなり近いです」「そこに気づけているのは良いです」。
            - ただし本文と関係ない長い応援文や締めメッセージは不要です。
            - 「超重要」「神」「どこよりも詳しく」などの誇張表現は禁止です。
            - 口調は丁寧でよいが、雑談をせず、読解に必要な説明だけを出してください。
            - 返答は原則250〜450字以内。長くなる場合は、まず要点だけを出し、詳しい説明は質問された部分だけに絞ってください。
            - 生徒の質問への答えを最初に出してください。基本の流れは「短い受け止め → 結論 → 形のヒント → 自然な訳」です。
            - 生徒の自主性を大切にし、まず生徒の読み方を受け止めてください。
            - 説明は必要な点を落とさず、ただし一度に全部を詰め込みすぎないでください。
            - 文法名から入らず、まず「形 → 意味」の直観的な読み方に落としてください。文法用語は必要なときだけ補足として使ってください。
            - 初学者向けには S/V/O/C や関係詞名よりも、「どこで切るか」「どの形がどの意味になるか」を優先してください。
            - 例: "the way + S + V" は「SがVする方法・やり方」と先に説明し、必要なら「how の省略」と一言だけ補足してください。
            - S/V/O/C、関係詞名、分詞構文などの文法名は、生徒が聞いたとき、または誤読修正に必要なときだけ使ってください。
            - 生徒が和訳を書いた場合は、良い点、ズレ、自然な訳、構文の根拠を示してください。
            - いきなり全訳だけで終わらず、なぜそう読めるのかを説明してください。

            【禁止する出力】
            - 知識を列挙するだけの長い講義
            - 「この文の骨組みは実はとてもシンプルです」のような講師っぽい前置き
            - 4項目以上の長い番号付き解説
            - 生徒が聞いていない文法知識の見せびらかし
            - 1回の返答で全文を分解し切ろうとすること
            """
            history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.reading_chat_logs[-4:]])
            ans = call_ai(
                f"直近の会話:\n{history_str}",
                sys_reading,
                use_pdf=False,
                model_name="gemini-2.5-flash-lite",
                max_output_tokens=650,
                timeout_seconds=35,
            )
            st.session_state.reading_chat_logs.append({"role": "assistant", "content": ans})
            st.rerun()


def render_reading_dictation_panel(current_sentence, idx):
    hide_key = f"dictation_hide_{idx}"
    if hide_key not in st.session_state:
        st.session_state[hide_key] = False
    st.checkbox("英文を隠して、音声だけで書き起こす", key=hide_key)
    render_reading_audio_controls(current_sentence, f"dictation_practice_{idx}", "この文を聞く")
    dictation_answer = st.text_area(
        "聞こえた英文を書いてください",
        height=90,
        key=f"dictation_answer_{idx}",
        placeholder="The old bookstore, which ...",
    )
    result_key = f"dictation_result_{idx}"
    if st.button("照合する", use_container_width=True, key=f"dictation_check_{idx}"):
        st.session_state[result_key] = compare_dictation_answer(current_sentence, dictation_answer)
    if result_key in st.session_state:
        result = st.session_state[result_key]
        score = result["score"]
        if score >= 90:
            st.success(f"一致度 {score}%：かなり聞き取れています。")
        elif score >= 70:
            st.warning(f"一致度 {score}%：大筋は取れています。抜けた語を確認しましょう。")
        else:
            st.info(f"一致度 {score}%：まず短いかたまりごとに聞くのがよさそうです。")
        st.progress(min(score, 100) / 100)
        if result["missing"]:
            st.caption("抜けやすかった語: " + ", ".join(result["missing"]))
        if result["extra"]:
            st.caption("余分に入った語: " + ", ".join(result["extra"]))


def render_reading_words_panel(current_sentence, idx):
    st.caption("調べたい語をタップ、または熟語を入力してください。")
    seen = set()
    words = []
    for w in re.findall(r"\b[a-zA-Z\-']+\b", current_sentence):
        low = w.lower()
        if low not in seen:
            seen.add(low)
            words.append(w)
    if words:
        sel_word = st.pills(
            "単語",
            words,
            selection_mode="single",
            key=f"word_pills_{idx}",
            label_visibility="collapsed",
        )
        done_key = f"word_pills_done_{idx}"
        if sel_word and sel_word != st.session_state.get(done_key):
            st.session_state[done_key] = sel_word
            with st.spinner(f"「{sel_word}」の意味を検索中..."):
                sys_dict = "あなたは辞書です。指定された単語の、この文脈における意味を簡潔に答えてください。"
                meaning = call_ai(
                    f"文脈: {current_sentence}\n単語: {sel_word}",
                    sys_dict,
                    use_pdf=False,
                    model_name="gemini-2.5-flash-lite",
                    max_output_tokens=220,
                    timeout_seconds=25,
                )
                st.session_state.temp_memo.append({"word": sel_word, "meaning": meaning.strip()})
                st.rerun()

    col_idiom1, col_idiom2 = st.columns([3, 1])
    search_idiom = col_idiom1.text_input("熟語・フレーズの検索", label_visibility="collapsed", placeholder="調べたい熟語を入力 (例: take care of)", key=f"search_idiom_{idx}")
    if col_idiom2.button("🔍 検索", key=f"search_idiom_btn_{idx}", use_container_width=True) and search_idiom:
        with st.spinner("検索中..."):
            sys_dict = "あなたは辞書です。指定された熟語・フレーズの、この文脈における意味を簡潔に答えてください。"
            meaning = call_ai(
                f"文脈: {current_sentence}\n熟語: {search_idiom}",
                sys_dict,
                use_pdf=False,
                model_name="gemini-2.5-flash-lite",
                max_output_tokens=220,
                timeout_seconds=25,
            )
            st.session_state.temp_memo.append({"word": search_idiom, "meaning": meaning.strip()})
            st.rerun()

    if st.session_state.get("temp_memo"):
        st.markdown("###### 一時メモ")
        for m in st.session_state.temp_memo[-6:]:
            st.markdown(f"- **{m['word']}**: {m['meaning']}")


def render_reading_save_panel(idx):
    with st.form(f"reading_memo_form_{idx}", clear_on_submit=True):
        note_title = st.text_input("📝 項目名（例：関係副詞whereの非制限用法）", key=f"close_read_note_title_{idx}")
        note_category = st.selectbox("分類", ["構文・文法", "単語", "熟語", "指示語", "論理展開", "和訳", "設問根拠", "その他"], key=f"close_read_note_category_{idx}")
        note_content = st.text_area("意味・ルール（AIの解説や単語メモを保存）", height=80, key=f"close_read_note_content_{idx}")
        also_grammar = st.checkbox("文法・語法ノートにも保存する", value=(note_category == "構文・文法"), key=f"close_read_note_also_grammar_{idx}")
        if st.form_submit_button("🏠 読解メモに保存", type="primary"):
            if note_title and note_content:
                save_reading_note(note_title, note_content, note_category, "精読・構文・和訳", also_grammar)
                st.success("読解メモに保存しました！")
            else:
                st.error("項目名と内容の両方を入力してください。")


def render_reading_overall_map(sentences, idx, hide_current_sentence):
    current_sentence_for_audio = sentences[idx] if idx < len(sentences) else ""
    audio_col1, audio_col2 = st.columns(2)
    with audio_col1:
        render_reading_audio_controls(st.session_state.get("reading_target_text", ""), f"full_{idx}", "全文")
    with audio_col2:
        render_reading_audio_controls(current_sentence_for_audio, f"current_{idx}", "今の文")
    with st.container(border=True):
        display_html = ""
        for i, s in enumerate(sentences):
            safe_sentence = html.escape(s)
            if i == idx:
                shown_sentence = "音声書き起こし中：この文は隠しています" if hide_current_sentence else safe_sentence
                display_html += f"<span style='background-color:#ffeb3b; color:black; font-weight:bold; padding:2px 4px; border-radius:3px;'>{shown_sentence}</span> "
            elif i in st.session_state.get("reading_translations", {}):
                safe_translation = html.escape(st.session_state['reading_translations'][i])
                display_html += f"<span style='color:#e7e7e7;'>{safe_sentence}</span><br><span style='color:#8ab4ff;'>↳ {safe_translation}</span> "
            else:
                display_html += f"<span style='color:gray;'>{safe_sentence}</span> "
        st.markdown(f"<div class='reading-map-box'>{display_html}</div>", unsafe_allow_html=True)


def render_close_reading_tab():
    st.markdown("#### 📖 精読・構文・和訳")
    st.caption("長文を読みながら、わからない文だけ対話式に構造・文法・和訳を確認します。")

    # 長文がまだ無い時だけ大きく入力欄を出し、読み込み済みのときは小さくたたむ
    if st.session_state.get("reading_target_text"):
        with st.expander("📄 別の長文に切り替える", expanded=False):
            render_reading_input_section()
    else:
        render_reading_input_section()

    if not st.session_state.get("reading_target_text"):
        return

    stored_reading_text = st.session_state.get("reading_target_text", "")
    cleaned_stored_text = clean_reading_text(stored_reading_text)
    if cleaned_stored_text != stored_reading_text:
        st.session_state["reading_target_text"] = cleaned_stored_text
        st.session_state["reading_sentences"] = split_reading_sentences(cleaned_stored_text)
        if st.session_state["reading_sentences"]:
            st.session_state["reading_current_idx"] = 0
        st.session_state["reading_chat_sentence_idx"] = -1
        st.session_state["reading_translations"] = {}
        st.session_state["reading_structures"] = {}

    st.markdown("""
    <style>
    .reading-map-box {
        max-height: min(55vh, 520px);
        overflow-y: auto;
        line-height: 1.75;
        font-size: 1.02rem;
        padding: 0.35rem 0.45rem;
    }
    .reading-target-card {
        background: #15344f;
        border: 1px solid #245174;
        border-radius: 8px;
        color: #9bd0ff;
        font-weight: 700;
        font-size: 1.05rem;
        line-height: 1.5;
        padding: 0.7rem 0.9rem;
        margin: 0.2rem 0 0.6rem;
    }
    @media (max-width: 640px) {
        .reading-target-card { font-size: 0.98rem; padding: 0.6rem 0.7rem; }
        .reading-map-box { font-size: 0.96rem; line-height: 1.6; }
    }
    </style>
    """, unsafe_allow_html=True)

    sentences = st.session_state.get("reading_sentences", [])
    idx = st.session_state.get("reading_current_idx", 0)

    if idx >= len(sentences):
        st.success("🎉 全ての文の精読が完了しました。")
        if st.button("🔄 長文をクリアする", use_container_width=True, key="close_read_finish_clear"):
            clear_reading_state()
            st.rerun()
        return

    current_sentence = sentences[idx]
    hide_current_sentence = st.session_state.get(f"dictation_hide_{idx}", False)

    # 文が変わったらチャットと表示ツールを初期化（先頭タブに戻す）
    TOOL_TABS = ["🤖 解説・チャット", "🎧 聞き取り", "🔎 単語・熟語", "🗺️ 全体マップ", "💾 保存"]
    if st.session_state.get("reading_chat_sentence_idx") != idx:
        st.session_state.reading_chat_sentence_idx = idx
        st.session_state.reading_chat_logs = [
            {"role": "assistant", "content": "まずはこの文の和訳に挑戦してみて。質問でも大丈夫です。必要なら構造・修飾・文法まで一緒にほどいていきます。"}
        ]
        st.session_state.show_word_selector = False
        st.session_state.temp_memo = []
        st.session_state["reading_tool_tab"] = TOOL_TABS[0]

    # === 上部に固定的に見せるヘッダー：今読む文＋ナビ ===
    st.markdown(f"##### 🎯 現在のターゲット ({idx + 1}/{len(sentences)})")
    target_card_text = "音声書き起こし中：英文を隠しています" if hide_current_sentence else html.escape(current_sentence)
    st.markdown(f"<div class='reading-target-card'>{target_card_text}</div>", unsafe_allow_html=True)

    nav_left, nav_right = st.columns([1, 1])
    if nav_left.button("⏭️ 完璧！次の文へ", use_container_width=True, type="primary", key=f"next_sentence_{idx}"):
        current_chat = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.reading_chat_logs if m['role'] in ["user", "assistant"]])
        st.session_state.reading_global_memory += f"\n【文 {idx + 1}: {current_sentence} の対話】\n{current_chat}\n"
        st.session_state.reading_current_idx += 1
        st.rerun()
    if nav_right.button("🗑️ 長文をクリア", use_container_width=True, key="close_read_clear"):
        clear_reading_state()
        st.rerun()

    # === 機能はタブ（横ラジオ）で切り替え。文は常に上に残る ===
    tool = st.radio(
        "ツール",
        TOOL_TABS,
        horizontal=True,
        label_visibility="collapsed",
        key="reading_tool_tab",
    )

    if tool == "🤖 解説・チャット":
        structures = st.session_state.get("reading_structures", {})
        hint_open = (idx in st.session_state.get("reading_translations", {})) or any(
            k in structures for k in (f"{idx}_intuitive", f"{idx}_theory", idx)
        )
        with st.expander("🔍 ヒント：訳・構造を見る（困ったら）", expanded=hint_open):
            render_reading_structure_buttons(current_sentence, idx)
        with st.expander("✍️ 構文ラベルを振る（任意）", expanded=False):
            render_inline_syntax_label_practice(current_sentence, idx)
        render_reading_chat_panel(current_sentence, idx)
    elif tool == "🎧 聞き取り":
        render_reading_dictation_panel(current_sentence, idx)
    elif tool == "🔎 単語・熟語":
        render_reading_words_panel(current_sentence, idx)
    elif tool == "🗺️ 全体マップ":
        render_reading_overall_map(sentences, idx, hide_current_sentence)
    elif tool == "💾 保存":
        render_reading_save_panel(idx)


def render_close_reading_tab_legacy_pc():
    st.markdown("#### 📖 精読・構文・和訳")
    st.caption("長文を読みながら、わからない文だけ対話式に構造・文法・和訳を確認します。")

    read_method = st.radio(
        "題材となる長文の準備方法",
        ["📝 自分でテキストを貼り付ける", "📄 PDF・写真から抽出する", "🤖 AIにウォームアップ文を作ってもらう"],
        horizontal=True,
        key="close_read_method",
    )

    if read_method == "📝 自分でテキストを貼り付ける":
        pasted = st.text_area("英語の長文を貼り付けてください", height=150, key="close_read_pasted")
        if st.button("✅ このテキストで精読を始める", type="primary", key="close_read_start_text") and pasted:
            set_reading_text(pasted)
            st.rerun()

    elif read_method == "📄 PDF・写真から抽出する":
        up_media = st.file_uploader(
            "PDFまたは写真をアップロード",
            type=["pdf", "png", "jpg", "jpeg", "webp"],
            key="close_read_media",
        )
        if st.button("🚀 PDF・写真から長文を抽出する", type="primary", key="close_read_extract_media") and up_media:
            with st.spinner("AIが英語の長文本文だけを抽出しています..."):
                extracted_text = extract_reading_text_from_uploaded_media(up_media)
                set_reading_text(extracted_text)
                st.rerun()

    else:
        st.info("ウォームアップ用です。実力を伸ばす本番練習では、過去問PDFか実際の英文を使うのがおすすめです。")
        if st.button("✨ ウォームアップ文を生成する", type="primary", key="close_read_generate_warmup"):
            with st.spinner("AIがウォームアップ用の英文を書いています..."):
                sys_gen_read = """
                あなたは大学受験用の英文作成者です。
                180〜250語の英語長文を1つだけ作成してください。
                出力は英語本文だけにしてください。
                タイトル、前置き、説明、日本語、箇条書き、注釈は禁止です。
                "Here is a passage..." や "This passage..." のような導入文も禁止です。
                1文字目から本文の最初の文を始めてください。
                関係詞、分詞、不定詞、比較、接続、指示語が自然に含まれる英文にしてください。
                """
                text = call_ai(
                    "Create one original passage for reading practice. Output only the passage itself.",
                    sys_gen_read,
                    use_pdf=False,
                    model_name="gemini-2.5-flash-lite",
                    max_output_tokens=650,
                    timeout_seconds=35,
                )
                set_reading_text(text)
                st.rerun()

    if st.session_state.get("reading_target_text"):
        stored_reading_text = st.session_state.get("reading_target_text", "")
        cleaned_stored_text = clean_reading_text(stored_reading_text)
        if cleaned_stored_text != stored_reading_text:
            st.session_state["reading_target_text"] = cleaned_stored_text
            st.session_state["reading_sentences"] = split_reading_sentences(cleaned_stored_text)
            if st.session_state["reading_sentences"]:
                st.session_state["reading_current_idx"] = 0
            st.session_state["reading_chat_sentence_idx"] = -1
            st.session_state["reading_translations"] = {}
            st.session_state["reading_structures"] = {}
        st.markdown("---")
        st.markdown("""
        <style>
        .reading-map-box {
            max-height: min(48vh, 500px);
            overflow-y: auto;
            line-height: 1.75;
            font-size: 1.05rem;
            padding: 0.35rem 0.45rem;
        }
        .reading-target-card {
            background: #15344f;
            border: 1px solid #245174;
            border-radius: 7px;
            color: #69b7ff;
            font-weight: 700;
            padding: 0.7rem 0.85rem;
            margin-bottom: 0.55rem;
        }
        .reading-mini-space {
            height: 0.35rem;
        }
        .syntax-token-word {
            border-bottom: 2px solid #6aa9ff;
            display: inline-block;
            padding: 0 0.15rem 0.1rem;
            margin-bottom: 0.2rem;
            font-weight: 700;
            color: #f4f7fb;
            max-width: 100%;
            overflow-wrap: anywhere;
        }
        div[data-testid="column"]:has(.reading-sticky-anchor),
        div[data-testid="column"]:has(.reading-work-anchor) {
            position: sticky;
            top: 0.55rem;
            align-self: flex-start;
            z-index: 2;
            height: calc(100vh - 1.1rem);
            overflow-y: auto;
            overflow-x: hidden;
            overscroll-behavior: contain;
            padding-right: 0.35rem;
        }
        div[data-testid="column"]:has(.reading-sticky-anchor)::-webkit-scrollbar,
        div[data-testid="column"]:has(.reading-work-anchor)::-webkit-scrollbar {
            width: 8px;
        }
        div[data-testid="column"]:has(.reading-sticky-anchor)::-webkit-scrollbar-thumb,
        div[data-testid="column"]:has(.reading-work-anchor)::-webkit-scrollbar-thumb {
            background: #3b4454;
            border-radius: 999px;
        }
        @media (max-width: 900px) {
            div[data-testid="column"]:has(.reading-sticky-anchor),
            div[data-testid="column"]:has(.reading-work-anchor) {
                position: static;
                height: auto;
                overflow-y: visible;
                padding-right: 0;
            }
            .reading-map-box {
                max-height: 230px;
                line-height: 1.55;
                font-size: 0.96rem;
                padding: 0.25rem 0.35rem;
            }
            .reading-target-card {
                padding: 0.55rem 0.65rem;
                margin-bottom: 0.35rem;
                font-size: 0.95rem;
            }
        }
        </style>
        """, unsafe_allow_html=True)
        col_read, col_chat = st.columns([1, 1])
        sentences = st.session_state.get("reading_sentences", [])
        idx = st.session_state.get("reading_current_idx", 0)
        hide_current_sentence = st.session_state.get(f"dictation_hide_{idx}", False)

        with col_read:
            st.markdown('<div class="reading-sticky-anchor"></div>', unsafe_allow_html=True)
            st.markdown("##### 📄 全体マップ")
            current_sentence_for_audio = sentences[idx] if idx < len(sentences) else ""
            audio_col1, audio_col2 = st.columns(2)
            with audio_col1:
                render_reading_audio_controls(st.session_state.get("reading_target_text", ""), f"full_{idx}", "全文")
            with audio_col2:
                render_reading_audio_controls(current_sentence_for_audio, f"current_{idx}", "今の文")
            with st.container(border=True):
                display_html = ""
                for i, s in enumerate(sentences):
                    safe_sentence = html.escape(s)
                    if i == idx:
                        shown_sentence = "音声書き起こし中：この文は隠しています" if hide_current_sentence else safe_sentence
                        display_html += f"<span style='background-color:#ffeb3b; color:black; font-weight:bold; padding:2px 4px; border-radius:3px;'>{shown_sentence}</span> "
                    elif i in st.session_state.get("reading_translations", {}):
                        safe_translation = html.escape(st.session_state['reading_translations'][i])
                        display_html += f"<span style='color:#e7e7e7;'>{safe_sentence}</span><br><span style='color:#8ab4ff;'>↳ {safe_translation}</span> "
                    else:
                        display_html += f"<span style='color:gray;'>{safe_sentence}</span> "
                st.markdown(f"<div class='reading-map-box'>{display_html}</div>", unsafe_allow_html=True)

            if st.button("🗑️ 長文をクリアして別のものを読む", use_container_width=True, key="close_read_clear"):
                clear_reading_state()
                st.rerun()

        with col_chat:
            st.markdown('<div class="reading-work-anchor"></div>', unsafe_allow_html=True)
            if idx < len(sentences):
                current_sentence = sentences[idx]
                st.markdown(f"##### 🎯 現在のターゲット ({idx + 1}/{len(sentences)})")
                target_card_text = "音声書き起こし中：英文を隠しています" if hide_current_sentence else html.escape(current_sentence)
                st.markdown(f"<div class='reading-target-card'>{target_card_text}</div>", unsafe_allow_html=True)

                if st.session_state.get("reading_chat_sentence_idx") != idx:
                    st.session_state.reading_chat_sentence_idx = idx
                    st.session_state.reading_chat_logs = [
                        {"role": "assistant", "content": "まずはこの文の和訳に挑戦してみて。質問でも大丈夫です。必要なら構造・修飾・文法まで一緒にほどいていきます。"}
                    ]
                    st.session_state.show_word_selector = False
                    st.session_state.temp_memo = []

                col_t1, col_t2, col_t3 = st.columns(3)
                if col_t1.button("🈁 この文の訳を見る", use_container_width=True, key=f"show_translation_{idx}"):
                    with st.spinner("自然な日本語訳を作成中..."):
                        sys_translation = "あなたは大学受験英語の講師です。英文を自然な日本語に訳してください。必要以上の解説はせず、訳だけを出力してください。"
                        if "reading_translations" not in st.session_state:
                            st.session_state.reading_translations = {}
                        st.session_state.reading_translations[idx] = call_ai(
                            current_sentence,
                            sys_translation,
                            model_name="gemini-2.5-flash-lite",
                            max_output_tokens=160,
                            timeout_seconds=25,
                        ).strip()
                        st.rerun()
                if col_t2.button("🧩 直感で構造", use_container_width=True, key=f"show_structure_intuitive_{idx}"):
                    with st.spinner("構文を分解中..."):
                        sys_structure = """
                        あなたは大学受験英語の構文解説者です。英文を読むために必要な形だけを、短く説明してください。

                        【文体ルール】
                        - 「受験生の皆さん、こんにちは」「今日の一文」などの定型あいさつは禁止。
                        - 励まし、雑談、長いテーマ紹介、まとめメッセージは禁止。
                        - 「超重要」「良問」「どこよりも詳しく」などの誇張表現は禁止。
                        - 参考書の欄外メモのように短く書く。
                        - 英文全体を再掲しない。必要な語句だけ引用する。
                        - 220〜360字以内。
                        - 説明は「結論 → 形 → 訳」の順にする。
                        - 文法名から説明を始めない。まず「この形はこう読めばよい」という直観的な型に落とし込む。
                        - 文法用語は必要な場合だけ、最後に1行で補足する。

                        【必ず含める項目】
                        1. 結論: この文の大まかな意味を1文
                        2. 形: 覚えるべき形を「形 = 意味」で1〜2個
                        3. 切り方: 2〜3区切り
                        4. 訳: 自然な日本語訳

                        【出力形式】
                        見出しは「結論」「形」「切り方」「訳」だけ。
                        各見出しは1〜2行まで。

                        【禁止】
                        「今回扱う英文」「ステップ」「まとめ」「講師から」などの長い教材風構成は禁止。
                        S/V/O/C の記号をメインにした説明は禁止。
                        """
                        if "reading_structures" not in st.session_state:
                            st.session_state.reading_structures = {}
                        st.session_state[f"reading_structure_mode_{idx}"] = "intuitive"
                        st.session_state.reading_structures[f"{idx}_intuitive"] = call_ai(
                            current_sentence,
                            sys_structure,
                            model_name="gemini-2.5-flash-lite",
                            max_output_tokens=460,
                            timeout_seconds=35,
                        ).strip()
                        st.rerun()
                if col_t3.button("📐 S/V/O/Cで整理", use_container_width=True, key=f"show_structure_theory_{idx}"):
                    with st.spinner("文型と修飾関係を整理中..."):
                        sys_structure_theory = """
                        あなたは大学受験英語の構文解説者です。英文をS/V/O/C・句・節ラベルで整理してください。

                        【文体】
                        - あいさつ、励まし、長い前置きは禁止。
                        - 参考書の解答欄のように短く正確に書く。
                        - 220〜420字以内。
                        - 文法名を使ってよいが、解説は最小限にする。
                        - 長い修飾説明は禁止。ラベルを振ってから、必要な注意だけ書く。

                        【必ず含める項目】
                        1. 文型: 第何文型かを1行
                        2. 骨格: S / V / O / C を1行
                        3. ラベル: 関係詞節・名詞句・動詞句・前置詞句・分詞句などを最大4つ
                        4. 接続: and / or / but が何をつなぐか。なければ「目立つ接続なし」
                        5. 訳: 自然な日本語訳を1文

                        【出力形式】
                        見出しは「文型」「骨格」「ラベル」「接続」「訳」だけ。
                        各見出しは最大2行。長い講義にしない。
                        """
                        if "reading_structures" not in st.session_state:
                            st.session_state.reading_structures = {}
                        st.session_state[f"reading_structure_mode_{idx}"] = "theory"
                        st.session_state.reading_structures[f"{idx}_theory"] = call_ai(
                            current_sentence,
                            sys_structure_theory,
                            model_name="gemini-2.5-flash-lite",
                            max_output_tokens=520,
                            timeout_seconds=40,
                        ).strip()
                        st.rerun()

                if idx in st.session_state.get("reading_translations", {}):
                    st.success(f"日本語訳: {st.session_state['reading_translations'][idx]}")
                structures = st.session_state.get("reading_structures", {})
                active_structure_mode = st.session_state.get(f"reading_structure_mode_{idx}", "intuitive")
                active_structure_key = f"{idx}_{active_structure_mode}"
                if active_structure_key in structures or idx in structures:
                    structure_text = structures.get(active_structure_key, structures.get(idx, ""))
                    structure_label = "S/V/O/C整理" if active_structure_mode == "theory" else "直感で読む構造"
                    old_long_style = (
                        "受験生の皆さん" in structure_text
                        or "今日の一文" in structure_text
                        or "今回扱う英文" in structure_text
                        or len(structure_text) > 850
                    )
                    with st.expander(f"🧩 この文の構造解説：{structure_label}", expanded=True):
                        if old_long_style:
                            st.info("長い形式の解説が残っています。上の構造ボタンを押すと、短い形式で作り直します。")
                        else:
                            with st.container(border=False, height=240):
                                st.markdown(structure_text)

                with st.expander("✍️ 和訳前に構文ラベルも振る（任意）", expanded=False):
                    render_inline_syntax_label_practice(current_sentence, idx)

                col_btn1, col_btn2 = st.columns(2)
                if col_btn1.button("🧰 練習ツールを開く", use_container_width=True, key=f"toggle_words_{idx}"):
                    st.session_state.show_word_selector = not st.session_state.get("show_word_selector", False)
                    st.rerun()

                if col_btn2.button("⏭️ 完璧！スキップして次へ", use_container_width=True, type="primary", key=f"next_sentence_{idx}"):
                    current_chat = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.reading_chat_logs if m['role'] in ["user", "assistant"]])
                    st.session_state.reading_global_memory += f"\n【文 {idx + 1}: {current_sentence} の対話】\n{current_chat}\n"
                    st.session_state.reading_current_idx += 1
                    st.rerun()

                col_chat_title, col_chat_reset = st.columns([3, 1])
                with col_chat_title:
                    st.markdown("##### 🤖 和訳チャレンジ＆伴走チャット")
                if col_chat_reset.button("🧹 リセット", use_container_width=True, key=f"reset_reading_chat_{idx}"):
                    st.session_state.reading_chat_logs = [
                        {"role": "assistant", "content": "まずはこの文を一緒に短くほどきます。和訳でも質問でも大丈夫です。"}
                    ]
                    st.rerun()
                chat_height = 175 if len(st.session_state.reading_chat_logs) <= 1 else 290
                with st.container(border=True, height=chat_height):
                    for msg in st.session_state.reading_chat_logs:
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])

                if user_req := st.chat_input("和訳を入力、または質問する...", key="reading_chat"):
                    st.session_state.reading_chat_logs.append({"role": "user", "content": user_req})
                    with st.spinner("解説を作成中..."):
                        sys_reading = f"""
                        あなたは大学受験英語のやさしい読解サポーターです。
                        知識を見せるためではなく、生徒が次に自分で読めるようになるためだけに答えてください。

                        【これまでの長文全体の対話履歴】
                        {st.session_state.get('reading_global_memory', 'まだありません')}

                        【現在のターゲット文】
                        {current_sentence}

                        【指導方針】
                        - 「受験生の皆さん」「今日の一文」などの定型あいさつは禁止です。
                        - 生徒の回答に対する短い励まし・受け止めは必ず入れてください。例:「その読み方でかなり近いです」「そこに気づけているのは良いです」。
                        - ただし本文と関係ない長い応援文や締めメッセージは不要です。
                        - 「超重要」「神」「どこよりも詳しく」などの誇張表現は禁止です。
                        - 口調は丁寧でよいが、雑談をせず、読解に必要な説明だけを出してください。
                        - 返答は原則250〜450字以内。長くなる場合は、まず要点だけを出し、詳しい説明は質問された部分だけに絞ってください。
                        - 生徒の質問への答えを最初に出してください。基本の流れは「短い受け止め → 結論 → 形のヒント → 自然な訳」です。
                        - 生徒の自主性を大切にし、まず生徒の読み方を受け止めてください。
                        - 説明は必要な点を落とさず、ただし一度に全部を詰め込みすぎないでください。
                        - 文法名から入らず、まず「形 → 意味」の直観的な読み方に落としてください。文法用語は必要なときだけ補足として使ってください。
                        - 初学者向けには S/V/O/C や関係詞名よりも、「どこで切るか」「どの形がどの意味になるか」を優先してください。
                        - 例: "the way + S + V" は「SがVする方法・やり方」と先に説明し、必要なら「how の省略」と一言だけ補足してください。
                        - S/V/O/C、関係詞名、分詞構文などの文法名は、生徒が聞いたとき、または誤読修正に必要なときだけ使ってください。
                        - 生徒が和訳を書いた場合は、良い点、ズレ、自然な訳、構文の根拠を示してください。
                        - いきなり全訳だけで終わらず、なぜそう読めるのかを説明してください。

                        【禁止する出力】
                        - 知識を列挙するだけの長い講義
                        - 「この文の骨組みは実はとてもシンプルです」のような講師っぽい前置き
                        - 4項目以上の長い番号付き解説
                        - 生徒が聞いていない文法知識の見せびらかし
                        - 1回の返答で全文を分解し切ろうとすること
                        """
                        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.reading_chat_logs[-4:]])
                        ans = call_ai(
                            f"直近の会話:\n{history_str}",
                            sys_reading,
                            use_pdf=False,
                            model_name="gemini-2.5-flash-lite",
                            max_output_tokens=650,
                            timeout_seconds=35,
                        )
                        st.session_state.reading_chat_logs.append({"role": "assistant", "content": ans})
                        st.rerun()

                with st.expander("🧰 必要な時だけ開く練習ツール", expanded=st.session_state.get("show_word_selector", False)):
                    tab_dictation, tab_words, tab_save = st.tabs([
                        "🎧 聞き取り",
                        "🔎 単語・熟語",
                        "💡 保存",
                    ])

                    with tab_dictation:
                        hide_key = f"dictation_hide_{idx}"
                        if hide_key not in st.session_state:
                            st.session_state[hide_key] = False
                        st.checkbox("英文を隠して、音声だけで書き起こす", key=hide_key)
                        render_reading_audio_controls(current_sentence, f"dictation_practice_{idx}", "この文を聞く")
                        dictation_answer = st.text_area(
                            "聞こえた英文を書いてください",
                            height=90,
                            key=f"dictation_answer_{idx}",
                            placeholder="The old bookstore, which ...",
                        )
                        result_key = f"dictation_result_{idx}"
                        if st.button("照合する", use_container_width=True, key=f"dictation_check_{idx}"):
                            st.session_state[result_key] = compare_dictation_answer(current_sentence, dictation_answer)
                        if result_key in st.session_state:
                            result = st.session_state[result_key]
                            score = result["score"]
                            if score >= 90:
                                st.success(f"一致度 {score}%：かなり聞き取れています。")
                            elif score >= 70:
                                st.warning(f"一致度 {score}%：大筋は取れています。抜けた語を確認しましょう。")
                            else:
                                st.info(f"一致度 {score}%：まず短いかたまりごとに聞くのがよさそうです。")
                            st.progress(min(score, 100) / 100)
                            if result["missing"]:
                                st.caption("抜けやすかった語: " + ", ".join(result["missing"]))
                            if result["extra"]:
                                st.caption("余分に入った語: " + ", ".join(result["extra"]))

                    with tab_words:
                        st.caption("調べたい語を押すか、熟語を入力してください。")
                        words = [w for w in re.findall(r"\b[a-zA-Z\-']+\b", current_sentence)]
                        cols = st.columns(6)
                        for i, w in enumerate(words):
                            if cols[i % 6].button(w, key=f"wbtn_{i}_{idx}"):
                                with st.spinner(f"「{w}」の意味を検索中..."):
                                    sys_dict = "あなたは辞書です。指定された単語の、この文脈における意味を簡潔に答えてください。"
                                    meaning = call_ai(
                                        f"文脈: {current_sentence}\n単語: {w}",
                                        sys_dict,
                                        use_pdf=False,
                                        model_name="gemini-2.5-flash-lite",
                                        max_output_tokens=220,
                                        timeout_seconds=25,
                                    )
                                    st.session_state.temp_memo.append({"word": w, "meaning": meaning.strip()})
                                    st.rerun()

                        col_idiom1, col_idiom2 = st.columns([3, 1])
                        search_idiom = col_idiom1.text_input("熟語・フレーズの検索", label_visibility="collapsed", placeholder="調べたい熟語を入力 (例: take care of)", key=f"search_idiom_{idx}")
                        if col_idiom2.button("🔍 検索", key=f"search_idiom_btn_{idx}", use_container_width=True) and search_idiom:
                            with st.spinner("検索中..."):
                                sys_dict = "あなたは辞書です。指定された熟語・フレーズの、この文脈における意味を簡潔に答えてください。"
                                meaning = call_ai(
                                    f"文脈: {current_sentence}\n熟語: {search_idiom}",
                                    sys_dict,
                                    use_pdf=False,
                                    model_name="gemini-2.5-flash-lite",
                                    max_output_tokens=220,
                                    timeout_seconds=25,
                                )
                                st.session_state.temp_memo.append({"word": search_idiom, "meaning": meaning.strip()})
                                st.rerun()

                        if st.session_state.get("temp_memo"):
                            st.markdown("###### 一時メモ")
                            for m in st.session_state.temp_memo[-6:]:
                                st.markdown(f"- **{m['word']}**: {m['meaning']}")

                    with tab_save:
                        with st.form("reading_memo_form", clear_on_submit=True):
                            note_title = st.text_input("📝 項目名（例：関係副詞whereの非制限用法）", key="close_read_note_title")
                            note_category = st.selectbox("分類", ["構文・文法", "単語", "熟語", "指示語", "論理展開", "和訳", "設問根拠", "その他"], key="close_read_note_category")
                            note_content = st.text_area("意味・ルール（AIの解説や単語メモを保存）", height=80, key="close_read_note_content")
                            also_grammar = st.checkbox("文法・語法ノートにも保存する", value=(note_category == "構文・文法"), key="close_read_note_also_grammar")
                            if st.form_submit_button("🏠 読解メモに保存", type="primary"):
                                if note_title and note_content:
                                    save_reading_note(note_title, note_content, note_category, "精読・構文・和訳", also_grammar)
                                    st.success("読解メモに保存しました！")
                                else:
                                    st.error("項目名と内容の両方を入力してください。")

            else:
                st.success("🎉 全ての文の精読が完了しました。")
                if st.button("🔄 長文をクリアする", use_container_width=True, key="close_read_finish_clear"):
                    clear_reading_state()
                    st.rerun()

def render_evidence_training_tab():
    st.markdown("#### 🎯 設問根拠トレーニング")
    st.caption("長文全体を読み、設問ごとに本文の根拠と選択肢を切る理由を考える練習です。")

    source = st.radio("題材の準備方法", ["📝 長文＋設問を貼り付ける", "📄 PDFから抽出する"], horizontal=True, key="evidence_source")
    if source == "📝 長文＋設問を貼り付ける":
        raw_text = st.text_area("長文・設問・選択肢をまとめて貼り付けてください", height=220, key="evidence_raw_text")
        if st.button("✅ この内容で実践読解を始める", type="primary", key="start_evidence_text") and raw_text:
            st.session_state.evidence_text = raw_text
            st.session_state.evidence_feedback = ""
            st.rerun()
    else:
        up_pdf = st.file_uploader("PDFをアップロード", type=["pdf"], key="evidence_pdf")
        if st.button("🚀 PDFから長文・設問を抽出する", type="primary", key="extract_evidence_pdf") and up_pdf:
            with st.spinner("AIが長文・設問・選択肢を抽出しています..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(up_pdf.getvalue())
                    tmp_path = tmp.name
                try:
                    g_file = genai.upload_file(tmp_path)
                    model = genai.GenerativeModel(model_name="gemini-2.5-pro")
                    prompt = "このPDFから英語長文、設問、選択肢を読みやすく抽出してください。解答や解説は作らず、本文・設問番号・選択肢が分かる形で整理してください。"
                    res = model.generate_content([g_file, prompt])
                    st.session_state.evidence_text = res.text
                    st.session_state.evidence_feedback = ""
                finally:
                    os.remove(tmp_path)
                st.rerun()

    if st.session_state.get("evidence_text"):
        st.markdown("---")
        st.markdown("""
        <style>
        div[data-testid="column"]:has(.evidence-sticky-anchor) {
            position: sticky;
            top: 0.75rem;
            align-self: flex-start;
            z-index: 2;
        }
        div[data-testid="column"]:has(.evidence-sticky-anchor) [data-testid="stVerticalBlock"] {
            max-height: calc(100vh - 1.5rem);
        }
        @media (max-width: 900px) {
            div[data-testid="column"]:has(.evidence-sticky-anchor) {
                position: static;
            }
        }
        </style>
        """, unsafe_allow_html=True)
        col_passage, col_work = st.columns([1, 1])
        with col_passage:
            st.markdown('<div class="evidence-sticky-anchor"></div>', unsafe_allow_html=True)
            st.markdown("##### 📄 本文・設問")
            with st.container(border=True, height=560):
                st.markdown(st.session_state.evidence_text)
            if st.button("🗑️ 実践読解をクリア", use_container_width=True, key="clear_evidence"):
                for k in ["evidence_text", "evidence_feedback", "evidence_summary"]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()

        with col_work:
            st.markdown("##### 🧠 自分の根拠を書く")
            question_ref = st.text_input("設問番号・設問名", placeholder="例: 問3 / Q2", key="evidence_question_ref")
            user_answer = st.text_input("自分の答え", placeholder="例: 2 / B", key="evidence_user_answer")
            evidence_sentence = st.text_area("根拠にした本文", height=110, key="evidence_sentence")
            elimination_reason = st.text_area("選択肢を切った理由・迷った理由", height=110, key="evidence_reason")
            col_e1, col_e2 = st.columns(2)
            if col_e1.button("🔍 根拠を添削してもらう", type="primary", use_container_width=True, key="check_evidence"):
                with st.spinner("根拠の取り方を確認中..."):
                    sys_evidence = """
                    あなたは大学受験英語の長文読解講師です。生徒の答えを頭ごなしに否定せず、本文根拠・選択肢処理・読み戻し方を丁寧に添削してください。
                    必ず以下を含めてください。
                    1. 生徒の根拠の良い点
                    2. 根拠として弱い点・読み違い
                    3. 本文のどこを見るべきか
                    4. 選択肢を切る判断基準
                    5. 次に同じタイプを解く時の読み方
                    """
                    prompt = f"""
                    【本文・設問】
                    {st.session_state.evidence_text}

                    【設問】
                    {question_ref}
                    【生徒の答え】
                    {user_answer}
                    【生徒が根拠にした本文】
                    {evidence_sentence}
                    【選択肢を切った理由・迷った理由】
                    {elimination_reason}
                    """
                    st.session_state.evidence_feedback = call_ai(prompt, sys_evidence, model_name="gemini-2.5-pro")
                    st.rerun()

            if col_e2.button("🧭 全体の読み方を見る", use_container_width=True, key="summarize_evidence"):
                with st.spinner("段落役割と要旨を整理中..."):
                    sys_summary = """
                    あなたは大学受験英語の長文読解講師です。本文全体について、段落の役割、筆者の主張、対比・因果・具体例、指示語、設問で狙われそうな根拠を整理してください。
                    説明は優しく詳しく、ただし本文から離れた一般論にしないでください。
                    """
                    st.session_state.evidence_summary = call_ai(st.session_state.evidence_text, sys_summary, model_name="gemini-2.5-pro")
                    st.rerun()

            if st.session_state.get("evidence_feedback"):
                with st.expander("🔍 根拠添削", expanded=True):
                    st.markdown(st.session_state.evidence_feedback)
            if st.session_state.get("evidence_summary"):
                with st.expander("🧭 全体の読み方", expanded=True):
                    st.markdown(st.session_state.evidence_summary)

            with st.form("evidence_note_form", clear_on_submit=True):
                st.markdown("##### 💾 この設問から学んだことを保存")
                title = st.text_input("項目名", placeholder="例: 指示語thisの根拠を前文に戻って確認する", key="evidence_note_title")
                content = st.text_area("内容", value=st.session_state.get("evidence_feedback", "")[:1200], height=120, key="evidence_note_content")
                if st.form_submit_button("読解メモに保存"):
                    if title and content:
                        save_reading_note(title, content, "設問根拠", "設問根拠トレーニング", False)
                        st.success("読解メモに保存しました。")
                    else:
                        st.error("項目名と内容を入力してください。")

def render_reading_notes_tab():
    st.markdown("#### 🗂️ 読解メモ・弱点ノート")
    st.caption("長文中で読めなかった構文・文法・論理・設問根拠を体系的に蓄積します。")

    categories = ["構文・文法", "単語", "熟語", "指示語", "論理展開", "和訳", "設問根拠", "その他"]
    with st.form("manual_reading_note_form", clear_on_submit=True):
        col_n1, col_n2 = st.columns([2, 1])
        title = col_n1.text_input("項目名", placeholder="例: 分詞構文が主節全体を説明する形", key="manual_reading_note_title")
        category = col_n2.selectbox("分類", categories, key="manual_reading_note_category")
        content = st.text_area("内容", height=120, placeholder="読めなかった理由、AIの解説、次に見るべきポイントなど", key="manual_reading_note_content")
        source = st.text_input("出典・メモ", placeholder="例: 日本大学 2024 N方式 第2問", key="manual_reading_note_source")
        also_grammar = st.checkbox("文法・語法ノートにも保存する", value=(category == "構文・文法"), key="manual_reading_note_also_grammar")
        if st.form_submit_button("💾 読解メモに保存", type="primary"):
            if title and content:
                save_reading_note(title, content, category, source or "読解メモ", also_grammar)
                st.success("保存しました。")
            else:
                st.error("項目名と内容を入力してください。")

    notes = my_data.get("reading_notes", [])
    if not notes:
        st.info("読解メモはまだありません。精読や設問根拠トレーニングから保存できます。")
        return

    st.markdown("---")
    col_f1, col_f2 = st.columns([1, 2])
    selected_category = col_f1.selectbox("分類で絞り込み", ["すべて"] + categories, key="reading_notes_filter_category")
    keyword = col_f2.text_input("検索", placeholder="関係詞、指示語、根拠など", key="reading_notes_filter_keyword")
    filtered = []
    for note in notes:
        if selected_category != "すべて" and note.get("category") != selected_category:
            continue
        haystack = f"{note.get('title', '')} {note.get('content', '')} {note.get('source', '')}"
        if keyword and keyword.lower() not in haystack.lower():
            continue
        filtered.append(note)

    st.write(f"表示中: **{len(filtered)}件** / 全 **{len(notes)}件**")
    for cat in categories:
        cat_notes = [n for n in filtered if n.get("category") == cat]
        if not cat_notes:
            continue
        with st.expander(f"{cat} ({len(cat_notes)}件)", expanded=(selected_category == cat)):
            for note in reversed(cat_notes):
                st.markdown(f"**{note.get('title', '')}**")
                st.caption(f"{note.get('source', '')} / {note.get('created_at', '')}")
                st.markdown(note.get("content", ""))
                st.markdown("---")

    if st.button("🧭 読解メモを体系化して弱点を整理する", type="primary", use_container_width=True):
        with st.spinner("読解メモを整理しています..."):
            compact_notes = "\n\n".join([
                f"分類: {n.get('category', '')}\n項目: {n.get('title', '')}\n内容: {n.get('content', '')}"
                for n in notes[-80:]
            ])
            sys_notes = """
            あなたは大学受験英語の学習設計者です。生徒の読解メモを体系化し、弱点を整理してください。
            出力は以下の形にしてください。
            1. 主要な弱点カテゴリ
            2. 似たメモの統合
            3. 次に優先して練習すべきこと
            4. 文法・語法ノートに移すべき知識
            5. 長文読解で毎回見るチェックリスト
            """
            st.session_state.reading_notes_summary = call_ai(compact_notes, sys_notes, model_name="gemini-2.5-pro")
            st.rerun()

    if st.session_state.get("reading_notes_summary"):
        with st.expander("🧭 体系化された弱点整理", expanded=True):
            st.markdown(st.session_state.reading_notes_summary)

# --- 共通UIコンポーネント（階層型過去問選択） ---
def render_exam_selector(options_dict, key_prefix):
    """階層型で過去問を選択するUI。選ばれたラベルのリストを返す"""
    if not options_dict: return []
    unis = sorted(list(set([v["u"] for v in options_dict.values()])))
    sel_u = st.selectbox("1️⃣ 大学を選択", ["-- 選択 --"] + unis, key=f"{key_prefix}_u")
    if sel_u == "-- 選択 --": return []
    
    facs = sorted(list(set([v["f"] for v in options_dict.values() if v["u"] == sel_u])))
    sel_f = st.selectbox("2️⃣ 学部を選択", ["-- 選択 --"] + facs, key=f"{key_prefix}_f")
    if sel_f == "-- 選択 --": return []
    
    methods = sorted(list(set([v["m"] for v in options_dict.values() if v["u"] == sel_u and v["f"] == sel_f])))
    # ▼ ここをマルチセレクト（複数選択）に変更！
    sel_m_list = st.multiselect("3️⃣ 方式を選択（複数選択可）", methods, default=methods, key=f"{key_prefix}_m")
    if not sel_m_list: return []
    
    years = sorted(list(set([v["y"] for v in options_dict.values() if v["u"] == sel_u and v["f"] == sel_f and v["m"] in sel_m_list])), reverse=True)
    sel_y_list = st.multiselect("4️⃣ 年度を選択（複数選択可）", years, default=years, key=f"{key_prefix}_y")
    
    return [lbl for lbl, d in options_dict.items() if d["u"] == sel_u and d["f"] == sel_f and d["m"] in sel_m_list and d["y"] in sel_y_list]

# --- 4. メイン画面 ---
st.title(mode)



# ==========================================
# ★真の完全版 モードD: 志望校別単語帳（本棚 ＋ 任意AIフィルター ＋ シミュレーター）
# ==========================================
if mode == "📖 志望校別単語帳":
    st.markdown("あなた専用の単語帳を作成し、本棚で管理します。")
    
    db_options = {}
    if exam_db:
        for cat, unis in exam_db.items():
            for uni, facs in unis.items():
                for fac, years in facs.items():
                    for year, methods in years.items():
                        for method, data in methods.items():
                            if data.get("frequencies"): # 🛡️ 単語データが空のものを除外！
                                label = f"[{cat}] {uni} {fac} ({year}年 {method})"
                                db_options[label] = {"c": cat, "u": uni, "f": fac, "y": year, "m": method}
                                
    if not db_options:
        st.warning("単語データが登録された過去問がありません。まずは db_manager.py で過去問を登録してください。")
    else:
                            
        tab_shelf, tab_output, tab_create, tab_sim = st.tabs(["📚 あなたの本棚", "📤 アウトプット学習", "✨ 新しい単語帳を作る", "📊 カバー率・難化シミュレーター"])
        
        # ------------------------------------------
        # タブ1: 本棚 (保存された単語帳の管理)
        # ------------------------------------------
        with tab_shelf:
            saved_books = my_data.get("vocab_books", [])
            fixed_book = build_frequency_strong_book(base_lexicon)
            books = ([fixed_book] if fixed_book else []) + saved_books
            if not books:
                st.info("まだ単語帳がありません。「新しい単語帳を作る」タブから作成してください。")
            else:
                book_titles = [b["title"] for b in books]
                selected_title = st.selectbox("📖 管理・学習する単語帳を選択してください", ["-- 選択してください --"] + book_titles)
                
                if selected_title != "-- 選択してください --":
                    book_idx = book_titles.index(selected_title)
                    current_book = books[book_idx]
                    is_fixed_book = current_book.get("fixed_source") == BASE_LEXICON_FILE
                    saved_book_idx = book_idx - 1 if fixed_book else book_idx
                    
                    st.markdown(f"### 📘 {current_book['title']}")
                    st.write(f"収録語数: メイン **{len(current_book['main_vocab'])}語** / 除外 **{len(current_book['excluded_vocab'])}語**")
                    if is_fixed_book:
                        st.info(f"この単語帳は `{BASE_LEXICON_FILE}` から固定読み込みしています。意味登録済み: **{len(current_book.get('enriched_vocab', [])):,} / {len(current_book['main_vocab']):,}語**")
                        last_generation = st.session_state.pop("last_frequency_strong_generation", None)
                        if last_generation:
                            batch_text = f" / {last_generation['batches']}回実行" if last_generation.get("batches") else ""
                            st.success(f"直前の意味生成: 依頼 {last_generation['requested']}語 / 保存 {last_generation['saved']}語{batch_text}")
                            if last_generation.get("missing"):
                                st.warning("AIが返さなかった単語があります。もう一度同じ語数で生成すると、未保存分から続きます: " + ", ".join(last_generation["missing"]))
                            if last_generation.get("extra"):
                                st.info("AIが指定外の単語も返しましたが、保存対象から外しました: " + ", ".join(last_generation["extra"]))
                            if last_generation.get("failed"):
                                st.warning("途中停止しました。次回は未保存分から続けられます。失敗時の先頭語: " + ", ".join(last_generation.get("failed_words", [])))
                        meaning_report = build_frequency_strong_meaning_report(base_lexicon)
                        with st.expander("🧪 頻度つよつよ単語3000の意味完成度チェック", expanded=False):
                            report_cols = st.columns(4)
                            report_cols[0].metric("固定語", f"{len(meaning_report['rows']):,}語")
                            report_cols[1].metric("意味未生成", f"{len(meaning_report['missing']):,}語")
                            report_cols[2].metric("薄い可能性", f"{len(meaning_report['thin']):,}語")
                            report_cols[3].metric("注意メモ候補", f"{len(meaning_report['no_alert_polysemy']):,}語")

                            st.download_button(
                                "📥 意味チェックCSVをダウンロード",
                                data=rows_to_csv(meaning_report["rows"]),
                                file_name="frequency_strong_meaning_check.csv",
                                mime="text/csv",
                                use_container_width=True,
                                key="download_frequency_strong_meaning_check",
                            )

                            check_tab1, check_tab2, check_tab3 = st.tabs(["未生成", "意味が薄い候補", "注意メモ候補"])
                            with check_tab1:
                                st.caption("ここに出る語は、次の意味生成で上から順に処理されます。")
                                st.dataframe(meaning_report["missing"][:200], use_container_width=True)
                            with check_tab2:
                                st.caption("高頻度なのに意味が1項目だけの語です。多義語の取りこぼし確認に使います。")
                                st.dataframe(meaning_report["thin"][:200], use_container_width=True)
                            with check_tab3:
                                st.caption("意味が3項目以上あるのに注意メモが空の語です。訳し分けの注意が必要か確認します。")
                                st.dataframe(meaning_report["no_alert_polysemy"][:200], use_container_width=True)

                        generation_logs = load_base_lexicon_generation_log()
                        if generation_logs:
                            with st.expander("🧾 頻度つよつよ単語の生成履歴", expanded=False):
                                st.dataframe(list(reversed(generation_logs[-20:])), use_container_width=True)
                    
                    # --- 管理機能 ---
                    with st.expander("⚙️ 単語帳の管理・編集（名前変更・追加・マージ）", expanded=False):
                        if is_fixed_book:
                            st.info("頻度つよつよ単語は固定単語帳です。ここでの名前変更・マージ・削除は無効です。意味生成だけ下のボタンから実行できます。")
                        st.markdown("#### 🏷️ 名前の変更")
                        col_rn1, col_rn2 = st.columns([3, 1])
                        new_title_input = col_rn1.text_input("新しい名前", value=current_book["title"], label_visibility="collapsed", key=f"rn_vocab_{book_idx}")
                        if col_rn2.button("名前を更新", use_container_width=True, key=f"btn_rn_vocab_{book_idx}") and not is_fixed_book:
                            if new_title_input and new_title_input != current_book["title"]:
                                current_book["title"] = new_title_input
                                my_data["vocab_books"][saved_book_idx] = current_book
                                save_data(my_data)
                                st.success("名前を変更しました！")
                                st.rerun()

                        st.markdown("---")
                        st.markdown("#### 📈 過去問データを追加して単語帳を強化 (マージ)")
                        add_labels = render_exam_selector(db_options, f"merge_vocab_{book_idx}")
                        
                        # ▼ AIフィルターのチェックボックスを追加
                        use_ai_filter_merge = st.checkbox("🤖 マージする新規単語から「人名」等をAIで除外する", value=False, key=f"ai_merge_v_{book_idx}")
                        exclude_frequency_strong_merge = st.checkbox("💪 頻度つよつよ単語3000をマージ対象から外す", value=True, key=f"strong_merge_v_{book_idx}")
                        
                        if st.button("✨ 選択したデータを追加 (マージ)", type="primary", key=f"btn_merge_vocab_{book_idx}") and add_labels and not is_fixed_book:
                            with st.spinner("単語データを結合し、AIフィルターで審査しています..."):
                                current_counts = Counter(current_book.get("counts", {}))
                                current_origins = defaultdict(list, current_book.get("origins", {}))
                                
                                new_words_candidate = set() # AI審査用の新規単語候補
                                
                                for label in add_labels:
                                    path = db_options[label]
                                    freqs = exam_db[path["c"]][path["u"]][path["f"]][path["y"]][path["m"]]["frequencies"]
                                    if exclude_frequency_strong_merge:
                                        freqs = {w: c for w, c in freqs.items() if normalize_vocab_word(w) not in frequency_strong_words}
                                    current_counts.update(freqs)
                                    short_label = f"{str(path['y'])[-2:]}年"
                                    for w, count in freqs.items():
                                        current_origins[w].append(f"{short_label}({count}回)")
                                        new_words_candidate.add(w)
                                
                                current_book["counts"] = dict(current_counts)
                                current_book["origins"] = dict(current_origins)
                                
                                excluded_set = set(current_book.get("excluded_vocab", []))
                                main_set = set(current_book.get("main_vocab", []))
                                
                                # まだ単語帳にない完全新規の単語だけを抽出
                                new_words = [w for w in new_words_candidate if w not in excluded_set and w not in main_set]
                                
                                # ▼ AIフィルター処理
                                if use_ai_filter_merge and api_key and new_words:
                                    sys_filter = "英単語リストを分類しJSON出力: {\"main_vocab\": [\"technology\"], \"excluded_vocab\": [\"david\"]}。人名・企業名などはexcludedへ。一般名詞・地名はmainへ。"
                                    try:
                                        res = call_ai(f"リスト:\n{new_words}", sys_filter, is_json=True)
                                        filtered_data = json.loads(res)
                                        main_set.update(filtered_data.get("main_vocab", []))
                                        excluded_set.update(filtered_data.get("excluded_vocab", []))
                                    except:
                                        main_set.update(new_words) # エラー時はとりあえず全部メインへ
                                else:
                                    main_set.update(new_words)
                                        
                                # 新しい頻度でソート
                                current_book["main_vocab"] = sorted(list(main_set), key=lambda x: current_counts[x], reverse=True)
                                current_book["excluded_vocab"] = list(excluded_set)
                                
                                my_data["vocab_books"][saved_book_idx] = current_book
                                save_data(my_data)
                                st.success("データをマージしました！頻度順に再ソートされています。")
                                st.rerun()
                        
                        st.markdown("---")
                        st.markdown("#### 📋 複製と削除")
                        col_dup, col_del = st.columns(2)
                        if col_dup.button("📋 この単語帳を複製する", use_container_width=True) and not is_fixed_book:
                            new_book = current_book.copy()
                            new_book["title"] = current_book["title"] + " (コピー)"
                            new_book.pop("fixed_source", None)
                            my_data["vocab_books"].append(new_book)
                            save_data(my_data); st.rerun()
                            
                        if col_del.button("🗑️ この単語帳を削除する", use_container_width=True) and not is_fixed_book:
                            my_data["vocab_books"].pop(saved_book_idx)
                            save_data(my_data); st.rerun()
                        
                        st.markdown("---")
                        st.markdown("#### ✏️ 単語の個別追加・削除")
                        col_edit1, col_edit2 = st.columns(2)
                        with col_edit1:
                            with st.form(f"add_word_form_{book_idx}"):
                                new_word = st.text_input("➕ 新しい単語を手動で追加")
                                if st.form_submit_button("追加する") and new_word and not is_fixed_book:
                                    new_word = new_word.lower().strip()
                                    if new_word not in current_book["main_vocab"]:
                                        current_book["main_vocab"].insert(0, new_word)
                                        if new_word not in current_book.get("counts", {}): current_book.setdefault("counts", {})[new_word] = 1
                                        my_data["vocab_books"][saved_book_idx] = current_book
                                        save_data(my_data); st.rerun()
                                    else:
                                        st.warning("登録済みです。")

                        with col_edit2:
                            with st.form(f"del_word_form_{book_idx}"):
                                del_word = st.selectbox("🗑️ 削除したい単語を選択", ["-- 選択 --"] + current_book["main_vocab"] + current_book["excluded_vocab"])
                                if st.form_submit_button("削除する") and del_word != "-- 選択 --" and not is_fixed_book:
                                    del_word = del_word.lower().strip()
                                    if del_word in current_book["main_vocab"]: current_book["main_vocab"].remove(del_word)
                                    elif del_word in current_book["excluded_vocab"]: current_book["excluded_vocab"].remove(del_word)
                                    my_data["vocab_books"][saved_book_idx] = current_book
                                    save_data(my_data); st.rerun()
                        
                        st.markdown("---")
                        st.markdown("#### 🗑️ データの完全初期化")
                        if st.button("🗑️ AI生成データ(意味・例文等)をすべてリセットする", use_container_width=True):
                            if is_fixed_book:
                                reset_count = reset_frequency_strong_enrichment(base_lexicon)
                                st.success(f"頻度つよつよ単語のAI生成データを {reset_count}語分リセットしました。")
                                st.rerun()
                            else:
                                if "enriched_vocab" in current_book: del current_book["enriched_vocab"]
                                if "skipped_vocab" in current_book: del current_book["skipped_vocab"]
                                my_data["vocab_books"][saved_book_idx] = current_book
                                save_data(my_data); st.rerun()
                                
                    st.markdown("---")
                    
                    # 🛡️ 安全装置：データにキーが無ければ空リストをセットしてクラッシュを防ぐ
                    if "enriched_vocab" not in current_book:
                        current_book["enriched_vocab"] = []
                    if "skipped_vocab" not in current_book:
                        current_book["skipped_vocab"] = []
                    
                    # 未生成（差分）の単語をリストアップ
                    enriched_words = [e["word"] for e in current_book["enriched_vocab"]]
                    missing_words = [w for w in current_book["main_vocab"] if w not in enriched_words and w not in current_book["skipped_vocab"]]
                    generation_candidates = missing_words
                    generation_mode = "未生成語"
                    if is_fixed_book:
                        fixed_generation_report = build_frequency_strong_meaning_report(base_lexicon)
                        generation_mode = st.selectbox(
                            "意味生成・補強の対象",
                            ["未生成語", "意味が薄い候補を補強", "注意メモ候補を補強"],
                            key=f"strong_generation_mode_{book_idx}",
                        )
                        if generation_mode == "意味が薄い候補を補強":
                            generation_candidates = [row["単語"] for row in fixed_generation_report["thin"]]
                        elif generation_mode == "注意メモ候補を補強":
                            generation_candidates = [row["単語"] for row in fixed_generation_report["no_alert_polysemy"]]
                    
                    if generation_candidates:
                        if is_fixed_book:
                            st.info(f"💡 {generation_mode} が **{len(generation_candidates)}** 語あります。")
                        else:
                            st.info(f"💡 未生成の単語が **{len(missing_words)}** 個あります。（過去問マージ等で追加された単語です）")
                        
                        # --- 生成数指定UI ---
                        col_gen1, col_gen2 = st.columns([1, 2])
                        default_gen_count = min(30, len(generation_candidates))
                        gen_help = "30語ずつが安全ですが、入力した語数ちょうど生成します。" if is_fixed_book else "入力した語数ちょうど生成します。"
                        gen_count = int(col_gen1.number_input("生成する単語数", min_value=1, max_value=len(generation_candidates), value=default_gen_count, step=1, help=gen_help, key=f"gen_v_{book_idx}"))
                        batch_count = 1
                        if is_fixed_book:
                            max_batches = max(1, min(20, (len(generation_candidates) + gen_count - 1) // gen_count))
                            batch_count = int(col_gen1.number_input("連続回数", min_value=1, max_value=max_batches, value=1, step=1, help="例: 30語 × 5回 = 150語を連続生成します。", key=f"gen_batch_v_{book_idx}"))
                        
                        col_gen2.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True) # ボタンの高さを合わせる
                        total_requested_words = min(gen_count * batch_count, len(generation_candidates))
                        button_label = f"✨ {generation_mode}を上位 {total_requested_words} 語生成・補強" if is_fixed_book else f"✨ 未生成の単語を上位 {gen_count} 語生成して追加"
                        if col_gen2.button(button_label, use_container_width=True, type="primary"):
                            with st.spinner(f"AIがネイティブの脳内ネットワークを作成中...（上位 {total_requested_words} 語）"):
                                target_words = generation_candidates[:total_requested_words]
                                if is_fixed_book:
                                    sys_enrich = build_frequency_strong_enrich_prompt()
                                else:
                                    sys_enrich = """
                                あなたは受験英語に精通したプロの予備校講師です。提供された英単語リストを精査し、以下のJSON形式を作成してください。
                                【絶対ルール】
                                1. 「大学受験レベル(B1以上)で重要な単語」は "enriched" へ、「中学レベル(A1〜A2)やゴミ」は "skipped" へ。
                                {
                                  "enriched": [
                                    {
                                      "word": "company",
                                      "forms": "複数形: companies",
                                      "meanings": "① 会社、企業 ② 仲間、同席",
                                      "chunks": ["① run a **company** (会社を経営する)", "② in the **company** of friends (友達と一緒に)"],
                                      "context": "ビジネス系で必須。",
                                      "alert": "「仲間」の意味では不可算名詞。"
                                    }
                                  ],
                                  "skipped": ["people", "don"]
                                }
                                """
                                try:
                                    if is_fixed_book:
                                        total_saved = 0
                                        all_missing = []
                                        all_extra = []
                                        actual_batches = 0
                                        failed_message = ""
                                        failed_words = []
                                        progress = st.progress(0) if batch_count > 1 else None
                                        for batch_index in range(batch_count):
                                            batch_words = target_words[batch_index * gen_count:(batch_index + 1) * gen_count]
                                            if not batch_words:
                                                break
                                            try:
                                                response_json = call_ai(f"処理対象:\n{batch_words}", sys_enrich, is_json=True)
                                                parsed_data = json.loads(response_json)
                                                batch_result = apply_frequency_strong_ai_response(base_lexicon, batch_words, parsed_data)
                                            except Exception as batch_error:
                                                failed_message = str(batch_error)
                                                failed_words = batch_words[:30]
                                                break
                                            total_saved += batch_result["saved"]
                                            all_missing.extend(batch_result["missing"])
                                            all_extra.extend(batch_result["extra"])
                                            actual_batches += 1
                                            if progress:
                                                progress.progress(actual_batches / batch_count)
                                        generation_result = {
                                            "requested": len(target_words),
                                            "saved": total_saved,
                                            "missing": all_missing[:30],
                                            "extra": all_extra[:30],
                                            "batches": actual_batches,
                                        }
                                        if failed_message:
                                            generation_result["failed"] = failed_message
                                            generation_result["failed_words"] = failed_words
                                        st.session_state["last_frequency_strong_generation"] = generation_result
                                        append_base_lexicon_generation_log({
                                            "time": datetime.now().isoformat(timespec="seconds"),
                                            "mode": generation_mode,
                                            "requested": len(target_words),
                                            "saved": total_saved,
                                            "batches": actual_batches,
                                            "missing_count": len(all_missing),
                                            "extra_count": len(all_extra),
                                            "failed": bool(failed_message),
                                            "failed_message": failed_message[:200],
                                        })
                                        if failed_message:
                                            st.warning(f"途中まで保存しました。保存 {total_saved}語 / 実行 {actual_batches}回。失敗箇所は次回もう一度回せます。")
                                        else:
                                            st.success(f"🎉 頻度つよつよ単語に {total_saved}語分の意味を保存しました！")
                                    else:
                                        response_json = call_ai(f"処理対象:\n{target_words}", sys_enrich, is_json=True)
                                        parsed_data = json.loads(response_json)
                                        current_book["enriched_vocab"].extend(parsed_data.get("enriched", []))
                                        current_book["skipped_vocab"].extend(parsed_data.get("skipped", []))
                                        my_data["vocab_books"][saved_book_idx] = current_book
                                        save_data(my_data)
                                        st.success("🎉 生成が完了しました！")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"生成エラー: {e}")
                    elif is_fixed_book:
                        st.success(f"{generation_mode} はありません。")
                    
                    # 生成済みデータが1つもない場合は生のリストを表示
                    if not current_book["enriched_vocab"]:
                        main_display = [{"単語": w, "出現回数": current_book.get("counts", {}).get(w, "-")} for w in current_book["main_vocab"]]
                        st.dataframe(main_display, use_container_width=True)
                        
                    # AI生成済みデータがある場合はカード型UIを表示
                    else:
                        if "selected_word_idx" not in st.session_state:
                            st.session_state.selected_word_idx = None
                            
                        # モード1：リスト
                        if st.session_state.selected_word_idx is None:
                            st.markdown("### 📚 単語リスト")
                            
                            # ★ 最新の main_vocab の順序に合わせて enriched_vocab をソートして表示
                            enriched_dict = {item["word"]: item for item in current_book["enriched_vocab"]}
                            enriched_list = [enriched_dict[w] for w in current_book["main_vocab"] if w in enriched_dict]
                            
                            WORDS_PER_PAGE = 20 
                            total_pages = max(1, (len(enriched_list) + WORDS_PER_PAGE - 1) // WORDS_PER_PAGE)
                            
                            page_key = f"page_{book_idx}"
                            if page_key not in st.session_state: st.session_state[page_key] = 1
                            current_page = st.session_state[page_key]
                            
                            col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
                            with col_p1:
                                if st.button("◀ 前の20件", use_container_width=True, disabled=(current_page == 1)): st.session_state[page_key] -= 1; st.rerun()
                            with col_p2:
                                jump_page = st.number_input("ページジャンプ", min_value=1, max_value=total_pages, value=current_page, label_visibility="collapsed")
                                if jump_page != current_page: st.session_state[page_key] = jump_page; st.rerun()
                                st.markdown(f"<div style='text-align: center; font-size: 0.8em; color: gray;'>全{len(enriched_list)}語 / {total_pages}ページ</div>", unsafe_allow_html=True)
                            with col_p3:
                                if st.button("次の20件 ▶", use_container_width=True, disabled=(current_page == total_pages)): st.session_state[page_key] += 1; st.rerun()
                                    
                            st.divider()
                            start_idx = (current_page - 1) * WORDS_PER_PAGE
                            end_idx = start_idx + WORDS_PER_PAGE
                            current_page_vocab = enriched_list[start_idx:end_idx]
                            
                            cols = st.columns(2)
                            for i, item in enumerate(current_page_vocab):
                                # 元の配列(current_book["enriched_vocab"])上の本当のインデックスを探す
                                word = item.get("word", "")
                                actual_idx = next((index for (index, d) in enumerate(current_book["enriched_vocab"]) if d["word"] == word), None)
                                
                                count = current_book.get("counts", {}).get(word, "-")
                                
                                col = cols[i % 2]
                                with col:
                                    with st.container(border=True):
                                        st.markdown(f"<div style='margin-bottom: 15px;'><span style='color:#1f77b4; font-size:2.8em; font-weight:900; line-height:1.1;'>{word}</span><span style='font-size:1.2em; color:gray; margin-left:10px;'>({count}回)</span></div>", unsafe_allow_html=True)
                                        if item.get("forms"):
                                            st.markdown(f"<div style='color:#d62728; font-weight:bold; font-size:1.0em; margin-bottom: 10px;'>🔄 {item.get('forms')}</div>", unsafe_allow_html=True)
                                        if item.get("meanings"):
                                            st.markdown(f"<div style='font-size:1.15em; font-weight:bold; color:#4caf50; margin-bottom: 10px;'>{item.get('meanings')}</div>", unsafe_allow_html=True)
                                        st.divider() 
                                        for chunk in item.get("chunks", []):
                                            colored_chunk = re.sub(r'\*\*(.*?)\*\*', r"<span style='color:#1f77b4; font-weight:bold; font-size:1.1em;'>\1</span>", chunk)
                                            st.markdown(f"<div style='margin-left: 0.5em; margin-bottom: 8px; font-size:0.95em;'>{colored_chunk}</div>", unsafe_allow_html=True)
                                        
                                        st.markdown("<br>", unsafe_allow_html=True)
                                        if st.button(f"👉 詳細・文脈・メモを開く", key=f"sel_{book_idx}_{actual_idx}", use_container_width=True):
                                            st.session_state.selected_word_idx = actual_idx
                                            st.rerun()
                            
                            if len(current_page_vocab) > 4:
                                st.divider()
                                col_p1_b, col_p2_b, col_p3_b = st.columns([1, 2, 1])
                                with col_p1_b:
                                    if st.button("◀ 前へ", key="prev_b", use_container_width=True, disabled=(current_page == 1)): st.session_state[page_key] -= 1; st.rerun()
                                with col_p3_b:
                                    if st.button("次へ ▶", key="next_b", use_container_width=True, disabled=(current_page == total_pages)): st.session_state[page_key] += 1; st.rerun()
                                            
                        # モード2：詳細ルーム
                        else:
                            i = st.session_state.selected_word_idx
                            item = current_book["enriched_vocab"][i]
                            word = item.get("word", "")
                            
                            if st.button("🔙 単語リストに戻る", type="primary"):
                                st.session_state.selected_word_idx = None
                                st.rerun()
                                
                            st.markdown(f"## 🔍 「{word}」の専用ルーム")
                            with st.container(border=True):
                                if item.get("forms"): st.markdown(f"**🔄 変化形・派生語:** {item.get('forms')}")
                                st.markdown("### 📚 意味とフレーズ")
                                if item.get("meanings"): st.markdown(f"**<span style='color:#4caf50; font-size:1.2em;'>{item.get('meanings')}</span>**", unsafe_allow_html=True)
                                for chunk in item.get("chunks", []):
                                    colored_ex = re.sub(r'\*\*(.*?)\*\*', r"<span style='color:#1f77b4; font-weight:bold;'>\1</span>", chunk)
                                    st.markdown(f"> {colored_ex}", unsafe_allow_html=True)
                                if item.get("context"): st.markdown(f"📖 **文脈:** {item.get('context')}")
                                if item.get("alert"): st.markdown(f"**⚠️ 混同注意:** {item.get('alert')}")
                            
                            st.markdown("#### 📝 マイ・メモ")
                            current_memo = item.get("user_memo", "")
                            col_m1, col_m2 = st.columns([4, 1])
                            new_memo = col_m1.text_input(f"メモ入力", value=current_memo, key=f"memo_{i}", label_visibility="collapsed")
                            if col_m2.button("💾 保存", key=f"save_memo_{i}", use_container_width=True):
                                current_book["enriched_vocab"][i]["user_memo"] = new_memo
                                save_data(my_data); st.success("保存しました！")
                            
                            st.divider()
                            st.markdown("#### 📖 例文アシスト")
                            if saved_ex := item.get("saved_examples", []):
                                st.markdown("**【保存済みの例文】**")
                                for ex in saved_ex: st.markdown(f"- {ex}")
                            
                            if st.button("➕ AIに新しい例文を3つ作ってもらう", key=f"gen_ex_{i}"):
                                with st.spinner("生成中..."):
                                    res_ex = call_ai(f"単語: {word}", "指定単語の例文と和訳を3つJSON配列で出力。例: [\"I have an apple. (私はリンゴを持っています。)\"]", is_json=True)
                                    try: st.session_state[f"temp_ex_{word}"] = json.loads(res_ex); st.rerun()
                                    except: st.error("失敗しました。")
                            
                            if f"temp_ex_{word}" in st.session_state:
                                st.markdown("**💡 保存したいものにチェック：**")
                                with st.form(f"save_ex_form_{i}"):
                                    selected_ex = [ex for idx, ex in enumerate(st.session_state[f"temp_ex_{word}"]) if st.checkbox(ex, key=f"chk_{word}_{idx}")]
                                    if st.form_submit_button("✅ 選択を保存"):
                                        current_book["enriched_vocab"][i].setdefault("saved_examples", []).extend(selected_ex)
                                        save_data(my_data)
                                        del st.session_state[f"temp_ex_{word}"]
                                        st.success("追加保存しました！"); st.rerun()

                            st.divider()
                            st.markdown(f"#### 🤖 AI講師に質問する")
                            chat_key = f"chat_{word}"
                            if chat_key not in st.session_state: st.session_state[chat_key] = []
                            with st.container(height=300):
                                for msg in st.session_state[chat_key]:
                                    with st.chat_message(msg["role"]): st.markdown(msg["content"])
                            if st.session_state[chat_key] and st.session_state[chat_key][-1]["role"] == "user":
                                with st.spinner("AI講師が考え中..."):
                                    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state[chat_key][-5:]])
                                    ans = call_ai(f"会話履歴:\n{history_str}", f"英単語「{word}」の専属講師として簡潔に答えてください。", use_pdf=False)
                                    st.session_state[chat_key].append({"role": "assistant", "content": ans}); st.rerun()
                            if user_q := st.chat_input("質問する...", key=f"chat_in_{i}"):
                                st.session_state[chat_key].append({"role": "user", "content": user_q}); st.rerun()
                            
                        # スキップ単語の復活
                        skipped = current_book.get("skipped_vocab", [])
                        if skipped:
                            with st.expander("👀 AIが「基本語・ゴミ」と判定して除外した単語（ここから復活可能）", expanded=False):
                                st.write(", ".join(skipped))
                                with st.form(f"restore_form_{book_idx}"):
                                    restore_word = st.selectbox("🔄 メインに復活させる単語", ["-- 選択 --"] + skipped)
                                    if st.form_submit_button("解説を生成して復活") and restore_word != "-- 選択 --":
                                        with st.spinner("生成中..."):
                                            sys_restore = "指定単語の解説をJSONで作成。{\"enriched\":[{\"word\":\"単語\",\"chunks\":[\"📌 塊1\"],\"details\":\"🧠...\",\"alert\":\"...\"}],\"skipped\":[]}"
                                            try:
                                                res_restore = call_ai(f"処理する単語: ['{restore_word}']", sys_restore, is_json=True)
                                                restored_data = json.loads(res_restore)
                                                if restored_data.get("enriched"):
                                                    current_book["enriched_vocab"].insert(0, restored_data["enriched"][0])
                                                    current_book["skipped_vocab"].remove(restore_word)
                                                    save_data(my_data)
                                                    st.success("復活しました！"); st.rerun()
                                            except: st.error("失敗しました。")


# ------------------------------------------
        # タブX: アウトプット学習 (DUQ & ランダム)
        # ------------------------------------------
        with tab_output:
            import time
            import math
            
            st.markdown("### 📤 単語アウトプット学習")
            st.caption("AI生成済みの単語データを使って、記憶の定着度をテストします。")

            fixed_book_for_output = build_frequency_strong_book(base_lexicon)
            books = ([fixed_book_for_output] if fixed_book_for_output else []) + my_data.get("vocab_books", [])
            if not books:
                st.info("単語帳がありません。")
            else:
                book_titles = [b["title"] for b in books]
                selected_title_out = st.selectbox("📖 テストする単語帳を選択", ["-- 選択してください --"] + book_titles, key="out_vocab_sel")

                if selected_title_out != "-- 選択してください --":
                    book_idx = book_titles.index(selected_title_out)
                    current_book = books[book_idx]

                    enriched_dict = {item["word"]: item for item in current_book.get("enriched_vocab", [])}
                    valid_words = [w for w in current_book["main_vocab"] if w in enriched_dict]

                    if len(valid_words) < 4:
                        st.warning("クイズを行うには、最低4つの単語をAIで生成（解説を追加）してください。本棚タブから生成できます。")
                    else:
                        st.markdown("#### 🎮 出題形式とモード選択")
                        
                        # 💡 追加：学習フェーズに合わせて出題形式を選べるトグル
                        quiz_format = st.radio("出題形式を選んでください:", ["🔘 4択クイズ (ヒントあり・初期学習向け)", "🃏 フラッシュカード (自力で思い出す・総仕上げ向け)"], horizontal=True)

                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            mode_random = st.button("🎲 リラックス・ランダム\n(プレッシャーなしで気軽に)", use_container_width=True)
                        with col_m2:
                            mode_duq = st.button("🧠 DUQ 記憶定着モード\n(忘却曲線をハックして最適化・プレッシャーが嫌いな人は非推奨)", use_container_width=True, type="primary")

                        if mode_random or mode_duq:
                            st.session_state.vocab_quiz_mode = "DUQ" if mode_duq else "Random"
                            st.session_state.vocab_quiz_format = "4択" if "4択" in quiz_format else "フラッシュ"
                            st.session_state.vocab_quiz_idx = 0
                            st.session_state.vocab_quiz_status = {}

                            # 🎯 出題単語の選定アルゴリズム
                            if mode_random:
                                st.session_state.vocab_quiz_qs = random.sample(valid_words, min(20, len(valid_words)))
                            else:
                                if "vocab_stats" not in my_data: my_data["vocab_stats"] = {}
                                now = time.time() / 86400
                                scores = []
                                for w in valid_words:
                                    stats = my_data["vocab_stats"].get(w, {"stability": 0.5, "last_review": 0})
                                    S = stats["stability"]
                                    t = now - stats["last_review"] if stats["last_review"] > 0 else 1000
                                    R = math.exp(-t / S)
                                    U = 1.0 - R
                                    scores.append((w, U))

                                scores.sort(key=lambda x: x[1], reverse=True)
                                st.session_state.vocab_quiz_qs = [w for w, u in scores[:20]]

                            questions = []
                            for w in st.session_state.vocab_quiz_qs:
                                correct_meaning = enriched_dict[w].get("meanings", "意味不明")
                                pool = [enriched_dict[dw].get("meanings", "意味不明") for dw in valid_words if dw != w]
                                dummies = random.sample(pool, min(3, len(pool)))
                                options = dummies + [correct_meaning]
                                random.shuffle(options)
                                options.append("🤔 わからない")
                                questions.append({"word": w, "options": options, "answer": correct_meaning})

                            st.session_state.vocab_quiz_data = questions
                            st.rerun()

                        # --- クイズ実行画面 ---
                        if "vocab_quiz_data" in st.session_state:
                            st.markdown("---")
                            qs = st.session_state.vocab_quiz_data
                            idx = st.session_state.vocab_quiz_idx
                            mode_name = st.session_state.vocab_quiz_mode
                            q_format = st.session_state.vocab_quiz_format

                            if idx < len(qs):
                                q = qs[idx]
                                word = q["word"]
                                st.markdown(f"#### 📝 Question {idx + 1} / {len(qs)} <span style='font-size:0.5em; color:gray;'>({mode_name} / {q_format})</span>", unsafe_allow_html=True)

                                with st.container(border=True):
                                    st.markdown(f"<div style='text-align: center; font-size: 3em; font-weight: bold; color: #1f77b4; margin-bottom: 20px;'>{word}</div>", unsafe_allow_html=True)

                                    is_answered = str(idx) in st.session_state.vocab_quiz_status

                                    # 💡 出題形式による分岐
                                    if q_format == "4択":
                                        user_ans = st.radio("最も適切な意味を選んでください:", q["options"], key=f"vq_{idx}", disabled=is_answered)
                                        st.markdown("<br>", unsafe_allow_html=True)

                                        if not is_answered:
                                            if st.button("📝 解答する", key=f"v_ans_{idx}", type="primary", use_container_width=True):
                                                is_correct = (user_ans == q["answer"])
                                                st.session_state.vocab_quiz_status[str(idx)] = {"user_ans": user_ans, "is_correct": is_correct}
                                                st.rerun()
                                    else:
                                        # フラッシュカードモード
                                        if not is_answered:
                                            st.info("頭の中で「意味」と「使われ方」を思い出してみてください。")
                                            if st.button("👀 答えと解説を見る", key=f"v_ans_{idx}", type="primary", use_container_width=True):
                                                # フラッシュカードは正誤判定なし（自己評価に委ねる）
                                                st.session_state.vocab_quiz_status[str(idx)] = {"user_ans": "確認済み", "is_correct": True}
                                                st.rerun()

                                    if is_answered:
                                        status = st.session_state.vocab_quiz_status[str(idx)]
                                        st.markdown("---")

                                        if q_format == "4択":
                                            if status["user_ans"] == "🤔 わからない":
                                                st.warning(f"💡 正解は **{q['answer']}** でした。")
                                            elif status["is_correct"]:
                                                st.success(f"⭕ **正解！** : {q['answer']}")
                                            else:
                                                st.error(
                                                    f"❌ **不正解** : あなたの解答「{status['user_ans']}」 ➔ 正解「{q['answer']}」"
                                                )
                                        else:
                                            st.success(f"💡 正解は **{q['answer']}** です。")

                                        item = enriched_dict[word]
                                        st.info(
                                            f"**🔄 変化形:** {item.get('forms', '-')}\n\n"
                                            f"**🎯 使い方:**\n" + "\n".join([f"- {c}" for c in item.get('chunks', [])])
                                        )

                                                                                    
                                                                                  
                                             

                                        
                                        
                                        if mode_name == "DUQ":
                                            st.markdown("#### 🧠 今の感覚に一番近いものは？（記憶の定着度を更新します）")
                                            st.caption("※消去法で当てたなど、自信がない場合は潔く「🔴 わからない・まぐれ」か「🟠 曖昧」を選んでください。")
                                            col_eval1, col_eval2, col_eval3, col_eval4 = st.columns(4)

                                            def update_stats(w, multiplier):
                                                if "vocab_stats" not in my_data: my_data["vocab_stats"] = {}
                                                if w not in my_data["vocab_stats"]: my_data["vocab_stats"][w] = {"stability": 0.5, "last_review": 0}
                                                
                                                S = my_data["vocab_stats"][w]["stability"]
                                                new_S = 0.5 if multiplier == 0 else S * multiplier

                                                my_data["vocab_stats"][w]["stability"] = min(new_S, 365.0) 
                                                my_data["vocab_stats"][w]["last_review"] = time.time() / 86400
                                                save_data(my_data)
                                                st.session_state.vocab_quiz_idx += 1
                                                st.rerun()

                                            if col_eval1.button("🔴 わからない・まぐれ\n(やり直し)", use_container_width=True): update_stats(word, 0)
                                            if col_eval2.button("🟠 曖昧\n(少し迷った)", use_container_width=True): update_stats(word, 1.2)
                                            if col_eval3.button("🟢 正解\n(思い出した)", use_container_width=True): update_stats(word, 2.5)
                                            if col_eval4.button("🔵 余裕\n(使い方まで即答)", use_container_width=True): update_stats(word, 3.5)
                                        else:
                                            # ランダムモードの場合は単純な「次へ」ボタンのみ表示
                                            st.markdown("<br>", unsafe_allow_html=True)
                                            if st.button("次の単語へ ▶", use_container_width=True, type="primary"):
                                                st.session_state.vocab_quiz_idx += 1
                                                st.rerun()

                            else:
                                st.success(f"### 🎉 1セット完了！お疲れ様でした。")
                                if mode_name == "DUQ":
                                    st.caption("裏側であなたの忘却曲線を更新し、次回の出題優先度を調整しました。")
                                if st.button("🔄 終了する", use_container_width=True):
                                    for k in ["vocab_quiz_data", "vocab_quiz_idx", "vocab_quiz_status", "vocab_quiz_mode", "vocab_quiz_format", "vocab_quiz_qs"]:
                                        if k in st.session_state: del st.session_state[k]
                                    st.rerun()
        # ------------------------------------------
        # タブ2: 新しい単語帳を作る
        # ------------------------------------------
        with tab_create:
            selected_labels = render_exam_selector(db_options, "create_vocab")
            use_ai_filter = st.checkbox("🤖 【テスト機能】AIで明らかな「人名」だけを除外する", value=False)
            exclude_frequency_strong_create = st.checkbox("💪 頻度つよつよ単語3000を単語帳対象から外す", value=True)
            
            if st.button("✨ 選択した過去問から単語帳を生成") and selected_labels:
                with st.spinner("単語を集計中..."):
                    combined_counter = Counter()
                    word_origins = defaultdict(list)
                    for label in selected_labels:
                        path = db_options[label]
                        freqs = exam_db[path["c"]][path["u"]][path["f"]][path["y"]][path["m"]]["frequencies"]
                        if exclude_frequency_strong_create:
                            freqs = {w: c for w, c in freqs.items() if normalize_vocab_word(w) not in frequency_strong_words}
                        combined_counter.update(freqs)
                        short_label = f"{str(path['y'])[-2:]}年"
                        for w, count in freqs.items(): word_origins[w].append(f"{short_label}({count}回)")
                    
                    sorted_words = [word for word, count in combined_counter.most_common()]
                    st.session_state.combined_counter = combined_counter
                    st.session_state.word_origins = dict(word_origins) 
                    
                    if use_ai_filter and api_key:
                        sys_filter = "英単語リストを分類しJSON出力: {\"main_vocab\": [\"technology\"], \"excluded_vocab\": [\"david\"]}。人名・企業名などはexcludedへ。一般名詞・地名はmainへ。"
                        try:
                            response_json = call_ai(f"リスト:\n{sorted_words}", sys_filter, is_json=True)
                            filtered_data = json.loads(response_json)
                            st.session_state.main_vocab = filtered_data.get("main_vocab", [])
                            st.session_state.excluded_vocab = filtered_data.get("excluded_vocab", [])
                        except:
                            st.session_state.main_vocab = sorted_words; st.session_state.excluded_vocab = []
                    else:
                        st.session_state.main_vocab = sorted_words; st.session_state.excluded_vocab = []

            if "main_vocab" in st.session_state:
                st.markdown("---")
                st.markdown("### 👀 単語帳のプレビュー")
                with st.form("save_book_form"):
                    new_title = st.text_input("💾 単語帳に名前をつけて保存 (例: 学習院2025 マスター)")
                    if st.form_submit_button("本棚に保存する") and new_title:
                        if "vocab_books" not in my_data: my_data["vocab_books"] = []
                        new_book = {
                            "title": new_title, 
                            "main_vocab": st.session_state.main_vocab, 
                            "excluded_vocab": st.session_state.excluded_vocab,
                            "counts": dict(st.session_state.combined_counter), 
                            "origins": dict(st.session_state.word_origins),
                            "enriched_vocab": [],  # ← 最初から空箱を持たせる
                            "skipped_vocab": []    # ← 最初から空箱を持たせる
                        }
                        my_data["vocab_books"].append(new_book)
                        save_data(my_data)
                        del st.session_state.main_vocab
                        st.success(f"🎉「{new_title}」を本棚に保存しました！"); st.rerun()

                main_display = [{"単語": w, "総出現回数": st.session_state.combined_counter[w], "出題内訳": ", ".join(st.session_state.word_origins.get(w, []))} for w in st.session_state.main_vocab]
                st.dataframe(main_display, use_container_width=True)

        # ------------------------------------------
        # タブ3: カバー率・難化シミュレーター（復元済）
        # ------------------------------------------
        with tab_sim:
            st.markdown("### 📊 過去問カバー率（定量分析）シミュレーター")
            st.info("あなたが作った「単語帳（武器）」が、特定の「過去問（敵）」にどれくらい通用するかを定量的にシミュレーションします。")
            
            fixed_book_for_sim = build_frequency_strong_book(base_lexicon)
            books = ([fixed_book_for_sim] if fixed_book_for_sim else []) + my_data.get("vocab_books", [])
            if not books:
                st.warning("まずは「✨ 新しい単語帳を作る」タブから、ベースとなる単語帳（例：2020〜2025年基礎マスター）を作成・保存してください。")
            else:
                col_base, col_target = st.columns(2)
                
                with col_base:
                    st.markdown("#### ⚔️ 武器（ベースライン）")
                    book_titles = [b["title"] for b in books]
                    selected_baseline = st.selectbox("学習済みの単語帳を選択", ["-- 選択 --"] + book_titles)
                
                with col_target:
                    st.markdown("#### 🎯 敵（ターゲット）")
                    target_options = {}
                    for cat, unis in exam_db.items():
                        for uni, facs in unis.items():
                            for fac, years in facs.items():
                                for year, methods in years.items():
                                    for method, data in methods.items():
                                        if data.get("frequencies"): # 🛡️ 空データ除外
                                            label = f"[{cat}] {uni} {fac} ({year}年 {method})"
                                            target_options[label] = {"c": cat, "u": uni, "f": fac, "y": year, "m": method}
                    
                    selected_target = st.selectbox("テストする未知の過去問を選択", ["-- 選択 --"] + list(target_options.keys()))
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                if st.button("🚀 カバー率を検証する（バックテスト）", use_container_width=True, type="primary"):
                    if selected_baseline == "-- 選択 --" or selected_target == "-- 選択 --":
                        st.error("武器と敵を両方選択してください。")
                    else:
                        with st.spinner("単語の照合中..."):
                            base_book = next(b for b in books if b["title"] == selected_baseline)
                            raw_base_words = base_book["main_vocab"]
                            base_words = set(lemmatizer.lemmatize(word.lower(), pos='v') for word in raw_base_words)
                            base_words = set(lemmatizer.lemmatize(word, pos='n') for word in base_words)
                            base_words.update(frequency_strong_words)

                            path = target_options[selected_target]
                            target_data = exam_db[path["c"]][path["u"]][path["f"]][path["y"]][path["m"]]
                            raw_target_freqs = target_data["frequencies"]
                            target_freqs = process_frequencies(raw_target_freqs)
                            target_freqs = {
                                w: c
                                for w, c in target_freqs.items()
                                if w not in frequency_excluded_words
                            }

                            total_tokens = sum(target_freqs.values())
                            covered_tokens = 0
                            missed_words = {}

                            for w, count in target_freqs.items():
                                if w in base_words:
                                    covered_tokens += count
                                else:
                                    missed_words[w] = count
               
                            coverage_rate = (covered_tokens / total_tokens) * 100 if total_tokens > 0 else 0
                            
                            st.session_state.sim_result = {
                                "coverage_rate": coverage_rate,
                                "total_tokens": total_tokens,
                                "covered_tokens": covered_tokens,
                                "missed_words": missed_words,
                                "target_name": selected_target
                            }
                            if "analysis_result" in st.session_state:
                                del st.session_state.analysis_result
                            st.rerun()
                            
                if "sim_result" in st.session_state:
                    res = st.session_state.sim_result
                    missed_words = res["missed_words"]
                    
                    st.markdown("---")
                    st.markdown(f"## 📊 カバー率: **{res['coverage_rate']:.1f}%** <span style='font-size:0.5em; color:gray;'>({res['target_name']})</span>", unsafe_allow_html=True)
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("ターゲットの総単語数", f"{res['total_tokens']}語")
                    col_m2.metric("カバーできた単語数", f"{res['covered_tokens']}語")
                    col_m3.metric("未知の単語（取りこぼし）", f"{len(missed_words)}種類")
                    
                    if not missed_words:
                        st.success("完璧です！未知の単語は1つもありません。")
                    else:
                        with st.expander("⚠️ 単語帳に載っていなかった「未知の単語リスト」を見る", expanded=False):
                            missed_display = [{"未知の単語": w, "出現回数": c} for w, c in sorted(missed_words.items(), key=lambda item: item[1], reverse=True)]
                            st.dataframe(missed_display, use_container_width=True)
                        
                        st.markdown("---")
                        st.markdown("### 🧠 致命傷チェッカー（単語帳・拡張判定）")
                        st.markdown("この未知の単語が実際の設問で**「即死レベルの致命傷」**になるか、推測可能な**「ノイズ」**かを分析し、**「今の単語帳に新しい単語を追加して覚えるべきか」**を結論付けます。")
                        
                        input_method = st.radio("入力方法を選択してください", ["📄 PDFをアップロード", "📝 テキストを貼り付け"], horizontal=True)
                        
                        exam_text = ""
                        local_pdf = None
                        if input_method == "📝 テキストを貼り付け":
                            exam_text = st.text_area("📄 過去問の全文（長文と設問・選択肢）をここに貼り付けてください", height=200, placeholder="※PDFのテキストをコピーして貼り付けてください")
                        else:
                            local_pdf = st.file_uploader("📄 分析する過去問のPDFをここにアップロード", type=["pdf"], key="fatal_pdf")

                        can_analyze = (input_method == "📝 テキストを貼り付け" and exam_text) or (input_method == "📄 PDFをアップロード" and local_pdf)
                        
                        if st.button("🔍 未知の単語の『致命度』を分析する", type="primary") and api_key and can_analyze:
                            with st.spinner("AIがPDFを読み込み、設問の構造と未知の単語の絡みを分析中...（約10〜30秒）"):
                                missed_list_str = ", ".join(missed_words.keys())
                                sys_prompt = """
                                あなたはプロの英語予備校講師であり、生徒の学習戦略コンサルタントです。
                                生徒は提示された「未知の単語リスト」の意味を知りません。
                                添付された「過去問のデータ（長文＋設問）」を精読し、これらの単語が「知らなくても推測・無視できるノイズ」か、「知らないと確実に失点する致命傷」かを判定してください。

                                【出力フォーマット（マークダウン）】
                                以下の3つのセクションで出力してください。

                                ### 🚨 致命傷アラート（追加学習が必須な単語）
                                （※正解選択肢の言い換えになっている、長文のメインテーマの核であるなど、失点に直結する単語とその理由を具体的に。なければ「なし」と記載）
                                - **[単語]**: [理由]

                                ### ⚠️ 推測可能なノイズ（無視してよい単語）
                                （※文脈から推測できる、あるいは設問に全く絡まない単語の例をいくつかピックアップして理由を記載）
                                - **[単語]**: [理由]

                                ### 💡 最終結論（学習戦略）
                                （※ズバリ、この志望校で合格点を取るために、「今の単語帳に新しい単語を追加して覚えるべきか（語彙力不足）」、それとも「現状の単語知識と推測力で十分戦えるか（読解戦略でカバー可能）」を結論付けてください。）
                                """
                                
                                prompt = f"【生徒が知らない未知の単語リスト】\n{missed_list_str}\n\n【過去問のデータ】\n"
                                
                                if input_method == "📝 テキストを貼り付け":
                                    prompt += exam_text
                                    st.session_state.analysis_result = call_ai(prompt, sys_prompt, use_pdf=False)
                                else:
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                        tmp.write(local_pdf.getvalue())
                                        tmp_path = tmp.name
                                    
                                    g_file = genai.upload_file(tmp_path)
                                    model = genai.GenerativeModel(model_name="gemini-2.5-pro", system_instruction=sys_prompt)
                                    res = model.generate_content([g_file, prompt])
                                    os.remove(tmp_path)
                                    st.session_state.analysis_result = res.text
                                    
                                st.rerun()
                                
                        if "analysis_result" in st.session_state:
                            st.info(st.session_state.analysis_result)

# ==========================================
# モードE: 志望校別熟語帳（★シミュレーター搭載版）
# ==========================================
elif mode == "🔗 志望校別熟語帳":
    st.markdown("### 🔗 志望校別熟語帳")
    st.caption("長文の文脈で覚える！志望校別熟語ドリルとリスト管理")
    
    if "idiom_books" not in my_data:
        my_data["idiom_books"] = []
        
    tab_shelf, tab_input, tab_output, tab_create, tab_sim = st.tabs(["📚 熟語帳", "📥 インプット用長文", "📤 アウトプット", "✨ 新しい熟語帳を作る", "📊 カバー率シミュレーター"])
    
    # ------------------------------------------
    # 共通データ生成
    # ------------------------------------------
    db_options = {}
    if exam_db:
        for cat, unis in exam_db.items():
            for uni, facs in unis.items():
                for fac, years in facs.items():
                    for year, methods in years.items():
                        for method, data in methods.items():
                            if data.get("idioms"): # 🛡️ 熟語データが空のものを除外！
                                label = f"[{cat}] {uni} {fac} ({year}年 {method})"
                                db_options[label] = {"c": cat, "u": uni, "f": fac, "y": year, "m": method}
                            
    book_titles = [b["title"] for b in my_data.get("idiom_books", [])]
    
    # ------------------------------------------
    # タブ4: 新しい熟語帳を作る
    # ------------------------------------------
    with tab_create:
        st.markdown("#### 📚 過去問データベースから熟語を抽出して熟語帳を作成")
        if not exam_db:
            st.warning("過去問データベースが空です。")
        else:
            selected_labels = render_exam_selector(db_options, "create_idiom")
            new_title = st.text_input("💾 この熟語帳に名前をつける (例: 学習院マスター熟語)")
            
            if st.button("✨ 熟語帳を生成して本棚に保存", type="primary") and selected_labels and new_title:
                combined_idioms = {}
                for label in selected_labels:
                    path = db_options[label]
                    idioms_data = exam_db[path["c"]][path["u"]][path["f"]][path["y"]][path["m"]].get("idioms", {})
                    for base_form, data in idioms_data.items():
                        if base_form in combined_idioms:
                            combined_idioms[base_form]["count"] += data["count"]
                        else:
                            combined_idioms[base_form] = {
                                "count": data["count"], "practice_count": 0, "correct_count": 0
                            }
                
                sorted_idioms = sorted([
                    {"base_form": k, "count": v["count"], "practice_count": v["practice_count"], "correct_count": v["correct_count"]} 
                    for k, v in combined_idioms.items()
                ], key=lambda x: x["count"], reverse=True)
                
                new_book = {"title": new_title, "idioms": sorted_idioms}
                my_data["idiom_books"].append(new_book)
                save_data(my_data)
                st.success(f"「{new_title}」を保存しました！")
                st.rerun()
    
    # ------------------------------------------
    # タブ1: 熟語帳（本棚・リスト・詳細ルーム）
    # ------------------------------------------
    with tab_shelf:
        if not my_data["idiom_books"]:
            st.info("まだ熟語帳がありません。「✨ 新しい熟語帳を作る」タブから作成してください。")
        else:
            selected_title_shelf = st.selectbox("📖 管理・学習する熟語帳を選択", ["-- 選択してください --"] + book_titles, key="shelf_sel")
            if selected_title_shelf != "-- 選択してください --":
                book_idx = book_titles.index(selected_title_shelf)
                current_idiom_book = my_data["idiom_books"][book_idx]
                
                st.markdown(f"### 📘 {current_idiom_book['title']}")
                st.write(f"収録熟語数: **{len(current_idiom_book['idioms'])} 個**")
                
                with st.expander("⚙️ 熟語帳の管理・編集（名前変更・追加・マージ）", expanded=False):
                    st.markdown("#### 🏷️ 名前の変更")
                    col_rn1, col_rn2 = st.columns([3, 1])
                    new_title_input = col_rn1.text_input("新しい名前", value=current_idiom_book["title"], label_visibility="collapsed", key=f"rn_idiom_{book_idx}")
                    if col_rn2.button("名前を更新", use_container_width=True, key=f"btn_rn_idiom_{book_idx}"):
                        if new_title_input and new_title_input != current_idiom_book["title"]:
                            current_idiom_book["title"] = new_title_input
                            my_data["idiom_books"][book_idx] = current_idiom_book
                            save_data(my_data)
                            st.success("名前を変更しました！")
                            st.rerun()
                            
                    st.markdown("---")
                    st.markdown("#### 📈 過去問データを追加して熟語帳を強化 (マージ)")
                    add_labels = render_exam_selector(db_options, f"merge_idiom_{book_idx}")
                    if st.button("✨ 選択したデータを追加 (マージ)", type="primary", key=f"btn_merge_idiom_{book_idx}") and add_labels:
                        with st.spinner("熟語データを結合し、頻度を再計算しています..."):
                            current_idiom_dict = {i["base_form"]: i for i in current_idiom_book["idioms"]}
                            
                            for label in add_labels:
                                path = db_options[label]
                                idioms_data = exam_db[path["c"]][path["u"]][path["f"]][path["y"]][path["m"]].get("idioms", {})
                                
                                for base_form, data in idioms_data.items():
                                    if base_form in current_idiom_dict:
                                        current_idiom_dict[base_form]["count"] += data["count"]
                                    else:
                                        current_idiom_dict[base_form] = {
                                            "base_form": base_form, "count": data["count"], "practice_count": 0, "correct_count": 0
                                        }
                            
                            # 新しい頻度でソート（AIデータは消さずに保持）
                            current_idiom_book["idioms"] = sorted(list(current_idiom_dict.values()), key=lambda x: x["count"], reverse=True)
                                
                            my_data["idiom_books"][book_idx] = current_idiom_book
                            save_data(my_data)
                            st.success("データをマージしました！頻度順に再ソートされています。")
                            st.rerun()

                    st.markdown("---")
                    st.markdown("#### 📋 複製と削除")
                    col_dup, col_del = st.columns(2)
                    if col_dup.button("📋 この熟語帳を複製する", use_container_width=True):
                        new_book = current_idiom_book.copy()
                        new_book["title"] = current_idiom_book["title"] + " (コピー)"
                        my_data["idiom_books"].append(new_book)
                        save_data(my_data); st.rerun()
                        
                    if col_del.button("🗑️ この熟語帳を削除する", use_container_width=True):
                        my_data["idiom_books"].pop(book_idx)
                        save_data(my_data); st.rerun()
                    
                    st.markdown("---")
                    st.markdown("#### ✏️ 熟語の個別追加・削除")
                    col_edit1, col_edit2 = st.columns(2)
                    with col_edit1:
                        with st.form(f"add_idiom_form_{book_idx}"):
                            new_idiom = st.text_input("➕ 新しい熟語を追加")
                            if st.form_submit_button("追加") and new_idiom:
                                new_idiom = new_idiom.lower().strip()
                                existing_bases = [i["base_form"] for i in current_idiom_book["idioms"]]
                                if new_idiom not in existing_bases:
                                    current_idiom_book["idioms"].insert(0, {"base_form": new_idiom, "count": 1, "practice_count": 0, "correct_count": 0})
                                    save_data(my_data); st.rerun()
                    with col_edit2:
                        with st.form(f"del_idiom_form_{book_idx}"):
                            idiom_list = [i["base_form"] for i in current_idiom_book["idioms"]]
                            del_idiom = st.selectbox("🗑️ 削除する熟語を選択", ["-- 選択 --"] + idiom_list)
                            if st.form_submit_button("削除") and del_idiom != "-- 選択 --":
                                current_idiom_book["idioms"] = [i for i in current_idiom_book["idioms"] if i["base_form"] != del_idiom]
                                if "enriched_idioms" in current_idiom_book:
                                    current_idiom_book["enriched_idioms"] = [i for i in current_idiom_book["enriched_idioms"] if i.get("base_form") != del_idiom]
                                save_data(my_data); st.rerun()
                                
                    st.markdown("---")
                    st.markdown("#### 🗑️ データの完全初期化")
                    if st.button("🗑️ AI生成データ(意味・例文等)をすべてリセットする", use_container_width=True):
                        if "enriched_idioms" in current_idiom_book: del current_idiom_book["enriched_idioms"]
                        save_data(my_data); st.rerun()
                
                st.markdown("---")
                
                # データの初期化チェック
                if "enriched_idioms" not in current_idiom_book: current_idiom_book["enriched_idioms"] = []
                
                # 未生成（差分）の熟語をリストアップ
                enriched_bases = [e["base_form"] for e in current_idiom_book["enriched_idioms"]]
                ordered_bases = [i["base_form"] for i in current_idiom_book["idioms"]]
                missing_idioms = [w for w in ordered_bases if w not in enriched_bases]
                
                if missing_idioms:
                    st.info(f"💡 未生成の熟語が **{len(missing_idioms)}** 個あります。（過去問マージ等で追加された熟語です）")
                    
                    # --- 生成数指定UI ---
                    col_gen_id1, col_gen_id2 = st.columns([1, 2])
                    gen_idiom_count = col_gen_id1.number_input("生成する熟語数", min_value=1, max_value=len(missing_idioms), value=min(30, len(missing_idioms)), step=10, key=f"gen_i_{book_idx}")
                    
                    col_gen_id2.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True) # ボタンの高さを合わせる
                    if col_gen_id2.button(f"✨ 未生成の熟語を上位 {gen_idiom_count} 個生成して追加", use_container_width=True, type="primary"):
                        with st.spinner(f"AIが複数の意味を考慮して例文を作成中...（上位 {gen_idiom_count} 個）"):
                            target_idioms = missing_idioms[:gen_idiom_count]
                            sys_enrich_idiom = """
                            あなたは受験英語に精通したプロ講師です。提供された英熟語リストから、以下のJSON形式で意味と例文を作成してください。
                            1. 複数の意味がある熟語は「①... ②...」のように1行にまとめる。
                            2. 例文は、1つの意味につき【必ず2つずつ】作成。行頭に番号を付与。
                            3. 例文中の熟語は **太字(Markdown)** にする。
                            {
                              "enriched": [
                                {
                                  "base_form": "take advantage of",
                                  "meanings": "① ～を利用する ② （人）につけこむ",
                                  "chunks": [
                                    "① We should **take advantage of** this. (これを利用すべきだ。)",
                                    "① He **took advantage of** his position. (彼は地位を利用した。)",
                                    "② Don't let them **take advantage of** you. (つけこまれるな。)",
                                    "② She felt **taken advantage of**. (彼女はつけこまれたと感じた。)"
                                  ]
                                }
                              ]
                            }
                            """
                            try:
                                res_json = call_ai(f"リスト:\n{target_idioms}", sys_enrich_idiom, is_json=True)
                                parsed_data = json.loads(res_json)
                                current_idiom_book["enriched_idioms"].extend(parsed_data.get("enriched", []))
                                my_data["idiom_books"][book_idx] = current_idiom_book
                                save_data(my_data)
                                st.success("🎉 生成して追加しました！")
                                st.rerun()
                            except Exception as e: 
                                st.error(f"エラー: {e}")
                
                # 生成済みデータがない場合
                if not current_idiom_book["enriched_idioms"]:
                    st.markdown("#### 📊 学習ステータス")
                    raw_display = [{"熟語・構文": i["base_form"], "頻度(重み)": i["count"], "正解数": i.get("correct_count", 0)} for i in current_idiom_book["idioms"]]
                    st.dataframe(raw_display, use_container_width=True)
                
                # 生成済みデータがある場合
                else:
                    if "selected_idiom_idx" not in st.session_state: st.session_state.selected_idiom_idx = None
                        
                    if st.session_state.selected_idiom_idx is None:
                        st.markdown("### 📚 熟語リスト")
                        
                        # ★ 最新の idioms の順序に合わせて enriched_idioms をソートして表示
                        enriched_dict = {item["base_form"]: item for item in current_idiom_book["enriched_idioms"]}
                        enriched_list = [enriched_dict[w] for w in ordered_bases if w in enriched_dict]
                        
                        WORDS_PER_PAGE = 20
                        total_pages = max(1, (len(enriched_list) + WORDS_PER_PAGE - 1) // WORDS_PER_PAGE)
                        
                        page_key = f"idiom_page_{book_idx}"
                        if page_key not in st.session_state: st.session_state[page_key] = 1
                        current_page = st.session_state[page_key]
                        
                        col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
                        with col_p1:
                            if st.button("◀ 前の20件", use_container_width=True, disabled=(current_page == 1)): st.session_state[page_key] -= 1; st.rerun()
                        with col_p2:
                            jump_page = st.number_input("ページジャンプ", min_value=1, max_value=total_pages, value=current_page, label_visibility="collapsed")
                            if jump_page != current_page: st.session_state[page_key] = jump_page; st.rerun()
                            st.markdown(f"<div style='text-align: center; font-size: 0.8em; color: gray;'>全{len(enriched_list)}語 / {total_pages}ページ</div>", unsafe_allow_html=True)
                        with col_p3:
                            if st.button("次の20件 ▶", use_container_width=True, disabled=(current_page == total_pages)): st.session_state[page_key] += 1; st.rerun()
                                
                        st.divider()
                        start_idx = (current_page - 1) * WORDS_PER_PAGE
                        end_idx = start_idx + WORDS_PER_PAGE
                        current_page_idioms = enriched_list[start_idx:end_idx]
                        
                        cols = st.columns(2)
                        for i, item in enumerate(current_page_idioms):
                            word = item.get("base_form", "")
                            # 実際のデータのインデックスを取得
                            actual_idx = next((index for (index, d) in enumerate(current_idiom_book["enriched_idioms"]) if d["base_form"] == word), None)
                            
                            count = "-"
                            for orig in current_idiom_book["idioms"]:
                                if orig["base_form"] == word:
                                    count = orig["count"]
                                    break
                            
                            col = cols[i % 2]
                            with col:
                                with st.container(border=True):
                                    st.markdown(f"<div style='margin-bottom: 15px;'><span style='color:#1f77b4; font-size:2.4em; font-weight:900; line-height:1.2;'>{word}</span><span style='font-size:1.1em; color:gray; margin-left:15px;'>({count}回)</span></div>", unsafe_allow_html=True)
                                    if item.get("meanings"): st.markdown(f"<div style='font-size:1.15em; font-weight:bold; color:#4caf50; margin-bottom: 10px;'>{item.get('meanings')}</div>", unsafe_allow_html=True)
                                    st.divider()
                                    for chunk in item.get("chunks", []):
                                        colored_chunk = re.sub(r'\*\*(.*?)\*\*', r"<span style='color:#1f77b4; font-weight:bold; font-size:1.1em;'>\1</span>", chunk)
                                        st.markdown(f"<div style='margin-left: 0.5em; margin-bottom: 8px; font-size:0.95em;'>{colored_chunk}</div>", unsafe_allow_html=True)
                                    
                                    st.markdown("<br>", unsafe_allow_html=True)
                                    if st.button(f"👉 詳細・文脈・メモを開く", key=f"idiom_sel_{book_idx}_{actual_idx}", use_container_width=True):
                                        st.session_state.selected_idiom_idx = actual_idx; st.rerun()
                        
                        if len(current_page_idioms) > 4:
                            st.divider()
                            col_p1_b, col_p2_b, col_p3_b = st.columns([1, 2, 1])
                            with col_p1_b:
                                if st.button("◀ 前へ", key="idiom_prev_b", use_container_width=True, disabled=(current_page == 1)): st.session_state[page_key] -= 1; st.rerun()
                            with col_p3_b:
                                if st.button("次へ ▶", key="idiom_next_b", use_container_width=True, disabled=(current_page == total_pages)): st.session_state[page_key] += 1; st.rerun()
                                        
                    else:
                        i = st.session_state.selected_idiom_idx
                        item = current_idiom_book["enriched_idioms"][i]
                        word = item.get("base_form", "")
                        
                        if st.button("🔙 熟語リストに戻る", type="primary"): st.session_state.selected_idiom_idx = None; st.rerun()
                            
                        st.markdown(f"## 🔍 「{word}」の専用ルーム")
                        with st.container(border=True):
                            st.markdown("### 📚 意味とフレーズ")
                            if item.get("meanings"): st.markdown(f"**<span style='color:#4caf50; font-size:1.2em;'>{item.get('meanings')}</span>**", unsafe_allow_html=True)
                            for chunk in item.get("chunks", []):
                                colored_ex = re.sub(r'\*\*(.*?)\*\*', r"<span style='color:#1f77b4; font-weight:bold;'>\1</span>", chunk)
                                st.markdown(f"> {colored_ex}", unsafe_allow_html=True)
                        
                        st.markdown("#### 📝 マイ・メモ")
                        col_m1, col_m2 = st.columns([4, 1])
                        new_memo = col_m1.text_input(f"メモ入力", value=item.get("user_memo", ""), key=f"idiom_memo_{i}", label_visibility="collapsed")
                        if col_m2.button("💾 保存", key=f"save_idiom_memo_{i}", use_container_width=True):
                            current_idiom_book["enriched_idioms"][i]["user_memo"] = new_memo
                            save_data(my_data); st.success("保存しました！")
                        
                        st.divider()
                        st.markdown("#### 🎯 実践！穴埋め4択クイズ (Flash-Lite搭載)")
                        if st.button("✨ この熟語でクイズを生成する", key=f"idiom_quiz_btn_{i}"):
                            with st.spinner("AI講師が問題を自動生成中..."):
                                try:
                                    res_quiz = call_ai(
                                        f"対象の熟語: {word}",
                                        "英語講師として指定熟語の4択問題を作成せよ。JSON出力: {\"question\": \"...\", \"options\": [\"...\"], \"answer\": \"...\", \"translation\": \"...\"}",
                                        is_json=True,
                                        model_name="gemini-2.5-flash-lite",
                                        max_output_tokens=500,
                                        timeout_seconds=30,
                                    )
                                    st.session_state[f"active_idiom_quiz_{i}"] = json.loads(res_quiz)
                                    st.session_state[f"idiom_quiz_answered_{i}"] = False
                                except Exception as e: st.error("生成に失敗しました。")
                        
                        if f"active_idiom_quiz_{i}" in st.session_state:
                            quiz_data = st.session_state[f"active_idiom_quiz_{i}"]
                            with st.container(border=True):
                                st.markdown(f"**Q.** {quiz_data['question']}")
                                user_choice = st.radio("選択してください:", quiz_data["options"], key=f"idiom_choice_{i}", disabled=st.session_state.get(f"idiom_quiz_answered_{i}", False))
                                if not st.session_state.get(f"idiom_quiz_answered_{i}", False):
                                    if st.button("📝 解答する", key=f"idiom_ans_btn_{i}"): st.session_state[f"idiom_quiz_answered_{i}"] = True; st.rerun()
                                if st.session_state.get(f"idiom_quiz_answered_{i}", False):
                                    st.markdown("---")
                                    if user_choice == quiz_data["answer"]: st.success(f"🎉 正解！ ({quiz_data['answer']})")
                                    else: st.error(f"❌ 惜しい！ 正解は **{quiz_data['answer']}** です。")
                                    st.markdown(f"**💡 和訳:** {quiz_data['translation']}")
                                    if st.button("🔄 もう一度", key=f"idiom_retry_btn_{i}"):
                                        del st.session_state[f"active_idiom_quiz_{i}"]
                                        del st.session_state[f"idiom_quiz_answered_{i}"]; st.rerun()

                        st.markdown("#### 📖 例文アシスト")
                        if saved_ex := item.get("saved_examples", []):
                            st.markdown("**【保存済みの例文】**")
                            for ex in saved_ex: st.markdown(f"- {ex}")
                        
                        if st.button("➕ AIに新しい例文を3つ作ってもらう", key=f"idiom_gen_ex_{i}"):
                            with st.spinner(f"生成中..."):
                                try:
                                    res_ex = call_ai(f"熟語: {word}", "実践的な例文と和訳を3つ、JSON配列形式で出力。例: [\"I look forward to seeing you. (楽しみにしています。)\"]", is_json=True)
                                    st.session_state[f"temp_idiom_ex_{word}"] = json.loads(res_ex); st.rerun()
                                except: st.error("生成に失敗しました。")
                        
                        if f"temp_idiom_ex_{word}" in st.session_state:
                            st.markdown("**💡 保存したいものにチェック：**")
                            with st.form(f"save_idiom_ex_form_{i}"):
                                selected_ex = [ex for idx, ex in enumerate(st.session_state[f"temp_idiom_ex_{word}"]) if st.checkbox(ex, key=f"idiom_chk_{word}_{idx}")]
                                if st.form_submit_button("✅ 選択を保存"):
                                    current_idiom_book["enriched_idioms"][i].setdefault("saved_examples", []).extend(selected_ex)
                                    save_data(my_data)
                                    del st.session_state[f"temp_idiom_ex_{word}"]
                                    st.success("追加保存しました！"); st.rerun()

                        st.divider()
                        st.markdown(f"#### 🤖 AI講師に質問する")
                        chat_key = f"idiom_chat_{word}"
                        if chat_key not in st.session_state: st.session_state[chat_key] = []
                        with st.container(height=300):
                            for msg in st.session_state[chat_key]:
                                with st.chat_message(msg["role"]): st.markdown(msg["content"])
                        if st.session_state[chat_key] and st.session_state[chat_key][-1]["role"] == "user":
                            with st.spinner("AI講師が考え中..."):
                                history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state[chat_key][-5:]])
                                ans = call_ai(f"会話履歴:\n{history_str}", f"英熟語「{word}」の専属講師として簡潔に答えてください。", use_pdf=False)
                                st.session_state[chat_key].append({"role": "assistant", "content": ans}); st.rerun()
                        if user_q := st.chat_input("質問する...", key=f"idiom_chat_in_{i}"):
                            st.session_state[chat_key].append({"role": "user", "content": user_q}); st.rerun()

    # ------------------------------------------
    # タブ2: インプット用長文（2カラムUI ＆ 漏れ防止プロンプト版）
    # ------------------------------------------
    with tab_input:
        if not my_data["idiom_books"]:
            st.info("熟語帳がありません。")
        else:
            st.markdown("### 📥 インプット用長文")
            st.caption("AIが選んだ熟語を自然な文脈に詰め込みます。テストではないのでリラックスして読んでください。")
            selected_title_input = st.selectbox("📖 学習する熟語帳を選択", ["-- 選択してください --"] + book_titles, key="input_sel")
            if selected_title_input != "-- 選択してください --":
                book_idx = book_titles.index(selected_title_input)
                current_idiom_book = my_data["idiom_books"][book_idx]
                
                if st.button("📖 新しいインプット長文を生成する（約10〜20秒）", type="primary"):
                    with st.spinner("熟語を散りばめた長文と詳細な解説を作成中..."):
                        sorted_idioms = sorted(current_idiom_book["idioms"], key=lambda x: x.get("practice_count", 0))
                        target_idioms = [item["base_form"] for item in sorted_idioms[:15]]
                        
                        # ⚠️ プロンプトを強化：解説の漏れを防ぐ
                        sys_input = """
                        あなたはプロ英語講師です。指定された英熟語が【全て】含まれる150〜200語程度の短い長文を作成してください。
                        【絶対ルール】
                        1. 指定熟語以外の単語は、極めて簡単な中学レベル（CEFR A1〜A2）のみを使用すること。「fortress」等の難単語は禁止。
                        2. 本文中の指定熟語は必ず **太字(Markdown)** で表記すること。
                        3. 「explanations（解説）」の配列には、今回指定された【すべての熟語】を、長文に登場する順番で【1つも漏らさず】記載すること。
                        
                        【出力JSON】
                        {
                          "passage": "This is a story... We must **deal with** the problem...",
                          "full_translation": "長文の和訳...",
                          "explanations": [
                            {"idiom": "deal with", "sentence_used": "We must deal with the problem...", "explanation": "この文脈では「〜に対処する」という意味で使われています。..."}
                          ]
                        }
                        """
                        try:
                            res_input = call_ai(
                                f"対象熟語（必ずすべて解説すること）:\n{', '.join(target_idioms)}",
                                sys_input,
                                is_json=True,
                                model_name="gemini-2.5-flash-lite",
                                max_output_tokens=1600,
                                timeout_seconds=45,
                            )
                            st.session_state.current_input_data = json.loads(res_input)
                        except Exception as e: 
                            st.error(f"生成失敗: {e}")

                if "current_input_data" in st.session_state:
                    data = st.session_state.current_input_data
                    st.markdown("---")
                    
                    # 💡 UIを2カラムに分割！左に長文、右に解説
                    col_passage, col_explanation = st.columns([3, 2])
                    
                    with col_passage:
                        st.markdown("#### 📖 Reading Passage (Input)")
                        with st.container(border=True, height=550):
                            # 太字を青色ハイライトに置換
                            colored_passage = re.sub(r'\*\*(.*?)\*\*', r"<span style='color:#1f77b4; font-weight:bold; font-size:1.05em;'>\1</span>", data.get('passage', ''))
                            st.markdown(f"<div style='line-height: 1.8; font-size: 1.1em;'>{colored_passage}</div>", unsafe_allow_html=True)
                        
                        with st.expander("👁️ 長文の全訳を確認する"):
                            st.write(data.get("full_translation", ""))

                    with col_explanation:
                        st.markdown("#### 💡 熟語の文脈解説")
                        st.caption("本文に登場した順にすべての熟語を解説しています。")
                        
                        # 高さを揃えるコンテナ
                        with st.container(height=550):
                            for item in data.get("explanations", []):
                                with st.expander(f"📌 {item.get('idiom', '')}", expanded=False):
                                    st.markdown(f"**使われ方:** {item.get('sentence_used', '')}")
                                    st.markdown(f"**解説:** {item.get('explanation', '')}")

    # ------------------------------------------
    # タブ3: アウトプット（難易度固定・重複バグ修正・3.5Flash搭載版）
    # ------------------------------------------
    with tab_output:
        if not my_data["idiom_books"]:
            st.info("熟語帳がありません。")
        else:
            st.markdown("### 📤 アウトプット")
            st.caption("正解するごとにその熟語の出題確率は減っていきます。知らない熟語や定着していない熟語が優先して出題されます。")
            selected_title_output = st.selectbox("📖 テストする熟語帳を選択", ["-- 選択してください --"] + book_titles, key="output_sel")
            if selected_title_output != "-- 選択してください --":
                book_idx = book_titles.index(selected_title_output)
                current_idiom_book = my_data["idiom_books"][book_idx]
                
                # 💡 難易度スライダーを廃止し、簡単なレベル固定のアナウンスを表示
                st.info("💡 一番簡単なレベル（中学〜高校基礎）の単語のみを使って、熟語の使い方の確認に集中できる長文テストを生成します。")
                
                if st.button("🚀 長文穴埋めテストを生成する", type="primary"):
                    with st.spinner("AIが論理的な長文と精巧なダミー選択肢を生成しています（約10〜20秒お待ちください）..."):
                        target_idioms = []
                        candidates = current_idiom_book["idioms"].copy()
                        for _ in range(min(15, len(candidates))):
                            weights = [item["count"] / (1 + item.get("correct_count", 0) ** 2) for item in candidates]
                            total_weight = sum(weights)
                            probs = [1/len(candidates)] * len(candidates) if total_weight == 0 else [w/total_weight for w in weights]
                            chosen = random.choices(candidates, weights=probs, k=1)[0]
                            target_idioms.append(chosen["base_form"])
                            candidates.remove(chosen)
                        
                        # ⚠️ プロンプトを大幅改良（重複防止と解説の充実化）
                        sys_output = """
                        あなたはプロの英語予備校講師です。指定された熟語が自然に含まれる150〜200語程度の短い長文を作成してください。
                        
                        【絶対ルール】
                        1. 長文の難易度は、英語が苦手な高校生向け（中学〜高校基礎、CEFR A1〜A2）に固定し、非常に簡単な単語のみを使用してください。
                        2. 指定された熟語の箇所を順番通りに ( 1 ), ( 2 )... と空欄にして問題を作成してください。
                        3. ⚠️【致命的エラー防止】空欄の前後と、正解の選択肢の間で、前置詞などの単語が「重複」しないように絶対確認してください。（悪い例：本文が「( 1 ) him into」で正解が「take into」など。熟語の塊をまるごと空欄に置き換えてください）
                        4. 「解説 (explanation)」には、「なぜその熟語が入るのか」を文脈から丁寧に説明し、不正解のダミー選択肢がなぜダメなのかも簡単に触れてください。
                        
                        【出力JSON形式】
                        {
                          "passage": "This is a story... We must ( 1 ) the issue...",
                          "questions": [
                            {"blank_id": 1, "options": ["deal with", "put off", "bring about", "stand for"], "answer": "deal with", "translation": "問題に対処する。", "explanation": "文脈的に「問題に〜する」となるため、「〜に対処する」という意味の deal with が正解です。put offは「延期する」、bring aboutは「引き起こす」なので文意に合いません。"}
                          ],
                          "full_translation": "全訳..."
                        }
                        """
                        try:
                            # 🚀 モデルを最新の 3.5-flash に固定
                            res_output = call_ai(
                                f"熟語リスト:\n{', '.join(target_idioms)}",
                                sys_output,
                                is_json=True,
                                model_name="gemini-2.5-flash-lite",
                                max_output_tokens=1800,
                                timeout_seconds=45,
                            )
                            st.session_state.current_test_drill = json.loads(res_output)
                            st.session_state.test_q_status = {}
                            st.session_state.current_q_index = 0
                            st.session_state.quiz_chat_logs = {} 
                        except Exception as e: st.error(f"生成失敗: {e}")

                if "current_test_drill" in st.session_state:
                    drill_data = st.session_state.current_test_drill
                    questions = drill_data["questions"]
                    current_idx = st.session_state.current_q_index
                    
                    st.markdown("---")
                    col_text, col_quiz = st.columns([3, 2])
                    
                    with col_text:
                        st.markdown("#### 📖 Reading Passage")
                        with st.container(border=True, height=550):
                            st.markdown(f"<div style='line-height: 1.8; font-size: 1.1em;'>{drill_data['passage']}</div>", unsafe_allow_html=True)
                        with st.expander("👁️ 長文の全訳を確認する"): st.write(drill_data["full_translation"])
                            
                    with col_quiz:
                        st.markdown("#### 📝 Question")
                        if current_idx < len(questions):
                            q = questions[current_idx]
                            q_id = q["blank_id"]
                            st.markdown(f"**Question {current_idx + 1} / {len(questions)}**")
                            st.markdown(f"**( {q_id} ) に入る最も適切な熟語を選べ。**")
                            
                            is_answered = q_id in st.session_state.test_q_status
                            user_ans = st.radio(label=f"q_{q_id}", options=q["options"], key=f"test_q_{q_id}", label_visibility="collapsed", disabled=is_answered)
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            if not is_answered:
                                if st.button("📝 解答して解説を見る", type="primary", use_container_width=True):
                                    correct_ans = q["answer"]
                                    is_correct = (user_ans == correct_ans)
                                    st.session_state.test_q_status[q_id] = {"user_ans": user_ans, "is_correct": is_correct}
                                    
                                    for item in current_idiom_book["idioms"]:
                                        if item["base_form"] == correct_ans:
                                            item["practice_count"] = item.get("practice_count", 0) + 1
                                            if is_correct: item["correct_count"] = item.get("correct_count", 0) + 1
                                            break
                                    save_data(my_data); st.rerun()
                            
                            if is_answered:
                                status = st.session_state.test_q_status[q_id]
                                if status["is_correct"]: st.success(f"⭕ **正解！** : {q['answer']}")
                                else: st.error(f"❌ **不正解** : あなたの解答「{status['user_ans']}」 ➔ 正解「{q['answer']}」")
                                st.info(f"**💡 和訳:** {q['translation']}\n\n**📘 解説:** {q.get('explanation', '解説なし')}")
                                
                                st.markdown("##### 🤖 この問題についてAI講師に質問する")
                                chat_key = f"quiz_chat_{q_id}"
                                if chat_key not in st.session_state.quiz_chat_logs: st.session_state.quiz_chat_logs[chat_key] = []
                                for msg in st.session_state.quiz_chat_logs[chat_key]:
                                    with st.chat_message(msg["role"]): st.markdown(msg["content"])
                                
                                if user_q := st.chat_input("例：他の選択肢の意味をもっと詳しく教えて！", key=f"chat_in_{q_id}"):
                                    st.session_state.quiz_chat_logs[chat_key].append({"role": "user", "content": user_q})
                                    with st.spinner("AI講師が考え中..."):
                                        # 🚀 長文全体を読み込ませ、文法機能と同じ「優しく寄り添う」プロンプトに進化
                                        sys_chat = f"""
                                        あなたは優しく、生徒に寄り添う英語の伴走講師です。絶対に説教や厳しい口調は避けてください。
                                        生徒からの熟語の穴埋め問題に関する質問に答えます。
                                        
                                        【長文の全体（生徒が読んでいる文脈）】
                                        {drill_data['passage']}
                                        
                                        【対象の問題の和訳と正解】
                                        和訳: {q['translation']}
                                        正解の熟語: {q['answer']}
                                        現在の解説: {q.get('explanation', '')}
                                        
                                        【指導のルール】
                                        1. 難しい文法用語は避け、パズルのように視覚的にカタチを教えてください。
                                        2. 熟語の丸暗記を強要するのではなく、「なぜその前置詞が使われるのか（例：onは接触のイメージだから…）」など、コアとなるニュアンスを優しく教えてあげてください。
                                        3. 他の選択肢について聞かれたら、それぞれの意味と、なぜこの文脈に合わないかを分かりやすく説明してください。
                                        """
                                        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.quiz_chat_logs[chat_key][-4:]])
                                        
                                        # 🚀 モデルを最新の 3.5-flash に固定
                                        ans = call_ai(
                                            f"会話:\n{history_str}",
                                            sys_chat,
                                            use_pdf=False,
                                            model_name="gemini-2.5-flash-lite",
                                            max_output_tokens=700,
                                            timeout_seconds=35,
                                        )
                                        st.session_state.quiz_chat_logs[chat_key].append({"role": "assistant", "content": ans}); st.rerun()
                                st.markdown("---")
                                if st.button("次の問題へ ▶", use_container_width=True): st.session_state.current_q_index += 1; st.rerun()
                        else:
                            score = sum(1 for v in st.session_state.test_q_status.values() if v["is_correct"])
                            st.success(f"### 🎉 全問終了！\n**スコア: {score} / {len(questions)}**")
                            if st.button("🔄 テストを終了する", use_container_width=True):
                                del st.session_state.current_test_drill; del st.session_state.test_q_status; del st.session_state.current_q_index; del st.session_state.quiz_chat_logs
                                st.rerun()

    # ------------------------------------------
    # タブ5: カバー率・難化シミュレーター（★熟語版）
    # ------------------------------------------
    with tab_sim:
        st.markdown("### 📊 過去問カバー率（定量分析）シミュレーター")
        st.info("あなたが作った「熟語帳（武器）」が、特定の「過去問（敵）」にどれくらい通用するかを定量的にシミュレーションします。")
        st.warning("⚠️ 熟語のカバー率は実際よりも大幅に低く出ることがあり、さらにすべての問題中の熟語を反映していないため参考程度にしてください。")
        
        books = my_data.get("idiom_books", [])
        if not books:
            st.warning("まずは「✨ 新しい熟語帳を作る」タブから、ベースとなる熟語帳を作成・保存してください。")
        else:
            col_base, col_target = st.columns(2)
            
            with col_base:
                st.markdown("#### ⚔️ 武器（ベースライン）")
                book_titles = [b["title"] for b in books]
                selected_baseline = st.selectbox("学習済みの熟語帳を選択", ["-- 選択 --"] + book_titles)
            
            with col_target:
                st.markdown("#### 🎯 敵（ターゲット）")
                target_options = {}
                for cat, unis in exam_db.items():
                    for uni, facs in unis.items():
                        for fac, years in facs.items():
                            for year, methods in years.items():
                                for method in methods.keys():
                                    label = f"[{cat}] {uni} {fac} ({year}年 {method})"
                                    target_options[label] = {"c": cat, "u": uni, "f": fac, "y": year, "m": method}
                
                selected_target = st.selectbox("テストする未知の過去問を選択", ["-- 選択 --"] + list(target_options.keys()))
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🚀 カバー率を検証する（バックテスト）", use_container_width=True, type="primary"):
                if selected_baseline == "-- 選択 --" or selected_target == "-- 選択 --":
                    st.error("武器と敵を両方選択してください。")
                else:
                    with st.spinner("熟語の照合中..."):
                        # 武器データの取得
                        base_book = next(b for b in books if b["title"] == selected_baseline)
                        base_words = set(item["base_form"] for item in base_book["idioms"])

                        # 敵データの取得
                        path = target_options[selected_target]
                        target_data = exam_db[path["c"]][path["u"]][path["f"]][path["y"]][path["m"]]
                        target_idioms_data = target_data.get("idioms", {})
                        
                        target_freqs = {k: v["count"] for k, v in target_idioms_data.items()}

                        total_tokens = sum(target_freqs.values())
                        covered_tokens = 0
                        missed_words = {}

                        for w, count in target_freqs.items():
                            if w in base_words:
                                covered_tokens += count
                            else:
                                missed_words[w] = count
           
                        coverage_rate = (covered_tokens / total_tokens) * 100 if total_tokens > 0 else 0
                        
                        st.session_state.idiom_sim_result = {
                            "coverage_rate": coverage_rate,
                            "total_tokens": total_tokens,
                            "covered_tokens": covered_tokens,
                            "missed_words": missed_words,
                            "target_name": selected_target
                        }
                        if "idiom_analysis_result" in st.session_state:
                            del st.session_state.idiom_analysis_result
                        st.rerun()
                        
            if "idiom_sim_result" in st.session_state:
                res = st.session_state.idiom_sim_result
                missed_words = res["missed_words"]
                
                st.markdown("---")
                st.markdown(f"## 📊 カバー率: **{res['coverage_rate']:.1f}%** <span style='font-size:0.5em; color:gray;'>({res['target_name']})</span>", unsafe_allow_html=True)
                
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("ターゲットの総熟語数", f"{res['total_tokens']}個")
                col_m2.metric("カバーできた熟語数", f"{res['covered_tokens']}個")
                col_m3.metric("未知の熟語（取りこぼし）", f"{len(missed_words)}種類")
                
                if not missed_words:
                    st.success("完璧です！未知の熟語は1つもありません。")
                else:
                    with st.expander("⚠️ 熟語帳に載っていなかった「未知の熟語リスト」を見る", expanded=False):
                        missed_display = [{"未知の熟語": w, "出現回数": c} for w, c in sorted(missed_words.items(), key=lambda item: item[1], reverse=True)]
                        st.dataframe(missed_display, use_container_width=True)
                    
                    st.markdown("---")
                    st.markdown("### 🧠 致命傷チェッカー（熟語版・拡張判定）")
                    st.markdown("この未知の熟語が実際の設問で**「即死レベルの致命傷」**になるか、推測可能な**「ノイズ」**かを分析し、**「今の熟語帳に新しい熟語を追加して覚えるべきか」**を結論付けます。")
                    
                    input_method = st.radio("入力方法を選択してください", ["📄 PDFをアップロード", "📝 テキストを貼り付け"], horizontal=True, key="idiom_fatal_input")
                    
                    exam_text = ""
                    local_pdf = None
                    if input_method == "📝 テキストを貼り付け":
                        exam_text = st.text_area("📄 過去問の全文（長文と設問・選択肢）をここに貼り付けてください", height=200, placeholder="※PDFのテキストをコピーして貼り付けてください", key="idiom_fatal_text")
                    else:
                        local_pdf = st.file_uploader("📄 分析する過去問のPDFをここにアップロード", type=["pdf"], key="idiom_fatal_pdf")

                    can_analyze = (input_method == "📝 テキストを貼り付け" and exam_text) or (input_method == "📄 PDFをアップロード" and local_pdf)
                    
                    if st.button("🔍 未知の熟語の『致命度』を分析する", type="primary", key="btn_idiom_fatal") and api_key and can_analyze:
                        with st.spinner("AIがPDFを読み込み、設問の構造と未知の熟語の絡みを分析中...（約10〜30秒）"):
                            missed_list_str = ", ".join(missed_words.keys())
                            sys_prompt = """
                            あなたはプロの英語予備校講師であり、生徒の学習戦略コンサルタントです。
                            生徒は提示された「未知の熟語リスト」の意味を知りません。
                            添付された「過去問のデータ（長文＋設問）」を精読し、これらの熟語が「知らなくても推測・無視できるノイズ」か、「知らないと確実に失点する致命傷」かを判定してください。

                            【出力フォーマット（マークダウン）】
                            以下の3つのセクションで出力してください。

                            ### 🚨 致命傷アラート（追加学習が必須な熟語）
                            （※正解選択肢の言い換えになっている、長文のメインテーマの核であるなど、失点に直結する熟語とその理由を具体的に。なければ「なし」と記載）
                            - **[熟語]**: [理由]

                            ### ⚠️ 推測可能なノイズ（無視してよい熟語）
                            （※文脈から推測できる、あるいは設問に全く絡まない熟語の例をいくつかピックアップして理由を記載）
                            - **[熟語]**: [理由]

                            ### 💡 最終結論（学習戦略）
                            （※ズバリ、この志望校で合格点を取るために、「今の熟語帳に新しい熟語を追加して覚えるべきか（知識不足）」、それとも「現状の知識と推測力で十分戦えるか」を結論付けてください。）
                            """
                            
                            prompt = f"【生徒が知らない未知の熟語リスト】\n{missed_list_str}\n\n【過去問のデータ】\n"
                            
                            if input_method == "📝 テキストを貼り付け":
                                prompt += exam_text
                                st.session_state.idiom_analysis_result = call_ai(prompt, sys_prompt, use_pdf=False)
                            else:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                    tmp.write(local_pdf.getvalue())
                                    tmp_path = tmp.name
                                
                                g_file = genai.upload_file(tmp_path)
                                model = genai.GenerativeModel(model_name="gemini-2.5-pro", system_instruction=sys_prompt)
                                res = model.generate_content([g_file, prompt])
                                os.remove(tmp_path)
                                st.session_state.idiom_analysis_result = res.text
                                
                            st.rerun()
                            
                    if "idiom_analysis_result" in st.session_state:
                        st.info(st.session_state.idiom_analysis_result)

# ==========================================
# モードF: 志望校別文法・語法ノート
# ==========================================
elif mode == "📝 志望校別文法・語法ノート":
    st.markdown("### 📝 志望校別文法・語法ノート")
    st.caption("過去問の文法・語法問題そのものを練習し、整序・空所補充・語法知識を固めます。")

    tab_drill = st.tabs(["📚 過去問オリジナルドリル"])[0]

    # ------------------------------------------
    # タブ1: 過去問オリジナルドリル（共通UI適用＆弱点克服アルゴリズム搭載版）
    # ------------------------------------------
    with tab_drill:
        st.markdown("#### 📚 過去問データベースから出題")
        
        # 🛡️ 文法データがある過去問だけを抽出
        db_options_grammar = {}
        if exam_db:
            for cat, unis in exam_db.items():
                for uni, facs in unis.items():
                    for fac, years in facs.items():
                        for year, methods in years.items():
                            for method, data in methods.items():
                                if data.get("grammar_questions"): # 🛡️ 空データ除外！
                                    label = f"[{cat}] {uni} {fac} ({year}年 {method})"
                                    db_options_grammar[label] = {"c": cat, "u": uni, "f": fac, "y": year, "m": method}

        if not db_options_grammar:
            st.info("文法問題のデータがありません。db_manager.py で過去問から文法を抽出してください。")
        else:
            st.markdown("##### 🎯 出題範囲の絞り込み")
            
            # ✨ 共通ツールを使ってスッキリ階層化！
            selected_labels = render_exam_selector(db_options_grammar, "grammar_drill")

            # --- 問題の収集とタグ絞り込み ---
            base_questions = []
            if selected_labels:
                for label in selected_labels:
                    path = db_options_grammar[label]
                    qs = exam_db[path["c"]][path["u"]][path["f"]][path["y"]][path["m"]].get("grammar_questions", [])
                    for q in qs:
                        q_copy = q.copy()
                        q_copy["source"] = label
                        base_questions.append(q_copy)
            
            # タグが存在すれば、さらにタグで絞り込めるようにする
            all_tags = set()
            for q in base_questions:
                tags = q.get("primary_tags", [])
                if isinstance(tags, str): tags = [tags]
                for t in tags: all_tags.add(t)
            
            selected_tag = "-- すべて --"
            if all_tags:
                selected_tag = st.selectbox("5️⃣ さらに文法テーマ（タグ）で絞り込む", ["-- すべて --"] + sorted(list(all_tags)))

            question_type_filter = st.selectbox(
                "6️⃣ 問題形式で絞り込む",
                ["すべて", "選択問題のみ", "整序問題のみ"]
            )

            # 最終的な出題候補
            final_candidates = []
            if selected_tag != "-- すべて --":
                final_candidates = [
                    q for q in base_questions
                    if selected_tag in q.get("primary_tags", q.get("required_knowledge", []))
                ]
            else:
                final_candidates = base_questions

            # 問題形式で絞り込み
            if question_type_filter == "選択問題のみ":
                final_candidates = [
                    q for q in final_candidates
                    if q.get("question_type", "multiple_choice") == "multiple_choice"
                ]
            elif question_type_filter == "整序問題のみ":
                final_candidates = [
                    q for q in final_candidates
                    if q.get("question_type") == "ordering"
                ]

            if selected_labels:
                st.write(f"該当する問題数: **{len(final_candidates)} 問**")

            if st.button("🚀 この範囲でドリルを開始する", type="primary") and selected_labels:
                if not final_candidates:
                    st.warning("選択した条件に合致する文法データがありません。")
                else:
                    # --- 🧠 弱点（不正解）優先アルゴリズム ---
                    if "grammar_stats" not in my_data: my_data["grammar_stats"] = {}
                    
                    weights = []
                    for q in final_candidates:
                        q_text = q.get("question", "")
                        stats = my_data["grammar_stats"].get(q_text, {"correct": 0, "incorrect": 0})
                        w = max(1.0, 10.0 + (stats["incorrect"] * 5.0) - (stats["correct"] * 2.0))
                        weights.append(w)

                    sample_size = min(10, len(final_candidates))
                    selected_qs = []
                    candidates_pool = list(zip(final_candidates, weights))
                    for _ in range(sample_size):
                        total_w = sum(w for q, w in candidates_pool)
                        r = random.uniform(0, total_w)
                        upto = 0
                        for i, (q, w) in enumerate(candidates_pool):
                            if upto + w >= r:
                                selected_qs.append(q)
                                candidates_pool.pop(i)
                                break
                            upto += w

                    st.session_state.grammar_drill_qs = selected_qs
                    st.session_state.grammar_drill_idx = 0
                    st.session_state.grammar_q_status = {}
                    st.session_state.grammar_chat_logs = {}
                    st.rerun()

            # --- ドリル実行画面 ---
            if "grammar_drill_qs" in st.session_state:
                qs = st.session_state.grammar_drill_qs
                idx = st.session_state.get("grammar_drill_idx", 0)
                
                st.markdown("---")
                if idx < len(qs):
                    q = qs[idx]
                    st.markdown(f"#### 📝 Question {idx + 1} / {len(qs)}")
                    st.caption(f"出題元: {q.get('source', '')}")
                    
                    with st.container(border=True):
                        st.markdown(f"**{q.get('question', '')}**")
                        
                                                # 🛡️ 安全装置: 辞書が存在しない場合は空辞書を作成してエラーを防ぐ
                        if "grammar_q_status" not in st.session_state:
                            st.session_state.grammar_q_status = {}

                        is_answered = str(idx) in st.session_state.grammar_q_status
                        q_type = q.get("question_type", "multiple_choice")
                        options = q.get("options", [])

                        # -----------------------------
                        # 選択問題
                        # -----------------------------
                        if q_type == "multiple_choice":
                            user_ans = st.radio(
                                "選択してください:",
                                options,
                                key=f"g_choice_{idx}",
                                disabled=is_answered
                            )

                        # -----------------------------
                        # 整序問題
                        # -----------------------------
                        elif q_type == "ordering":
                            st.markdown("**語句一覧**")
                            for n, word in enumerate(options, start=1):
                                st.markdown(f"{n}. {word}")

                            user_ans = st.text_input(
                                "正しい順番を番号で入力してください。例: 1,5,7,6,4,2,3",
                                key=f"g_order_{idx}",
                                disabled=is_answered
                            )

                        else:
                            st.warning("未対応の問題形式です。")
                            user_ans = ""

                        st.markdown("<br>", unsafe_allow_html=True)
                        if not is_answered:
                            if st.button("📝 解答して解説を見る", key=f"g_ans_btn_{idx}", type="primary", use_container_width=True):
                                correct_ans = q.get("answer")
                                user_display_ans = user_ans

                                if q_type == "multiple_choice":
                                    is_correct = (user_ans == correct_ans)

                                elif q_type == "ordering":
                                    try:
                                        nums = [
                                            int(x.strip())
                                            for x in str(user_ans).replace("，", ",").split(",")
                                            if x.strip()
                                        ]

                                        user_words = [options[n - 1] for n in nums]
                                        user_sentence = " ".join(user_words)

                                        normalized_user = re.sub(
                                            r"\s+",
                                            " ",
                                            re.sub(r"[?.!,，。！？]", "", user_sentence.lower()).strip()
                                        )
                                        normalized_correct = re.sub(
                                            r"\s+",
                                            " ",
                                            re.sub(r"[?.!,，。！？]", "", str(correct_ans).lower()).strip()
                                        )

                                        is_correct = (normalized_user == normalized_correct)
                                        user_display_ans = user_sentence

                                    except Exception:
                                        is_correct = False
                                        user_display_ans = "入力形式エラー"

                                else:
                                    is_correct = False

                                st.session_state.grammar_q_status[str(idx)] = {
                                    "user_ans": user_display_ans,
                                    "raw_user_ans": user_ans,
                                    "is_correct": is_correct
                                }

                                # 正答・誤答の履歴を保存（次回の出題確率に影響）
                                q_text = q.get("question", "")
                                if "grammar_stats" not in my_data:
                                    my_data["grammar_stats"] = {}
                                if q_text not in my_data["grammar_stats"]:
                                    my_data["grammar_stats"][q_text] = {"correct": 0, "incorrect": 0}
                                if is_correct:
                                    my_data["grammar_stats"][q_text]["correct"] += 1
                                else:
                                    my_data["grammar_stats"][q_text]["incorrect"] += 1
                                save_data(my_data)

                                st.rerun()

                        if is_answered:
                            status = st.session_state.grammar_q_status[str(idx)]
                            st.markdown("---")

                            if status["is_correct"]:
                                st.success(f"⭕ **正解！** : {q.get('answer')}")
                            else:
                                st.error(f"❌ **不正解** : あなたの解答「{status['user_ans']}」 ➔ 正解「{q.get('answer')}」")

                            if q.get("question_type") == "ordering" and q.get("answer_order"):
                                st.caption(f"正しい番号順: {','.join(map(str, q.get('answer_order', [])))}")

                            st.info(f"**💡 和訳:** {q.get('translation', '記載なし')}\n\n**📘 解説・背景知識:**\n{q.get('explanation', '記載なし')}")

                            if st.button("💾 この文法知識を「マイ教訓ノート」に保存", key=f"g_save_note_{idx}"):
                                if "grammar" not in my_data:
                                    my_data["grammar"] = []
                                my_data["grammar"].append({
                                    "title": "文法ドリルからの教訓",
                                    "content": q.get("explanation", ""),
                                    "source": q.get("source", "")
                                })
                                save_data(my_data)
                                st.success("保存しました！")

                            # --- チャット機能 ---
                            st.markdown("##### 🤖 この問題についてAI講師に深掘り質問する")
                            chat_key = f"g_chat_{idx}"
                            if "grammar_chat_logs" not in st.session_state:
                                st.session_state.grammar_chat_logs = {}
                            if chat_key not in st.session_state.grammar_chat_logs:
                                st.session_state.grammar_chat_logs[chat_key] = []
                            for msg in st.session_state.grammar_chat_logs[chat_key]:
                                with st.chat_message(msg["role"]): st.markdown(msg["content"])
                            if user_q := st.chat_input("例：この動詞の他の用法も教えて", key=f"g_chat_in_{idx}"):
                                st.session_state.grammar_chat_logs[chat_key].append({"role": "user", "content": user_q})
                                with st.spinner("AI講師が思考中..."):
                                    sys_chat = f"問題: {q.get('question', '')}\n和訳: {q.get('translation', '')}\n正解: {q.get('answer', '')}\n解説: {q.get('explanation', '')}\n生徒の質問に簡潔に答えてください。"
                                    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.grammar_chat_logs[chat_key][-4:]])
                                    ans = call_ai(
                                        f"会話:\n{history_str}",
                                        sys_chat,
                                        use_pdf=False,
                                        model_name="gemini-2.5-flash-lite",
                                        max_output_tokens=700,
                                        timeout_seconds=35,
                                    )
                                    st.session_state.grammar_chat_logs[chat_key].append({"role": "assistant", "content": ans})
                                    st.rerun()

                            st.markdown("---")
                            if st.button("次の問題へ ▶", use_container_width=True):
                                st.session_state.grammar_drill_idx += 1
                                st.rerun()
                else:
                    # 🛡️ 安全にスコアを集計（存在しない場合は無視する）
                    status_dict = st.session_state.get("grammar_q_status", {})
                    score = sum(1 for v in status_dict.values() if v.get("is_correct"))
                    
                    st.success(f"### 🎉 ドリル終了！\n**スコア: {score} / {len(qs)}**")
                    if st.button("🔄 終了する", use_container_width=True):
                        # 🧹 関連するセッション（裏の記憶）をすべて綺麗に削除
                        for k in ["grammar_drill_qs", "grammar_drill_idx", "grammar_q_status", "grammar_chat_logs"]:
                            if k in st.session_state:
                                del st.session_state[k]
                        st.rerun()
# ==========================================
# モードG: 長文読解
# ==========================================
elif mode == "📚 長文読解":
    st.markdown("### 📚 長文読解")
    st.caption("長文を読むための文法、1文精読、設問根拠、読解メモをここに集約します。")

    tab_close, tab_evidence, tab_notes = st.tabs([
        "📖 精読・構文・和訳",
        "🎯 設問根拠トレーニング",
        "🗂️ 読解メモ・弱点ノート",
    ])

    with tab_close:
        render_close_reading_tab()

    with tab_evidence:
        render_evidence_training_tab()

    with tab_notes:
        render_reading_notes_tab()

# ==========================================
# ★最終章 モードH: 過去問演習・合格分析（長文＋スコア管理＋コンパス）
# ==========================================
elif mode == "🏆 過去問演習・合格分析":
    import time
    
    st.markdown("### 🏆 過去問演習・合格分析")
    st.caption("表面的なスピードではなく、「知識の解像度」と「時間の使い方」を多角的に分析し、合格最低点を超えるための最終ダッシュボードです。")

    if "exam_records" not in my_data:
        my_data["exam_records"] = []

    tab_score, tab_deep, tab_ai_sim, tab_compass = st.tabs(["📊 時間＆戦績データ管理", "🧠 全問ディープ分析＆オウトプシー", "🤖 AI代行受験 (スコア推定)", "🧭 総合戦略コンパス (全データ連携)"])

    # ------------------------------------------
    # タブ1: 時間＆戦績データ管理
    # ------------------------------------------
    with tab_score:
        st.markdown("#### 📝 過去問のスコアとタイムを記録")
        st.info("まずは「時間内」「時間無制限」の得点を記録し、後からディープ分析を経て判明した「実力点」や「期待値得点」を編集・追加できます。")
        
        # --- 階層データツリーの構築 ---
        tree = {}
        if exam_db:
            for cat, unis in exam_db.items():
                for u, facs in unis.items():
                    if u not in tree: tree[u] = {}
                    for f, years in facs.items():
                        if f not in tree[u]: tree[u][f] = {}
                        for y, methods in years.items():
                            y_str = str(y)
                            if y_str not in tree[u][f]: tree[u][f][y_str] = []
                            for m in methods.keys():
                                if m not in tree[u][f][y_str]: tree[u][f][y_str].append(m)
        
        for r in my_data.get("exam_records", []):
            u, f, m, y = r.get("uni"), r.get("fac"), r.get("method"), str(r.get("year", ""))
            if u and f and m and y:
                if u not in tree: tree[u] = {}
                if f not in tree[u]: tree[u][f] = {}
                if y not in tree[u][f]: tree[u][f][y] = []
                if m not in tree[u][f][y]: tree[u][f][y].append(m)

        # --- カスケード選択 UI ---
        st.markdown("##### 📍 受験した過去問の指定")
        col_u, col_f, col_m, col_y = st.columns(4)
        
        with col_u:
            uni_opts = ["-- 選択/新規 --"] + sorted(list(tree.keys()))
            sel_u = st.selectbox("1️⃣ 大学", uni_opts, key="rec_u")
            input_uni = st.text_input("大学名を入力", key="in_u") if sel_u == "-- 選択/新規 --" else sel_u

        with col_f:
            fac_opts = ["-- 選択/新規 --"]
            if sel_u in tree: fac_opts += sorted(list(tree[sel_u].keys()))
            sel_f = st.selectbox("2️⃣ 学部", fac_opts, key="rec_f")
            input_fac = st.text_input("学部を入力", key="in_f") if sel_f == "-- 選択/新規 --" else sel_f

        with col_m:
            method_opts = ["-- 選択/新規 --"]
            if sel_u in tree and sel_f in tree[sel_u]: 
                all_m = set()
                for y_dict in tree[sel_u][sel_f].values(): all_m.update(y_dict)
                method_opts += sorted(list(all_m))
            sel_m = st.selectbox("3️⃣ 方式", method_opts, key="rec_m")
            input_method = st.text_input("方式を入力", key="in_m") if sel_m == "-- 選択/新規 --" else sel_m

        with col_y:
            year_opts = ["-- 選択/新規 --"]
            if sel_u in tree and sel_f in tree[sel_u]:
                if sel_m != "-- 選択/新規 --":
                    valid_years = [y for y, m_list in tree[sel_u][sel_f].items() if sel_m in m_list]
                    year_opts += sorted(valid_years, reverse=True)
                else:
                    year_opts += sorted(list(tree[sel_u][sel_f].keys()), reverse=True)
            sel_y = st.selectbox("4️⃣ 年度", year_opts, key="rec_y")
            input_year = st.text_input("年度を入力 (例: 2025)", key="in_y") if sel_y == "-- 選択/新規 --" else sel_y

        st.markdown("##### 📊 スコアとタイムの入力")
        with st.container(border=True):
            col_s1, col_s2, col_s3 = st.columns(3)
            target_score = col_s1.number_input("合格最低点(目標)", min_value=0, max_value=1000, value=70, key="rec_ts")
            score_in_time = col_s2.number_input("制限時間内の得点", min_value=0, max_value=1000, value=50, key="rec_st")
            score_unlimited = col_s3.number_input("時間無制限での得点", min_value=0, max_value=1000, value=65, key="rec_su")
            
            st.caption("※分析後に判明した「実力点」「期待値得点」は、保存後に履歴から編集・追加できます。")
            
            col_t1, col_t2 = st.columns(2)
            time_limit = col_t1.number_input("制限時間（分）", min_value=1, max_value=300, value=60, key="rec_tl")
            time_taken = col_t2.number_input("完答までにかかった総時間（分）", min_value=1, max_value=500, value=85, key="rec_tt")
            
        if st.button("💾 戦績とタイムを初記録する", type="primary", use_container_width=True):
            if input_uni and input_fac and input_method and input_year:
                exam_name = f"{input_year}年 {input_uni} {input_fac} {input_method}"
                my_data["exam_records"].append({
                    "date": time.strftime("%Y-%m-%d"),
                    "uni": input_uni,
                    "fac": input_fac,
                    "method": input_method,
                    "year": input_year,
                    "exam_name": exam_name,
                    "target_score": target_score,
                    "score_in_time": score_in_time,
                    "score_unlimited": score_unlimited,
                    "time_limit": time_limit,
                    "time_taken": time_taken,
                    "true_score": 0, # 初期値
                    "expected_score": 0 # 初期値
                })
                save_data(my_data)
                st.success("記録しました！")
                st.rerun()
            else:
                st.error("大学・学部・方式・年度をすべて入力してください。")

        st.divider()
        st.markdown("#### 📈 あなたの戦績履歴 ＆ 分析後データ追記")
        if not my_data["exam_records"]:
            st.info("まだ記録がありません。過去問を解いたらデータを記録しましょう。")
        else:
            recorded_unis = set([r.get("uni", "その他 (旧データ)") for r in my_data["exam_records"]])
            filter_uni = st.selectbox("🏫 表示する大学を絞り込む", ["-- すべての大学 --"] + sorted(list(recorded_unis)))
            
            filtered_records = []
            for i, r in enumerate(my_data["exam_records"]):
                r_with_idx = r.copy()
                r_with_idx["_orig_idx"] = i 
                if filter_uni == "-- すべての大学 --" or r.get("uni", "その他 (旧データ)") == filter_uni:
                    filtered_records.append(r_with_idx)
                    
            if not filtered_records:
                st.info("指定された大学の記録はありません。")
            else:
                filtered_records.sort(key=lambda x: str(x.get("year", "0")), reverse=True)
                
            for record in filtered_records:
                s_in_time = record.get('score_in_time', record.get('actual_score', 0))
                s_unlim = record.get('score_unlimited', s_in_time)
                t_limit = record.get('time_limit', 0)
                t_taken = record.get('time_taken', 0)
                
                # 新指標
                true_score = record.get('true_score', 0)
                expected_score = record.get('expected_score', 0)
                
                gap_time = s_in_time - record.get('target_score', 0)
                knowledge_gap = s_unlim - s_in_time
                time_over = t_taken - t_limit
                
                with st.container(border=True):
                    st.markdown(f"**{record.get('exam_name', '名称不明')}** <span style='color:gray; font-size:0.8em;'>({record.get('date', '')})</span>", unsafe_allow_html=True)
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("時間内得点 (目標)", f"{s_in_time}点", f"{gap_time}点", delta_color="normal" if gap_time >= 0 else "inverse")
                    c2.metric("無制限得点 (純粋力)", f"{s_unlim}点", f"+{knowledge_gap}点 (時間外)", delta_color="off")
                    c3.metric("かかった時間", f"{t_taken}分", f"{time_over}分オーバー" if time_over > 0 else f"{abs(time_over)}分余り", delta_color="inverse" if time_over > 0 else "normal")
                    
                    # 💡 ディープ分析後の「真の実力」を表示
                    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
                    ce1, ce2, ce3 = st.columns([1, 1, 2])
                    ce1.metric("🟢 実力点", f"{true_score}点", help="まぐれを排除し、根拠を持って正解できた点数")
                    ce2.metric("🟡 期待値得点", f"{expected_score}点", help="絞り込めた選択肢から確率論的に算出した期待値（例：2択まで絞れた問題×0.5）")
                    
                    with st.expander("✏️ ディープ分析後の実力点・期待値を編集"):
                        with st.form(f"edit_form_{record['_orig_idx']}"):
                            st.caption("分析タブでまぐれを排除した後の「実力点」や、選択肢を絞り込んだことによる「期待値得点」を記録します。")
                            col_e1, col_e2 = st.columns(2)
                            new_true = col_e1.number_input("実力点", value=true_score, min_value=0, max_value=1000)
                            new_exp = col_e2.number_input("期待値得点", value=expected_score, min_value=0, max_value=1000)
                            
                            col_f1, col_f2 = st.columns(2)
                            if col_f1.form_submit_button("更新する"):
                                my_data["exam_records"][record['_orig_idx']]["true_score"] = new_true
                                my_data["exam_records"][record['_orig_idx']]["expected_score"] = new_exp
                                save_data(my_data)
                                st.success("更新しました！"); st.rerun()
                            
                            if col_f2.form_submit_button("🗑️ この戦績ごと削除"):
                                my_data["exam_records"].pop(record['_orig_idx'])
                                save_data(my_data); st.rerun()

    # ------------------------------------------
    # タブ2: 全問ディープ分析＆総合戦略 (インタラクティブ＆チェックリスト搭載版)
    # ------------------------------------------
    with tab_deep:
        st.markdown("#### 🧠 全問ディープ・オウトプシー（死因究明）")
        st.caption("AIと対話して「なぜ間違えたか」「何が足りないか」を解剖しつつ、右側のチェックリストで正直な自分の『実力』を記録します。")
        
        # --- 中断データの復元 ---
        if "saved_review_session" in my_data and my_data["saved_review_session"]:
            st.info("💾 中断されたレビューセッションがあります。")
            col_res1, col_res2 = st.columns([1, 3])
            if col_res1.button("🔄 前回から再開する", type="primary", use_container_width=True):
                saved = my_data["saved_review_session"]
                st.session_state.exam_review_text = saved["text"]
                st.session_state.exam_review_chat = saved["chat"]
                st.session_state.exam_review_questions = saved["questions"]
                st.session_state.exam_review_checklist = saved["checklist"]
                st.session_state.exam_review_name = saved.get("name", "過去問")
                st.rerun()
            if col_res2.button("🗑️ 中断データを破棄する"):
                my_data["saved_review_session"] = {}
                save_data(my_data)
                st.rerun()

        # --- PDF抽出関数 ---
        def extract_exam_text_for_review(uploaded_file):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            try:
                g_file = genai.upload_file(tmp_path)
                model = genai.GenerativeModel(model_name="gemini-2.5-pro")
                prompt = "このPDF（英語の過去問）から、長文、設問、選択肢などのテキストデータをすべて抽出してください。不要なレイアウト情報は省き、問題の構造がわかるように綺麗にテキスト化してください。"
                res = model.generate_content([g_file, prompt])
                return res.text
            finally:
                os.remove(tmp_path)

        # 1️⃣ PDFアップロードと抽出フェーズ
        if "exam_review_text" not in st.session_state and "draft_review_text" not in st.session_state:
            exam_name_input = st.text_input("🎓 復習する過去問の名前（例：2025 日本大学 文理学部）", key="new_review_name")
            up_pdf = st.file_uploader("📄 過去問のPDFをアップロード", type=["pdf"], key="review_pdf")
            
            if st.button("🚀 PDFからテキストを抽出する", type="primary", key="btn_extract_review_pdf"):
                if up_pdf and api_key and exam_name_input:
                    with st.spinner("AIが過去問PDFを解析中...（約10〜20秒）"):
                        try:
                            extracted_text = extract_exam_text_for_review(up_pdf)
                            st.session_state.draft_review_text = extracted_text
                            st.session_state.exam_review_name = exam_name_input
                            st.rerun()
                        except Exception as e:
                            st.error(f"PDFの読み込みに失敗しました: {e}")
                else:
                    st.error("⚠️ 過去問の名前とPDFの両方を入力してください。")
                    
        # 2️⃣ 抽出テキストの確認 ＆ 問題番号の自動抽出フェーズ
        if "draft_review_text" in st.session_state and "exam_review_text" not in st.session_state:
            st.markdown("##### 📝 抽出されたテキストの確認・修正")
            st.info("文字化けや抜けがあれば修正してください。OKを押すと、AIが自動で「問題数（チェックリスト）」を作成します。")
            edited_text = st.text_area("テキストデータを修正", st.session_state.draft_review_text, height=300)
            
            col_ok, col_cancel = st.columns(2)
            if col_ok.button("✅ このテキストでレビューを開始する", type="primary", use_container_width=True):
                with st.spinner("AIが問題構成を把握し、チェックリストを作成中..."):
                    sys_q_extract = "入力された過去問テキストから、問題番号（例: 大問1 (1), 大問2 問1 など）をすべて抽出し、JSONの配列として出力してください。出力例: [\"大問1 (1)\", \"大問1 (2)\", \"大問2 (A)\"]"
                    try:
                        q_res = call_ai(edited_text, sys_q_extract, is_json=True, model_name="gemini-2.5-pro")
                        q_list = json.loads(q_res)
                    except:
                        q_list = ["問題1", "問題2", "問題3"] # エラー時のフォールバック
                        
                    st.session_state.exam_review_questions = q_list
                    st.session_state.exam_review_checklist = {q: "未チェック" for q in q_list}
                    st.session_state.exam_review_text = edited_text
                    st.session_state.exam_review_chat = [
                        {"role": "assistant", "content": "準備完了！さっそく一問ずつ復習していこう。わからないところや、勘違いしていた選択肢について何でも聞いてね。右側のチェックリストは君の自由に使っていいよ！"}
                    ]
                    del st.session_state.draft_review_text
                    st.rerun()
                    
            if col_cancel.button("🗑️ やり直す", use_container_width=True):
                del st.session_state.draft_review_text
                st.rerun()

        # 3️⃣ レビューセッション（左：チャット、右：自己評価チェックリスト）
        if "exam_review_text" in st.session_state:
            st.markdown(f"### 🎯 {st.session_state.exam_review_name} の復習セッション")
            
            with st.expander("👀 確定した過去問テキストを確認する", expanded=False):
                st.text_area("抽出データ", st.session_state.exam_review_text, height=150, disabled=True)
                
            col_chat, col_check = st.columns([3, 2])
            
            # --- 左側：AIとのディープチャット ---
            with col_chat:
                st.markdown("##### 🗣️ AI講師との壁打ち")
                chat_model_choice = st.radio("🧠 分析モデル", ["推論モデル (2.5 Pro) - 深掘り", "高速モデル (2.5 Flash Lite)"], horizontal=True, label_visibility="collapsed")
                
                with st.container(border=True, height=500):
                    for msg in st.session_state.exam_review_chat:
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])
                
                if user_req := st.chat_input("例：問1は②にした。なぜ他の選択肢がダメなの？", key="review_chat_in"):
                    st.session_state.exam_review_chat.append({"role": "user", "content": user_req})
                    with st.spinner("AI講師が分析中..."):
                        selected_model_name = "gemini-2.5-pro" if "Pro" in chat_model_choice else "gemini-2.5-flash-lite"
                        
                        sys_review = f"""
                        あなたは生徒に寄り添う予備校講師です。以下の【過去問テキスト】を共有のコンテキストとして復習を行います。
                        
                        【絶対ルール】
                        1. 生徒の「実力評価」は生徒自身に委ねられています。あなたは生徒の言葉を否定せず、知識の補強（どの単語が必要だったか、どの文法を勘違いしていたか）に徹してください。
                        2. 日本の受験特有の省略（「最初は2にした」「問3は4」等）は空気を読んで文脈から察してください。
                        3. 「なぜその選択肢はダメなのか」「どういう知識があれば解けたのか」を論理的かつ簡潔に教えてください。

                        【過去問テキスト】
                        {st.session_state.exam_review_text}
                        """
                        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.exam_review_chat[-6:]])
                        
                        ans = call_ai(f"会話履歴:\n{history_str}", sys_review, use_pdf=False, model_name=selected_model_name)
                        st.session_state.exam_review_chat.append({"role": "assistant", "content": ans})
                        st.rerun()

            # --- 右側：自己評価とセッション管理 ---
            with col_check:
                st.markdown("##### 📝 自己評価チェックリスト")
                st.caption("AIの顔色を気にせず、正直な実力を記録しよう。")
                
                with st.container(border=True, height=500):
                    options = ["未チェック", "🟢 実力で正解", "🟡 まぐれ・他選択肢の勘違いあり", "🔴 不正解（知識・精読不足）"]
                    for q in st.session_state.exam_review_questions:
                        current_val = st.session_state.exam_review_checklist.get(q, "未チェック")
                        # UI上で状態を更新
                        new_val = st.radio(q, options, index=options.index(current_val), key=f"chk_{q}", horizontal=False)
                        st.session_state.exam_review_checklist[q] = new_val

                st.markdown("<br>", unsafe_allow_html=True)
                
                if st.button("💾 この状態を一時保存して中断する", use_container_width=True):
                    my_data["saved_review_session"] = {
                        "name": st.session_state.exam_review_name,
                        "text": st.session_state.exam_review_text,
                        "chat": st.session_state.exam_review_chat,
                        "questions": st.session_state.exam_review_questions,
                        "checklist": st.session_state.exam_review_checklist
                    }
                    save_data(my_data)
                    st.success("セッションを一時保存しました！次回このタブを開いた時に復元できます。")

            st.markdown("---")
            st.markdown("##### 💡 判明した課題をストック")
            with st.form("exam_insight_form", clear_on_submit=True):
                col_i1, col_i2 = st.columns([1, 2])
                note_title = col_i1.text_input("📝 足りなかった知識・項目名（例：仮定法過去完了の倒置）")
                note_content = col_i2.text_input("教訓・戦略（例：Had S p.p. が見えたらIfの省略を疑う！）")
                
                if st.form_submit_button("🏠 マイ教訓ノート(戦略)に保存", type="primary"):
                    if note_title and note_content:
                        if "strategy" not in my_data: my_data["strategy"] = []
                        my_data["strategy"].append({"title": note_title, "content": note_content, "source": st.session_state.exam_review_name})
                        save_data(my_data)
                        st.success("マイ教訓ノートに保存しました！")
                    else:
                        st.error("項目名と教訓の両方を入力してください。")
                
            if st.button("🗑️ レビューを完全終了する（保存したチェックリストとチャットを破棄）", use_container_width=True):
                if "saved_review_session" in my_data: my_data["saved_review_session"] = {}
                save_data(my_data)
                for k in ["exam_review_text", "exam_review_chat", "exam_review_questions", "exam_review_checklist", "exam_review_name"]:
                    if k in st.session_state: del st.session_state[k]
                st.rerun()
    
    # ------------------------------------------
    # タブX: AI代行受験 (スコア＆期待値シミュレーター)
    # ------------------------------------------
    with tab_ai_sim:
        st.markdown("#### 🤖 AI代行受験 (合格期待値シミュレーター)")
        st.caption("現在のあなたの「単語・熟語・文法知識（武器）」だけを持たせたAIのクローンに、初見の過去問を解かせ、数学的な期待値スコアを算出します。")

        # 武器の選択
        fixed_book_for_ai_sim = build_frequency_strong_book(base_lexicon)
        vocab_book_objects = ([fixed_book_for_ai_sim] if fixed_book_for_ai_sim else []) + my_data.get("vocab_books", [])
        vocab_books = [b["title"] for b in vocab_book_objects]
        idiom_books = [b["title"] for b in my_data.get("idiom_books", [])]

        col_w1, col_w2, col_w3 = st.columns([2, 2, 1.5])
        sel_vocab = col_w1.selectbox("⚔️ 装備する単語帳", ["-- なし --"] + vocab_books)
        sel_idiom = col_w2.selectbox("⚔️ 装備する熟語帳", ["-- なし --"] + idiom_books)
        
        col_w3.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        use_grammar = col_w3.checkbox("⚔️ 過去問DBの文法知識を装備", value=True, help="過去問データベースから抽出・蓄積した「必須知識タグ」をAIに持たせます")

        st.markdown("##### ⚙️ 本番環境のデバフ（ペナルティ）設定")
        st.caption("※「ケアレスミス」と「焦り」は、どちらも『実力で取れたはずの持ち点』からの失点として合算されます。")
        col_p1, col_p2, col_p3 = st.columns(3)
        careless_rate = col_p1.slider("😰 ケアレスミス率", 0, 30, 5, format="%d%%", help="マークミスや読み間違いによる失点率") / 100.0
        panic_rate = col_p2.slider("🌀 焦り失点率", 0, 50, 10, format="%d%%", help="プレッシャーで普段の論理的思考ができず落としてしまう割合") / 100.0
        timeout_rate = col_p3.slider("⏳ 時間切れ(塗り絵)率", 0, 50, 10, format="%d%%", help="時間が足りず未着手となり、完全な勘でマークする割合") / 100.0

        exam_text_sim = st.text_area("📄 仮想受験させる過去問のテキスト（長文＋設問＋選択肢）", height=200, placeholder="ここに解かせたい過去問のテキストを貼り付けてください")

        if st.button("🚀 クローンAIに仮想受験させる", type="primary", use_container_width=True):
            if not exam_text_sim:
                st.error("過去問のテキストを入力してください。")
            else:
                with st.spinner("あなたの知識セットをAIにコピーし、仮想受験を実行中...（約20〜40秒）"):
                    # 知識の抽出
                    known_words = sorted(frequency_strong_words)
                    if sel_vocab != "-- なし --":
                        selected_vocab_book = next((b for b in vocab_book_objects if b["title"] == sel_vocab), None)
                        if selected_vocab_book:
                            known_words = sorted(set(known_words) | set(selected_vocab_book.get("main_vocab", [])))
                    known_idioms = []
                    if sel_idiom != "-- なし --":
                        known_idioms = [i["base_form"] for i in next((b["idioms"] for b in my_data["idiom_books"] if b["title"] == sel_idiom), [])]
                    
                    known_grammar = []
                    if use_grammar and exam_db:
                        tags_set = set()
                        for cat, unis in exam_db.items():
                            for u, facs in unis.items():
                                for f, years in facs.items():
                                    for y, methods in years.items():
                                        for m, data in methods.items():
                                            if "grammar_tags" in data:
                                                tags_set.update(data["grammar_tags"].keys())
                        known_grammar = sorted(list(tags_set))

                    sys_sim = f"""
                    あなたは、指定された「習得済み語彙・文法リスト」のみを知識として持つ仮想の受験生です。
                    リストにない単語や文法事項は「未知」として扱い、文脈から必死に推測して問題を解いてください。
                    【AIの超人化防止ルール】推測の際は、文章全体から俯瞰して逆算するのではなく「その未知語が含まれる文とその前後の1文」のみを根拠として局所的に推測すること。

                    【あなたの武器（習得済み）】
                    ■ 単語: {', '.join(known_words[:800]) if known_words else 'なし'} ...
                    ■ 熟語: {', '.join(known_idioms[:300]) if known_idioms else 'なし'} ...
                    ■ 文法・語法テーマ: {', '.join(known_grammar) if known_grammar else 'なし'}

                    【指示】
                    入力された試験問題を解き、各設問について以下のJSONを出力してください。
                    {{
                      "results": [
                        {{
                          "question_id": "問1",
                          "total_options": 4,
                          "narrowed_down_to": 2, 
                          "reasoning": "〇〇は未知語だったが、前後の文脈からプラスの意味だと推測。選択肢1と3を消去し、2択に絞った。"
                        }}
                      ]
                    }}
                    ※ `total_options` はその問題の本来の選択肢の数（3択なら3、4択なら4、5択なら5）を整数で出力。
                    ※ `narrowed_down_to` は、既知の語彙・文法と推測で「何択まで絞れたか」を整数で出力（完全に自信があれば1、2択なら2、全く分からなければ total_options と同じ値）。
                    """
                    try:
                        res_sim = call_ai(exam_text_sim, sys_sim, is_json=True, model_name="gemini-2.5-pro")
                        st.session_state.sim_exam_results = json.loads(res_sim)
                        st.session_state.sim_penalties = {"careless": careless_rate, "panic": panic_rate, "timeout": timeout_rate}
                    except Exception as e:
                        st.error(f"シミュレーション失敗: {e}")

        if "sim_exam_results" in st.session_state:
            results = st.session_state.sim_exam_results.get("results", [])
            penalties = st.session_state.sim_penalties
            if results:
                total_q = len(results)
                pts_per_q = 100.0 / total_q  # 均等配点（100点満点換算）

                timeout_rate = penalties.get("timeout", 0.0)
                careless_rate = penalties.get("careless", 0.0)
                panic_rate = penalties.get("panic", 0.0)

                raw_true_total = 0.0
                raw_expected_total = 0.0
                blind_guess_score = 0.0

                for r in results:
                    tot_opt = max(1, r.get("total_options", 4)) # ゼロ割防止（デフォルト4択）
                    n = max(1, r.get("narrowed_down_to", tot_opt))

                    # 1. 時間切れ(未着手)分の期待値（問題ごとの選択肢数に応じた純粋な勘）
                    blind_guess_score += (pts_per_q * timeout_rate) * (1.0 / tot_opt)
                    
                    # 2. 着手できた割合による獲得見込み点数
                    attempted_pts = pts_per_q * (1.0 - timeout_rate)
                    
                    if n <= 1:
                        raw_true_total += attempted_pts
                    else:
                        raw_expected_total += attempted_pts * (1.0 / n)

                # 3. デバフ（ケアレスミス ＋ 焦り）の適用
                # ユーザー仕様: どちらも「実力でとれるはずのもの（True + Expected）」からの失点とする
                debuff_rate = careless_rate + panic_rate
                
                final_true = raw_true_total * (1.0 - debuff_rate)
                final_expected = raw_expected_total * (1.0 - debuff_rate)
                debuff_loss = (raw_true_total + raw_expected_total) * debuff_rate
                
                final_total = final_true + final_expected + blind_guess_score

                st.markdown("---")
                st.markdown(f"### 📊 仮想受験スコア（推定期待値）: **{final_total:.1f} / 100 点**")

                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                col_s1.metric("🟢 実力点 (確保分)", f"{final_true:.1f}点", f"元の実力: {raw_true_total:.1f}点", delta_color="off")
                col_s2.metric("🟡 期待値点 (確保分)", f"{final_expected:.1f}点", f"元の期待値: {raw_expected_total:.1f}点", delta_color="off")
                col_s3.metric("🎲 時間切れ (塗り絵)", f"{blind_guess_score:.1f}点", "問題ごとの確率計算", delta_color="off")
                col_s4.metric("⚠️ デバフ失点", f"-{debuff_loss:.1f}点", f"ケアレス+焦りのロス", delta_color="inverse")

                st.markdown("#### 🧠 AIクローンの全問オウトプシー（思考・根拠）")
                st.caption(f"※全問題のうち {(1.0 - timeout_rate)*100:.0f}% に着手できたと仮定した解答プロセスです。")
                for r in results:
                    tot_opt = r.get("total_options", 4)
                    n = r.get("narrowed_down_to", tot_opt)
                    
                    if n <= 1:
                        badge = "🟢 確信 (1択)"
                    elif n < tot_opt:
                        badge = f"🟡 絞り込み成功 ({n}/{tot_opt}択)"
                    else:
                        badge = f"🔴 完全な勘 ({tot_opt}択のまま)"

                    with st.expander(f"{r.get('question_id', '問題')} - {badge}", expanded=(1 < n < tot_opt)):
                        st.markdown(f"**思考プロセス（根拠）:**\n{r.get('reasoning', '')}")

    # ------------------------------------------
    # タブ3: 総合戦略コンパス (全データ連携)
    # ------------------------------------------
    with tab_compass:
        st.markdown("#### 🧭 総合戦略コンパス")
        st.caption("アプリ内に蓄積されたすべてのデータ（単語・熟語・文法・教訓・過去問の戦績）をAIが集約し、現在の立ち位置と次にやるべきアクションをコンサルティングします。")

        if not st.session_state.get("compass_permission"):
            st.warning("⚠️ AIがあなたの全学習データにアクセスして、志望校に向けた専用の戦略を立てます。実行しますか？")
            if st.button("🔓 全データへのアクセスを許可して戦略会議を始める", type="primary"):
                st.session_state.compass_permission = True
                st.rerun()
        else:
            st.success("✅ 全データへのアクセスが許可されています。")
            
            if "compass_chat" not in st.session_state:
                st.session_state.compass_chat = [
                    {"role": "assistant", "content": "データの読み込みが完了したよ！これまでの君の頑張り（単語帳、過去問の戦績、マイ教訓ノートの内容など）はすべて把握しています。\n今の率直な悩みや、「この志望校に届くか不安」といった目標について教えてくれる？一緒に最短ルートの作戦を立てよう！"}
                ]

            with st.container(border=True, height=500):
                for msg in st.session_state.compass_chat:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

            if user_req := st.chat_input("例：〇〇大学に受かりたいけど、何から手をつければいい？"):
                st.session_state.compass_chat.append({"role": "user", "content": user_req})
                
                with st.spinner("全データを横断分析し、戦略を構築中...（推論モデル使用）"):
                    # データを要約してAIに渡す（トークン溢れ防止）
                    vocab_count = len(frequency_strong_words) + sum([len(b.get("main_vocab", [])) for b in my_data.get('vocab_books', [])])
                    idiom_count = sum([len(b.get("idioms", [])) for b in my_data.get('idiom_books', [])])
                    strat_summary = "\n".join([f"- {s['title']}: {s['content']}" for s in my_data.get('strategy', [])[-15:]]) # 最新15件の教訓
                    
                    exam_summary = ""
                    for r in my_data.get('exam_records', [])[-10:]: # 最新10件の過去問
                        exam_summary += f"[{r.get('exam_name')}] 時間内:{r.get('score_in_time')}点 / 目標:{r.get('target_score')}点 / 実力点:{r.get('true_score', 0)}点 / 期待値:{r.get('expected_score', 0)}点\n"

                    sys_compass = f"""
                    あなたは「コンパス」と呼ばれる、受験生の最高峰の戦略AIメンターです。
                    生徒の全学習データにアクセスし、マクロな視点で戦略を決定・指導してください。
                    
                    【生徒の現在地データ】
                    ■ 単語のストック数: 約 {vocab_count} 語
                    ■ 熟語のストック数: 約 {idiom_count} 個
                    ■ 最近のマイ教訓ノート（弱点や気づき）:
                    {strat_summary}
                    ■ 過去問の戦績（直近）:
                    {exam_summary}

                    【指導ルール】
                    1. データに基づき、現状何が足りないか（語彙力か、文法力か、長文の精読力か、処理スピードか）を論理的に分析してください。
                    2. 「実力点」や「期待値得点」と「実際の得点」に乖離がある場合、「まぐれに頼っている」「選択肢は絞れているからあと一歩」など、具体的な伸びしろを示唆してください。
                    3. 一方的に長文を語るのではなく、対話形式で1つずつ課題を潰していくように優しく回答してください。説教は厳禁です。
                    """
                    
                    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.compass_chat[-4:]])
                    
                    # コンパスは戦略決定のため、常に賢い 2.5-pro を使用
                    ans = call_ai(f"会話:\n{history_str}", sys_compass, use_pdf=False, model_name="gemini-2.5-pro")
                    st.session_state.compass_chat.append({"role": "assistant", "content": ans})
                    st.rerun()

            if st.button("🔒 アクセス権をリセットして会議を終了する"):
                del st.session_state.compass_permission
                if "compass_chat" in st.session_state:
                    del st.session_state.compass_chat
                st.rerun()
