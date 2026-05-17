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


def extract_idioms_with_gemini(text):
    """AIを使って長文から熟語・イディオムを抽出し、JSONで返す関数"""
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
        return json.loads(res.text)
    except Exception as e:
        st.error(f"熟語の抽出中にエラーが発生しました: {e}")
        return {"idioms": []}
    

def extract_grammar_with_gemini(text):
    """AIを使って長文・文法問題から設問と文法要素を抽出し、JSONで返す関数"""
    sys_prompt = """
    あなたは大学受験英語のプロ予備校講師であり、入試問題の緻密な分析官です。
    提供された過去問のテキストから、必ず【設問として問われている問題（空所補充、整序問題、下線部言い換え、内容一致問題など）】だけを抽出し、以下のJSON形式で出力してください。

    【絶対ルール：抽出対象の制限】
    長文の「本文（地の文）」を勝手に文法解析してはいけません。必ず「選択肢が用意されている設問」だけを対象としてください。

    【絶対ルール：タグの統一（表記揺れの防止）】
    分析結果の集計を正確に行うため、問題の核となる文法・語法テーマを `primary_tags`（配列）として出力してください。
    タグは【必ず】以下の標準カテゴリからのみ選択してください（勝手に新しいタグを作らないこと）。
    [標準カテゴリ]: 時制, 助動詞, 仮定法, 受動態, 不定詞, 動名詞, 分詞, 分詞構文, 関係詞, 比較, 接続詞, 前置詞, 名詞・代名詞・冠詞, 形容詞・副詞, 動詞の語法, 無生物主語, 倒置・省略・強調, 否定, 内容一致・要旨把握, 指示語把握

    【絶対ルール：解説の書き方】
    `explanation` の中に、「なぜそれが正解なのか（正解の根拠）」と、「なぜ他の選択肢はダメなのか（不正解の除外プロセス）」の両方を具体的に記述してください。

    【JSONフォーマット】
    {
      "grammar_questions": [
        {
          "question": "The man (      ) I thought was a doctor turned out to be a teacher.",
          "options": ["who", "whose", "whom", "which"],
          "answer": "whom",
          "translation": "私が医者だと思っていたその男は、実は教師だった。",
          "primary_tags": ["関係詞"],
          "explanation": "【正解の根拠】I thought が挿入された連鎖関係代名詞の構造であり、目的語が欠落しているため目的格 whom が入る。\\n【不正解の除外】whoは主格のため不可。whichは先行詞が人のため不可。whoseは後ろに名詞が必要なため不可。"
        }
      ]
    }
    """
    model = genai.GenerativeModel(model_name="gemini-2.5-pro", system_instruction=sys_prompt, generation_config={"response_mime_type": "application/json"})
    try:
        res = model.generate_content(f"以下のテキストから設問を抽出し、分析してください:\n\n{text}")
        return json.loads(res.text)
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
                                merged_idioms[base_form]["quotes"] = list(set(merged_idioms[base_form].get("quotes", []) + item["quotes_in_text"]))
                            else:
                                merged_idioms[base_form] = {"count": item["count"], "quotes": item["quotes_in_text"]}
                                
                        target_db["idioms"] = merged_idioms

                    # -----------------------------------------
                    # ルートC: 文法・語法の抽出（AI処理）
                    # -----------------------------------------
                    if ext_grammar:
                        extracted_grammar_data = extract_grammar_with_gemini(edited_text)
                        
                        merged_grammar_questions = target_db.get("grammar_questions", [])
                        merged_grammar_tags = Counter(target_db.get("grammar_tags", {}))
                        
                        for q in extracted_grammar_data.get("grammar_questions", []):
                            merged_grammar_questions.append(q)
                            for tag in q.get("required_knowledge", []):
                                merged_grammar_tags[tag] += 1
                                
                        target_db["grammar_questions"] = merged_grammar_questions
                        target_db["grammar_tags"] = dict(merged_grammar_tags)

                    # データベースを上書き保存
                    save_db(db)
                    st.success(f"🎉 登録完了！ ({', '.join(executed_tasks)}を抽出・統合しました)")
                    st.session_state.draft_text = None
                    st.rerun()
                    
                    # -----------------------------------------
                    # ルートB: 熟語の抽出（AI処理）
                    # -----------------------------------------
                    if extract_target in ["単語と熟語を両方抽出", "熟語だけ抽出 (AI使用)"]:
                        extracted_idioms_data = extract_idioms_with_gemini(edited_text)
                        merged_idioms = target_db.get("idioms", {})
                        
                        for item in extracted_idioms_data.get("idioms", []):
                            base_form = item["base_form"]
                            if base_form in merged_idioms:
                                merged_idioms[base_form]["count"] += item["count"]
                                # 引用リストの重複を省いて結合
                                merged_idioms[base_form]["quotes"] = list(set(merged_idioms[base_form].get("quotes", []) + item["quotes_in_text"]))
                            else:
                                merged_idioms[base_form] = {"count": item["count"], "quotes": item["quotes_in_text"]}
                                
                        target_db["idioms"] = merged_idioms

                    # データベースを上書き保存
                    save_db(db)
                    st.success(f"登録完了！ ({extract_target})")
                    st.session_state.draft_text = None
                    st.rerun()

                    # -----------------------------------------
                    # ルートC: 文法の抽出（AI処理）
                    # -----------------------------------------
                    if extract_target == "文法・語法問題の抽出と分析 (AI使用)":
                        extracted_grammar_data = extract_grammar_with_gemini(edited_text)
                        
                        # 既存の文法データとタグ集計の取得（無ければ初期化）
                        merged_grammar_questions = target_db.get("grammar_questions", [])
                        merged_grammar_tags = Counter(target_db.get("grammar_tags", {}))
                        
                        for q in extracted_grammar_data.get("grammar_questions", []):
                            merged_grammar_questions.append(q)
                            # 修正: 統一された標準カテゴリ(primary_tags)をカウントする
                            for tag in q.get("primary_tags", []):
                                merged_grammar_tags[tag] += 1
                                
                        target_db["grammar_questions"] = merged_grammar_questions
                        target_db["grammar_tags"] = dict(merged_grammar_tags)
        
        if col_cancel.button("🗑️ キャンセル"):
            st.session_state.draft_text = None
            st.rerun()
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
                        "本文中での使われ方 (Quote)": " / ".join(data["quotes"])
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
                        tags = q.get("primary_tags", q.get("required_knowledge", []))
                        for t in tags:
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