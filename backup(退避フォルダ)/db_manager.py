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
        col_save, col_cancel = st.columns(2)
        
        if col_save.button("💾 2. この内容でデータベースに登録", type="primary"):
            if not (cat_val and uni_val and fac_val and year_val and method_val):
                st.error("階層をすべて入力してください")
            else:
                if cat_val not in db: db[cat_val] = {}
                if uni_val not in db[cat_val]: db[cat_val][uni_val] = {}
                if fac_val not in db[cat_val][uni_val]: db[cat_val][uni_val][fac_val] = {}
                if year_val not in db[cat_val][uni_val][fac_val]: db[cat_val][uni_val][fac_val][year_val] = {}
                
                new_words = extract_words_from_text(edited_text)
                word_counts = Counter(new_words)
                
                if method_val in db[cat_val][uni_val][fac_val][year_val]:
                    existing = db[cat_val][uni_val][fac_val][year_val][method_val]
                    merged = Counter(existing["frequencies"]) + word_counts
                    db[cat_val][uni_val][fac_val][year_val][method_val] = {
                        "total_words": existing["total_words"] + len(new_words),
                        "unique_words": len(merged),
                        "frequencies": dict(merged.most_common())
                    }
                else:
                    db[cat_val][uni_val][fac_val][year_val][method_val] = {
                        "total_words": len(new_words),
                        "unique_words": len(word_counts),
                        "frequencies": dict(word_counts.most_common())
                    }
                save_db(db)
                st.success("登録完了！")
                st.session_state.draft_text = None
                st.rerun()
        
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

            # --- 復活箇所2: 単語個別削除と中身クリア ---
            col_del, col_clear = st.columns(2)
            with col_del:
                st.markdown("#### 🗑️ 不要な単語を個別削除")
                del_word = st.text_input("削除したい単語を入力")
                if st.button("この単語をリストから消す") and del_word:
                    if del_word in target_data["frequencies"]:
                        target_data["total_words"] -= target_data["frequencies"][del_word]
                        del target_data["frequencies"][del_word]
                        target_data["unique_words"] = len(target_data["frequencies"])
                        save_db(db)
                        st.success(f"「{del_word}」を削除しました")
                        st.rerun()
                    else:
                        st.error("その単語は見つかりませんでした")
            
            with col_clear:
                st.markdown("#### 🧹 データの中身を空にする")
                if st.button("単語だけ全て消去（階層の枠は残す）"):
                    target_data["frequencies"] = {}
                    target_data["total_words"] = 0
                    target_data["unique_words"] = 0
                    save_db(db)
                    st.warning("中身を空にしました")
                    st.rerun()

            st.markdown("#### 🏆 頻出単語ランキング")
            st.write(f"総語数: **{target_data['total_words']} 語** / 種類: **{target_data['unique_words']} 種類**")
            top_words = [{"単語": k, "回数": v} for k, v in list(target_data['frequencies'].items())[:200]]
            st.dataframe(pd.DataFrame(top_words), use_container_width=True)