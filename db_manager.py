import streamlit as st
import google.generativeai as genai
import tempfile
import re
import json
import os
from collections import Counter

# --- 設定 ---
DB_FILE = "past_exams_db.json"

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("APIキーが見つかりません。.streamlit/secrets.toml を確認してください。")

STOP_WORDS = {
    "a", "an", "the", "and", "but", "or", "for", "nor", "on", "at", "to", "from", "by", 
    "is", "are", "was", "were", "am", "be", "been", "being", 
    "in", "of", "with", "as", "it", "this", "that", "these", "those",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "their", "our", "its", "which", "who", "whom", "whose",
    "have", "has", "had", "do", "does", "did", "can", "will", "would", "could", "should",
    "not", "no", "if", "then", "than", "so", "very", "too", "all", "any", "some"
}

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_db(db):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

# --- AIによるPDF抽出 ---
def extract_words_with_gemini(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        g_file = genai.upload_file(tmp_path)
        model = genai.GenerativeModel(model_name="gemini-2.5-pro")
        prompt = """
        このPDFファイル（英語の試験問題）から、英語の単語のみを抽出してください。
        【絶対ルール】
        1. 日本語の文字、記号、数字、段落番号などはすべて無視すること。
        2. 抽出した英単語をすべて小文字にし、スペース区切りで出力すること。
        3. 余計な説明や挨拶は一切不要。英単語の羅列だけを返すこと。
        """
        res = model.generate_content([g_file, prompt])
        words = re.findall(r'\b[a-z]+\b', res.text.lower())
        filtered_words = [word for word in words if word not in STOP_WORDS and len(word) > 1]
        return filtered_words
    finally:
        os.remove(tmp_path)

def extract_words_from_text(text):
    words = re.findall(r'\b[a-z]+\b', text.lower())
    filtered_words = [word for word in words if word not in STOP_WORDS and len(word) > 1]
    return filtered_words

# --- 🌟 UI用ヘルパー：選択 or 新規作成 ---
def select_or_create(label, options, key):
    # ★ 既存の選択肢を先に表示し、最後に「+ 新規作成」を配置
    choices = list(options) + ["+ 新規作成"]
    selected = st.selectbox(label, choices, key=key+"_sel")
    
    # ★ 「+ 新規作成」が選ばれた時だけ入力ボックスを出す
    if selected == "+ 新規作成":
        return st.text_input(f"🆕 新しい{label}を入力", key=key+"_new")
    return selected

# ==========================================
# UI 画面構築
# ==========================================
st.set_page_config(page_title="過去問DBマネージャー", page_icon="📂", layout="wide")
st.title("📂 過去問データベース管理システム")

tab1, tab2 = st.tabs(["📥 データ登録・追加", "🔍 データベース閲覧・編集"])

db = load_db()

# ------------------------------------------
# タブ1: データ登録・追加（UI大刷新）
# ------------------------------------------
with tab1:
    st.markdown("既存の階層は自動で選択されます。新しい分岐を作りたい時だけドロップダウンから「+ 新規作成」を選んでください。")
    
    input_method = st.radio("データの入力方法を選んでください", ["PDFをアップロード (AI抽出)", "テキストを直接貼り付け (正確な検証用)"], horizontal=True)
    
    st.markdown("#### 📂 登録先（階層）の指定")
    
    # ★ フォーム（st.form）の外に出すことで、ドロップダウンの連動（カスケーディング）を即座に反映させる
    col1, col2, col3 = st.columns(3)
    with col1:
        cats = list(db.keys())
        cat_val = select_or_create("カテゴリ", cats, "c")
        
        unis = list(db.get(cat_val, {}).keys()) if cat_val and cat_val != "+ 新規作成" else []
        uni_val = select_or_create("大学名", unis, "u")
        
    with col2:
        facs = list(db.get(cat_val, {}).get(uni_val, {}).keys()) if uni_val and uni_val != "+ 新規作成" else []
        fac_val = select_or_create("学部", facs, "f")
        
        # ★ 年度は打つのではなく選ぶ（固定リスト）
        year_options = [str(y) for y in range(2030, 1990, -1)]
        default_index = year_options.index("2026") if "2026" in year_options else 0
        year_val = st.selectbox("年度", year_options, index=default_index, key="y_sel")
        
    with col3:
        methods = list(db.get(cat_val, {}).get(uni_val, {}).get(fac_val, {}).get(year_val, {}).keys()) if fac_val and fac_val != "+ 新規作成" else []
        method_val = select_or_create("方式・日程", methods, "m")

    st.markdown("---")
    
    # ★ データ入力と保存ボタンだけをフォームにする
    with st.form("upload_form"):
        st.markdown("#### 📄 データの入力")
        
        if input_method == "PDFをアップロード (AI抽出)":
            uploaded_pdf = st.file_uploader("過去問PDFをアップロード", type=["pdf"])
            pasted_text = ""
        else:
            uploaded_pdf = None
            pasted_text = st.text_area("テキストを貼り付けてください", height=150)
            
        submitted = st.form_submit_button("📊 抽出してデータベースに保存・追加", use_container_width=True)
        
        if submitted:
            if not (cat_val and uni_val and fac_val and year_val and method_val):
                st.error("階層の名前をすべて入力または選択してください！")
            elif input_method == "PDFをアップロード (AI抽出)" and not uploaded_pdf:
                st.error("PDFをアップロードしてください。")
            elif input_method == "テキストを直接貼り付け (正確な検証用)" and not pasted_text:
                st.error("テキストを貼り付けてください。")
            else:
                with st.spinner("解析・集計中..."):
                    # 階層の確保
                    if cat_val not in db: db[cat_val] = {}
                    if uni_val not in db[cat_val]: db[cat_val][uni_val] = {}
                    if fac_val not in db[cat_val][uni_val]: db[cat_val][uni_val][fac_val] = {}
                    if year_val not in db[cat_val][uni_val][fac_val]: db[cat_val][uni_val][fac_val][year_val] = {}
                    
                    if uploaded_pdf:
                        new_words = extract_words_with_gemini(uploaded_pdf)
                    else:
                        new_words = extract_words_from_text(pasted_text)
                        
                    # 既存データがあればマージ（合体）する
                    if method_val in db[cat_val][uni_val][fac_val][year_val]:
                        existing_data = db[cat_val][uni_val][fac_val][year_val][method_val]
                        current_counter = Counter(existing_data["frequencies"])
                        merged_counter = current_counter + Counter(new_words)
                        
                        db[cat_val][uni_val][fac_val][year_val][method_val] = {
                            "total_words": existing_data["total_words"] + len(new_words),
                            "unique_words": len(merged_counter),
                            "frequencies": dict(merged_counter.most_common())
                        }
                        st.success(f"✅ 既存の「{method_val}」にデータを追加（マージ）しました！ 追加語数: {len(new_words)}語")
                    else:
                        # 新規作成
                        word_freq = dict(Counter(new_words).most_common())
                        db[cat_val][uni_val][fac_val][year_val][method_val] = {
                            "total_words": len(new_words),
                            "unique_words": len(word_freq),
                            "frequencies": word_freq
                        }
                        st.success(f"✅ 新規登録完了！ 抽出された総単語数: {len(new_words)}語")
                    
                    save_db(db)
# ------------------------------------------
# タブ2: データベース閲覧・編集
# ------------------------------------------
with tab2:
    if not db:
        st.info("まだデータベースに何も登録されていません。")
    else:
        col_c, col_u, col_f, col_y, col_m = st.columns(5)
        with col_c: sel_cat = st.selectbox("📁 カテゴリ", list(db.keys()), key="v_c")
        with col_u:
            uni_list = list(db[sel_cat].keys()) if sel_cat else []
            sel_uni = st.selectbox("🏫 大学名", uni_list, key="v_u") if uni_list else None
        with col_f:
            fac_list = list(db[sel_cat][sel_uni].keys()) if sel_uni else []
            sel_fac = st.selectbox("📚 学部", fac_list, key="v_f") if fac_list else None
        with col_y:
            year_list = list(db[sel_cat][sel_uni][sel_fac].keys()) if sel_fac else []
            sel_year = st.selectbox("🗓️ 年度", year_list, key="v_y") if year_list else None
        with col_m:
            method_list = list(db[sel_cat][sel_uni][sel_fac][sel_year].keys()) if sel_year else []
            sel_method = st.selectbox("📝 方式", method_list, key="v_m") if method_list else None
            
        st.markdown("---")
        
        if sel_method:
            target_data = db[sel_cat][sel_uni][sel_fac][sel_year][sel_method]
            st.markdown(f"### 📊 【{sel_uni} {sel_fac}】{sel_year}年 {sel_method} のデータ")
            
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
                        del db[sel_cat][sel_uni][sel_fac][sel_year][sel_method]
                        if not db[sel_cat][sel_uni][sel_fac][sel_year]: del db[sel_cat][sel_uni][sel_fac][sel_year]
                        if not db[sel_cat][sel_uni][sel_fac]: del db[sel_cat][sel_uni][sel_fac]
                        if not db[sel_cat][sel_uni]: del db[sel_cat][sel_uni]
                        if not db[sel_cat]: del db[sel_cat]
                        
                        if e_cat not in db: db[e_cat] = {}
                        if e_uni not in db[e_cat]: db[e_cat][e_uni] = {}
                        if e_fac not in db[e_cat][e_uni]: db[e_cat][e_uni][e_fac] = {}
                        if e_year not in db[e_cat][e_uni][e_fac]: db[e_cat][e_uni][e_fac][e_year] = {}
                        db[e_cat][e_uni][e_fac][e_year][e_method] = data_to_move
                        save_db(db)
                        st.success("✅ 移動・変更が完了しました！")
                        st.rerun()
                        
                    if col_d.form_submit_button("🗑️ この階層ごと削除"):
                        del db[sel_cat][sel_uni][sel_fac][sel_year][sel_method]
                        if not db[sel_cat][sel_uni][sel_fac][sel_year]: del db[sel_cat][sel_uni][sel_fac][sel_year]
                        if not db[sel_cat][sel_uni][sel_fac]: del db[sel_cat][sel_uni][sel_fac]
                        if not db[sel_cat][sel_uni]: del db[sel_cat][sel_uni]
                        if not db[sel_cat]: del db[sel_cat]
                        save_db(db)
                        st.success("🗑️ 階層を削除しました。")
                        st.rerun()

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
                        st.success(f"「{del_word}」を削除しました。")
                        st.rerun()
                    else:
                        st.error("その単語は見つかりませんでした。")
            with col_clear:
                st.markdown("#### 🧹 データの中身を空にする")
                if st.button("単語だけ全て消去（階層の枠は残す）"):
                    target_data["frequencies"] = {}
                    target_data["total_words"] = 0
                    target_data["unique_words"] = 0
                    save_db(db)
                    st.warning("中身を空にしました。")
                    st.rerun()

            st.markdown("#### 🏆 頻出単語ランキング")
            st.write(f"総単語数: **{target_data['total_words']} 語** / 種類: **{target_data['unique_words']} 種類**")
            top_words = [{"単語": k, "出現回数": v} for k, v in list(target_data['frequencies'].items())[:200]]
            st.dataframe(top_words, use_container_width=True)