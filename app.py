import streamlit as st
import google.generativeai as genai
import tempfile
import os
import json
import re
import random
from collections import Counter
import nltk
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

# NLTKの辞書データをダウンロード（初回のみ裏で自動実行されます）
try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet')

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

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"vocabulary": [], "grammar": [], "strategy": [], "meta": []}

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_exam_db():
    if os.path.exists(EXAM_DB_FILE):
        with open(EXAM_DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

my_data = load_data()
exam_db = load_exam_db()

# --- 2. 設定とUI ---
st.set_page_config(page_title="自律型AI塾", page_icon="🧭", layout="wide")
st.sidebar.title("設定")

if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = st.sidebar.text_input("Gemini API Key", type="password")
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

mode = st.sidebar.radio("モード", ["💬 対話で分析", "☕ 学習の作戦会議", "🏠 マイ教訓ノート", "📖 志望校別単語帳", "🔗 志望校別熟語帳", "📝 志望校別文法・語法ノート", "🏆 過去問演習・合格分析"])

if "messages" not in st.session_state: st.session_state.messages = []
if "auto_insight" not in st.session_state: st.session_state.auto_insight = ""
if "current_quiz_question" not in st.session_state: st.session_state.current_quiz_question = ""
if "current_quiz_data" not in st.session_state: st.session_state.current_quiz_data = ""
if "quiz_chat_history" not in st.session_state: st.session_state.quiz_chat_history = []

# --- 3. AI呼び出し関数 ---
def call_ai(prompt, sys_msg, use_pdf=False, is_json=False, model_name="gemini-2.5-pro"):
    genai.configure(api_key=api_key)
    
    # モデルの指定（デフォルトは 2.5-pro、クイズ等で 3.1-flash-lite を渡せるようにする）
    if is_json:
        model = genai.GenerativeModel(model_name=model_name, system_instruction=sys_msg, generation_config={"response_mime_type": "application/json"})
    else:
        model = genai.GenerativeModel(model_name=model_name, system_instruction=sys_msg)

    if use_pdf and uploaded_pdf:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.getvalue())
            tmp_path = tmp.name
        g_file = genai.upload_file(tmp_path)
        res = model.generate_content([g_file, prompt])
        os.remove(tmp_path)
        return res.text
    else:
        res = model.generate_content(prompt)
        return res.text

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
# モードA: 対話で分析
# ==========================================
if mode == "💬 対話で分析":
    st.markdown("間違えた問題を教えてください。一緒に原因を探りましょう。")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if user_text := st.chat_input("例：第6問の問2を①にして間違えました"):
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"): st.markdown(user_text)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
                sys = "あなたは生徒の思考を引き出す塾講師です。PDFと対話履歴を見て、いきなり正解を教えず「なぜそう思った？」と2〜3文で問いかけてください。"
                response = call_ai(history_text, sys, use_pdf=True)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
                
    st.markdown("---")
    st.markdown("### 💡 気づきをストック")
    if st.button("✨ この対話から教訓を自動生成") and api_key:
        with st.spinner("教訓を要約中..."):
            history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
            sys_summary = "この対話履歴から、生徒が次に活かすべき教訓を1文（20文字程度）で簡潔に出力してください。"
            st.session_state.auto_insight = call_ai(history_text, sys_summary, use_pdf=False).strip()
            st.rerun()

    insight_in = st.text_input("教訓", value=st.session_state.auto_insight)
    if st.button("💾 保存する") and insight_in:
        my_data["meta"].append({"title": "対話からの気づき", "content": insight_in, "source": "対話分析"})
        save_data(my_data)
        st.session_state.auto_insight = ""
        st.success("✅ 教訓をノートに追加しました！")

# ==========================================
# モードB: 学習の作戦会議
# ==========================================
elif mode == "☕ 学習の作戦会議":
    st.markdown("今の勉強法や進捗を自由に報告してください。")
    report_in = st.text_area("例：ターゲット1900を1日1周しています。でも長文が遅いです。", height=100)
    if st.button("AIに報告・相談する") and api_key and report_in:
        with st.spinner("作戦を考え中..."):
            sys = "あなたは生徒の自主性を重んじるメンターです。生徒の学習報告を肯定し、さらに良くなるための具体的なアドバイスを1つだけ、2〜3文で提案してください。説教は禁止です。"
            st.info(call_ai(report_in, sys, use_pdf=False))

# ==========================================
# モードC: マイ教訓ノート
# ==========================================
elif mode == "🏠 マイ教訓ノート":
    st.title("🏠 究極のマイ教訓データベース")
    
    with st.form("manual_add_form"):
        st.markdown("#### ✨ 知識を構造化して登録（商用化セーフ版）")
        col1, col2 = st.columns([1, 2])
        with col1:
            category_options = {"🔤 語彙（単語・熟語）": "vocabulary", "📖 文法・構文": "grammar", "🧠 解法・読解戦略": "strategy", "⚠️ メタ認知・その他": "meta"}
            selected_cat = st.selectbox("カテゴリ", list(category_options.keys()))
            source_tag = st.text_input("出題元タグ（例：25年日大I-3）")
        with col2:
            item_title = st.text_input("項目名（例：pop, 倒置法）")
            item_content = st.text_area("意味・ルール（例：ひょっこり現れる, 否定の副詞が文頭に来るとVSになる）")
        
        if st.form_submit_button("💾 データベースにクリーン登録") and item_title and item_content:
            my_data[category_options[selected_cat]].append({"title": item_title, "content": item_content, "source": source_tag if source_tag else "タグなし"})
            save_data(my_data)
            st.success(f"「{item_title}」を追加しました！")
            st.rerun() 
            
    st.markdown("---")

    def display_entries(category_key, icon_name):
        with st.expander(icon_name, expanded=True):
            if my_data.get(category_key):
                for item in my_data[category_key]:
                    if isinstance(item, dict):
                        st.markdown(f"**{item['title']}** （🏷️ {item['source']}）\n↳ {item['content']}")
                    else:
                        st.markdown(f"- {item}")
            else:
                st.info("まだ登録されていません。")

    display_entries("vocabulary", "🔤 語彙（単語・熟語）")
    display_entries("grammar", "📖 文法・構文")
    display_entries("strategy", "🧠 解法・読解戦略（テクニック）")
    display_entries("meta", "⚠️ メタ認知・その他（メンタル・教訓）")

    st.markdown("---")
    
    col_a, col_b = st.columns([1, 1])
    with col_a: create_quiz = st.button("🔄 教訓から復習テストを作る", use_container_width=True)
    with col_b:
        if st.session_state.current_quiz_question and st.button("🗑️ テストを終了・クリア", use_container_width=True):
            st.session_state.current_quiz_question = ""
            st.session_state.current_quiz_data = ""
            st.session_state.quiz_chat_history = []
            st.rerun()

    if create_quiz and api_key:
        sys_gen = "ユーザーの教訓リストを踏まえて、シンプルで素直な『短い英語の和訳クイズ』または『穴埋めクイズ』を1問だけ出してください。ひっかけ問題や理不尽な問題は【絶対に】作らないでください。解説や答えはまだ書かないでください。"
        all_insights = my_data.get("vocabulary", []) + my_data.get("grammar", []) + my_data.get("strategy", []) + my_data.get("meta", [])
        insight_texts = [f"{i['title']}: {i['content']}" if isinstance(i, dict) else i for i in all_insights]
        prompt_text = f"私の教訓リスト: {', '.join(insight_texts)}"
        
        with st.spinner("クイズを生成中..."):
            st.session_state.current_quiz_question = call_ai(prompt_text, sys_gen, use_pdf=False)
            st.session_state.current_quiz_data = prompt_text
            st.session_state.quiz_chat_history = []
            st.rerun()

    if st.session_state.current_quiz_question:
        st.markdown("---")
        st.markdown("### 📝 今日の復習クイズ")
        st.info(st.session_state.current_quiz_question)
        st.markdown("#### 🗣️ AI先生との対話・添削")
        
        for chat in st.session_state.quiz_chat_history:
            with st.chat_message(chat["role"]): st.write(chat["content"])

        if user_input := st.chat_input("ここに回答や質問を入力..."):
            st.session_state.quiz_chat_history.append({"role": "user", "content": user_input})
            with st.spinner("AI先生が思考中..."):
                sys_grading = f"""
                あなたは優しくフレンドリーな伴走者です。プレッシャーは与えないでください。
                生徒の『回答』を見てフィードバックしてください。間違えても否定せずヒントを1つ出し、正解したら大げさに褒めてください。
                ■ 出題された問題: {st.session_state.current_quiz_question}
                ■ 教訓データ: {st.session_state.current_quiz_data}
                """
                history_context = "\n".join([f"{h['role']}: {h['content']}" for h in st.session_state.quiz_chat_history[-6:]])
                ai_response = call_ai(f"これまでの会話:\n{history_context}\n生徒: {user_input}", sys_grading, use_pdf=False)
                st.session_state.quiz_chat_history.append({"role": "assistant", "content": ai_response})
            st.rerun()

# ==========================================
# ★真の完全版 モードD: 志望校別単語帳（本棚 ＋ 任意AIフィルター ＋ シミュレーター）
# ==========================================
# ==========================================
# ★真の完全版 モードD: 志望校別単語帳（本棚 ＋ 任意AIフィルター ＋ シミュレーター）
# ==========================================
elif mode == "📖 志望校別単語帳":
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
            books = my_data.get("vocab_books", [])
            if not books:
                st.info("まだ単語帳がありません。「新しい単語帳を作る」タブから作成してください。")
            else:
                book_titles = [b["title"] for b in books]
                selected_title = st.selectbox("📖 管理・学習する単語帳を選択してください", ["-- 選択してください --"] + book_titles)
                
                if selected_title != "-- 選択してください --":
                    book_idx = book_titles.index(selected_title)
                    current_book = books[book_idx]
                    
                    st.markdown(f"### 📘 {current_book['title']}")
                    st.write(f"収録語数: メイン **{len(current_book['main_vocab'])}語** / 除外 **{len(current_book['excluded_vocab'])}語**")
                    
                    # --- 管理機能 ---
                    with st.expander("⚙️ 単語帳の管理・編集（名前変更・追加・マージ）", expanded=False):
                        st.markdown("#### 🏷️ 名前の変更")
                        col_rn1, col_rn2 = st.columns([3, 1])
                        new_title_input = col_rn1.text_input("新しい名前", value=current_book["title"], label_visibility="collapsed", key=f"rn_vocab_{book_idx}")
                        if col_rn2.button("名前を更新", use_container_width=True, key=f"btn_rn_vocab_{book_idx}"):
                            if new_title_input and new_title_input != current_book["title"]:
                                current_book["title"] = new_title_input
                                my_data["vocab_books"][book_idx] = current_book
                                save_data(my_data)
                                st.success("名前を変更しました！")
                                st.rerun()

                        st.markdown("---")
                        st.markdown("#### 📈 過去問データを追加して単語帳を強化 (マージ)")
                        add_labels = render_exam_selector(db_options, f"merge_vocab_{book_idx}")
                        
                        # ▼ AIフィルターのチェックボックスを追加
                        use_ai_filter_merge = st.checkbox("🤖 マージする新規単語から「人名」等をAIで除外する", value=False, key=f"ai_merge_v_{book_idx}")
                        
                        if st.button("✨ 選択したデータを追加 (マージ)", type="primary", key=f"btn_merge_vocab_{book_idx}") and add_labels:
                            with st.spinner("単語データを結合し、AIフィルターで審査しています..."):
                                current_counts = Counter(current_book.get("counts", {}))
                                current_origins = defaultdict(list, current_book.get("origins", {}))
                                
                                new_words_candidate = set() # AI審査用の新規単語候補
                                
                                for label in add_labels:
                                    path = db_options[label]
                                    freqs = exam_db[path["c"]][path["u"]][path["f"]][path["y"]][path["m"]]["frequencies"]
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
                                
                                my_data["vocab_books"][book_idx] = current_book
                                save_data(my_data)
                                st.success("データをマージしました！頻度順に再ソートされています。")
                                st.rerun()
                        
                        st.markdown("---")
                        st.markdown("#### 📋 複製と削除")
                        col_dup, col_del = st.columns(2)
                        if col_dup.button("📋 この単語帳を複製する", use_container_width=True):
                            new_book = current_book.copy()
                            new_book["title"] = current_book["title"] + " (コピー)"
                            my_data["vocab_books"].append(new_book)
                            save_data(my_data); st.rerun()
                            
                        if col_del.button("🗑️ この単語帳を削除する", use_container_width=True):
                            my_data["vocab_books"].pop(book_idx)
                            save_data(my_data); st.rerun()
                        
                        st.markdown("---")
                        st.markdown("#### ✏️ 単語の個別追加・削除")
                        col_edit1, col_edit2 = st.columns(2)
                        with col_edit1:
                            with st.form(f"add_word_form_{book_idx}"):
                                new_word = st.text_input("➕ 新しい単語を手動で追加")
                                if st.form_submit_button("追加する") and new_word:
                                    new_word = new_word.lower().strip()
                                    if new_word not in current_book["main_vocab"]:
                                        current_book["main_vocab"].insert(0, new_word)
                                        if new_word not in current_book.get("counts", {}): current_book.setdefault("counts", {})[new_word] = 1
                                        save_data(my_data); st.rerun()
                                    else:
                                        st.warning("登録済みです。")

                        with col_edit2:
                            with st.form(f"del_word_form_{book_idx}"):
                                del_word = st.selectbox("🗑️ 削除したい単語を選択", ["-- 選択 --"] + current_book["main_vocab"] + current_book["excluded_vocab"])
                                if st.form_submit_button("削除する") and del_word != "-- 選択 --":
                                    del_word = del_word.lower().strip()
                                    if del_word in current_book["main_vocab"]: current_book["main_vocab"].remove(del_word)
                                    elif del_word in current_book["excluded_vocab"]: current_book["excluded_vocab"].remove(del_word)
                                    save_data(my_data); st.rerun()
                        
                        st.markdown("---")
                        st.markdown("#### 🗑️ データの完全初期化")
                        if st.button("🗑️ AI生成データ(意味・例文等)をすべてリセットする", use_container_width=True):
                            if "enriched_vocab" in current_book: del current_book["enriched_vocab"]
                            if "skipped_vocab" in current_book: del current_book["skipped_vocab"]
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
                    
                    if missing_words:
                        st.info(f"💡 未生成の単語が **{len(missing_words)}** 個あります。（過去問マージ等で追加された単語です）")
                        
                        # --- 生成数指定UI ---
                        col_gen1, col_gen2 = st.columns([1, 2])
                        gen_count = col_gen1.number_input("生成する単語数", min_value=1, max_value=len(missing_words), value=min(30, len(missing_words)), step=10, key=f"gen_v_{book_idx}")
                        
                        col_gen2.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True) # ボタンの高さを合わせる
                        if col_gen2.button(f"✨ 未生成の単語を上位 {gen_count} 語生成して追加", use_container_width=True, type="primary"):
                            with st.spinner(f"AIがネイティブの脳内ネットワークを作成中...（上位 {gen_count} 語）"):
                                target_words = missing_words[:gen_count] 
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
                                    response_json = call_ai(f"処理対象:\n{target_words}", sys_enrich, is_json=True)
                                    parsed_data = json.loads(response_json)
                                    current_book["enriched_vocab"].extend(parsed_data.get("enriched", []))
                                    current_book["skipped_vocab"].extend(parsed_data.get("skipped", []))
                                    my_data["vocab_books"][book_idx] = current_book
                                    save_data(my_data)
                                    st.success("🎉 生成が完了しました！")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"生成エラー: {e}")
                    
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

            books = my_data.get("vocab_books", [])
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
                                                st.success(f"⭕ **正解！**")
                                            else:
                                                st.error(f"❌ **不正解** : あなたの解答「{status['user_ans']}」 ➔ 正解「{q['answer']}」")
                                        else:
                                            st.success(f"💡 正解は **{q['answer']}** です。")

                                        item = enriched_dict[word]
                                        st.info(f"**🔄 変化形:** {item.get('forms', '-')}\n\n**🎯 使い方:**\n" + "\n".join([f"- {c}" for c in item.get('chunks', [])]))

                                        
                                        
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
            
            if st.button("✨ 選択した過去問から単語帳を生成") and selected_labels:
                with st.spinner("単語を集計中..."):
                    combined_counter = Counter()
                    word_origins = defaultdict(list)
                    for label in selected_labels:
                        path = db_options[label]
                        freqs = exam_db[path["c"]][path["u"]][path["f"]][path["y"]][path["m"]]["frequencies"]
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
            
            books = my_data.get("vocab_books", [])
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
                                        if data.get("idioms"): # 🛡️ 空データ除外
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

                            path = target_options[selected_target]
                            target_data = exam_db[path["c"]][path["u"]][path["f"]][path["y"]][path["m"]]
                            raw_target_freqs = target_data["frequencies"]
                            target_freqs = process_frequencies(raw_target_freqs)

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
                                    res_quiz = call_ai(f"対象の熟語: {word}", "英語講師として指定熟語の4択問題を作成せよ。JSON出力: {\"question\": \"...\", \"options\": [\"...\"], \"answer\": \"...\", \"translation\": \"...\"}", is_json=True, model_name="gemini-3.1-flash-lite")
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
                            # 🚀 モデルを最新の 3.5-flash に固定して高速・高精度化
                            res_input = call_ai(f"対象熟語（必ずすべて解説すること）:\n{', '.join(target_idioms)}", sys_input, is_json=True, model_name="gemini-3.5-flash")
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
                            res_output = call_ai(f"熟語リスト:\n{', '.join(target_idioms)}", sys_output, is_json=True, model_name="gemini-3.5-flash")
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
                                        ans = call_ai(f"会話:\n{history_str}", sys_chat, use_pdf=False, model_name="gemini-3.5-flash")
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
    st.caption("過去問のエッセンスから作られたオリジナルドリルと、実践的な長文精読で文法をマスターします。")

    tab_drill, tab_reading = st.tabs(["📚 過去問オリジナルドリル", "📖 実践！長文精読アシスト"])

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

            # 最終的な出題候補
            final_candidates = []
            if selected_tag != "-- すべて --":
                final_candidates = [q for q in base_questions if selected_tag in q.get("primary_tags", [])]
            else:
                final_candidates = base_questions

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
                        user_ans = st.radio("選択してください:", q.get("options", []), key=f"g_choice_{idx}", disabled=is_answered)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        if not is_answered:
                            if st.button("📝 解答して解説を見る", key=f"g_ans_btn_{idx}", type="primary", use_container_width=True):
                                correct_ans = q.get("answer")
                                is_correct = (user_ans == correct_ans)
                                st.session_state.grammar_q_status[str(idx)] = {
                                    "user_ans": user_ans, 
                                    "is_correct": is_correct
                                }
                                
                                # 正答・誤答の履歴を保存（次回の出題確率に影響）
                                q_text = q.get("question", "")
                                if "grammar_stats" not in my_data: my_data["grammar_stats"] = {}
                                if q_text not in my_data["grammar_stats"]: my_data["grammar_stats"][q_text] = {"correct": 0, "incorrect": 0}
                                if is_correct: my_data["grammar_stats"][q_text]["correct"] += 1
                                else: my_data["grammar_stats"][q_text]["incorrect"] += 1
                                save_data(my_data)
                                
                                st.rerun()
                        
                        if is_answered:
                            status = st.session_state.grammar_q_status[str(idx)]
                            st.markdown("---")
                            if status["is_correct"]:
                                st.success(f"⭕ **正解！** : {q.get('answer')}")
                            else:
                                st.error(f"❌ **不正解** : あなたの解答「{status['user_ans']}」 ➔ 正解「{q.get('answer')}」")
                            
                            st.info(f"**💡 和訳:** {q.get('translation', '記載なし')}\n\n**📘 解説・背景知識:**\n{q.get('explanation', '記載なし')}")
                            
                            if st.button("💾 この文法知識を「マイ教訓ノート」に保存", key=f"g_save_note_{idx}"):
                                if "grammar" not in my_data: my_data["grammar"] = []
                                my_data["grammar"].append({"title": "文法ドリルからの教訓", "content": q.get('explanation', ''), "source": q.get('source', '')})
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
                                    ans = call_ai(f"会話:\n{history_str}", sys_chat, use_pdf=False, model_name="gemini-3.5-flash")
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
   # ------------------------------------------
    # タブ2: 実践！長文精読アシスト（裏メモリ搭載版）
    # ------------------------------------------
    with tab_reading:
        st.markdown("#### 📖 長文を1文ずつ精読し、実践的に文法を学ぶ")
        
        read_method = st.radio("題材となる長文の準備方法", ["🤖 AIにランダム生成してもらう", "📝 自分でテキストを貼り付ける", "📄 過去問PDFから抽出する"], horizontal=True)
        
        if read_method == "🤖 AIにランダム生成してもらう":
            st.info("💡 一番簡単なレベル（中学〜高校基礎）の単語のみを使って、純粋な構文把握に集中できる長文を生成します。")
            if st.button("✨ ランダムなテーマで長文を生成する", type="primary"):
                with st.spinner("AIがランダムなテーマで長文を書き下ろしています..."):
                    sys_gen_read = "難易度 CEFR A1〜A2 (中学〜高校基礎レベル) で、英語が苦手な高校生が基本構文を学ぶのに適した論理的な英語長文（150〜200語程度）を作成してください。テーマは毎回ランダムに。英語の本文のみを出力し、タイトルや改行は含めないでください。"
                    st.session_state.reading_target_text = call_ai("ランダムなテーマで長文を作成してください。", sys_gen_read, use_pdf=False, model_name="gemini-3.5-flash")
                    st.session_state.reading_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', st.session_state.reading_target_text.replace('\n', ' ')) if s.strip()]
                    st.session_state.reading_current_idx = 0
                    st.session_state.reading_chat_sentence_idx = -1 
                    st.session_state.temp_memo = []
                    st.session_state.show_word_selector = False
                    st.session_state.reading_global_memory = "" # 🚀 ここに裏メモリを初期化
                    st.rerun()
                    
        elif read_method == "📝 自分でテキストを貼り付ける":
            pasted = st.text_area("英語の長文を貼り付けてください", height=150)
            if st.button("✅ このテキストで精読を始める", type="primary") and pasted:
                st.session_state.reading_target_text = pasted
                st.session_state.reading_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', pasted.replace('\n', ' ')) if s.strip()]
                st.session_state.reading_current_idx = 0
                st.session_state.reading_chat_sentence_idx = -1
                st.session_state.temp_memo = []
                st.session_state.show_word_selector = False
                st.session_state.reading_global_memory = "" # 🚀 ここに裏メモリを初期化
                st.rerun()
                
        elif read_method == "📄 過去問PDFから抽出する":
            up_pdf = st.file_uploader("PDFをアップロード", type=["pdf"], key="reading_pdf")
            if st.button("🚀 PDFから長文を抽出する", type="primary") and up_pdf:
                with st.spinner("AIがPDFから長文部分だけを抽出しています..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(up_pdf.getvalue())
                        tmp_path = tmp.name
                    g_file = genai.upload_file(tmp_path)
                    model = genai.GenerativeModel(model_name="gemini-3.5-flash")
                    res = model.generate_content([g_file, "このPDFから英語の長文（本文）部分のみを抽出して出力してください。設問、選択肢、日本語の指示文は完全に除外してください。改行は極力なくしてください。"])
                    os.remove(tmp_path)
                    st.session_state.reading_target_text = res.text
                    st.session_state.reading_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', res.text.replace('\n', ' ')) if s.strip()]
                    st.session_state.reading_current_idx = 0
                    st.session_state.reading_chat_sentence_idx = -1
                    st.session_state.temp_memo = []
                    st.session_state.show_word_selector = False
                    st.session_state.reading_global_memory = "" # 🚀 ここに裏メモリを初期化
                    st.rerun()

        # 精読エリア
        if st.session_state.get("reading_target_text"):
            st.markdown("---")
            col_read, col_chat = st.columns([1, 1])
            
            sentences = st.session_state.get("reading_sentences", [])
            idx = st.session_state.get("reading_current_idx", 0)
            
            with col_read:
                st.markdown("##### 📄 全体マップ")
                with st.container(border=True, height=500):
                    display_html = ""
                    for i, s in enumerate(sentences):
                        if i == idx:
                            display_html += f"<span style='background-color: #ffeb3b; color: black; font-weight: bold; padding: 2px 4px; border-radius: 3px;'>{s}</span> "
                        else:
                            display_html += f"<span style='color: gray;'>{s}</span> "
                    st.markdown(f"<div style='line-height:1.8; font-size:1.1em;'>{display_html}</div>", unsafe_allow_html=True)
                
                if st.button("🗑️ 長文をクリアして別のものを読む", use_container_width=True):
                    for k in ["reading_target_text", "reading_sentences", "reading_current_idx", "reading_chat_logs", "reading_chat_sentence_idx", "temp_memo", "show_word_selector", "reading_global_memory"]:
                        if k in st.session_state: del st.session_state[k]
                    st.rerun()

            with col_chat:
                if idx < len(sentences):
                    current_sentence = sentences[idx]
                    st.markdown(f"##### 🎯 現在のターゲット ({idx+1}/{len(sentences)})")
                    st.info(f"**{current_sentence}**")
                    
                    if st.session_state.get("reading_chat_sentence_idx") != idx:
                        st.session_state.reading_chat_sentence_idx = idx
                        st.session_state.reading_chat_logs = [
                            {"role": "assistant", "content": "まずはこの文の**和訳**に挑戦してみて！間違えても全然大丈夫、一緒にどこが難しかったか考えていこうね。"}
                        ]
                        st.session_state.show_word_selector = False
                        st.session_state.temp_memo = []

                    st.markdown("##### 🤖 和訳チャレンジ＆伴走チャット")
                    with st.container(border=True, height=450):
                        for msg in st.session_state.reading_chat_logs:
                            with st.chat_message(msg["role"]): st.markdown(msg["content"])
                    
                    if user_req := st.chat_input("和訳を入力、または質問する...", key="reading_chat"):
                        st.session_state.reading_chat_logs.append({"role": "user", "content": user_req})
                        with st.spinner("AI講師が優しく添削中..."):
                            sys_reading = f"""
                            あなたは優しく、生徒に寄り添う英語の伴走講師です。スパルタや説教は絶対にやめてください。
                            
                            【これまでの長文全体の対話履歴（あなたはこれを記憶しています）】
                            {st.session_state.get('reading_global_memory', 'まだありません')}

                            【現在のターゲット文】
                            {current_sentence}
                            
                            【指導のルール（A1〜A2レベル向け）】
                            1. 難しい文法用語は避け、【英語の構文ルール（カタチ・公式）】をパズルのように視覚的に教えてください。
                            2. 以前の文で登場した話題（裏メモリ参照）と関連があれば、「前の文でも出てきたね」と触れてあげると生徒は喜びます。
                            3. 生徒が「わからない」と言った場合は、いきなり全訳を教えず、構文のカタチを先に示し、そこに単語を当てはめさせるヒントを出してください。
                            """
                            history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.reading_chat_logs[-4:]])
                            ans = call_ai(f"直近の会話:\n{history_str}", sys_reading, use_pdf=False, model_name="gemini-3.5-flash")
                            st.session_state.reading_chat_logs.append({"role": "assistant", "content": ans})
                            st.rerun()

                    st.markdown("---")
                    
                    col_btn1, col_btn2 = st.columns(2)
                    if col_btn1.button("❓ わからない単語・熟語がある", use_container_width=True):
                        st.session_state.show_word_selector = not st.session_state.get("show_word_selector", False)
                        st.rerun()
                        
                    if col_btn2.button("⏭️ 完璧！スキップして次へ", use_container_width=True, type="primary"):
                        # 🚀 次の文へ行く前に、今の会話を裏メモリに退避させて記憶させる
                        current_chat = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.reading_chat_logs if m['role'] == 'user' or m['role'] == 'assistant'])
                        st.session_state.reading_global_memory += f"\n【文 {idx+1}: {current_sentence} の対話】\n{current_chat}\n"
                        
                        st.session_state.reading_current_idx += 1
                        st.rerun()

                    # 単語・熟語検索モード
                    if st.session_state.get("show_word_selector"):
                        st.markdown("###### 👆 調べたい単語をクリック、または熟語を検索！")
                        words = [w for w in re.findall(r"\b[a-zA-Z\-']+\b", current_sentence)]
                        cols = st.columns(6)
                        for i, w in enumerate(words):
                            if cols[i % 6].button(w, key=f"wbtn_{i}_{idx}"):
                                with st.spinner(f"「{w}」の意味を検索中..."):
                                    sys_dict = "あなたは辞書です。指定された単語の、この文脈における意味を簡潔に（10文字以内で）答えてください。"
                                    meaning = call_ai(f"文脈: {current_sentence}\n単語: {w}", sys_dict, use_pdf=False, model_name="gemini-3.5-flash")
                                    st.session_state.temp_memo.append({"word": w, "meaning": meaning.strip()})
                                    st.rerun()
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        col_idiom1, col_idiom2 = st.columns([3, 1])
                        search_idiom = col_idiom1.text_input("熟語・フレーズの検索", label_visibility="collapsed", placeholder="調べたい熟語を入力 (例: take care of)")
                        if col_idiom2.button("🔍 検索", key=f"search_idiom_btn_{idx}", use_container_width=True):
                            if search_idiom:
                                with st.spinner("検索中..."):
                                    sys_dict = "あなたは辞書です。指定された熟語・フレーズの、この文脈における意味を簡潔に答えてください。"
                                    meaning = call_ai(f"文脈: {current_sentence}\n熟語: {search_idiom}", sys_dict, use_pdf=False, model_name="gemini-3.5-flash")
                                    st.session_state.temp_memo.append({"word": search_idiom, "meaning": meaning.strip()})
                                    st.rerun()
                                    
                    # 一時メモ表示
                    if st.session_state.get("temp_memo"):
                        st.markdown("###### 📝 単語・熟語の一時メモ")
                        with st.container(border=True):
                            for m in st.session_state.temp_memo:
                                st.markdown(f"- **{m['word']}**: {m['meaning']}")

                    st.markdown("---")
                    st.markdown("##### 💡 学びをストック")
                    with st.form("reading_memo_form", clear_on_submit=True):
                        note_title = st.text_input("📝 項目名（例：関係副詞whereの非制限用法）")
                        note_content = st.text_area("意味・ルール（AIの解説や単語メモを保存）", height=80)
                        if st.form_submit_button("🏠 マイ教訓ノート(文法)に保存", type="primary"):
                            if note_title and note_content:
                                if "grammar" not in my_data: my_data["grammar"] = []
                                my_data["grammar"].append({"title": note_title, "content": note_content, "source": "長文精読アシスト"})
                                save_data(my_data)
                                st.success(f"教訓ノートに保存しました！")
                            else:
                                st.error("項目名とルールの両方を入力してください。")
                                
                    with st.expander("📔 学習途中にマイ教訓ノートを参照する"):
                        if my_data.get("grammar"):
                            for item in my_data["grammar"]:
                                st.markdown(f"**{item['title']}**\n> {item['content']}")
                        else:
                            st.info("ノートはまだ空です。")

                else:
                    st.success("🎉 全ての文の精読が完了しました！お疲れ様でした。")
                    if st.button("🔄 長文をクリアする", use_container_width=True):
                        for k in ["reading_target_text", "reading_sentences", "reading_current_idx", "reading_chat_logs", "reading_chat_sentence_idx", "temp_memo", "show_word_selector", "reading_global_memory"]:
                            if k in st.session_state: del st.session_state[k]
                        st.rerun()

# ==========================================
# ★最終章 モードG: 過去問演習・合格分析（長文＋スコア管理＋コンパス）
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
                chat_model_choice = st.radio("🧠 分析モデル", ["推論モデル (2.5 Pro) - 深掘り", "高速モデル (3.5 Flash)"], horizontal=True, label_visibility="collapsed")
                
                with st.container(border=True, height=500):
                    for msg in st.session_state.exam_review_chat:
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])
                
                if user_req := st.chat_input("例：問1は②にした。なぜ他の選択肢がダメなの？", key="review_chat_in"):
                    st.session_state.exam_review_chat.append({"role": "user", "content": user_req})
                    with st.spinner("AI講師が分析中..."):
                        selected_model_name = "gemini-2.5-pro" if "Pro" in chat_model_choice else "gemini-3.5-flash"
                        
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
        vocab_books = [b["title"] for b in my_data.get("vocab_books", [])]
        idiom_books = [b["title"] for b in my_data.get("idiom_books", [])]

        col_w1, col_w2, col_w3 = st.columns([2, 2, 1.5])
        sel_vocab = col_w1.selectbox("⚔️ 装備する単語帳", ["-- なし --"] + vocab_books)
        sel_idiom = col_w2.selectbox("⚔️ 装備する熟語帳", ["-- なし --"] + idiom_books)
        
        
        col_w3.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        use_grammar = col_w3.checkbox("⚔️ 過去問DBの文法知識を装備", value=True, help="過去問データベースから抽出・蓄積した「必須知識タグ」をAIに持たせます")

        st.markdown("##### ⚙️ 本番環境のデバフ（ペナルティ）設定")
        col_p1, col_p2, col_p3 = st.columns(3)
        careless_rate = col_p1.slider("😰 ケアレスミス率 (実力点ロス)", 0, 30, 5, format="%d%%", help="1択まで絞り切れた問題でも、マークミス等で落としてしまう確率") / 100.0
        panic_rate = col_p2.slider("🌀 焦り失点率 (期待値ロス)", 0, 50, 20, format="%d%%", help="2択等に絞れても、プレッシャーで本来の確率通りに点が取れないロス率") / 100.0
        timeout_rate = col_p3.slider("⏳ 時間切れ(塗り絵)率", 0, 50, 10, format="%d%%", help="時間が足りず、問題を読めずに完全な勘（4択=25%の確率）でマークする割合") / 100.0

        exam_text_sim = st.text_area("📄 仮想受験させる過去問のテキスト（長文＋設問＋選択肢）", height=200, placeholder="ここに解かせたい過去問のテキストを貼り付けてください")

        if st.button("🚀 クローンAIに仮想受験させる", type="primary", use_container_width=True):
            if not exam_text_sim:
                st.error("過去問のテキストを入力してください。")
            else:
                with st.spinner("あなたの知識セットをAIにコピーし、仮想受験を実行中...（約20〜40秒）"):
                    # 知識の抽出（プロンプト制限回避のため上位を抽出）
                    known_words = []
                    if sel_vocab != "-- なし --":
                        known_words = next((b["main_vocab"] for b in my_data["vocab_books"] if b["title"] == sel_vocab), [])
                    known_idioms = []
                    if sel_idiom != "-- なし --":
                        known_idioms = [i["base_form"] for i in next((b["idioms"] for b in my_data["idiom_books"] if b["title"] == sel_idiom), [])]
                    
                    # ▼▼ 修正: 過去問DBから文法タグ（必須知識）を全取得 ▼▼
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
                          "narrowed_down_to": 2, 
                          "reasoning": "〇〇という単語はリストになく未知語だったが、前後の文からプラスの意味だと推測。また、文法テーマ「仮定法」は習得済みのため形から選択肢1と3を消去し、2択に絞った。"
                        }}
                      ]
                    }}
                    ※ `narrowed_down_to` は、既知の語彙・文法知識と文脈推測で「何択まで絞れたか」を整数で出力（完全に自信があれば1、2択なら2、全く分からなければ4など）。
                    """

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
                    vocab_count = sum([len(b.get("main_vocab", [])) for b in my_data.get('vocab_books', [])])
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