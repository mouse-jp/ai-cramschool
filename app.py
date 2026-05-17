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

mode = st.sidebar.radio("モード", ["💬 対話で分析", "☕ 学習の作戦会議", "🏠 マイ教訓ノート", "📖 志望校別単語帳", "🔗 志望校別熟語帳"])

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
elif mode == "📖 志望校別単語帳":
    st.markdown("あなた専用の単語帳を作成し、本棚で管理します。")
    
    if not exam_db:
        st.warning("過去問データベースが空です。まずは db_manager.py を起動して過去問を登録してください。")
    else:
        db_options = {}
        for cat, unis in exam_db.items():
            for uni, facs in unis.items():
                for fac, years in facs.items():
                    for year, methods in years.items():
                        for method in methods.keys():
                            label = f"[{cat}] {uni} {fac} ({year}年 {method})"
                            db_options[label] = {"c": cat, "u": uni, "f": fac, "y": year, "m": method}
                            
        tab_shelf, tab_create, tab_sim = st.tabs(["📚 あなたの本棚", "✨ 新しい単語帳を作る", "📊 カバー率・難化シミュレーター"])
        
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
                        add_labels = st.multiselect("追加したい過去問を選んでください", list(db_options.keys()), key=f"merge_vocab_{book_idx}")
                        if st.button("✨ 選択したデータを追加 (マージ)", type="primary", key=f"btn_merge_vocab_{book_idx}") and add_labels:
                            with st.spinner("単語データを結合し、頻度を再計算しています..."):
                                current_counts = Counter(current_book.get("counts", {}))
                                current_origins = defaultdict(list, current_book.get("origins", {}))
                                
                                for label in add_labels:
                                    path = db_options[label]
                                    freqs = exam_db[path["c"]][path["u"]][path["f"]][path["y"]][path["m"]]["frequencies"]
                                    current_counts.update(freqs)
                                    short_label = f"{str(path['y'])[-2:]}年"
                                    for w, count in freqs.items():
                                        current_origins[w].append(f"{short_label}({count}回)")
                                
                                current_book["counts"] = dict(current_counts)
                                current_book["origins"] = dict(current_origins)
                                
                                excluded_set = set(current_book.get("excluded_vocab", []))
                                main_set = set(current_book.get("main_vocab", []))
                                for w in current_counts.keys():
                                    if w not in excluded_set and w not in main_set:
                                        main_set.add(w)
                                        
                                # 新しい頻度でソート（※AIデータは消さずに保持する）
                                current_book["main_vocab"] = sorted(list(main_set), key=lambda x: current_counts[x], reverse=True)
                                
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
                            st.markdown("#### 🎯 実践！穴埋め4択クイズ (Flash-Lite搭載)")
                            if st.button("✨ この単語でクイズを生成", key=f"quiz_btn_{i}"):
                                with st.spinner("生成中..."):
                                    sys_quiz = "英語の予備校講師として、指定された単語の4択穴埋め問題を作成せよ。JSON出力: {\"question\": \"...\", \"options\": [\"...\"], \"answer\": \"...\", \"translation\": \"...\"}"
                                    try:
                                        res_quiz = call_ai(f"単語: {word}", sys_quiz, is_json=True, model_name="gemini-3.1-flash-lite")
                                        st.session_state[f"active_quiz_{i}"] = json.loads(res_quiz)
                                        st.session_state[f"quiz_answered_{i}"] = False
                                    except: st.error("失敗しました。")
                            
                            if f"active_quiz_{i}" in st.session_state:
                                quiz_data = st.session_state[f"active_quiz_{i}"]
                                with st.container(border=True):
                                    st.markdown(f"**Q.** {quiz_data['question']}")
                                    user_choice = st.radio("選択:", quiz_data["options"], key=f"choice_{i}", disabled=st.session_state.get(f"quiz_answered_{i}", False))
                                    if not st.session_state.get(f"quiz_answered_{i}", False):
                                        if st.button("📝 解答する", key=f"ans_btn_{i}"):
                                            st.session_state[f"quiz_answered_{i}"] = True; st.rerun()
                                    if st.session_state.get(f"quiz_answered_{i}", False):
                                        st.markdown("---")
                                        if user_choice == quiz_data["answer"]: st.success(f"🎉 正解！ ({quiz_data['answer']})")
                                        else: st.error(f"❌ 惜しい！ 正解は **{quiz_data['answer']}**")
                                        st.markdown(f"**💡 和訳:** {quiz_data['translation']}")
                                        if st.button("🔄 もう一度", key=f"retry_btn_{i}"):
                                            del st.session_state[f"active_quiz_{i}"]
                                            del st.session_state[f"quiz_answered_{i}"]; st.rerun()

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
        # タブ2: 新しい単語帳を作る
        # ------------------------------------------
        with tab_create:
            selected_labels = st.multiselect("📚 組み合わせたい過去問を選んでください", list(db_options.keys()))
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
                            "title": new_title, "main_vocab": st.session_state.main_vocab, "excluded_vocab": st.session_state.excluded_vocab,
                            "counts": dict(st.session_state.combined_counter), "origins": dict(st.session_state.word_origins)
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
                                    for method in methods.keys():
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
                        for method in methods.keys():
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
            selected_labels = st.multiselect("📚 組み合わせたい過去問を選んでください", list(db_options.keys()), key="idiom_multi")
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
                    add_labels = st.multiselect("追加したい過去問を選んでください", list(db_options.keys()), key=f"merge_idiom_{book_idx}")
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
    # タブ2: インプット用長文
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
                
                if st.button("📖 新しいインプット長文を生成する（約20〜30秒）"):
                    with st.spinner("熟語を散りばめた長文と詳細な解説を作成中..."):
                        sorted_idioms = sorted(current_idiom_book["idioms"], key=lambda x: x.get("practice_count", 0))
                        target_idioms = [item["base_form"] for item in sorted_idioms[:15]]
                        sys_input = """
                        あなたはプロ英語講師です。指定された英熟語が全て含まれる300語程度の長文を作成してください。
                        【絶対ルール】指定熟語以外の単語は、極めて簡単な中学レベル（CEFR A1〜A2）のみを使用すること。「fortress」等の難単語は禁止。
                        【出力JSON】
                        {
                          "passage": "This is a story... We must **deal with** the problem...",
                          "full_translation": "長文の和訳...",
                          "explanations": [{"idiom": "deal with", "sentence_used": "...", "explanation": "..."}]
                        }
                        """
                        try:
                            res_input = call_ai(f"対象熟語:\n{', '.join(target_idioms)}", sys_input, is_json=True, model_name="gemini-2.5-pro")
                            st.session_state.current_input_data = json.loads(res_input)
                        except Exception as e: st.error(f"生成失敗: {e}")

                if "current_input_data" in st.session_state:
                    data = st.session_state.current_input_data
                    st.markdown("---")
                    st.markdown("#### 📖 Reading Passage (Input)")
                    with st.container(border=True):
                        colored_passage = re.sub(r'\*\*(.*?)\*\*', r"<span style='color:#1f77b4; font-weight:bold; font-size:1.05em;'>\1</span>", data['passage'])
                        st.markdown(f"<div style='line-height: 1.8; font-size: 1.1em;'>{colored_passage}</div>", unsafe_allow_html=True)
                    st.markdown("#### 👁️ 全訳")
                    st.info(data["full_translation"])
                    st.markdown("#### 💡 熟語の文脈解説")
                    for item in data.get("explanations", []):
                        with st.expander(f"📌 {item['idiom']}", expanded=False):
                            st.markdown(f"**使われ方:** {item['sentence_used']}")
                            st.markdown(f"**解説:** {item['explanation']}")

    # ------------------------------------------
    # タブ3: アウトプット
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
                
                difficulty_level = st.slider("📊 長文自体の難易度を指定 (1:易しい 〜 10:難関大レベル)", min_value=1, max_value=10, value=5)
                
                if st.button("🚀 レベルに応じた長文テストを生成する"):
                    with st.spinner("AIが論理的な長文と精巧なダミー選択肢を生成しています（約20〜30秒お待ちください）..."):
                        target_idioms = []
                        candidates = current_idiom_book["idioms"].copy()
                        for _ in range(min(15, len(candidates))):
                            weights = [item["count"] / (1 + item.get("correct_count", 0) ** 2) for item in candidates]
                            total_weight = sum(weights)
                            probs = [1/len(candidates)] * len(candidates) if total_weight == 0 else [w/total_weight for w in weights]
                            chosen = random.choices(candidates, weights=probs, k=1)[0]
                            target_idioms.append(chosen["base_form"])
                            candidates.remove(chosen)
                        
                        sys_output = f"""
                        プロ英語講師として、指定熟語が自然に含まれる300〜400語の長文を作成せよ。
                        【絶対ルール】構成難易度は10段階中【レベル {difficulty_level}】だが、ターゲット以外の周辺単語は絶対に中学〜高校基礎（CEFR A1〜B1）の平易な単語のみに限定すること。
                        熟語箇所を順番通りに ( 1 ), ( 2 )... と空欄にせよ。
                        {{
                          "passage": "This is a story... We must ( 1 ) the issue...",
                          "questions": [
                            {{"blank_id": 1, "options": ["deal with", "put off", "bring about", "stand for"], "answer": "deal with", "translation": "問題に対処する。", "explanation": "deal with は〜に対処する。..."}}
                          ],
                          "full_translation": "全訳..."
                        }}
                        """
                        try:
                            res_output = call_ai(f"熟語リスト:\n{', '.join(target_idioms)}", sys_output, is_json=True, model_name="gemini-2.5-pro")
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
                                        sys_chat = f"問題の和訳: {q['translation']}\n正解: {q['answer']}\n生徒の質問に高速かつ簡潔に答えてください。"
                                        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.quiz_chat_logs[chat_key][-4:]])
                                        ans = call_ai(f"会話:\n{history_str}", sys_chat, use_pdf=False, model_name="gemini-2.5-flash")
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