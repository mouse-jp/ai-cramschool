import streamlit as st
import google.generativeai as genai
import tempfile
import os
import json
import re
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

mode = st.sidebar.radio("モード", ["💬 対話で分析", "☕ 学習の作戦会議", "🏠 マイ教訓ノート", "📖 志望校別単語帳"])

if "messages" not in st.session_state: st.session_state.messages = []
if "auto_insight" not in st.session_state: st.session_state.auto_insight = ""
if "current_quiz_question" not in st.session_state: st.session_state.current_quiz_question = ""
if "current_quiz_data" not in st.session_state: st.session_state.current_quiz_data = ""
if "quiz_chat_history" not in st.session_state: st.session_state.quiz_chat_history = []

# --- 3. AI呼び出し関数 ---
def call_ai(prompt, sys_msg, use_pdf=False, is_json=False):
    genai.configure(api_key=api_key)
    if is_json:
        model = genai.GenerativeModel(model_name="gemini-2.5-pro", system_instruction=sys_msg, generation_config={"response_mime_type": "application/json"})
    else:
        model = genai.GenerativeModel(model_name="gemini-2.5-pro", system_instruction=sys_msg)

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
# ★真の完全版 モードD: 志望校別単語帳（本棚 ＋ 任意AIフィルター）
# ==========================================
elif mode == "📖 志望校別単語帳":
    st.markdown("あなた専用の単語帳を作成し、本棚で管理します。")
    
    if not exam_db:
        st.warning("過去問データベースが空です。まずは db_manager.py を起動して過去問を登録してください。")
    else:
        # ★ タブで「本棚」「新規作成」「シミュレーター」を分ける
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
                    
                    # --- 管理機能をアコーディオンに収納 ---
                    with st.expander("⚙️ 単語帳の管理・編集（単語の追加/削除・データリセット）", expanded=False):
                        col_test, col_dup, col_del = st.columns(3)
                        col_test.button("▶️ この単語帳でテストする (準備中)", use_container_width=True, disabled=True)
                        
                        if col_dup.button("📋 この単語帳を複製する", use_container_width=True):
                            new_book = current_book.copy()
                            new_book["title"] = current_book["title"] + " (コピー)"
                            my_data["vocab_books"].append(new_book)
                            save_data(my_data)
                            st.success("単語帳を複製しました！")
                            st.rerun()
                            
                        if col_del.button("🗑️ この単語帳を削除する", use_container_width=True):
                            my_data["vocab_books"].pop(book_idx)
                            save_data(my_data)
                            st.success("単語帳を削除しました！")
                            st.rerun()
                        
                        st.markdown("---")
                        st.markdown("#### ✏️ 単語の追加・削除")
                        col_edit1, col_edit2 = st.columns(2)
                        
                        with col_edit1:
                            with st.form(f"add_word_form_{book_idx}"):
                                new_word = st.text_input("➕ 新しい単語を手動で追加")
                                if st.form_submit_button("追加する") and new_word:
                                    new_word = new_word.lower().strip()
                                    if new_word not in current_book["main_vocab"]:
                                        current_book["main_vocab"].insert(0, new_word)
                                        if new_word not in current_book.get("counts", {}):
                                            current_book.setdefault("counts", {})[new_word] = 1
                                        my_data["vocab_books"][book_idx] = current_book
                                        save_data(my_data)
                                        st.success(f"「{new_word}」を追加しました！")
                                        st.rerun()
                                    else:
                                        st.warning("その単語は既に登録されています。")

                        with col_edit2:
                            with st.form(f"del_word_form_{book_idx}"):
                                del_word = st.selectbox("🗑️ 削除したい単語を検索して選択", ["-- 選択 --"] + current_book["main_vocab"] + current_book["excluded_vocab"])
                                if st.form_submit_button("削除する") and del_word != "-- 選択 --":
                                    del_word = del_word.lower().strip()
                                    if del_word in current_book["main_vocab"]:
                                        current_book["main_vocab"].remove(del_word)
                                        my_data["vocab_books"][book_idx] = current_book
                                        save_data(my_data)
                                        st.success(f"メインリストから「{del_word}」を削除しました！")
                                        st.rerun()
                                    elif del_word in current_book["excluded_vocab"]:
                                        current_book["excluded_vocab"].remove(del_word)
                                        my_data["vocab_books"][book_idx] = current_book
                                        save_data(my_data)
                                        st.success(f"除外リストから「{del_word}」を削除しました！")
                                        st.rerun()
                                    else:
                                        st.error("その単語は見つかりませんでした。")
                        
                        # AI生成データのリセットボタン
                        if "enriched_vocab" in current_book:
                            st.markdown("---")
                            st.markdown("#### 🗑️ データの初期化")
                            if st.button("🗑️ AI生成データをリセットして最初からやり直す", use_container_width=True):
                                del current_book["enriched_vocab"]
                                if "skipped_vocab" in current_book:
                                    del current_book["skipped_vocab"]
                                my_data["vocab_books"][book_idx] = current_book
                                save_data(my_data)
                                st.rerun()
                                
                    st.markdown("---")
                    
                    # まだAI生成されていない場合
                    if "enriched_vocab" not in current_book:
                        st.info("💡 まだ単語の意味やイメージが生成されていません。（現在は英単語のリストのみです）")
                        
                        if st.button("✨ 上位30語の意味・フレーズをAI生成する（実験）", use_container_width=True):
                            with st.spinner("AIがネイティブの脳内ネットワークを作成中...（約10〜20秒）"):
                                target_words = current_book["main_vocab"][:30] 
                                
                                sys_enrich = """
                                あなたは受験英語に精通したプロの予備校講師です。提供された英単語リストを精査し、以下のJSON形式（辞書型）を作成してください。
                                
                                【絶対ルール：分類と抽出】
                                1. 「大学受験レベル(B1以上)で重要な単語」と「中学レベルの基本単語(A1〜A2)やOCRのゴミ」に分ける。
                                2. 重要単語は "enriched" へ、基本単語やゴミ(donなど)は "skipped" (文字列の配列) へ。
                                
                                【JSONフォーマット】
                                {
                                  "enriched": [
                                    {
                                      "word": "company",
                                      "forms": "複数形: companies",
                                      "meanings": "① 会社、企業 ② 仲間、同席 ③ 劇団、一座",
                                      "chunks": [
                                        "① run a **company** (会社を経営する)",
                                        "② in the **company** of friends (友達と一緒に)"
                                      ],
                                      "context": "ビジネス系で必須。②の意味での出題も多いので注意。",
                                      "alert": "「仲間」の意味では不可算名詞なので a/an がつかない。"
                                    }
                                  ],
                                  "skipped": ["people", "don"]
                                }
                                
                                【プロンプト指示】
                                ・forms: 不規則な複数形、特殊な活用、注意すべき派生語があれば記載。なければ空文字("")。
                                ・meanings: 大学受験において覚えるべき意味を ① ② ③... のように番号を振って【1行で】列挙すること。
                                ・chunks: 上記の意味番号（①, ②...）に対応する実践的な例文やフレーズを作成すること。行頭は必ず対応する番号（①, ②...）にし、対象の単語部分は **太字(Markdown)** にすること。
                                ・context: どんな長文テーマでよく出るか。
                                ・alert: 受験生が間違えやすいポイントへの警告。なければ空文字("")。
                                """
                                try:
                                    response_json = call_ai(f"以下のリストを処理してください:\n{target_words}", sys_enrich, is_json=True)
                                    parsed_data = json.loads(response_json)
                                    
                                    current_book["enriched_vocab"] = parsed_data.get("enriched", [])
                                    current_book["skipped_vocab"] = parsed_data.get("skipped", [])
                                    
                                    my_data["vocab_books"][book_idx] = current_book
                                    save_data(my_data)
                                    st.success("🎉 生成が完了しました！")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"生成中にエラーが発生しました。（無料枠の制限などに引っかかった可能性があります）詳細: {e}")
                        
                        main_display = [{"単語": w, "出現回数": current_book.get("counts", {}).get(w, "-")} for w in current_book["main_vocab"]]
                        st.dataframe(main_display, use_container_width=True)
                        
                    # AI生成済みの場合
                    else:
                        if "selected_word_idx" not in st.session_state:
                            st.session_state.selected_word_idx = None
                            
                        # 📚 モード1：単語リスト一覧画面
                        if st.session_state.selected_word_idx is None:
                            st.markdown("### 📚 単語リスト")
                            
                            enriched_list = current_book.get("enriched_vocab", [])
                            WORDS_PER_PAGE = 20 # 1ページあたりの表示件数
                            total_pages = max(1, (len(enriched_list) + WORDS_PER_PAGE - 1) // WORDS_PER_PAGE)
                            
                            # ページ番号の記憶（本棚ごとに記憶させる）
                            page_key = f"page_{book_idx}"
                            if page_key not in st.session_state:
                                st.session_state[page_key] = 1
                            current_page = st.session_state[page_key]
                            
                            # ページネーション（上部ボタンとジャンプ機能）
                            col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
                            with col_p1:
                                if st.button("◀ 前の20件", use_container_width=True, disabled=(current_page == 1)):
                                    st.session_state[page_key] -= 1
                                    st.rerun()
                                    
                            with col_p2:
                                jump_page = st.number_input(
                                    "ページジャンプ", 
                                    min_value=1, 
                                    max_value=total_pages, 
                                    value=current_page, 
                                    label_visibility="collapsed"
                                )
                                if jump_page != current_page:
                                    st.session_state[page_key] = jump_page
                                    st.rerun()
                                    
                                st.markdown(f"<div style='text-align: center; font-size: 0.8em; color: gray;'>全{len(enriched_list)}語 / {total_pages}ページ</div>", unsafe_allow_html=True)
                                
                            with col_p3:
                                if st.button("次の20件 ▶", use_container_width=True, disabled=(current_page == total_pages)):
                                    st.session_state[page_key] += 1
                                    st.rerun()
                                    
                            st.divider()
                            # 現在のページに表示する20件を切り出す
                            start_idx = (current_page - 1) * WORDS_PER_PAGE
                            end_idx = start_idx + WORDS_PER_PAGE
                            current_page_vocab = enriched_list[start_idx:end_idx]
                            
                            # 単語を2列に並べて表示
                            cols = st.columns(2)
                            for i, item in enumerate(current_page_vocab):
                                actual_idx = start_idx + i # 全体における本当のインデックス番号
                                word = item.get("word", "")
                                count = current_book.get("counts", {}).get(word, "-")
                                
                                col = cols[i % 2]
                                with col:
                                    with st.container(border=True):
                                        # 1. 見出し語を約3倍のサイズで強調 ＆ 出現回数
                                        st.markdown(
                                            f"<div style='margin-bottom: 15px;'>"
                                            f"<span style='color:#1f77b4; font-size:2.8em; font-weight:900; line-height:1.1;'>{word}</span>"
                                            f"<span style='font-size:1.2em; color:gray; margin-left:10px;'>({count}回)</span>"
                                            f"</div>", 
                                            unsafe_allow_html=True
                                        )
                                        
                                        # 2. 変化形・派生語（赤字で目立たせる）
                                        if item.get("forms"):
                                            st.markdown(f"<div style='color:#d62728; font-weight:bold; font-size:1.0em; margin-bottom: 10px;'>🔄 {item.get('forms')}</div>", unsafe_allow_html=True)
                                            
                                        # 3. 意味と例文を1対1でセット表示（色分けして見やすく）
                                        # 3. 意味を緑色でコンパクトに表示
                                        if item.get("meanings"):
                                            st.markdown(f"<div style='font-size:1.15em; font-weight:bold; color:#4caf50; margin-bottom: 10px;'>{item.get('meanings')}</div>", unsafe_allow_html=True)
                                            
                                        st.divider() # 情報の区切り線
                                        
                                        # 4. チャンク（例文）の表示（①、②の番号付き）
                                        for chunk in item.get("chunks", []):
                                            # 対象単語を青色太字に変換
                                            colored_chunk = re.sub(r'\*\*(.*?)\*\*', r"<span style='color:#1f77b4; font-weight:bold; font-size:1.1em;'>\1</span>", chunk)
                                            # ダークモードでも見やすいように文字色を自動調整(colorを外す)
                                            st.markdown(f"<div style='margin-left: 0.5em; margin-bottom: 8px; font-size:0.95em;'>{colored_chunk}</div>", unsafe_allow_html=True)
                                        
                                        st.markdown("<br>", unsafe_allow_html=True)
                                        if st.button(f"👉 詳細・文脈・メモを開く", key=f"sel_{book_idx}_{actual_idx}", use_container_width=True):
                                            st.session_state.selected_word_idx = actual_idx
                                            st.rerun()
                            
                            # ページネーション（下部にもボタンを設置すると便利）
                            if len(current_page_vocab) > 4:
                                st.divider()
                                col_p1_b, col_p2_b, col_p3_b = st.columns([1, 2, 1])
                                with col_p1_b:
                                    if st.button("◀ 前へ", key="prev_b", use_container_width=True, disabled=(current_page == 1)):
                                        st.session_state[page_key] -= 1
                                        st.rerun()
                                with col_p3_b:
                                    if st.button("次へ ▶", key="next_b", use_container_width=True, disabled=(current_page == total_pages)):
                                        st.session_state[page_key] += 1
                                        st.rerun()
                                            
                        # 🔍 モード2：詳細ルーム（単語専用の個別画面）
                        else:
                            i = st.session_state.selected_word_idx
                            item = current_book["enriched_vocab"][i]
                            word = item.get("word", "")
                            
                            if st.button("🔙 単語リストに戻る", type="primary"):
                                st.session_state.selected_word_idx = None
                                st.rerun()
                                
                            st.markdown(f"## 🔍 「{word}」の専用ルーム")
                            with st.container(border=True):
                                if item.get("forms"):
                                    st.markdown(f"**🔄 変化形・派生語:** {item.get('forms')}")
                                
                                st.markdown("### 📚 意味とフレーズ")
                                if item.get("meanings"):
                                    st.markdown(f"**<span style='color:#4caf50; font-size:1.2em;'>{item.get('meanings')}</span>**", unsafe_allow_html=True)
                                    
                                for chunk in item.get("chunks", []):
                                    colored_ex = re.sub(r'\*\*(.*?)\*\*', r"<span style='color:#1f77b4; font-weight:bold;'>\1</span>", chunk)
                                    st.markdown(f"> {colored_ex}", unsafe_allow_html=True)
                                        
                                if item.get("context"):
                                    st.markdown(f"📖 **文脈:** {item.get('context')}")
                                if item.get("alert"):
                                    st.markdown(f"**⚠️ 混同注意:** {item.get('alert')}")
                            
                            # 📝 マイ・メモ機能
                            st.markdown("#### 📝 マイ・メモ")
                            current_memo = item.get("user_memo", "")
                            col_m1, col_m2 = st.columns([4, 1])
                            new_memo = col_m1.text_input(f"メモ入力", value=current_memo, key=f"memo_{i}", label_visibility="collapsed", placeholder="例：過去問での出題、自分なりの覚え方など")
                            if col_m2.button("💾 保存", key=f"save_memo_{i}", use_container_width=True):
                                current_book["enriched_vocab"][i]["user_memo"] = new_memo
                                my_data["vocab_books"][book_idx] = current_book
                                save_data(my_data)
                                st.success("保存しました！")
                            
                            st.divider()
                            
                            # 📖 AI例文アシスト（生成＆選択保存）
                            st.markdown("#### 📖 例文アシスト")
                            
                            saved_ex = item.get("saved_examples", [])
                            if saved_ex:
                                st.markdown("**【保存済みの例文】**")
                                for ex in saved_ex:
                                    st.markdown(f"- {ex}")
                            
                            if st.button("➕ AIに新しい例文を3つ作ってもらう", key=f"gen_ex_{i}"):
                                with st.spinner(f"「{word}」の実践的な例文を生成中..."):
                                    sys_ex = "指定された英単語の実践的な例文と和訳を3つ、JSONの配列形式で出力してください。例: [\"I have an apple. (私はリンゴを持っています。)\"]"
                                    res_ex = call_ai(f"単語: {word}", sys_ex, is_json=True)
                                    try:
                                        st.session_state[f"temp_ex_{word}"] = json.loads(res_ex)
                                        st.rerun()
                                    except:
                                        st.error("生成に失敗しました。")
                            
                            if f"temp_ex_{word}" in st.session_state:
                                st.markdown("**💡 保存したいものにチェック：**")
                                with st.form(f"save_ex_form_{i}"):
                                    selected_ex = []
                                    for idx, ex in enumerate(st.session_state[f"temp_ex_{word}"]):
                                        if st.checkbox(ex, key=f"chk_{word}_{idx}"):
                                            selected_ex.append(ex)
                                    
                                    if st.form_submit_button("✅ 選択を保存"):
                                        if "saved_examples" not in current_book["enriched_vocab"][i]:
                                            current_book["enriched_vocab"][i]["saved_examples"] = []
                                        current_book["enriched_vocab"][i]["saved_examples"].extend(selected_ex)
                                        my_data["vocab_books"][book_idx] = current_book
                                        save_data(my_data)
                                        del st.session_state[f"temp_ex_{word}"]
                                        st.success("追加保存しました！")
                                        st.rerun()

                            st.divider()
                            
                            # 🤖 AIチャット
                            st.markdown(f"#### 🤖 AI講師に質問する")
                            st.caption("「これと似たスペルの単語は？」「adaptとの違いは？」「覚え方を教えて」")
                            
                            chat_key = f"chat_{word}"
                            if chat_key not in st.session_state:
                                st.session_state[chat_key] = []
                                
                            with st.container(height=300):
                                for msg in st.session_state[chat_key]:
                                    with st.chat_message(msg["role"]):
                                        st.markdown(msg["content"])
                                        
                            if st.session_state[chat_key] and st.session_state[chat_key][-1]["role"] == "user":
                                with st.spinner("AI講師が考え中..."):
                                    sys_chat = f"あなたは英単語「{word}」の専属講師です。生徒からの質問に対して、簡潔に、わかりやすく答えてください。"
                                    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state[chat_key][-5:]])
                                    ans = call_ai(f"会話履歴:\n{history_str}", sys_chat, use_pdf=False)
                                    st.session_state[chat_key].append({"role": "assistant", "content": ans})
                                    st.rerun()
                            
                            if user_q := st.chat_input(f"{word}について質問する...", key=f"chat_in_{i}"):
                                st.session_state[chat_key].append({"role": "user", "content": user_q})
                                st.rerun()
                            
                        # --- 🌟 AIがスキップした単語の個別復活機能 ---
                        skipped = current_book.get("skipped_vocab", [])
                        if skipped:
                            with st.expander("👀 AIが「基本語・ゴミ」と判定して除外した単語（ここから復活可能）", expanded=False):
                                st.info("以下の単語は基礎的すぎるかゴミと判定されましたが、ドロップダウンから選んで個別に解説を生成できます。")
                                st.write(", ".join(skipped))
                                
                                with st.form(f"restore_form_{book_idx}"):
                                    restore_word = st.selectbox("🔄 メインリストに復活させたい単語を選んでください", ["-- 選択 --"] + skipped)
                                    if st.form_submit_button("この単語の解説を生成して復活させる") and restore_word != "-- 選択 --":
                                        with st.spinner(f"「{restore_word}」の解説をピンポイントで生成中..."):
                                            sys_restore = """
                                            あなたは英語講師です。提供された1つの英単語について、以下のJSONを作成してください。
                                            【絶対ルール】ユーザーが指定した重要単語です。絶対に "enriched" 配列に入れて解説を作成してください。
                                            【JSONフォーマット】
                                            {
                                              "enriched": [
                                                {
                                                  "word": "指定された単語",
                                                  "chunks": ["📌 塊1 (超簡単な現代語訳)", "📌 塊2", "📌 塊3"],
                                                  "details": "🧠【コア】15文字以内の根本イメージ\n📖【文脈】どんな長文テーマで出るか一言",
                                                  "alert": "注意点"
                                                }
                                              ],
                                              "skipped": []
                                            }
                                            """
                                            try:
                                                res_restore = call_ai(f"処理する単語: ['{restore_word}']", sys_restore, is_json=True)
                                                restored_data = json.loads(res_restore)
                                                
                                                if restored_data.get("enriched"):
                                                    current_book["enriched_vocab"].insert(0, restored_data["enriched"][0])
                                                    current_book["skipped_vocab"].remove(restore_word)
                                                    my_data["vocab_books"][book_idx] = current_book
                                                    save_data(my_data)
                                                    st.success(f"「{restore_word}」を復活させました！")
                                                    st.rerun()
                                            except Exception as e:
                                                st.error(f"復活中にエラーが発生しました: {e}")

                    with st.expander("➕ 人名・固有名詞などの除外リストを確認する", expanded=False):
                        if not current_book["excluded_vocab"]:
                            st.write("除外された単語はありません。")
                        else:
                            excluded_display = [{"単語": w, "出現回数": current_book.get("counts", {}).get(w, "-")} for w in current_book["excluded_vocab"]]
                            st.dataframe(excluded_display, use_container_width=True)

        # ------------------------------------------
        # タブ2: 新しい単語帳を作る
        # ------------------------------------------
        with tab_create:
            db_options = {}
            for cat, unis in exam_db.items():
                for uni, facs in unis.items():
                    for fac, years in facs.items():
                        for year, methods in years.items():
                            for method in methods.keys():
                                label = f"[{cat}] {uni} {fac} ({year}年 {method})"
                                db_options[label] = {"c": cat, "u": uni, "f": fac, "y": year, "m": method}
            
            selected_labels = st.multiselect("📚 組み合わせたい過去問を選んでください（複数選択可）", list(db_options.keys()))
            
            use_ai_filter = st.checkbox("🤖 【テスト機能】AIで明らかな「人名」だけを除外する", value=False)
            
            if st.button("✨ 選択した過去問から単語帳を生成") and selected_labels:
                with st.spinner("単語を集計中..."):
                    combined_counter = Counter()
                    word_origins = defaultdict(list) # ★追加：出題元の内訳を記録する辞書
                    
                    for label in selected_labels:
                        path = db_options[label]
                        freqs = exam_db[path["c"]][path["u"]][path["f"]][path["y"]][path["m"]]["frequencies"]
                        combined_counter.update(freqs)
                        
                        # ★追加：どの年度で何回出たかを記録（例: "25年(3回)"）
                        short_label = f"{str(path['y'])[-2:]}年"
                        for w, count in freqs.items():
                            word_origins[w].append(f"{short_label}({count}回)")
                    
                    sorted_words = [word for word, count in combined_counter.most_common()]
                    st.session_state.combined_counter = combined_counter
                    st.session_state.word_origins = dict(word_origins) # セッションに保存
                    
                    # ★修正：制限を撤廃し、すべての単語をAIに渡す
                    words_to_filter = sorted_words 
                    
                    if use_ai_filter and api_key:
                        # (sys_filter の定義は先ほど修正した絶対ルールのままでOK)
                        sys_filter = """
                        あなたは英語学習の専門家であり、データ整理のアシスタントです。
                        入力された英単語リストを以下の2つに分類し、JSON形式で出力してください。
                        
                        {
                          "main_vocab": ["technology", "people", "chicago", "japan", ...],
                          "excluded_vocab": ["david", "sarah", "macintyre", "pathways", ...]
                        }
                        
                        【分類の絶対ルール】
                        1. excluded_vocab に入れるもの（完全な除外対象）
                        ・「人名」「登場人物名」
                        ・「著者名」「出典元の名称（出版社など）」
                        ・「特定の企業・ブランド名」
                        
                        2. main_vocab に残すもの（学習対象）
                        ・「地名（国名、都市名、地域名）」は残す。
                        ・「一般名詞」「動詞」「形容詞」「専門用語」「抽象概念」は絶対に消さない。
                        """
                        try:
                            # 制限なしのリストをそのまま渡す
                            response_json = call_ai(f"以下のリストを分類してください:\n{words_to_filter}", sys_filter, is_json=True)
                            filtered_data = json.loads(response_json)
                            st.session_state.main_vocab = filtered_data.get("main_vocab", [])
                            st.session_state.excluded_vocab = filtered_data.get("excluded_vocab", [])
                        except Exception as e:
                            st.error("AIの振り分け中にエラーが発生しました。そのまま全単語を表示します。")
                            st.session_state.main_vocab = sorted_words
                            st.session_state.excluded_vocab = []
                    else:
                        st.session_state.main_vocab = sorted_words
                        st.session_state.excluded_vocab = []

            # --- プレビューと保存 ---
            if "main_vocab" in st.session_state:
                st.markdown("---")
                st.markdown("### 👀 単語帳のプレビュー")
                
                with st.form("save_book_form"):
                    new_title = st.text_input("💾 この単語帳に名前をつけて本棚に保存 (例: 学習院2025 マスター)")
                    if st.form_submit_button("本棚に保存する") and new_title:
                        if "vocab_books" not in my_data:
                            my_data["vocab_books"] = []
                            
                        new_book = {
                            "title": new_title,
                            "main_vocab": st.session_state.main_vocab,
                            "excluded_vocab": st.session_state.excluded_vocab,
                            "counts": dict(st.session_state.combined_counter),
                            "origins": dict(st.session_state.word_origins) # 本棚にも内訳データを保存
                        }
                        my_data["vocab_books"].append(new_book)
                        save_data(my_data)
                        del st.session_state.main_vocab
                        st.success(f"🎉「{new_title}」を本棚に保存しました！「📚 あなたの本棚」タブから確認できます。")
                        st.rerun()

                st.info("※出現頻度が高い順に並んでいます。")
                
                # ★修正：プレビューの表に「出題内訳」の列を追加
                main_display = [
                    {
                        "単語": w, 
                        "総出現回数": st.session_state.combined_counter[w],
                        "出題内訳": ", ".join(st.session_state.word_origins.get(w, []))
                    } 
                    for w in st.session_state.main_vocab
                ]
                st.dataframe(main_display, use_container_width=True)

        # ------------------------------------------
        # タブ3: カバー率・難化シミュレーター（★致命傷チェッカー追加）
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
                
                # --- ① 定量分析（カバー率計算）の実行 ---
                if st.button("🚀 カバー率を検証する（バックテスト）", use_container_width=True, type="primary"):
                    if selected_baseline == "-- 選択 --" or selected_target == "-- 選択 --":
                        st.error("武器と敵を両方選択してください。")
                    else:
                        with st.spinner("単語の照合中..."):
                            # --- 1. 武器（自分の単語帳）のデータ取得と原形変換 ---
                            base_book = next(b for b in books if b["title"] == selected_baseline)
                            raw_base_words = base_book["main_vocab"]
                            # 小文字にして動詞・名詞の原形に変換
                            base_words = set(lemmatizer.lemmatize(word.lower(), pos='v') for word in raw_base_words)
                            base_words = set(lemmatizer.lemmatize(word, pos='n') for word in base_words)

                            # --- 2. 敵（過去問）のデータ取得と原形変換・ゴミ除去 ---
                            path = target_options[selected_target]
                            target_data = exam_db[path["c"]][path["u"]][path["f"]][path["y"]][path["m"]]
                            raw_target_freqs = target_data["frequencies"]
                            # NLTK関数を通して et などのゴミを除去
                            target_freqs = process_frequencies(raw_target_freqs)

                            # --- 3. 照合準備 ---
                            total_tokens = sum(target_freqs.values())
                            covered_tokens = 0
                            missed_words = {}

                            # --- 4. 照合処理 ---
                            for w, count in target_freqs.items():
                                if w in base_words:
                                    covered_tokens += count
                                else:
                                    missed_words[w] = count
               
                                    
                            coverage_rate = (covered_tokens / total_tokens) * 100 if total_tokens > 0 else 0
                            
                            # 分析結果をSessionStateに保存（画面リロードで消えないようにする）
                            st.session_state.sim_result = {
                                "coverage_rate": coverage_rate,
                                "total_tokens": total_tokens,
                                "covered_tokens": covered_tokens,
                                "missed_words": missed_words,
                                "target_name": selected_target
                            }
                            # 前回までの定性分析結果があればリセットする
                            if "analysis_result" in st.session_state:
                                del st.session_state.analysis_result
                            st.rerun()
                            
                # --- ② 定量結果の表示と、定性分析（致命傷チェック）UI ---
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
                        
                        # ★ PDFとテキスト入力を選べるように変更
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
                                    # PDFの場合はGeminiに直接アップロードして解析させる
                                    import tempfile, os
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                        tmp.write(local_pdf.getvalue())
                                        tmp_path = tmp.name
                                    
                                    genai.configure(api_key=api_key)
                                    g_file = genai.upload_file(tmp_path)
                                    model = genai.GenerativeModel(model_name="gemini-2.5-pro", system_instruction=sys_prompt)
                                    res = model.generate_content([g_file, prompt])
                                    os.remove(tmp_path)
                                    st.session_state.analysis_result = res.text
                                    
                                st.rerun()
                                
                        if "analysis_result" in st.session_state:
                            st.info(st.session_state.analysis_result)