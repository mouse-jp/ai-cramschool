import io
import json
import os
import re
from collections import defaultdict
from datetime import datetime

import pandas as pd
import streamlit as st


BASE_LEXICON_FILE = "base_lexicon.json"
EXAM_DB_FILE = "past_exams_db.json"

BASE_STATUSES = {"core_verified", "exam_format", "watch_known"}
STRICT_EXCLUDED_STATUS = "strict_excluded"

STATUS_LABELS = {
    "pending": "未判定",
    "core_verified": "基礎語",
    "exam_format": "試験語",
    "watch_known": "注意多義語",
    "learning_target": "単語帳対象",
    "strict_excluded": "完全除外",
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


def add_frequency(counter, sources, word, count, source_label):
    clean_word = normalize_word(word)
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


def build_candidates(uploaded_files, include_exam_db):
    counter = defaultdict(int)
    sources = defaultdict(set)

    if uploaded_files:
        collect_from_csv_files(uploaded_files, counter, sources)

    if include_exam_db:
        exam_db = load_json_file(EXAM_DB_FILE, {})
        collect_from_exam_db_node(exam_db, counter, sources, [])

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
    return "pending"


def coverage_policy_for_status(status):
    if status in BASE_STATUSES:
        return "auto_covered"
    if status == STRICT_EXCLUDED_STATUS:
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


def filtered_candidates(df, status_filter, min_count, search_text, max_rows):
    result = df.copy()
    if status_filter != "すべて":
        reverse_labels = {label: status for status, label in STATUS_LABELS.items()}
        status = reverse_labels.get(status_filter, status_filter)
        result = result[result["status"] == status]
    if min_count:
        result = result[result["count"] >= min_count]
    if search_text:
        result = result[result["word"].str.contains(search_text.lower(), case=False, regex=False)]
    return result.head(max_rows)


def main():
    st.set_page_config(page_title="基礎語3000 選定プラットフォーム", page_icon="📚", layout="wide")
    st.title("基礎語3000 選定プラットフォーム")
    st.caption("過去問やCSVから候補語を集め、AIに回さない基礎語・試験語・注意多義語を選ぶための作業台です。")

    if "base_lexicon_work" not in st.session_state:
        st.session_state.base_lexicon_work = load_json_file(BASE_LEXICON_FILE, {})

    lexicon = st.session_state.base_lexicon_work

    with st.sidebar:
        st.header("候補データ")
        uploaded_files = st.file_uploader(
            "単語CSVを追加",
            type=["csv"],
            accept_multiple_files=True,
            help="列名は「単語」「出現回数」、または word/count 形式に対応します。",
        )
        include_exam_db = st.checkbox(
            "past_exams_db.json も読む",
            value=os.path.exists(EXAM_DB_FILE),
            disabled=not os.path.exists(EXAM_DB_FILE),
        )
        target_count = st.number_input("基礎語の目標数", min_value=100, max_value=10000, value=3000, step=100)
        st.divider()
        if st.button("保存済みJSONを再読み込み", use_container_width=True):
            st.session_state.base_lexicon_work = load_json_file(BASE_LEXICON_FILE, {})
            st.rerun()

    try:
        raw_candidates = build_candidates(uploaded_files, include_exam_db)
    except Exception as e:
        st.error(f"候補データの読み込みでエラーが出ました: {e}")
        raw_candidates = pd.DataFrame()

    candidates = add_lexicon_columns(raw_candidates, lexicon)

    base_count = count_base_words(lexicon)
    strict_count = count_status(lexicon, STRICT_EXCLUDED_STATUS)
    learning_count = count_status(lexicon, "learning_target")
    remaining = max(int(target_count) - base_count, 0)

    if candidates.empty:
        total_tokens = 0
        covered_tokens = 0
        excluded_tokens = 0
    else:
        total_tokens = int(candidates["count"].sum())
        covered_tokens = int(candidates[candidates["status"].isin(BASE_STATUSES)]["count"].sum())
        excluded_tokens = int(candidates[candidates["status"] == STRICT_EXCLUDED_STATUS]["count"].sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("候補語数", f"{len(candidates):,}")
    col2.metric("基礎語登録", f"{base_count:,}", delta=f"目標まで {remaining:,}")
    col3.metric("完全除外", f"{strict_count:,}")
    col4.metric("単語帳対象", f"{learning_count:,}")

    if total_tokens:
        cov1, cov2 = st.columns(2)
        cov1.progress(min(covered_tokens / total_tokens, 1.0), text=f"基礎語でカバー: {covered_tokens / total_tokens:.1%}")
        cov2.progress(min(excluded_tokens / total_tokens, 1.0), text=f"完全除外: {excluded_tokens / total_tokens:.1%}")

    tab_pick, tab_base, tab_save = st.tabs(["候補を選ぶ", "基礎語リスト", "保存・出力"])

    with tab_pick:
        st.subheader("候補一覧")

        if candidates.empty:
            st.info("CSVをアップロードするか、同じフォルダに past_exams_db.json を置くと候補が表示されます。")
        else:
            with st.expander("自動で仮分類する", expanded=True):
                a1, a2, a3 = st.columns(3)
                if a1.button("上位語を基礎語に仮登録", use_container_width=True):
                    eligible = candidates[~candidates["word"].isin(STRICT_EXCLUDED_WORDS)]
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

                b1, b2 = st.columns(2)
                if b1.button("完全除外語を分類", use_container_width=True):
                    changed = auto_mark_words(lexicon, candidates, "strict_excluded", word_set=STRICT_EXCLUDED_WORDS, allow_override=True)
                    st.success(f"{changed}語を完全除外として分類しました。")
                    st.rerun()
                if b2.button("保存済みでない低頻度語を単語帳対象にする", use_container_width=True):
                    low_freq = candidates[(candidates["count"] <= 2) & (candidates["status"] == "pending")]
                    changed = auto_mark_words(lexicon, low_freq, "learning_target")
                    st.success(f"{changed}語を単語帳対象として分類しました。")
                    st.rerun()

            f1, f2, f3, f4 = st.columns([1.2, 1, 1, 1])
            status_filter = f1.selectbox("状態", ["すべて"] + list(STATUS_LABELS.values()))
            min_count = f2.number_input("最低回数", min_value=0, max_value=10000, value=0, step=1)
            search_text = f3.text_input("検索")
            max_rows = f4.number_input("表示行数", min_value=50, max_value=10000, value=500, step=50)

            view_df = filtered_candidates(candidates, status_filter, min_count, search_text, int(max_rows))
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
        rows = []
        for word, entry in sorted(lexicon.items()):
            if entry.get("status") in BASE_STATUSES:
                rows.append(
                    {
                        "word": word,
                        "status": entry.get("status"),
                        "meanings": meanings_to_text(entry.get("meanings", [])),
                        "note": entry.get("note", ""),
                        "last_seen_count": entry.get("last_seen_count", ""),
                        "last_seen_rank": entry.get("last_seen_rank", ""),
                    }
                )

        base_df = pd.DataFrame(rows)
        if base_df.empty:
            st.info("まだ基礎語は登録されていません。候補タブから仮登録してください。")
        else:
            st.write(f"登録数: **{len(base_df):,}語**")
            st.dataframe(base_df, use_container_width=True, hide_index=True)

    with tab_save:
        st.subheader("保存・出力")
        st.write(f"`{BASE_LEXICON_FILE}` に保存すると、次回も同じ分類を続きから編集できます。")

        save_col, download_col = st.columns(2)
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

        with st.expander("分類の意味"):
            st.markdown(
                """
- **基礎語**: AIに毎回意味付けさせず、登録済み意味でカバー済みにする語。
- **試験語**: question, option など、試験形式として頻出する語。
- **注意多義語**: 基礎語だが、文脈で意味がずれると読解に効く語。
- **単語帳対象**: 基礎語には入れず、生徒の単語帳やAI解析の対象にする語。
- **完全除外**: 冠詞・代名詞・助動詞の破片など、語彙カバー率の分母から外す語。
                """.strip()
            )


if __name__ == "__main__":
    main()
