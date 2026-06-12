# -*- coding: utf-8 -*-
"""
auth.py — ログイン認証 ＋ ユーザーごとのデータ保存（ローカルファイル版）

設計方針:
- パスワードは pbkdf2_hmac(sha256) でハッシュ化して users.json に保存（平文保存しない）。
- 学習データはユーザーごとに user_data/<ユーザー名>.json に分離（データの“構造”は従来と同じ）。
- 最初に登録したユーザーは、既存の my_data.json を自動で引き継ぐ。
- 将来クラウド（Supabase 等）へ移す時は、このファイルの保存処理だけ差し替えれば
  app.py 側（ログイン画面・本体）はそのまま使える、という分離構成。

依存は標準ライブラリ＋streamlitのみ（追加インストール不要）。
"""

import os
import re
import json
import hmac
import hashlib
import secrets
from datetime import datetime

import streamlit as st

# --- 保存先 ---
USERS_FILE = "users.json"            # アカウント情報（ハッシュ化済みパスワード）
USER_DATA_DIR = "user_data"          # ユーザーごとの学習データ置き場
LEGACY_DATA_FILE = "my_data.json"    # 既存の単一データ（最初のユーザーが引き継ぐ）
DEFAULT_DATA = {"vocabulary": [], "grammar": [], "strategy": [], "meta": []}

_PBKDF2_ITERATIONS = 200_000


# ============================================================
# 低レベル: ユーザー情報の読み書き・パスワードハッシュ
# ============================================================
def _load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def _hash_password(password, salt_hex):
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), _PBKDF2_ITERATIONS
    )
    return dk.hex()


def _verify_password(password, salt_hex, expected_hash):
    return hmac.compare_digest(_hash_password(password, salt_hex), expected_hash)


def _is_valid_username(username):
    # 英数字とアンダースコア、3〜20文字
    return bool(re.fullmatch(r"[A-Za-z0-9_]{3,20}", username or ""))


# よく使われる/流出済みの弱いパスワード（小文字で比較）
_COMMON_PASSWORDS = {
    "password", "passw0rd", "p@ssword", "p@ssw0rd", "password1", "password123",
    "123456", "1234567", "12345678", "123456789", "1234567890", "11111111",
    "qwerty", "qwerty123", "abc123", "abcd1234", "111111", "000000", "123123",
    "654321", "admin", "admin123", "manager", "manager123", "root", "letmein",
    "welcome", "welcome1", "iloveyou", "monkey", "dragon", "sunshine", "master",
    "test", "test123", "aaaaaaaa", "asdfghjk", "1q2w3e4r", "zxcvbnm",
}


def _check_password_strength(username, password):
    """パスワードの強さをチェック。(OK, メッセージ) を返す。"""
    pw = password or ""
    if len(pw) < 8:
        return False, "パスワードは8文字以上にしてください。"
    if pw.lower() in _COMMON_PASSWORDS:
        return False, "そのパスワードはよく使われる／流出済みのため使えません。推測されにくいものにしてください。"
    if len(set(pw)) <= 2:
        return False, "同じ文字の繰り返しなど単純すぎます。別のパスワードにしてください。"
    if username and username.lower() in pw.lower():
        return False, "パスワードにユーザー名を含めないでください。"
    has_letter = any(c.isalpha() for c in pw)
    has_digit = any(c.isdigit() for c in pw)
    if not (has_letter and has_digit):
        return False, "英字と数字の両方を含めてください（記号も足すとさらに安全です）。"
    return True, ""


# ============================================================
# データファイルのパス（ユーザーごと）
# ============================================================
def user_data_path(username):
    os.makedirs(USER_DATA_DIR, exist_ok=True)
    return os.path.join(USER_DATA_DIR, f"{username}.json")


def _ensure_user_data_file(username, inherit_legacy=False):
    """ユーザーのデータファイルが無ければ作る。
    inherit_legacy=True かつ既存 my_data.json があればそれを引き継ぐ。"""
    path = user_data_path(username)
    if os.path.exists(path):
        return
    seed = dict(DEFAULT_DATA)
    if inherit_legacy and os.path.exists(LEGACY_DATA_FILE):
        try:
            with open(LEGACY_DATA_FILE, "r", encoding="utf-8-sig") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                seed = loaded
        except Exception:
            seed = dict(DEFAULT_DATA)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(seed, f, ensure_ascii=False, indent=2)


# ============================================================
# 登録・ログイン
# ============================================================
def register(username, password, password_confirm):
    username = (username or "").strip()
    if not _is_valid_username(username):
        return False, "ユーザー名は半角英数字とアンダースコア（_）の3〜20文字にしてください。"
    if password != password_confirm:
        return False, "確認用パスワードが一致しません。"
    ok, msg = _check_password_strength(username, password)
    if not ok:
        return False, msg

    users = _load_users()
    if username in users:
        return False, "そのユーザー名は既に使われています。別の名前にしてください。"

    is_first_user = (len(users) == 0)
    salt_hex = secrets.token_hex(16)
    users[username] = {
        "salt": salt_hex,
        "hash": _hash_password(password, salt_hex),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    # 最初のユーザーは既存の学習データ(my_data.json)を引き継ぐ
    _ensure_user_data_file(username, inherit_legacy=is_first_user)
    _save_users(users)
    inherited = " 既存の学習データを引き継ぎました。" if is_first_user and os.path.exists(LEGACY_DATA_FILE) else ""
    return True, "登録しました。ログインしてください。" + inherited


def login(username, password):
    username = (username or "").strip()
    users = _load_users()
    info = users.get(username)
    if not info:
        return False
    return _verify_password(password, info.get("salt", ""), info.get("hash", ""))


# ============================================================
# クラウド(Supabase)対応：キーがあればクラウド、無ければローカルファイル
# ============================================================
def _secret(key):
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""


def is_cloud():
    """SUPABASE_URL と SUPABASE_KEY が secrets にあればクラウドモード。"""
    return bool(_secret("SUPABASE_URL") and _secret("SUPABASE_KEY"))


def _get_supabase():
    """Supabaseクライアントを毎回生成し、保存済みトークンでユーザー認証を復元する。
    （共有キャッシュにすると他ユーザーとセッションが混ざるため、毎回生成する）"""
    from supabase import create_client
    client = create_client(_secret("SUPABASE_URL"), _secret("SUPABASE_KEY"))
    access = st.session_state.get("sb_access_token")
    refresh = st.session_state.get("sb_refresh_token")
    if access and refresh:
        try:
            client.auth.set_session(access, refresh)
        except Exception:
            pass
    return client


# --- Cookie（毎回ログイン不要にする「ログイン保持」）---
_REMEMBER_COOKIE = "sb_refresh"
_REMEMBER_DAYS = 14


def _cookie_manager():
    """Cookie操作用マネージャ。1run内で同じインスタンスを使う（key重複を避ける）。"""
    if "_cookie_mgr" not in st.session_state:
        try:
            import extra_streamlit_components as stx
            st.session_state["_cookie_mgr"] = stx.CookieManager(key="auth_cookie_mgr")
        except Exception:
            st.session_state["_cookie_mgr"] = None
    return st.session_state["_cookie_mgr"]


def _save_login_cookie(refresh_token):
    cm = _cookie_manager()
    if not cm or not refresh_token:
        return
    try:
        from datetime import timedelta
        cm.set(
            _REMEMBER_COOKIE,
            refresh_token,
            expires_at=datetime.now() + timedelta(days=_REMEMBER_DAYS),
            key="cookie_set",
        )
    except Exception:
        pass


def _clear_login_cookie():
    cm = _cookie_manager()
    if not cm:
        return
    try:
        cm.delete(_REMEMBER_COOKIE, key="cookie_del")
    except Exception:
        pass


def _try_cookie_login():
    """Cookieのrefresh_tokenでセッション復元を試みる。成功でTrue。"""
    if st.session_state.get("_cookie_login_failed"):
        return False
    cm = _cookie_manager()
    if not cm:
        return False
    try:
        token = cm.get(_REMEMBER_COOKIE)
    except Exception:
        token = None
    if not token:
        return False
    try:
        res = _get_supabase().auth.refresh_session(token)
        if _cloud_store_session(res):
            return True
    except Exception:
        pass
    # トークンが無効なら以後試さない＆Cookie掃除
    st.session_state["_cookie_login_failed"] = True
    _clear_login_cookie()
    return False


def _cloud_store_session(res):
    """sign_in / sign_up / refresh の戻り値からトークンとユーザー情報を session_state に保存。"""
    session = getattr(res, "session", None)
    user = getattr(res, "user", None)
    if session:
        st.session_state["sb_access_token"] = session.access_token
        st.session_state["sb_refresh_token"] = session.refresh_token
        _save_login_cookie(session.refresh_token)  # 「ログイン保持」Cookieを更新
    if user:
        st.session_state["auth_user"] = getattr(user, "email", None) or "user"
        st.session_state["auth_user_id"] = user.id
    return bool(session)


def _cloud_sign_in(email, password):
    try:
        res = _get_supabase().auth.sign_in_with_password(
            {"email": (email or "").strip(), "password": password or ""}
        )
        if _cloud_store_session(res):
            st.rerun()
        else:
            st.error("ログインに失敗しました。メール確認が必要な場合は、確認を済ませてからお試しください。")
    except Exception as e:
        st.error(f"ログインに失敗しました：{e}")


def _cloud_sign_up(email, password, password_confirm):
    if password != password_confirm:
        st.error("確認用パスワードが一致しません。")
        return
    ok, msg = _check_password_strength(None, password)
    if not ok:
        st.error(msg)
        return
    try:
        res = _get_supabase().auth.sign_up(
            {"email": (email or "").strip(), "password": password or ""}
        )
        if _cloud_store_session(res):
            st.success("登録してログインしました。")
            st.rerun()
        else:
            st.success("登録しました。確認メールが届いた場合は、リンクを開いてからログインしてください。")
    except Exception as e:
        st.error(f"登録に失敗しました：{e}")


def _cloud_require_login():
    if st.session_state.get("auth_user") and st.session_state.get("auth_user_id"):
        return st.session_state["auth_user"]
    # Cookie（前回ログイン）から自動ログインを試す
    if _try_cookie_login() and st.session_state.get("auth_user_id"):
        return st.session_state["auth_user"]
    st.markdown("## 🔐 自律型AI塾")
    st.caption("メールアドレスとパスワードでログインします。初めての方は「新規登録」から。")
    tab_login, tab_signup = st.tabs(["ログイン", "新規登録"])
    with tab_login:
        with st.form("cloud_login_form"):
            email = st.text_input("メールアドレス")
            pw = st.text_input("パスワード", type="password")
            if st.form_submit_button("ログイン", type="primary", use_container_width=True):
                _cloud_sign_in(email, pw)
    with tab_signup:
        with st.form("cloud_signup_form"):
            su_email = st.text_input("メールアドレス", key="cloud_su_email")
            su_pw = st.text_input("パスワード（8文字以上・英字と数字を含む）", type="password", key="cloud_su_pw")
            su_pw2 = st.text_input("パスワード（確認）", type="password", key="cloud_su_pw2")
            if st.form_submit_button("登録する", use_container_width=True):
                _cloud_sign_up(su_email, su_pw, su_pw2)
    st.stop()


def cloud_load_data():
    uid = st.session_state.get("auth_user_id")
    if not uid:
        return dict(DEFAULT_DATA)
    try:
        res = _get_supabase().table("user_data").select("data").eq("user_id", uid).limit(1).execute()
        rows = res.data or []
        if rows and isinstance(rows[0].get("data"), dict) and rows[0]["data"]:
            return rows[0]["data"]
    except Exception as e:
        st.error(f"クラウドからのデータ読み込みに失敗しました：{e}")
    return dict(DEFAULT_DATA)


def cloud_save_data(data):
    uid = st.session_state.get("auth_user_id")
    if not uid:
        return
    try:
        _get_supabase().table("user_data").upsert(
            {"user_id": uid, "data": data}, on_conflict="user_id"
        ).execute()
    except Exception as e:
        st.error(f"クラウドへの保存に失敗しました：{e}")


# ============================================================
# データの読み書き（モードに応じて自動切替）— app.py から呼ぶ
# ============================================================
def load_user_data():
    if is_cloud():
        return cloud_load_data()
    user = st.session_state.get("auth_user")
    path = user_data_path(user) if user else None
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                return json.load(f)
        except Exception:
            pass
    return dict(DEFAULT_DATA)


def save_user_data(data):
    if is_cloud():
        cloud_save_data(data)
        return
    user = st.session_state.get("auth_user")
    if not user:
        return
    with open(user_data_path(user), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================
# Streamlit 用ゲート / ログアウト
# ============================================================
def require_login():
    """モードに応じてログイン画面を表示。ログイン済みなら識別子を返す。
    ※ 呼び出し前に st.set_page_config を済ませておくこと。"""
    if is_cloud():
        return _cloud_require_login()
    return _local_require_login()


def _local_require_login():
    """（ローカルモード）ユーザー名＋パスワードのログイン/登録。"""
    if st.session_state.get("auth_user"):
        return st.session_state["auth_user"]

    st.markdown("## 🔐 自律型AI塾")
    st.caption("学習データはアカウントごとに保存されます。初めての方は「新規登録」から。")

    tab_login, tab_signup = st.tabs(["ログイン", "新規登録"])

    with tab_login:
        with st.form("login_form"):
            lu = st.text_input("ユーザー名")
            lp = st.text_input("パスワード", type="password")
            if st.form_submit_button("ログイン", type="primary", use_container_width=True):
                if login(lu, lp):
                    st.session_state["auth_user"] = lu.strip()
                    st.rerun()
                else:
                    st.error("ユーザー名またはパスワードが違います。")

    with tab_signup:
        with st.form("signup_form"):
            su = st.text_input("ユーザー名（半角英数字3〜20文字）", key="signup_user")
            sp = st.text_input("パスワード（8文字以上・英字と数字を含む）", type="password", key="signup_pw")
            sp2 = st.text_input("パスワード（確認）", type="password", key="signup_pw2")
            if st.form_submit_button("登録する", use_container_width=True):
                ok, msg = register(su, sp, sp2)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    st.stop()


def logout_button():
    """サイドバーにログイン中ユーザー名とログアウトボタンを表示。"""
    user = st.session_state.get("auth_user")
    if not user:
        return
    st.sidebar.markdown(f"👤 **{user}** さんでログイン中")
    if st.sidebar.button("🚪 ログアウト", use_container_width=True):
        if is_cloud():
            try:
                _get_supabase().auth.sign_out()
            except Exception:
                pass
            _clear_login_cookie()
            for k in ("sb_access_token", "sb_refresh_token", "auth_user_id", "_cookie_login_failed"):
                st.session_state.pop(k, None)
        st.session_state.pop("auth_user", None)
        st.rerun()
