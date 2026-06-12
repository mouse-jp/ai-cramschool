# Supabase セットアップ手順（クラウド化の準備）

この手順を終えると、ログインと学習データがクラウド（Supabase）に保存され、
ブラウザを更新しても・別の端末からでも、同じアカウントで続きから使えるようになります。
**所要時間 約10分。無料プランでOKです。**

> 終わったら「終わった」と教えてください。あなたのプロジェクトのキーを使って、私がアプリ側の接続を実装します。

---

## ステップ1：アカウントとプロジェクトを作る

1. ブラウザで **https://supabase.com** を開き、右上の **「Start your project」** からサインアップ
   （GitHubアカウント or メールでOK）。
2. ログインしたら **「New project」** をクリック。
3. 次を入力して作成：
   - **Name**：`ai-cram-school`（何でもOK）
   - **Database Password**：強いパスワードを入力（**メモしておく**。後で必要になることがあります）
   - **Region**：`Northeast Asia (Tokyo)` を選ぶ（日本から速い）
4. 「Create new project」を押す → 1〜2分で準備完了。

---

## ステップ2：接続キーを2つコピーする

左メニューの **歯車（Project Settings）→ API** を開き、次の2つを控えます：

| 名前 | どこにある | 用途 |
|---|---|---|
| **Project URL** | 「Project URL」欄（`https://xxxx.supabase.co`） | 接続先 |
| **anon public key** | 「Project API keys」の **`anon` `public`** の長い文字列 | 公開用キー |

> ⚠️ `service_role` キーは**絶対に使わない／共有しない**でください（管理者全権キーです）。使うのは `anon public` の方だけです。

---

## ステップ3：データ保存テーブルを作る（SQLを貼るだけ）

左メニューの **SQL Editor → New query** を開き、下を**そのまま貼り付けて「Run」**を押します。

```sql
-- 学習データ保存テーブル（1ユーザー1行。従来の my_data の中身を JSONB にそのまま入れる）
create table if not exists public.user_data (
  user_id uuid primary key references auth.users(id) on delete cascade,
  data jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now()
);

-- RLS（行レベルセキュリティ）：各ユーザーは自分の行だけ読み書きできる
alter table public.user_data enable row level security;

create policy "own_select" on public.user_data
  for select using (auth.uid() = user_id);

create policy "own_insert" on public.user_data
  for insert with check (auth.uid() = user_id);

create policy "own_update" on public.user_data
  for update using (auth.uid() = user_id) with check (auth.uid() = user_id);
```

「Success. No rows returned」と出れば成功です。

---

## ステップ4：テスト中はメール確認をオフにする（任意・推奨）

初期設定だと、新規登録時にメールのリンク確認が必要で、テストが面倒です。
**Authentication → Sign In / Providers（または Settings）→ Email** で
**「Confirm email」を OFF** にしておくと、登録してすぐログインできます。
（本番公開時は ON に戻すのがおすすめ）

---

## ステップ5：キーをアプリに設定する

プロジェクトフォルダ（app.py と同じ場所）の中に **`.streamlit`** フォルダを作り、
その中に **`secrets.toml`** というファイルを作って、次の内容を書きます
（`xxxx` と `eyJ...` は、ステップ2でコピーした値に置き換え）：

```toml
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "eyJhbGciOi...（anon public キー）"
```

- 既に `.streamlit/secrets.toml` がある場合は、この2行を**追記**してください。
- このファイルは秘密情報なので、GitHub等には上げないようにします（`.gitignore` に `.streamlit/secrets.toml` を追加推奨）。

---

## ステップ6：パッケージを入れる

ターミナル（コマンドプロンプト）でプロジェクトフォルダに移動し：

```
pip install supabase
```

（`requirements.txt` には追記済みなので、`pip install -r requirements.txt` でもOK）

---

## 完了チェック

- [ ] プロジェクトを作成した
- [ ] Project URL と anon public key を控えた
- [ ] SQL を実行して `user_data` テーブルができた
- [ ] （任意）Confirm email を OFF にした
- [ ] `.streamlit/secrets.toml` に2つのキーを書いた
- [ ] `pip install supabase` した

ここまで終わったら教えてください。
アプリ側を **「キーがあればSupabase、無ければ今のローカル保存」** に切り替えて接続し、一緒に動作確認します。

---

### 補足：何が変わる？何が変わらない？

- **変わらない**：単語帳・熟語帳など `my_data` の**データの形**。Gemini APIキーの扱い。アプリの画面。
- **変わる**：保存先がローカルの `users/◯◯.json` → Supabaseの `user_data` テーブル（JSONB列に同じ形で入る）。
  ログインがユーザー名 → **メールアドレス＋パスワード**になります（Supabase認証のため）。
