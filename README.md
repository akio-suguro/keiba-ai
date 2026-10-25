# keiba-ai

## データベースの設定

アプリとJRAクローラーは、標準ではローカルのSQLiteを使用します。
PostgreSQLを使う場合は、`DATABASE_URL`に接続URLを設定してください。
接続URLは環境変数からのみ読み込み、リポジトリにコミットしないでください。

### 移行後の設定

SQLiteからNeonへの初回データ移行は完了しています。一度限りの移行スクリプトは削除済みです。
アプリをNeonに接続するには、リポジトリ直下の`.env`に`DATABASE_URL`を設定して、
次のコマンドで起動します。

```powershell
docker compose up -d --build
```

アプリは`http://localhost:8501`で開けます。`.env`はGitの追跡対象外ですが、
このリポジトリをOneDriveなどの同期フォルダーに置いている場合、接続URLも同期されることがあります。
接続情報を共有せず、Neonのパスワードを定期的に管理してください。

### GitHub Actionsで日次クローラーを実行する

GitHubリポジトリの **Settings → Secrets and variables → Actions** を開き、
**New repository secret** を選択します。名前を`NEON_DATABASE_URL`、
値をNeonの新しい接続URLにして保存してください。接続URLをリポジトリやチャットに記載しないでください。

設定後、**Actions → Daily Keiba Update → Run workflow** から手動実行して成功することを確認します。
以後は毎日日本時間の午前0時30分ごろに、GitHub ActionsがNeonへ直接接続してクローラーを実行します。
PCを起動しておく必要はありません。Secretが未設定の場合はワークフローが早い段階で失敗します。
GitHub ActionsはDBファイルをリポジトリにコミットしません。

ローカルアプリでは、`.env`の`DATABASE_URL`が設定されていればNeonを使用します。
未設定の場合、ComposeとPythonスクリプトはローカルSQLiteを使用します。
