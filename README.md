# keiba-ai

競馬レース情報を収集し、上位3頭と波乱を予測するStreamlitアプリです。

## 起動

Neonを使う場合は、`.env`に`DATABASE_URL`を設定して起動します。

```powershell
docker compose up -d --build
```

アプリ: http://localhost:8501

## Streamlit Community Cloud

アプリの **Settings → Secrets** に以下を設定し、再起動または再デプロイします。

```toml
DATABASE_URL = "Neonの接続URL"
```

## 週次自動更新

GitHubリポジトリの **Settings → Secrets and variables → Actions** に
`NEON_DATABASE_URL`を登録してください。毎週土曜日の日本時間午前9時ごろに
クローラーと2種類の学習モデルを実行し、更新したモデルをリポジトリへpushします。
実行状況は **Actions → Weekly Keiba Update** で確認できます。

接続URLやパスワードはGitやチャットに記載しないでください。
