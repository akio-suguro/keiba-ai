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
`NEON_DATABASE_URL`を登録してください。毎週の更新では、まずpytestを実行し、
成功した場合だけクロール・学習・予測保存を続けます。上位3頭モデルは
レース単位のLightGBM LambdaRankで学習し、直近5年で学習・評価しつつ、
過去10年分の結果から出走前特徴量を作り、
人気・オッズを除いたモデルとの比較、ウォークフォワード評価、
単勝・馬連・3連複の払戻シミュレーションを記録します。
波乱モデルは時系列検証期間で確率を校正し、上位10%の波乱率も評価します。
今後の出走表の予測は`predictions`テーブルへ先に保存し、アプリは保存済み結果を
読み込みます。結果が確定した週次予測は実績と払戻で追跡評価します。
失敗したクロールURLはバックオフ付きの再取得キューに残します。

更新済みモデルと評価指標は、週次ジョブが GitHub Release に公開します。
アプリは公開中の最新 Release からモデル一式を取得し、1時間キャッシュします。
この方式を使うにはリポジトリを公開状態にしてください。モデルファイルは
今後の週次コミットには含めません。初回 Release を作るときや手動で更新したいときは
**Actions → Weekly Keiba Update → Run workflow** を実行してください。
手動実行は週内の完了状態に関係なくクロール・学習・予測・Release 公開を行います。
GitHub ActionsのcronはUTC指定のため、`23:17 Friday UTC`を日本時間の土曜日午前8時17分に設定しています。予定時刻に起動しない場合に備え、午前10時17分にも再試行します。週内に一度正常完了していれば、再試行は自動でスキップされます。
実行状況は **Actions → Weekly Keiba Update** で確認できます。

接続URLやパスワードはGitやチャットに記載しないでください。

`src/keiba-ai/schema.sql`はクローラーとアプリのDBスキーマです。
正規化CSV取込専用のスキーマは`src/keiba-ai/import_schema.sql`に分離しています。
依存パッケージは`requirements.txt`で直接依存を固定し、
CIとDockerは推移依存も固定した`requirements.lock`を使います。
