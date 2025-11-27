# Animal 20Q: Taxonomy Edition

**動物分類ゲーム** - Yes/Noの質問に答えて、思い浮かべた動物を当てるゲームです。

分類学的特徴（形態・解剖学的特徴）のみに基づく質問で、動物界（Animalia）のルートから種レベルまで降りていきます。

## 遊び方

1. 心の中で動物を1つ思い浮かべます
2. 出題される質問に「はい」「いいえ」で回答します
3. 質問は分類学の特徴（羽毛、乳腺、外骨格など）に基づきます
4. 正解にたどり着くか、候補が絞り込まれます

### ゲームURL

GitHub Pages: `https://<username>.github.io/animal_quiz/app/`

## プロジェクト構成

```
animal_quiz/
├── app/                    # ゲーム（静的SPA）
│   ├── index.html
│   ├── main.js
│   └── style.css
├── data/
│   ├── backbone/           # 参照データ（git管理外）
│   ├── taxonomy.jsonl      # 分類木
│   └── questions.jsonl     # Yes/No設問集
├── rules/
│   ├── templates.yaml      # 質問テンプレート
│   ├── traits_core.yaml    # 決定形質カタログ
│   └── mapping_aliases.csv # 同物異名
├── scripts/
│   ├── fetch_backbone.py   # CoL/NCBIデータ取得
│   ├── build_taxonomy.py   # taxonomy.jsonl生成
│   ├── draft_questions.py  # 設問自動ドラフト
│   └── copy_to_pages.py    # docs/へ同期
├── docs/                   # GitHub Pages公開用
├── .github/workflows/      # GitHub Actions
├── README.md
├── CREDITS.md              # 出典・ライセンス
└── CONTRIBUTING.md         # 貢献ガイド
```

## データ形式

### taxonomy.jsonl

1行1ノードのJSON形式:

```json
{"taxonID":"Phocidae","parentID":"Carnivora","rank":"family","scientificName":"Phocidae","vernacularName":"アザラシ科","children":["Phoca","Pusa"]}
```

### questions.jsonl

1行1設問のJSON形式:

```json
{"node_id":"Phocidae","rank":"family","q_id":"Phocidae_P1","question":"成獣雄の鼻部がフード状に膨らみますか？","yes_next":"Cystophora","no_next":"Phocidae_minus_Cystophora","sources":["traits_core"],"confidence":0.95}
```

## 自動化（GitHub Actions）

### 週1回: 参照データ更新

- CoL（Catalogue of Life）とNCBI Taxonomyからデータを取得
- `taxonomy.jsonl`を再生成
- PRを自動作成

### 毎日: 設問ドラフト

- 未充足ノードにYes/No設問を自動提案
- **Claude Haiku 4.5** を使ったLLM生成（オプション）
- `questions.jsonl`を更新
- PRを自動作成（レビュー後にマージ）

### LLM設問生成の設定

1. GitHubリポジトリの Settings → Secrets and variables → Actions
2. `ANTHROPIC_API_KEY` シークレットを追加
3. Actions の手動実行で `use_llm: true` を選択

LLMが生成した設問は `sources: ["llm", "claude-haiku-4.5"]` でマークされます。

## ローカル開発

### 必要環境

- Python 3.11+
- pip (pyyaml, pandas, requests)

### セットアップ

```bash
# リポジトリをクローン
git clone https://github.com/<username>/animal_quiz.git
cd animal_quiz

# Python依存関係をインストール
pip install pyyaml pandas requests

# 初期データを生成（ルールベース）
python scripts/draft_questions.py \
  --taxonomy data/taxonomy.jsonl \
  --questions data/questions.jsonl \
  --traits rules/traits_core.yaml \
  --max 100 \
  --out data/questions.jsonl

# LLMを使って設問生成（要APIキー）
export ANTHROPIC_API_KEY='your-api-key'
python scripts/draft_questions.py \
  --taxonomy data/taxonomy.jsonl \
  --questions data/questions.jsonl \
  --traits rules/traits_core.yaml \
  --max 20 \
  --use-llm true \
  --out data/questions.jsonl

# docs/にコピー
python scripts/copy_to_pages.py

# ローカルサーバーで確認
cd app
python -m http.server 8000
# ブラウザで http://localhost:8000 を開く
```

### 参照データの取得（オプション）

```bash
# CoLデータ取得
python scripts/fetch_backbone.py --source col --out data/backbone/col/

# NCBIデータ取得
python scripts/fetch_backbone.py --source ncbi --out data/backbone/ncbi/

# taxonomy.jsonl生成
python scripts/build_taxonomy.py --col data/backbone/col/ --out data/taxonomy.jsonl
```

## 貢献方法

設問の追加・修正を歓迎します！詳細は [CONTRIBUTING.md](CONTRIBUTING.md) を参照してください。

### 設問の基準

1. **分類学的特徴のみ**: 形態・解剖学的特徴に基づく質問
2. **Yes/No明確**: 曖昧さのない回答が可能
3. **出典明記**: 参考文献・情報源を記載
4. **決定形質優先**: その分類群を特徴づける形質を使用

## ライセンス

MIT License

データの出典については [CREDITS.md](CREDITS.md) を参照してください。
