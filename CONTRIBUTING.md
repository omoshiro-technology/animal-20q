# Contributing Guide

Animal 20Q への貢献を歓迎します！

## 貢献の方法

### 1. 設問の追加・修正

設問の品質がゲームの面白さを左右します。以下の基準を満たす設問を追加してください。

#### 設問の基準（必須）

1. **分類学的特徴のみ**
   - 形態・解剖学的特徴に基づく質問のみ
   - NG例: 「北極に住んでいますか？」（地理）
   - NG例: 「夜行性ですか？」（行動）
   - OK例: 「羽毛を持っていますか？」
   - OK例: 「外耳介が外から見えますか？」

2. **Yes/No で明確に回答可能**
   - 曖昧さのない質問
   - 年齢・性差で変わる場合は注記
   - 例: 「成獣雄で鼻が長大に伸長しますか？」

3. **出典の明記**
   - `sources` フィールドに情報源を記載
   - 書籍・論文・信頼できるウェブサイト
   - 例: `["FAO Marine Mammals", "Riedman 1990"]`

4. **決定形質の優先使用**
   - その分類群を特徴づける形質
   - 子孫すべてに共通する形質
   - または子孫を明確に二分できる形質

#### 設問の形式

```json
{
  "node_id": "Phocidae",
  "rank": "family",
  "q_id": "Phocidae_P1",
  "question": "成獣雄の鼻部がフード状に膨らみますか？",
  "yes_next": "Cystophora",
  "no_next": "Phocidae_minus_Cystophora",
  "sources": ["traits_core", "FAO Marine Mammals"],
  "confidence": 0.95
}
```

#### confidence の目安

| confidence | 意味 |
|------------|------|
| 0.95+ | 決定形質。例外なく適用 |
| 0.85-0.94 | 高信頼。稀な例外あり |
| 0.70-0.84 | 中信頼。要確認 |
| 0.70未満 | 低信頼。ドラフト段階 |

### 2. 形質カタログの拡充

`rules/traits_core.yaml` に新しい分類群の形質を追加できます。

```yaml
NewTaxon:
  hint: "形質の説明（日本語）"
  rank: family
  parent: ParentTaxon
  vernacular: "和名"
```

### 3. バグ報告・機能提案

- Issue を作成してください
- バグの場合: 再現手順を記載
- 機能提案の場合: ユースケースを説明

## Pull Request の作成

### 自動生成されたPRのレビュー

GitHub Actions が毎日生成する設問ドラフトPRをレビューしてください：

1. `data/questions.jsonl` の diff を確認
2. 各設問が上記の基準を満たすかチェック
3. 問題があれば：
   - 修正をコミットしてからマージ
   - または、コメントで指摘

### 手動でPRを作成する場合

1. このリポジトリをフォーク
2. 新しいブランチを作成: `git checkout -b feature/my-feature`
3. 変更を加える
4. コミット: `git commit -m "Add: 説明"`
5. プッシュ: `git push origin feature/my-feature`
6. Pull Request を作成

### PRのテンプレート

```markdown
## Summary
- 追加/変更したノードID
- 変更の概要

## 参考文献
- 書籍・論文・URL

## チェックリスト
- [ ] 質問は形質のみに基づいている
- [ ] Yes/No で明確に回答できる
- [ ] 出典を記載した
- [ ] confidence を適切に設定した
```

## コードスタイル

### Python

- Python 3.11+ 対応
- 型ヒント推奨
- docstring 推奨

### YAML

- インデント: スペース2つ
- 日本語コメント可

### JSON/JSONL

- UTF-8 エンコーディング
- 1行1レコード（JSONL形式）

## 質問・相談

- Issue で質問を作成してください
- Discussion（有効な場合）も利用可能

---

貢献いただきありがとうございます！
