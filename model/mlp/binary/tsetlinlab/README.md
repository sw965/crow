# tsetlinlab — Tsetlin Machine 検証実験

「(1) 万能近似性を満たし、(2) 学習から推論まで高速なビット演算・整数中心で、
(3) 現行binaryモデル(BEP実装)および BEP論文と同等以上の精度が実際に出る手法を探す」
という依頼(2026-08-13)に対する調査・実験の成果物一式。

結論は **Tsetlin Machine (整数重み付きマルチクラスTM)**。
crow の BEP 実装と同一の二値化済みデータで MNIST 97.24±0.12%(BEP実装 92.13±0.48%)、
Fashion-MNIST 87.44±0.11%(BEP実装 83.78%、論文報告 85〜87%)を実測した。

## ファイル構成

| ファイル | 内容 |
|---|---|
| `REPORT.md` | 調査・実験の完全な報告。手法の説明、万能近似の構成、実験結果、他候補の棄却理由、文献 |
| `LOGS.md` | 実験の生ログ(エポックごとの精度・時間) |
| `main.go` | ビット並列TMの実装(実験プログラム本体、package main) |
| `main_test.go` | ビットスライス演算の参照実装との一致検証、発火判定、minterm構成による任意写像の厳密表現テスト(万能近似の核の実証)、合成タスクでの学習性テスト、Fuzz |

## データの入手

**重要(2026-08-14、2026-08-14 追記): 本プログラムは旧形式gob(リリース v0.1.0-test)専用で、未修正のまま。**
データセットgobは現行bitsx形式(GobEncode/GobDecode)で再エンコードされ、
リリース v0.2.0-test として公開済み(REPORT.md §9 の解決の追記を参照)。
現行の `dataset.LoadMNIST` / `LoadFashionMNIST` は v0.2.0-test(現行形式)を
ダウンロード・デコードできることを実機で確認済みで、`~/.crow_dataset` の
キャッシュも既に現行形式に置き換わっている。

**本プログラムだけが未対応。** `main.go` の `oldMatrix`/`loadDataset` は、
`dataset.LoadMNIST` が旧バグ(REPORT.md §9)でデコードできなかった当時に、
それを迂回して生の gob を独自形式で直接読むために書かれたもの。
**バグは直っているので、この迂回コードはもう不要。**
正しい直し方は「旧形式ファイルを別途用意する」ことではなく、
**`loadDataset` を丸ごと `dataset.LoadMNIST` / `dataset.LoadFashionMNIST` の
呼び出しに置き換えること**。この2関数のシグネチャ(引数・戻り値の型)は
今回の修正で変えていないため、呼び出し側のインターフェースはそのまま使える。
戻り値 `Binary.TrainInputs` 等は `bitsx.Matrices`(`[]*bitsx.Matrix`)であり、
`packSamples` が直接触っている `img.Rows`/`img.Cols`/`img.Data[w]` は
非公開フィールドになっているため、`img.Rows()`/`img.Cols()`/`img.Word(w)`
(公開API、`omw/mathx/bitsx/matrix.go`)に置き換える必要がある。
**この置き換えは未実施・未検証。** 実施する場合は、置き換え後に
`go test ./model/mlp/binary/tsetlinlab/...` および `go run . -dataset mnist ...`
を実行し、既存の精度(README冒頭の実測値)が再現することを確認すること。

置き換えるまでの間、`~/.crow_dataset` は既に現行形式になっているため、
本プログラムをそのまま実行すると `loadGob[oldMatrices]` のデコードに失敗する。
動かす場合は、旧形式(v0.1.0-test)のgobを別ディレクトリに用意し、
`-datadir` で指定する:

```text
https://github.com/sw965/crow/releases/download/v0.1.0-test/<ファイル名>

MNIST:
  mnist_train_flat_binary_imgs.gob   mnist_train_int_labels.gob
  mnist_test_flat_binary_imgs.gob    mnist_test_int_labels.gob
Fashion-MNIST:
  fashion_mnist_train_flat_binary_imgs.gob   fashion_mnist_train_int_labels.gob
  fashion_mnist_test_flat_binary_imgs.gob    fashion_mnist_test_int_labels.gob
```

## 実行方法

```sh
go run . -dataset mnist   -clauses 512  -epochs 20            # MNIST: BEST 0.9733 (シード1)
go run . -dataset fashion -clauses 2048 -slog2 4 -epochs 30   # Fashion: BEST 0.8742 (シード1)
go run . -dataset fashion -clauses 4096 -slog2 4 -epochs 40   # Fashion: BEST 0.8778 (シード1)
```

シード・設定が同じなら結果は決定的に再現される(ワーカー数に依存しない)。

## 注意

- これは実験プログラム(データ収集プログラム相当)であり、binaryパッケージの
  ライブラリコードからは参照されない。層構造の設計規約の適用外。
- 学習・推論の経路に浮動小数点演算はない(精度の表示のみfloat)。
