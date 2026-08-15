# revlab: 可逆な二値バックボーン実験

全結合BNNの中間表現を、情報を捨てない可逆変換に置き換える実験です。既存のcrow実装は変更せず、このフォルダだけで完結します。

## モデル

MNIST/Fashion-MNISTの784ビット入力を1024ビットへ単射的に埋め込み、可逆カップリングを重ね、最後だけ通常のBEP Denseで固定ETFプロトタイプへ写します。

```text
784 bit input
  -> 1024 bitへ固定値(-1)でpadding（入力はそのまま残る）
  -> 可逆coupling x N
  -> BEP Dense 1024 -> 1024
  -> 固定ETFプロトタイプとのHamming距離で分類
```

状態を半分の `left, right` に分け、1ブロックを次のように定義します。

```text
q       = sign(W * left + b)
toggled = right * q                 // ±1の積。ビット表現ではXNOR
output  = (toggled, left)           // 半分を交換
```

逆変換は次のとおりです。

```text
left  = output.right
q     = sign(W * left + b)
right = output.left * q
input = (left, right)
```

`q*q=1` なので、重みやバイアスの値に関係なく、推論時のブロックはビット単位で厳密に元へ戻せます。これは残差接続ではなく、二値状態上の全単射です。

## 学習

- 最終Denseは従来BEPと同じく、正解ETFプロトタイプを希望出力として局所更新します。
- 可逆ブロックは、後段から届いた希望状態を現在のブロックで逆写像し、前段へ渡します。
- 同時に `right` を希望する `toggled` へ変えるための `q` を求め、ブロック内部のDenseをBEPで更新します。
- バイアス有りでは、ニューロンごとの更新時に重みか整数バイアスの一方だけを選びます。バイアスは選ばれたときだけ±1動きます。
- 推論と可逆性検査は浮動小数点を使わず、ビット演算・整数加算・整数比較だけです。

既定の `revnoise=0` は重要です。ブロック内部へ学習時ノイズを入れると、そのサンプルで使った写像と決定論的な逆写像が一致しなくなります。

## 比較モード

- `dense`: 従来型の `784 -> 512 -> 1024` BEP Dense。
- `project`: 784ビットを1024ビットへ埋め、可逆ブロックなしで最終Denseだけ学習。単純な読み出し対照。
- `reversible -learnrev=false`: 固定ランダム可逆特徴 + 学習する最終Dense。
- `reversible -learnrev=true`: 可逆ブロックと最終Denseを一貫して学習。

## 実行例

```powershell
go run ./model/mlp/binary/revlab --mode reversible --blocks 2 --epochs 20 --seed 1
```

バイアスを切って可逆構造だけ比較する場合:

```powershell
go run ./model/mlp/binary/revlab --mode reversible --blocks 2 --bias=false --biaschoice 0 --epochs 20 --seed 1
```

主要フラグは `--dataset mnist|fashion`、`--mode dense|project|reversible`、`--blocks`、`--learnrev`、`--bias`、`--biaschoice`、`--state`、`--output`、`--epochs`、`--seed` です。

## 境界

- 可逆なのはpadding後のバックボーンです。最終Denseとクラス決定は可逆ではありません。
- 784から1024へのpaddingは入力情報を保存しますが、1024状態の全域から784入力への全単射ではありません。
- この実装は、深い二値NNを可逆に構成して実際に学習できるかを試すものです。この特定の有限幅・整数制約モデルについて万能近似定理を証明するものではありません。
- 結果の詳細は [REPORT.md](REPORT.md)、実行条件は [LOGS.md](LOGS.md) にあります。

