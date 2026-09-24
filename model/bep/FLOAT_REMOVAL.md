# BEP 本体から浮動小数を取り除く手順書（二値のみ）

`model/bep` の学習カーネルから浮動小数を取り除くための作業手順です。
精度を上げるのではなく、**機構と浮動小数を削って精度を保つ**ことが目的です。

各ステップは独立して完結し、**どこで中断してもビルドとテストが通る**ように並べてあります。

> 三値版（`model/mlp/ternary`）は 2026-09-24 に削除されたため、本書は二値専用です。
> 根拠となる実験結果は [glulab/](glulab/) にあります。
> 2026-09-24 に `model/mlp/binary`（package binary）から `model/bep`（package bep）へ移動した。本書のパスは移動後のもの。
> gob に保存される型名が `*binary.Dense` から `*bep.Dense` に変わったため、移動前に保存したモデルは読み込めない（互換性は持たせない判断をした）。

## 進捗

| ステップ | 状態 |
|---|---|
| 0. 基準値の記録 | — |
| **1. `math.Abs` → べた書き** | **✅ 完了（2026-09-24）** |
| **2. `margin` → 整数ビット数** | **✅ 完了（2026-09-24）** |
| **3. `lr` → 2の冪の確率（`LRHalfPow`）** | **✅ 完了（2026-09-24）** |
| **4. ゲート廃止** | **✅ 完了（2026-09-24）** |
| **5. ガウス → 整数一様ノイズ** | **✅ 完了（2026-09-24）** |
| **6. 仕上げ** | **✅ 完了（2026-09-24）** |

---

## 現状の棚卸し

### 学習ループの中（ホットパス）

| 箇所 | 内容 | 頻度 | 対応 |
|---|---|---|---|
| ~~`layer.go:190`~~ | ~~`int(math.Abs(float64(zi)))`~~ | ニューロン×サンプル | **✅ ステップ1で完了** |
| ~~`train.go:224`~~ | ~~`marginBits := int(float32(totalBits) * margin / 2)`~~ | サンプル | **✅ ステップ2で完了** |
| ~~`layer.go:338`~~ | ~~`rng.Float32() > lr`~~ | 重み×更新 | **✅ ステップ3で完了** |
| ~~`layer.go:168`~~ | ~~`d.GateDropThreshold()`~~ | 層×サンプル | **✅ ステップ4で完了** |
| ~~`layer.go:133`~~ | ~~`randx.IntNorm(..., noiseStd, ...)`~~ | ニューロン×サンプル | **✅ ステップ5で完了** |

### 構築時（1回だけ）

[layer.go:100](layer.go):

```go
noiseStdBase := float32(math.Sqrt(float64(w.Cols())))
```

ステップ4でゲート側の `gateDropThresholdBase := int(noiseStdBase)` は削除済み。残りはステップ5で扱う。

### カーネル外（触らない）

`Accuracy` / `Loss` / `PredictSoftmax` / `PredictValue` / `Model.Values` /
`SplitTrainValidation` / `GridSearch` の集計。評価と復号なので浮動小数のままでよい。

---

## 作業を始める前に知っておくこと（地雷）

### 転置行列の同期漏れ

`Update` で可視重みのビットを反転したら、**必ず転置 `WT` も `Toggle` で同期する**
（[layer.go:341](layer.go) 付近がやっている）。

これを落とすと逆伝播が**初期のランダム転置を使い続ける**。実際にこのバグを出し、
「ゲート廃止で +2.13pt」という結果を全撤回したことがある。
**精度は90%前後出てしまうので、精度では気づけない。**

[glulab/main_test.go](glulab/main_test.go) の
`TestTransposeStaysInSyncAfterUpdate` が不変条件を固定している。

### 乱数消費が変わる変更は単一シードで比較できない

`rng.Float32()` / `rng.IntN()` / `randx.IntNorm()` は消費ビット数が違う。
同じシードでも以降の乱数列がずれるので、**必ず3シード以上の平均で比較する。**
ステップ3と5が該当する。

### 検証の作法

- 条件を変えたら**値が動くこと**を短い試走で確認してから本実験に入る。
  値が完全一致したら「差が無い」ではなく「**分岐が効いていない**」を先に疑う
- スクリプトで一括置換する場合は必ず存在確認を入れる（無言失敗で実験が丸ごと無効になったことがある）
- Python で書き戻すときは `io.open(p,'w',encoding='utf-8',newline='')`。
  `newline` を指定しないと Windows で CRLF になり `gofmt` が落ちる

---

## ステップ0: 基準値の記録（必須）

変更の前後を比べられるようにします。**これをやらないと退行に気づけません。**

```powershell
go test ./... 2>&1 | Tee-Object -FilePath baseline_test.txt
go run ./examples/mnist 2>&1 | Tee-Object -FilePath baseline_mnist.txt
```

**`examples/mnist` は50エポック回ります**（[examples/mnist/main.go:65](../../examples/mnist/main.go)）。
時間を惜しむなら `glulab` を `-epochs` を絞って回すのでも代用できます。

参考値（`glulab -arch plain`、20エポック、3シード）:

| | MNIST | Fashion |
|---|---:|---:|
| 現行相当（ゲート有り） | 90.41 ± 0.16 | 82.46 ± 0.22 |
| ゲート無し | 90.64 ± 0.08 | 82.89 ± 0.25 |

---

## ステップ1: `math.Abs` の float 往復を削る ✅ 完了

**2026-09-24 に適用済み。** 差分は1箇所のみ。

```go
// layer.go:190 付近
  zi := zWord[i]
- absZi := int(math.Abs(float64(zi)))
+ absZi := zi
+ if absZi < 0 {
+ 	absZi = -absZi
+ }
```

ヘルパー関数は作らず、べた書きにしてある。
`biaslab` / `e2elab` / `swalab` が既に同じ書き方をしており、
差分テストで本体とラボを見比べるときに揃っている方が読みやすいため。

`math` インポートは残る（`math.Sqrt` と `math.MinInt8` / `math.MaxInt8` で使用中）。

検査: `gofmt` / `go vet` / `staticcheck` / `errcheck` 指摘なし、テスト全パス、`-race` OK。
計算結果は完全に同一。

---

## ステップ2: `margin` を整数ビット数にする ✅ 完了

**2026-09-24 に適用済み。**

### 分かったこと: `Margin` は2つの意味で使い回されていた

| 呼び出し | 意味 | 整数化 |
|---|---|---|
| `SatisfiesUpdateCriterion`（分類・回帰の更新判定） | 出力ビット数に対する比率 | 可能 |
| `PairwiseLabels`（ランキング学習） | 復号した値の値域に対する比率 | 不可（値空間が float32） |

回帰（サーモメータ符号 + `Values`）は専用の学習経路を持たず `Train` を通るので、
**分類と同じ更新判定のマージンを使う。**

### 実装

**マージンを「出力の総ビット数」ではなく「プロトタイプ間の最小距離」に対する比率で持つ。**

```go
type Trainer struct {
	MiniBatchSize  int
	LR             float32
	LogitMargin    float32   // 最小距離に対する比率(0〜1)。分類・回帰共通
	PairwiseMargin float32   // ランキング用。値の空間なので単位が違う
	...
}

// ComputeSeqSignDelta の先頭で、バッチごとに1回だけ計算する
minDist, err := minPrototypeHammingDistance(prototypes)
marginBits := int(t.LogitMargin * float32(minDist))

func SatisfiesUpdateCriterion(..., marginBits int) (bool, error)   // 旧: margin float32
```

理由: 正解のプロトタイプが他より何ビット近いか（リード）は、三角不等式により
「プロトタイプ間の距離」を超えられない。総ビット数を基準にすると、隣接段が近い
温度計（回帰）では小さな値でも上限を超え、判定が常に true になっていた。
最小距離を基準にすれば、ratio は「正解の領域のどれだけ内側に入ったら止めるか」
という、符号によらない意味になる（出力が正解から `(1 - ratio) × 最小距離 / 2` 以内なら確実に止まる）。

- ビット数は Trainer に保存しない（プロトタイプから毎回導出するので古くならない）
- `Validate` は `0 <= LogitMargin <= 1` を確かめるだけで、値を書き換えない
- セッターは不要。`LR` と同じ公開フィールド
- 学習の内側（サンプルごと）の浮動小数乗算は消えた
- 最小距離はプロトタイプごとではなく全体の最小を使う（実測でばらつきは最大6%程度）

### 波及させた箇所

| ファイル | 対応 |
|---|---|
| `pairwise.go:111` | `t.Margin` → `t.PairwiseMargin` |
| `examples/mnist/main.go`、`nnlab/bepreadout/main.go` | `trainer.Margin = 0.5` → `trainer.LogitMargin = 0.5` |
| `margin_internal_test.go`（新規） | `minPrototypeHammingDistance` のテスト |
| `bep_test.go` の `TestTrainerValidate` | `LogitMargin` の範囲外を弾くことのテストを追加 |
| `bep_test.go` | `0.5` → `marginBits = 2`（元コメントの `8 × 0.5 / 2 = 2` のとおり） |
| `biaslab` / `e2elab` / `revlab` の `main_test.go` | crow を呼ぶ側で比率→ビット数に変換（差分テストの意味を保つため） |

`search.go` / `search_test.go` は依頼者の指示で削除した（参照元ゼロ）。

### 注意1: 等価なのは「入力の行数 = `XRows`」のとき

旧実装は実際の出力 `y` の形状からビット数を計算していたが、
新実装はモデルが宣言する出力形状（`OutputShape(XRows, XCols)`）から計算する。
`Train` は入力の行数を `XRows` と照合していないため、**宣言と違う行数の入力で学習すると
ビット数が旧実装と変わる。** 通常の使い方（`examples/mnist` など）では一致している。

### 注意2: 分類と回帰で適正値が桁違いに違う（実測）

更新判定を満たせる上限は `ratio <= 2 × プロトタイプ間の最小距離 / L`。
L = 1024 での実測:

| 符号 | n | 最小距離 | ratio の上限 |
|---|---:|---:|---:|
| ETF（分類） | 10 | 563 | 1.10 |
| ETF（分類） | 101 | 453 | 0.88 |
| 温度計（回帰） | 10 | 113 | 0.22 |
| 温度計（回帰） | 21 | 51 | 0.10 |
| 温度計（回帰） | 101 | 10 | **0.0195** |

温度計の上限は理論上 `2/(n-1)` で、L に依存せず段数 n だけで決まる。
**既定値 0.5 は回帰では n=10 ですら2.3倍超過し、判定が常に true（=全サンプル更新）になる。**
[PROTOTYPES.md](PROTOTYPES.md) B-3 と同じ結論。

### 実験結果（分類）と既定値

glulab に同じ計算を入れて、3シード×2データセット×20エポックで比べた。

| 基準 | 設定値 | marginBits（10クラス・1024ビット） | MNIST | Fashion |
|---|---:|---:|---:|---:|
| 旧（総ビット数） | 0.5 | 256 | 90.64 ± 0.08 | 82.89 ± 0.25 |
| 新（最小距離） | 0.5 | 約281 | 90.33 ± 0.28 | 82.70 ± 0.26 |
| 新（最小距離） | **0.45** | 約253 | **90.64 ± 0.06** | **82.73 ± 0.04** |

同じ 0.5 だと判定が約1割厳しくなって少し下がるが、0.45 で厳しさを揃えると精度は戻る。
**`LogitMargin` の既定値を 0.45 にした**（`PairwiseMargin` は 0.5 のまま。定数も分けた）。
`examples/mnist` と `nnlab/bepreadout` の明示的な設定も 0.45 にした。

旧方式と同じ厳しさになる値はクラス数で変わる（101クラスなら約0.55）。基準を変えた以上、
全条件で旧方式と一致する単一の値は無い。

### 未検証

回帰では「常に更新」から「マージンが効く」状態に変わっている（n=101 で約4ビット）。
回帰は現在どこからも使われていないので、使い始める前に確かめればよい。

---|---|---|
| 分類 | marginBits が 256 → 約281（n=10）、約231（n=101） | 3シードで精度が落ちないか |
| 回帰 | 常に更新 → 5ビット（n=101） | 0 / 0.5 / 1 と「常に更新」のどれが良いか |

回帰の既存の実測結果（PROTOTYPES.md）はすべて「常に更新」の状態で得られたもので、
マージンが効いた方が良いかどうかは分かっていない。

---

## ステップ3: `lr` を2の冪の確率にする ✅ 完了

**2026-09-24 に反映済み。** 当初は有理数（`lrNum/lrDen`）にする予定だったが、速度を優先して
更新確率を (1/2)^n に限る形に変えた。

### なぜ有理数ではなく2の冪か

有理数版（`rng.IntN(lrDen) >= lrNum`）は、要素ごとに乱数を1回引く点が `rng.Float32()` と変わらず、
速くならない。2の冪に限ると、「各ビットが確率 1/2 の乱数を n 個 AND する」だけで
**64要素分の更新対象を一度に決められる**。

`Dense.Update` 単体（784→512）の実測:

| 方式 | 時間 |
|---|---:|
| 旧（要素ごとに `rng.Float32() > lr`） | 1.75ms |
| 新（1ワードごとにマスクを作り、1 のビットだけをたどる） | 約0.36ms |

### 変更したもの

* omw: `bitsx.RandHalfPow[B](n, rng)` を追加（各ビットが確率 (1/2)^n で 1）。
  `NewRandMatrix(rows, cols, k, rng)` を `NewRandMatrix(rows, cols, rng)`（確率 1/2）と
  `NewRandMatrixHalfPow(rows, cols, n, rng)` に分け、OR で確率を上げる方向（k > 0）は削除した。
* [layer.go](layer.go) の `Dense.Update`: `lr float32` → `lrHalfPow int`。
  1ワードごとに `RandHalfPow` でマスクを作り、`bits.TrailingZeros64` で 1 のビットだけを更新する。
  行の最後のワードは要素が64個未満のことがあるので、マスクを要素数に切り詰めている。
  `ScanBits` は全要素を回すため使わず、`WT.Toggle` の列は `ctx.ColStart+i` で求める。
* `Layer` インターフェース、`Sequence.Update` も `int` に変更。
* [train.go](train.go): `LR float32` → `LRHalfPow int`（既定値 3 = 1/8）。
  `Validate` は `LRHalfPow < 0` を弾く（0 は「毎回更新」として有効）。
* `bep_test.go`: `TestTrainerValidate` の `LR = 0.0` → `LRHalfPow = -1`。
  `TestDenseUpdate` を追加（lrHalfPow = 0 で全要素が更新されること、1 と 3 で更新割合が
  1/2・1/8 になること、端数ワード、W = sign(H) と WT の同期、負の値のエラー）。

### 検証済みの結果

`glulab -arch plain` で3シード×2データセット×20エポック:

| 更新確率 | MNIST | Fashion |
|---|---:|---:|
| 0.1（旧既定値） | 90.64 ± 0.06 | 82.73 ± 0.04 |
| **1/8**（新既定値） | 90.55 ± 0.15 | 82.83 ± 0.12 |
| 1/16 | 90.56 ± 0.15 | 82.61 ± 0.22 |

**差はどれもばらつきの範囲内。** 0.1 に最も近い 1/8 を既定値にした。

### 注意

* 乱数の消費の仕方が変わったので、同じシードでも旧実装とはビット単位で一致しない。
* biaslab の `TestUpdateMatchesCrow` は crow の Update と乱数消費まで一致することを確かめるため、
  biaslab の重み側の update も同じマスク方式にそろえ、`-lr`（小数）を `-lrhalfpow`（整数）に置き換えた。
  テストは lrHalfPow = 1 で比べる。biaslab の `REPORT.md` / `LOGS.md` の結果は旧 `-lr` で測定したもの。
* ほかの lab（e2elab / glulab / revlab / swalab）の独自 update は `lr float32` のまま。

---

## ステップ4: ゲートを廃止する ✅ 完了

**リスク: 中（換算ミスに注意）。実験では両データセットでわずかに精度が上がりました。**

### 根拠

3シード×2データセット、20エポック:

| | ゲート有り（現行） | ゲート無し | 差 |
|---|---:|---:|---:|
| MNIST | 90.41 ± 0.16 | **90.64 ± 0.08** | **+0.23** |
| Fashion | 82.46 ± 0.22 | **82.89 ± 0.25** | **+0.43** |

改善幅は小さい（SDに対して2σ前後）ので「はっきり良い」とは言えませんが、
**削って落ちない**という目的には足ります。

しきい値の決め方を変えた他の条件（分位点ベース・順位ベース・反転など）も測りましたが、
全モードがMNISTで約0.7pt、Fashionで約0.9ptの幅に収まりました。
**逆伝播経路が正しい限り、ゲートは低レバレッジな機構です。**

### 反映後の確認（crow 本体、2026-09-24）

crow 本体（784→512→1024、ミニバッチ1024、`LRHalfPow` 3、`LogitMargin` 0.45、20エポック、
モデル初期化シード 1〜3）で、ゲート有り（反映前）とゲート無し（反映後）を比べた:

| | ゲート有り | ゲート無し | 差 |
|---|---:|---:|---:|
| MNIST | 90.26 ± 0.12 | **90.69 ± 0.21** | **+0.43** |
| Fashion | 82.32 ± 0.49 | **82.92 ± 0.33** | **+0.60** |

seed 別（ゲート有り）: MNIST 0.9022 / 0.9017 / 0.9040、Fashion 0.8195 / 0.8214 / 0.8287
seed 別（ゲート無し）: MNIST 0.9087 / 0.9073 / 0.9046、Fashion 0.8263 / 0.8285 / 0.8328

glulab と同じ向きで、全シードでゲート無しが上回った。学習時の乱数（ワーカーとシャッフル）は固定していない。

2段階の2（`Dot` への置き換え）は、1（全ビット1の `DotTernary`）と固定シードの Forward → backward の
出力（y・nextT・デルタ）が、奇数の列幅（37 / 63 / 101）を含む5形状×5シードの全25ケースでビット単位で一致した。

### 削除するもの（`model/bep/layer.go`）

| 行 | 削除対象 |
|---:|---|
| 25 | `SharedHyperparameters.GateDropThresholdScale float32` |
| 32 | `NewSharedHyperparameters` の `GateDropThresholdScale: 1.0` |
| 66 | `Dense.GateDropThresholdBase int` |
| 104 | `gateDropThresholdBase := int(noiseStdBase)` |
| 110 | `GateDropThresholdBase: gateDropThresholdBase,` |
| 115-117 | `func (d *Dense) GateDropThreshold() int` |
| 164 | `keepGate, err := bitsx.NewZerosMatrix(yRows, yCols)` |
| 168 | `gateDropThreshold := d.GateDropThreshold()` |
| 185 | `var keepGateWord uint64` |
| 196-198 | `if absZi <= gateDropThreshold { keepGateWord \|= (1 << uint64(i)) }` |
| 217-219 | `if err := keepGate.SetWord(tCtx.WordIndex, keepGateWord); err != nil { ... }` |
| 263 | `d.WT.DotTernary(t, keepGate)` → `d.WT.Dot(t)`（後述の換算が必要） |

**注意: 103行の `noiseStdBase := ...` は残します。** 104行がこれに依存していますが、
`noiseStdBase` 自体はノイズ側でも使われているので、消せるのはステップ5と合わせてからです。

### 換算（ここが一番危険）

`bitsx` のカーネル（`kernels.go`）が返す値:

```text
Dot(t)              = 一致ビット数 u
DotTernary(t, mask) = nonZero数 - 2*不一致数
```

`mask` が全ビット1のとき後者は `2u - cols` になります（`cols` は `d.WT.Cols()`、
つまりこの層の出力幅）。したがって符号判定は:

```go
if rawNextTT[colT] >= 0 {        // DotTernary 版（現行）
if 2*u >= cols {                 // Dot 版（移行後）
```

**`u >= cols/2` と書いてはいけません。** `cols` が奇数のとき整数除算で切り捨てられ、
境界のニューロンで符号が反転します（`cols = 37`, `u = 18` なら `2*18 - 37 = -1 < 0` なのに
`18 >= 18` は真）。現在の出力幅は512/1024で偶数のため偶然動いてしまい、
**幅を奇数にした瞬間に静かに壊れます。**

この恒等式は [glulab/main_test.go](glulab/main_test.go) の
`TestDotTernaryAllOnesEqualsDot` で、奇数 `cols` を含む4形状について検証済みです。

### 2段階で進める

1. **`keepGate` を全ビット1にして `DotTernary` のまま**動かす（195-197の条件を無条件にする）。
   3シードで 90.6 ± 0.1 付近になることを確認
2. `Dot` へ置き換え、上の換算を適用。**1とビット単位で一致するはず**
   （数学的に等価で、乱数消費も変わらないため）

2で値が変わったら換算を間違えています。

### 壊れるもの

| ファイル | 内容 |
|---|---|
| `biaslab/main_test.go:364` | `mine.gateScale = ctx.GateDropThresholdScale` → **コンパイルエラー** |

**`biaslab` は crow 本体との1ステップ差分テストを持つ唯一のラボです。**
ここが動かなくなると本体の検証手段を失うので、**同時に追従させてください。**
具体的には `biaslab` の `dense.gateScale` を廃止し、`keepGate` を常に全ビット1にします。

`e2elab` / `revlab` / `swalab` の `-gate` フラグはラボ独自なのでコンパイルは通ります。

---

## ステップ5: ガウスノイズを整数一様ノイズに置き換える ✅ 完了

### 反映内容（2026-09-24）

* `SharedHyperparameters.NoiseStdScale`、`Dense.NoiseStdBase`、`Dense.NoiseStd()` を削除し、
  層ごとの `Dense.MaxAbsNoise int`（ノイズは [-MaxAbsNoise, MaxAbsNoise] の整数一様）に置き換えた（全層共通の倍率は持たない）。
  名前は UNIVERSAL_APPROXIMATION.md の `maxAbsNoise` に合わせた。
  ノイズを止めたいときは各層の `MaxAbsNoise` を 0 にする。
* 既定値は非公開の `maxAbsNoiseForScale(fanIn, 1, 2) = isqrt(fanIn) * 1732 * 1 / 1000 / 2`（0.5 × √3 ≈ 0.866）。
  `isqrt` は layer.go 内に置いた整数平方根で、構築時の `math.Sqrt` も無くなった。
  fanIn 784 → 24、512 → 19 で、旧実装の小数計算を切り捨てた値と一致する。
* 順伝播は `z[i] = zi + rng.IntN(2*w+1) - w`。`isNoisy` の分岐は残し、w = 0 では乱数を消費しない。
* `Trainer.Validate` で `0 <= MaxAbsNoise <= 入力数` を検証する。負の値は黙ってノイズ無しになり、
  入力数を超える値は最も確信の強いニューロンまで反転させて出力を無作為に近づけるだけなので弾く。
* biaslab の差分テストは `ctx.NoiseStdScale = 0` → `crowDense.MaxAbsNoise = 0`。
  biaslab 自身のノイズはガウスのままなので、crow と一致するのはノイズ0のときだけ（README に注記）。
* テスト追加: `TestIsqrt`、`TestMaxAbsNoiseForScale`（layer_internal_test.go）、`TestSetNoiseScale`、
  `TestDenseForwardNoise`（MaxAbsNoise が 0 で Predict と一致し乱数を消費しない／|z| が MaxAbsNoise を超えるニューロンは反転しない）。
* 保存済みの gob モデルを読み込むと `MaxAbsNoise` は 0（ノイズ無し）になる。続けて学習する場合は設定し直す。

crow 本体（ステップ4と同じ条件: 784→512→1024、ミニバッチ1024、`LRHalfPow` 3、`LogitMargin` 0.45、
ゲート無し、20エポック、モデル初期化シード 1〜3）での比較:

| ノイズ | MNIST | Fashion |
|---|---:|---:|
| ガウス（反映前） | 90.59 ± 0.10 | 82.65 ± 0.16 |
| **整数一様（反映後）** | 90.49 ± 0.17 | 82.76 ± 0.12 |

差は −0.10 / +0.11 で、どちらもばらつきの範囲内。同じガウスのコードでもステップ4の計測（90.69 / 82.92）と
0.1〜0.3pt ずれており、学習時の乱数を固定していない分の揺れがこの程度ある。

seed 別（ガウス）: MNIST 0.9064 / 0.9066 / 0.9047、Fashion 0.8250 / 0.8281 / 0.8265
seed 別（一様）: MNIST 0.9061 / 0.9029 / 0.9057、Fashion 0.8284 / 0.8283 / 0.8262

### 倍率の確認（整数一様、同じ条件）

`MaxAbsNoise = 倍率 × √3 × isqrt(入力数)` を全層に入れて比べた:

| 倍率 | MaxAbsNoise（784 / 512 入力） | MNIST | Fashion |
|---|---|---:|---:|
| **0.5（既定値）** | 24 / 19 | **90.49 ± 0.17** | **82.76 ± 0.12** |
| 0.75 | 36 / 29 | 90.44 ± 0.15 | 82.41 ± 0.13 |
| 1.0 | 48 / 38 | 90.21 ± 0.29 | 82.26 ± 0.20 |

seed 別（0.75）: MNIST 0.9031 / 0.9061 / 0.9041、Fashion 0.8231 / 0.8255 / 0.8236
seed 別（1.0）: MNIST 0.9052 / 0.9014 / 0.8996、Fashion 0.8244 / 0.8229 / 0.8204

強めるほど下がり（Fashion は 0.75 で −0.35、1.0 で −0.50）、glulab の 0.25 も 0.5 より悪かったので、
既定値 0.5 のままとした。

倍率を試しやすくするため、`Dense.SetNoiseScale(num, denom int) error` と、全層をまとめて設定する
`Sequence.SetNoiseScale(num, denom int) error` を追加した（例: `model.Backbone.SetNoiseScale(3, 4)` で倍率 0.75）。
倍率は「ノイズの標準偏差が √入力数 の何倍か」で、計算式は `maxAbsNoiseForScale` の1か所にまとめている。
範囲外（負・分母0以下・入力数を超える）はその場でエラーにして値を変えない。`Sequence` 版は全層の値を計算してから代入するので、
一部の層だけ変わることはない。

以下は反映前に書いた手順（記録として残す）。

**リスク: 低。実測で両分布に有意差はありませんでした。ただしノイズ機構そのものは残します。**

### 根拠

`glulab -arch plain` で3シード×2データセット×20エポック:

| 条件 | MNIST | Fashion |
|---|---:|---:|
| `norm 0.5`（現行のガウス） | 90.64 ± 0.08 | **82.89 ± 0.25** |
| **`uniform 0.5`（整数一様）** | **90.67 ± 0.07** | 82.59 ± 0.23 |
| `uniform 0.25` | 90.37 ± 0.07 | 82.79 ± 0.09 |
| `norm 0.25` | 90.21 ± 0.06 | 82.80 ± 0.25 |
| **`noise 0`（完全撤廃）** | **90.40 ± 0.17** | **82.34 ± 0.25** |

同じスケールで揃えた uniform − norm の差は +0.03 / +0.16 / −0.30 / −0.01 で、**平均 −0.03**。
すべてSDの1.2倍以内で、**分布による差は検出できません。**

一方で**ノイズを完全に削ると MNIST −0.24pt / Fashion −0.55pt 落ちます。**
Fashion の落ち幅はSDの2倍を超えるので、これは実在の劣化です。
**ノイズ機構は残し、分布だけ置き換えるのが正解**という結論になります。

### 変更するもの

`model/bep/layer.go`:

```go
type SharedHyperparameters struct {
	NoiseStdScale float32   // 25行: 削除
}

type Dense struct {
	NoiseStdBase float32    // 64行: 削除
}

func (d *Dense) NoiseStd() float32 { ... }   // 110-112行: 削除
```
↓
```go
type Dense struct {
	...
	// NoiseHalfWidth は活性前値へ加える整数一様ノイズの半幅。0 ならノイズ無し。
	NoiseHalfWidth int
}
```

順伝播（[layer.go:123-142](layer.go)）:

```go
noiseStd := d.NoiseStd()
isNoisy := noiseStd > 0.0
...
noise, err := randx.IntNorm(minZi, maxZi, 0, noiseStd, rng)
if err != nil {
	return nil, nil, err
}
z[i] = zi + noise
```
↓
```go
w := d.NoiseHalfWidth
isNoisy := w > 0
...
z[i] = zi + rng.IntN(2*w+1) - w
```

**`isNoisy` の分岐は残してください。** `w = 0` でも `rng.IntN(1)` は 0 を返すので
結果は正しいのですが、**乱数を1つ消費してしまいます。**現行コードは `isNoisy` でループごと
分けており、ノイズ無効時に乱数を消費しません。ここを崩すと `biaslab` の差分テスト
（`NoiseStdScale = 0` で決定的にしている）が壊れます。

**エラー戻り値が1つ減ります。** `randx.IntNorm` は `(int, error)` を返しますが
`rng.IntN` は `int` だけなので、周辺のエラー処理が簡略化できます。

### 既定値の決め方

従来の `NoiseStdScale = 0.5` に対応する半幅は、一様分布 `[-w, w]` の標準偏差が `w/√3` なので

```text
w = 0.5 × √3 × √fanIn ≈ 0.866 × √fanIn
```

整数演算で書くなら（`isqrt` は整数平方根）:

```go
// NoiseHalfWidthFor は、従来の NoiseStdScale = num/den に対応する半幅を返す。
// 一様分布 [-w, w] の標準偏差は w/√3 なので、√3 ≈ 1732/1000 を掛ける。
func NoiseHalfWidthFor(fanIn, num, den int) int {
	return num * isqrt(fanIn) * 1732 / (den * 1000)
}
```

実際の値: `fanIn = 784` → `w = 24`、`fanIn = 512` → `w = 19`。

### 壊れる呼び出し元

| ファイル | 内容 |
|---|---|
| `biaslab/main_test.go:349` | `ctx.NoiseStdScale = 0` → `NoiseHalfWidth = 0`（差分テストを決定的にするため） |

ラボの `-noise` フラグ（`biaslab/main.go:1122` 等）はラボ独自なのでコンパイルは通ります。

### 補足: CP+R について

[REPORT.md](REPORT.md) の差分5に、論文のCP+R強化ステップを省略した理由として
「ノイズ注入とSign集約が安定化の役割を代替している」とあります。
**今回はノイズ機構を残すので、CP+Rを戻す必要はありません。**
（ノイズを完全に削る場合のみ、この論点が復活します。）

---

## ステップ6: 仕上げの確認 ✅ 完了

### 結果（2026-09-24）

| 確認項目 | 結果 |
|---|---|
| 学習カーネル（`Forward` / backward / `Update` / `SatisfiesUpdateCriterion`）に浮動小数が無い | ✅ |
| 機械検査（gofmt / vet / staticcheck / errcheck / test / race / fuzz / modernize / mound / govulncheck） | ✅（govulncheck は go1.26.5 の標準ライブラリの4件のみ。go1.26.6 で修正済み） |
| 精度（3シード×2データセット×20エポック） | ✅ MNIST 90.49 ± 0.17、Fashion 82.76 ± 0.12（目標 90.4 / 82.6 以上） |

精度はステップ5（整数一様ノイズ）反映後の crow 本体での計測。ステップ0の基準（ゲート有り、
90.41 / 82.46）を上回っている。

### 意図して残した浮動小数

`model/bep/*.go`（ラボを除く）に残っているのは次だけで、いずれも残す判断をした。

| 場所 | 内容 | 残す理由 |
|---|---|---|
| train.go `LogitMargin` | 更新判定のマージン（比率、既定値 0.45） | 利用者が指定する値として比率の方が分かりやすい。判定自体は `ComputeSeqSignDelta` でバッチごとに1回だけ整数ビット数へ変換しており、サンプルごとの判定は整数 |
| train.go `PairwiseMargin`、pairwise.go `PairwiseLabels` | ランキング学習の、復号した値どうしの比較 | 学習ループ内に残る唯一の浮動小数。整数化するには回帰の値の表し方から設計し直す必要があり、「精度を保ったまま機構を削る」範囲を超えるため別課題とする |
| model.go `Values` / `SetValues` / `ValueToLabel` / `PredictValue` | 回帰の値（ラベル ↔ 実数値）の変換 | 利用者とやり取りする値。分類・回帰の学習中は使わない |
| model.go `PredictSoftmax` / `Accuracy` / `Loss`、pairwise.go `PairwiseAccuracy` | 確率・評価指標 | 利用者に返す値で、小数の方が自然。学習中は使わない |
| model.go `SetClassPrototypes` 内の ETF 反復回数（`10 × n × log n`） | プロトタイプ生成の反復回数 | 構築時に1回だけで、整数化しても得るものがほぼ無い |

（当初の本書は `SplitTrainValidation` / `GridSearch` の集計も挙げていたが、search.go は削除済み。）

### 確認方法

```powershell
Select-String -Path model/bep/*.go -Pattern "float32|float64|math\.Sqrt|math\.Round|math\.Abs|math\.Log|math\.Exp|Float32\(\)|Float64\(\)"
```

上の表以外がヒットしたら、学習カーネルに浮動小数が戻っていないか確認する。

---

## 作業順序のまとめ

| 順 | ステップ | 影響範囲 | リスク | 状態 |
|---:|---|---|---|---|
| 0 | 基準値の記録 | なし | — | — |
| 1 | `math.Abs` → べた書き | `layer.go` 1箇所 | なし | **✅ 完了** |
| 2 | `margin` → 整数ビット数 | 本体 + examples + labs | なし（等価） | **✅ 完了** |
| 3 | `lr` → 2の冪の確率 | 本体 + tests | 低（乱数列が変わる） | **✅ 完了** |
| 4 | ゲート廃止 | 本体 + **biaslab** | 中（換算ミスに注意） | **✅ 完了** |
| 5 | ガウス → 整数一様 | 本体 + tests | 低（有意差なし） | **✅ 完了** |
| 6 | 仕上げ | — | — | **✅ 完了** |

2と3は互いに独立なので順序を入れ替えても構いません。
**4は3の後にやってください。**乱数消費の変化が重なると原因の切り分けが難しくなります。

各ステップの後に `go build ./... && go test ./...` が通ることを必ず確認してください。
