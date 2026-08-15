# 実験の生ログ

実行日: 2026-08-14。環境: Windows 11 / Go 1.26.5 / 12論理コアCPU。
データは `dataset.LoadMNIST` / `LoadFashionMNIST`(初回実行時に自動ダウンロード)と、
`../PROTOTYPES.md` B-8 と同じ生成規則の合成回帰データ。

**本ログは方法論を修正した後の再測定である**(`REPORT.md` §0)。
学習乱数は seed で固定、最良エポックと温度は検証集合で選び、テスト集合は選択に使わない。
`BEST_VAL ... / TEST ...` の `TEST` が報告値(検証で選んだエポックのテスト値)。

比較する3条件:

| 名前 | 設定 |
|---|---|
| 従来BEP | `-bias=false -biaschoice 0` |
| 対照(更新半減) | `-bias=false -biaschoice 0.5` |
| バイアス有り | `-bias -biaschoice 0.5` |

---

## 0. 再実装の妥当性確認

### 0.1 crow 本体との1ステップ差分テスト

`main_test.go` が、同じ初期値・同じ乱数列で crow の `binary.Dense` と
ビット単位で一致する事を検証している。

```text
$ go test -run MatchesCrow -v ./
=== RUN   TestSatisfiesUpdateCriterionMatchesCrow
--- PASS: TestSatisfiesUpdateCriterionMatchesCrow (0.00s)
=== RUN   TestPredictMatchesCrow
--- PASS: TestPredictMatchesCrow (0.00s)
=== RUN   TestForwardBackwardDeltaMatchesCrow
--- PASS: TestForwardBackwardDeltaMatchesCrow (0.00s)
=== RUN   TestUpdateMatchesCrow
--- PASS: TestUpdateMatchesCrow (0.00s)
ok
```

比較対象は `Predict` 出力 / `Forward` 出力 / 逆伝播で返る希望活性 /
逆伝播で溜まるデルタ / `Update` 後の `H`・`W`・`WT`。形状は 8x64、16x100、33x37 の3種。

### 0.2 seed による再現性

同一 seed の2回実行が完全に一致する。

```text
$ go run . -task regress -bias -cellmask -epochs 3 -lr 0.3 -margin 0.01 -gsize 16 -gate 4 -seed 77
BEST_VAL_MAE 20.097 @epoch 3 (T=40) / TEST_MAE 20.478
BEST_VAL_MAE 20.097 @epoch 3 (T=40) / TEST_MAE 20.478
```

(修正前は同じコマンドの1エポック後MAEが 24.285 / 24.655 と割れていた)

---

## 1. 分類(20エポック、784→512→1024、lr=0.1 margin=0.5 gsize=4 gate=1.0 noise=0.5、検証10%)

```text
########## 分類 mnist ##########
seed=1 cellmask=false  従来BEP        BEST_VAL 0.8942 @epoch 14 / TEST 0.9055
seed=1 cellmask=false  対照(更新半減)  BEST_VAL 0.8910 @epoch 20 / TEST 0.8999
seed=1 cellmask=false  バイアス有り    BEST_VAL 0.8918 @epoch 20 / TEST 0.9026
seed=1 cellmask=true   従来BEP        BEST_VAL 0.8907 @epoch 19 / TEST 0.9014
seed=1 cellmask=true   対照(更新半減)  BEST_VAL 0.8893 @epoch 17 / TEST 0.9018
seed=1 cellmask=true   バイアス有り    BEST_VAL 0.8892 @epoch 18 / TEST 0.9014
seed=2 cellmask=false  従来BEP        BEST_VAL 0.8972 @epoch 18 / TEST 0.9024
seed=2 cellmask=false  対照(更新半減)  BEST_VAL 0.8965 @epoch 16 / TEST 0.9030
seed=2 cellmask=false  バイアス有り    BEST_VAL 0.8960 @epoch 18 / TEST 0.9020
seed=2 cellmask=true   従来BEP        BEST_VAL 0.8982 @epoch 19 / TEST 0.9027
seed=2 cellmask=true   対照(更新半減)  BEST_VAL 0.8990 @epoch 19 / TEST 0.9039
seed=2 cellmask=true   バイアス有り    BEST_VAL 0.8963 @epoch 19 / TEST 0.9037
seed=3 cellmask=false  従来BEP        BEST_VAL 0.8908 @epoch 16 / TEST 0.9045
seed=3 cellmask=false  対照(更新半減)  BEST_VAL 0.8892 @epoch 18 / TEST 0.9006
seed=3 cellmask=false  バイアス有り    BEST_VAL 0.8917 @epoch 20 / TEST 0.9026
seed=3 cellmask=true   従来BEP        BEST_VAL 0.8917 @epoch 18 / TEST 0.9052
seed=3 cellmask=true   対照(更新半減)  BEST_VAL 0.8900 @epoch 20 / TEST 0.9006
seed=3 cellmask=true   バイアス有り    BEST_VAL 0.8893 @epoch 18 / TEST 0.9015
########## 分類 fashion ##########
seed=1 cellmask=false  従来BEP        BEST_VAL 0.8347 @epoch 15 / TEST 0.8239
seed=1 cellmask=false  対照(更新半減)  BEST_VAL 0.8325 @epoch 20 / TEST 0.8238
seed=1 cellmask=false  バイアス有り    BEST_VAL 0.8332 @epoch 18 / TEST 0.8244
seed=1 cellmask=true   従来BEP        BEST_VAL 0.8315 @epoch 15 / TEST 0.8256
seed=1 cellmask=true   対照(更新半減)  BEST_VAL 0.8315 @epoch 17 / TEST 0.8264
seed=1 cellmask=true   バイアス有り    BEST_VAL 0.8347 @epoch 19 / TEST 0.8255
seed=2 cellmask=false  従来BEP        BEST_VAL 0.8367 @epoch 12 / TEST 0.8271
seed=2 cellmask=false  対照(更新半減)  BEST_VAL 0.8352 @epoch 16 / TEST 0.8215
seed=2 cellmask=false  バイアス有り    BEST_VAL 0.8368 @epoch 19 / TEST 0.8274
seed=2 cellmask=true   従来BEP        BEST_VAL 0.8380 @epoch 16 / TEST 0.8256
seed=2 cellmask=true   対照(更新半減)  BEST_VAL 0.8383 @epoch 18 / TEST 0.8270
seed=2 cellmask=true   バイアス有り    BEST_VAL 0.8392 @epoch 15 / TEST 0.8264
seed=3 cellmask=false  従来BEP        BEST_VAL 0.8355 @epoch 19 / TEST 0.8229
seed=3 cellmask=false  対照(更新半減)  BEST_VAL 0.8362 @epoch 19 / TEST 0.8246
seed=3 cellmask=false  バイアス有り    BEST_VAL 0.8362 @epoch 17 / TEST 0.8256
seed=3 cellmask=true   従来BEP        BEST_VAL 0.8350 @epoch 20 / TEST 0.8250
seed=3 cellmask=true   対照(更新半減)  BEST_VAL 0.8343 @epoch 20 / TEST 0.8234
seed=3 cellmask=true   バイアス有り    BEST_VAL 0.8387 @epoch 19 / TEST 0.8293
```

集計(テスト精度の3シード平均):

| データセット | セルマスク | 従来BEP | 対照(更新半減) | バイアス有り |
|---|---|---:|---:|---:|
| MNIST | 無し | 0.9041 | 0.9012 | 0.9024 |
| MNIST | 有り | 0.9031 | 0.9021 | 0.9022 |
| Fashion | 無し | 0.8246 | 0.8233 | 0.8258 |
| Fashion | 有り | 0.8254 | 0.8256 | 0.8271 |

全ての差が ±0.3pt 以内。

---

## 2. 回帰(30エポック、64x128→256→64、lr=0.3 margin=0.01 gsize=16 gate=4、加重平均復号、検証10%)

```text
seed=1 cellmask=false  従来BEP        BEST_VAL_MAE 12.978 @epoch 25 (T=160) / TEST_MAE 12.635
seed=1 cellmask=false  対照(更新半減)  BEST_VAL_MAE 13.184 @epoch 30 (T=160) / TEST_MAE 12.467
seed=1 cellmask=false  バイアス有り    BEST_VAL_MAE 12.744 @epoch 21 (T=160) / TEST_MAE 12.094
seed=1 cellmask=true   従来BEP        BEST_VAL_MAE 5.244 @epoch 26 (T=120) / TEST_MAE 5.350
seed=1 cellmask=true   対照(更新半減)  BEST_VAL_MAE 5.294 @epoch 17 (T=120) / TEST_MAE 5.502
seed=1 cellmask=true   バイアス有り    BEST_VAL_MAE 5.909 @epoch 29 (T=60) / TEST_MAE 6.031
seed=2 cellmask=false  従来BEP        BEST_VAL_MAE 12.476 @epoch 24 (T=160) / TEST_MAE 13.165
seed=2 cellmask=false  対照(更新半減)  BEST_VAL_MAE 12.499 @epoch 30 (T=160) / TEST_MAE 13.125
seed=2 cellmask=false  バイアス有り    BEST_VAL_MAE 11.741 @epoch 17 (T=160) / TEST_MAE 12.427
seed=2 cellmask=true   従来BEP        BEST_VAL_MAE 5.340 @epoch 28 (T=120) / TEST_MAE 5.470
seed=2 cellmask=true   対照(更新半減)  BEST_VAL_MAE 5.320 @epoch 26 (T=120) / TEST_MAE 5.434
seed=2 cellmask=true   バイアス有り    BEST_VAL_MAE 5.769 @epoch 21 (T=40) / TEST_MAE 6.122
seed=3 cellmask=false  従来BEP        BEST_VAL_MAE 12.412 @epoch 30 (T=160) / TEST_MAE 11.178
seed=3 cellmask=false  対照(更新半減)  BEST_VAL_MAE 12.641 @epoch 25 (T=160) / TEST_MAE 11.275
seed=3 cellmask=false  バイアス有り    BEST_VAL_MAE 12.342 @epoch 19 (T=160) / TEST_MAE 11.100
seed=3 cellmask=true   従来BEP        BEST_VAL_MAE 5.726 @epoch 17 (T=120) / TEST_MAE 5.445
seed=3 cellmask=true   対照(更新半減)  BEST_VAL_MAE 5.675 @epoch 23 (T=120) / TEST_MAE 5.736
seed=3 cellmask=true   バイアス有り    BEST_VAL_MAE 6.093 @epoch 27 (T=60) / TEST_MAE 6.425
```

集計(テストMAEの3シード平均):

| セルマスク | 従来BEP | 対照(更新半減) | バイアス有り |
|---|---:|---:|---:|
| 無し | 12.326 | 12.289 | **11.874** |
| 有り | **5.422** | 5.557 | 6.193 |

- セルマスク: 12.326 → 5.422 で 2.27倍。3シードの範囲(11.18〜13.17 対 5.35〜5.47)が重ならない
- 重み更新の半減はほぼ無害(12.326 → 12.289、5.422 → 5.557)
- バイアスは対照に対し、マスク無しで −0.415(3/3改善)、マスク有りで +0.636(3/3悪化)

---

## 3. 診断(回帰、バイアス有り、10エポック、seed=1)

```text
=== セルマスク有り ===
  層0 活性前値(バイアス抜き): 平均 -0.01 / 標準偏差(実測) 11.37 / √fanIn 11.31
  層1 活性前値(バイアス抜き): 平均 -8.76 / 標準偏差(実測) 17.85 / √fanIn 16.00
  層0 バイアス: 平均 -8.68 / 平均絶対値 8.84 / 最大絶対値 19 / 非ゼロ 254/256 / fanIn 128
  層1 バイアス: 平均 -3.47 / 平均絶対値 5.12 / 最大絶対値 18 / 非ゼロ 59/64 / fanIn 256
希望活性の極性(累計): +1 = 22034138 / -1 = 24506662
BEST_VAL_MAE 6.718 @epoch 10 (T=40) / TEST_MAE 6.687

=== セルマスク無し ===
  層0 活性前値(バイアス抜き): 平均 -0.02 / 標準偏差(実測) 11.36 / √fanIn 11.31
  層1 活性前値(バイアス抜き): 平均 -1.44 / 標準偏差(実測) 23.79 / √fanIn 16.00
  層0 バイアス: 平均 -10.32 / 平均絶対値 10.41 / 最大絶対値 17 / 非ゼロ 256/256 / fanIn 128
  層1 バイアス: 平均 1.97 / 平均絶対値 4.47 / 最大絶対値 13 / 非ゼロ 64/64 / fanIn 256
希望活性の極性(累計): +1 = 22098071 / -1 = 24153976
BEST_VAL_MAE 13.064 @epoch 10 (T=160) / TEST_MAE 12.519
```

- バイアスはほぼ全ニューロンで非ゼロ。動いていないわけではない
- 活性前値の標準偏差は**実測**。第1層は √fanIn とほぼ一致するが、
  第2層は 17.85 対 16.00(+12%)、23.79 対 16.00(+49%)とずれる。
  初版が √fanIn を基準にしていたのは不適切だった
- 希望活性の −1 の割合は 52.2〜52.7% で、**偏りは5%程度しかない**。
  それでも層全体が負へ寄るのは、バッチ集約後に符号を取って ±1 動かす更新則が
  微小な偏りを増幅し、数十バッチ積み重ねるためと考えられる

---

## 参考: 比較基準

- crow BEP 実装(`../REPORT.md`): MNIST 92.13 ± 0.48、Fashion 83.78 ± 0.03
  (3シード、50エポック、検証分離)。本実験は20エポックなので低めに出る
- 回帰の加重平均復号(`../PROTOTYPES.md` B-11a): 素の温度計 12.70、セルマスク 5.41。
  本実験(従来BEP)は 12.33 / 5.42 でよく一致する

---

## 4. 回帰の復号方式の比較

復号は学習済みモデルへの後処理なので、1回の学習で全方式を同じモデルに対して測れる。
`検証MAE / テストMAE` の順。パラメータを持つ方式は検証集合で選び、テスト集合は選択に使わない。

コマンド(セルマスク有り):

```text
go run . -task regress -bias=false -biaschoice 0 -cellmask -epochs 30 \
         -lr 0.3 -margin 0.01 -gsize 16 -gate 4 -seed 1
```

### 4.1 セルマスク有り、seed=1

```text
一致率の散らばり(テスト集合): 最大 0.5557 / 最小 0.4455 / 幅 0.1102 (総ビット 4096)
復号方式                                    検証MAE   テストMAE   パラメータ
同点平均(現行)                               17.243    17.381        -
提案: 一致率をそのまま正規化                     23.070    23.159        -
提案+: 最小を引いて正規化 (p=1)                 12.012    11.917        -
提案+: 最小を引いて2乗 (p=2)                   10.731    10.792        -
提案+: 最小を引いてp乗 (pを検証で選ぶ)             10.731    10.792        2
加重平均 softmax (Tを検証で選ぶ)                 5.362     5.438      120
点灯数から直接(校正なし)                        20.034    20.118        -
点灯数 + 検証で線形校正                          3.569     3.582      a,b
  診断: 校正なしの予測平均 49.85 / 真値の平均 50.00 (ずれ -0.15)
  校正: 素の傾き 0.02441 -> 検証での最小二乗 a=0.11708 b=-189.57
```

### 4.2 セルマスク有り、seed=2

```text
同点平均(現行)                               16.788    16.831        -
提案: 一致率をそのまま正規化                     23.430    23.170        -
提案+: 最小を引いて正規化 (p=1)                 11.872    12.032        -
提案+: 最小を引いて2乗 (p=2)                   10.250    10.738        -
提案+: 最小を引いてp乗                          10.250    10.738        2
加重平均 softmax                              5.452     5.520      120
点灯数から直接(校正なし)                        20.255    20.098        -
点灯数 + 検証で線形校正                          3.479     3.421      a,b
  診断: 校正なしの予測平均 49.94 / 真値の平均 50.00 (ずれ -0.06)
  校正: 素の傾き 0.02441 -> 検証での最小二乗 a=0.11523 b=-186.24
```

### 4.3 セルマスク無し、seed=1

```text
同点平均(現行)                               21.085    19.119        -
提案: 一致率をそのまま正規化                     22.730    22.776        -
提案+: 最小を引いて正規化 (p=1)                 16.284    15.943        -
提案+: 最小を引いて2乗 (p=2)                   14.142    13.834        -
提案+: 最小を引いてp乗                          14.074    13.654        3
加重平均 softmax                             12.995    12.435      160
点灯数から直接(校正なし)                        19.612    19.585        -
点灯数 + 検証で線形校正                         13.454    12.964      a,b
  診断: 校正なしの予測平均 51.21 / 真値の平均 50.00 (ずれ +1.21)
  校正: 素の傾き 0.02441 -> 検証での最小二乗 a=0.07037 b=-97.82
```

セルマスク無しでは `prototypes[0]` が全ゼロなので、点灯数復号は
`popcount(y)` そのもの(素の温度計をそのまま数える形)になる。

### 4.4 まとめ

| 復号 | マスク有り(seed1/seed2) | マスク無し(seed1) |
|---|---:|---:|
| 一致率をそのまま正規化 | 23.16 / 23.17 | 22.78 |
| 点灯数から直接(校正なし) | 20.12 / 20.10 | 19.59 |
| 同点平均(現行) | 17.38 / 16.83 | 19.12 |
| 最小を引いて2乗 | 10.79 / 10.74 | 13.83 |
| 加重平均 softmax | 5.44 / 5.52 | 12.44 |
| **点灯数 + 線形校正** | **3.58 / 3.42** | 12.96 |

点灯数の振れ幅(値域0〜100に対応する count の幅):

| | 実測 | 理論値 |
|---|---:|---:|
| セルマスク有り | 854ビット | 4096ビット |
| セルマスク無し | 1421ビット | 4096ビット |

マスク無しの方が振れ幅は広いのに精度は3.6倍悪い。
振れ幅ではなく、同じレベルのサンプル間での点灯数のばらつきが効いている。
