# 実験の生ログ

実行日: 2026-08-14。環境: Windows 11 / Go 1.26.5 / 12論理コアCPU。
構成は 784→512→1024、ETFプロトタイプ初期値、lr=0.1 / clslr=0.1 / margin=0.5 /
gsize=4 / gate=1.0 / noise=0.5、ミニバッチ1024、20エポック、学習データの10%を検証。

学習乱数は seed で固定。最良エポックは検証集合で選び、テスト集合は選択に使わない。
`BEST_VAL ... / TEST ...` の `TEST` が報告値。

条件:

| 名前 | 設定 |
|---|---|
| 従来BEP | `-target proto -learnq=false` |
| 目標のみdiff | `-target diff -learnq=false` |
| 分類器のみ学習 | `-target proto -learnq` |
| 提案 | `-target diff -learnq` |
| diffgate | `-target diff -hard`(指定なしビットをゲートからも除外) |
| 提案 n個 | `-target randdiff -n N`(競合差分から無作為に N 個) |
| 対照 n個 | `-target randall -n N`(全ビットから無作為に N 個) |

---

## 1. 4条件 x 3シード

```text
########## mnist ##########
seed=1 従来BEP(proto,固定)     BEST_VAL 0.8942 @epoch 14 / TEST 0.9055
seed=1 目標のみdiff(固定)       BEST_VAL 0.8835 @epoch 20 / TEST 0.8911
seed=1 分類器のみ学習(proto)    BEST_VAL 0.8928 @epoch 19 / TEST 0.9004
seed=1 提案(diff+学習)          分類器行の距離(学習後): 最小 539 / 平均 560.2
                                BEST_VAL 0.8835 @epoch 12 / TEST 0.8950
seed=2 従来BEP(proto,固定)     BEST_VAL 0.8972 @epoch 18 / TEST 0.9024
seed=2 目標のみdiff(固定)       BEST_VAL 0.8867 @epoch 20 / TEST 0.8938
seed=2 分類器のみ学習(proto)    BEST_VAL 0.8965 @epoch 16 / TEST 0.9038
seed=2 提案(diff+学習)          分類器行の距離(学習後): 最小 536 / 平均 560.6
                                BEST_VAL 0.8873 @epoch 17 / TEST 0.8958
seed=3 従来BEP(proto,固定)     BEST_VAL 0.8908 @epoch 16 / TEST 0.9045
seed=3 目標のみdiff(固定)       BEST_VAL 0.8818 @epoch 20 / TEST 0.8940
seed=3 分類器のみ学習(proto)    BEST_VAL 0.8938 @epoch 20 / TEST 0.9047
seed=3 提案(diff+学習)          分類器行の距離(学習後): 最小 536 / 平均 559.6
                                BEST_VAL 0.8828 @epoch 16 / TEST 0.8942
########## fashion ##########
seed=1 従来BEP(proto,固定)     BEST_VAL 0.8347 @epoch 15 / TEST 0.8239
seed=1 目標のみdiff(固定)       BEST_VAL 0.8285 @epoch 19 / TEST 0.8230
seed=1 分類器のみ学習(proto)    BEST_VAL 0.8313 @epoch 12 / TEST 0.8215
seed=1 提案(diff+学習)          分類器行の距離(学習後): 最小 523 / 平均 558.6
                                BEST_VAL 0.8293 @epoch 19 / TEST 0.8211
seed=2 従来BEP(proto,固定)     BEST_VAL 0.8367 @epoch 12 / TEST 0.8271
seed=2 目標のみdiff(固定)       BEST_VAL 0.8337 @epoch 20 / TEST 0.8205
seed=2 分類器のみ学習(proto)    BEST_VAL 0.8372 @epoch 17 / TEST 0.8243
seed=2 提案(diff+学習)          分類器行の距離(学習後): 最小 522 / 平均 558.7
                                BEST_VAL 0.8358 @epoch 20 / TEST 0.8211
seed=3 従来BEP(proto,固定)     BEST_VAL 0.8355 @epoch 19 / TEST 0.8229
seed=3 目標のみdiff(固定)       BEST_VAL 0.8332 @epoch 19 / TEST 0.8193
seed=3 分類器のみ学習(proto)    BEST_VAL 0.8360 @epoch 19 / TEST 0.8220
seed=3 提案(diff+学習)          分類器行の距離(学習後): 最小 528 / 平均 559.3
                                BEST_VAL 0.8310 @epoch 19 / TEST 0.8195
```

集計(テスト精度の3シード平均):

| 条件 | MNIST | Fashion |
|---|---:|---:|
| 従来BEP | **0.9041** | **0.8246** |
| 目標のみdiff | 0.8930 | 0.8209 |
| 分類器のみ学習 | 0.9030 | 0.8226 |
| 提案(diff+学習) | 0.8950 | 0.8206 |

分類器行の距離は初期が最小565・平均569で、学習後は最小522〜539・平均559〜561。
行どうしが近づきすぎて崩壊することはなかった。

---

## 2. diffgate(指定なしビットをゲートからも外す)

```text
########## mnist ##########
seed=1 diffgate(固定)   BEST_VAL 0.8703 @epoch 19 / TEST 0.8820
seed=1 diffgate+学習     BEST_VAL 0.8662 @epoch 11 / TEST 0.8744
seed=2 diffgate(固定)   BEST_VAL 0.8702 @epoch 16 / TEST 0.8781
seed=2 diffgate+学習     BEST_VAL 0.8712 @epoch 12 / TEST 0.8802
seed=3 diffgate(固定)   BEST_VAL 0.8645 @epoch 16 / TEST 0.8770
seed=3 diffgate+学習     BEST_VAL 0.8668 @epoch 16 / TEST 0.8827
########## fashion ##########
seed=1 diffgate(固定)   BEST_VAL 0.8167 @epoch 19 / TEST 0.8132
seed=1 diffgate+学習     BEST_VAL 0.8195 @epoch 17 / TEST 0.8098
seed=2 diffgate(固定)   BEST_VAL 0.8278 @epoch 20 / TEST 0.8100
seed=2 diffgate+学習     BEST_VAL 0.8257 @epoch 18 / TEST 0.8110
seed=3 diffgate(固定)   BEST_VAL 0.8238 @epoch 10 / TEST 0.8119
seed=3 diffgate+学習     BEST_VAL 0.8242 @epoch 20 / TEST 0.8157
```

集計:

| 条件 | MNIST | Fashion |
|---|---:|---:|
| diffgate(固定) | 0.8790 | 0.8117 |
| diffgate+学習 | 0.8791 | 0.8122 |

diff よりさらに悪い。「指定なしビットが後方射影を汚染している」という仮説は
これで否定された(除外するとむしろ悪化する)。

---

## 3. 最終活性の凝集度(MNIST、seed=1、テスト集合、クラスあたり60件)

```text
proto     最終活性の距離: 級内 235.9 / 級間 538.9 / 比 0.438 (総ビット 1024)
diff      最終活性の距離: 級内 283.4 / 級間 530.8 / 比 0.534 (総ビット 1024)
diffgate  最終活性の距離: 級内 300.9 / 級間 527.2 / 比 0.571 (総ビット 1024)
diff+学習 最終活性の距離: 級内 275.0 / 級間 527.8 / 比 0.521 (総ビット 1024)
```

| 目標 | 級内/級間 | テスト精度 |
|---|---:|---:|
| proto | 0.438 | 0.9055 |
| diff+学習 | 0.521 | 0.8950 |
| diff | 0.534 | 0.8911 |
| diffgate | 0.571 | 0.8820 |

級内/級間の比の順序と、精度の順序が完全に一致する。

---

## 4. 実験B: 目標を指定するビットを絞る

### 4.1 n の掃引(MNIST、seed=1、20エポック)

```text
randdiff n=8    最終活性の距離: 級内 428.2 / 級間 475.3 / 比 0.901  BEST_VAL 0.5000 @epoch 20 / TEST 0.5023
randall  n=8    最終活性の距離: 級内 332.0 / 級間 517.6 / 比 0.641  BEST_VAL 0.7377 @epoch 20 / TEST 0.7551
randdiff n=32   最終活性の距離: 級内 330.5 / 級間 524.2 / 比 0.630  BEST_VAL 0.8503 @epoch 20 / TEST 0.8665
randall  n=32   最終活性の距離: 級内 299.4 / 級間 528.8 / 比 0.566  BEST_VAL 0.8582 @epoch 20 / TEST 0.8675
randdiff n=128  最終活性の距離: 級内 312.9 / 級間 526.2 / 比 0.595  BEST_VAL 0.8742 @epoch 18 / TEST 0.8850
randall  n=128  最終活性の距離: 級内 276.2 / 級間 534.0 / 比 0.517  BEST_VAL 0.8805 @epoch 20 / TEST 0.8908
randdiff n=512  最終活性の距離: 級内 287.2 / 級間 530.8 / 比 0.541  BEST_VAL 0.8845 @epoch 20 / TEST 0.8911
randall  n=512  最終活性の距離: 級内 255.5 / 級間 535.9 / 比 0.477  BEST_VAL 0.8900 @epoch 14 / TEST 0.9004
```

n を増やすほど単調に精度が上がる。全部(= 元のBEP)の 0.9055 が最良。

### 4.2 提案 vs 対照(3シード、20エポック)

```text
########## mnist 3シード ##########
seed=1 n=128  randdiff BEST_VAL 0.8742 @epoch 18 / TEST 0.8850
seed=1 n=128  randall  BEST_VAL 0.8805 @epoch 20 / TEST 0.8908
seed=1 n=512  randdiff BEST_VAL 0.8845 @epoch 20 / TEST 0.8911
seed=1 n=512  randall  BEST_VAL 0.8900 @epoch 14 / TEST 0.9004
seed=2 n=128  randdiff BEST_VAL 0.8763 @epoch 18 / TEST 0.8856
seed=2 n=128  randall  BEST_VAL 0.8843 @epoch 18 / TEST 0.8917
seed=2 n=512  randdiff BEST_VAL 0.8872 @epoch 18 / TEST 0.8928
seed=2 n=512  randall  BEST_VAL 0.8945 @epoch 19 / TEST 0.8978
seed=3 n=128  randdiff BEST_VAL 0.8702 @epoch 16 / TEST 0.8858
seed=3 n=128  randall  BEST_VAL 0.8785 @epoch 16 / TEST 0.8888
seed=3 n=512  randdiff BEST_VAL 0.8795 @epoch 16 / TEST 0.8916
seed=3 n=512  randall  BEST_VAL 0.8883 @epoch 14 / TEST 0.8987
########## fashion 3シード ##########
seed=1 n=128  randdiff BEST_VAL 0.8123 @epoch 16 / TEST 0.8019
seed=1 n=128  randall  BEST_VAL 0.8038 @epoch 15 / TEST 0.7933
seed=1 n=512  randdiff BEST_VAL 0.8268 @epoch 14 / TEST 0.8203
seed=1 n=512  randall  BEST_VAL 0.8293 @epoch 18 / TEST 0.8198
seed=2 n=128  randdiff BEST_VAL 0.8227 @epoch 19 / TEST 0.8051
seed=2 n=128  randall  BEST_VAL 0.8112 @epoch 18 / TEST 0.7975
seed=2 n=512  randdiff BEST_VAL 0.8327 @epoch 20 / TEST 0.8199
seed=2 n=512  randall  BEST_VAL 0.8378 @epoch 17 / TEST 0.8238
seed=3 n=128  randdiff BEST_VAL 0.8135 @epoch 17 / TEST 0.8012
seed=3 n=128  randall  BEST_VAL 0.8098 @epoch 20 / TEST 0.7991
seed=3 n=512  randdiff BEST_VAL 0.8255 @epoch 20 / TEST 0.8186
seed=3 n=512  randall  BEST_VAL 0.8357 @epoch 19 / TEST 0.8250
```

集計(テスト精度の3シード平均):

| データセット | n | 提案(randdiff) | 対照(randall) | 勝者 |
|---|---|---:|---:|---|
| MNIST | 128 | 0.8855 | **0.8904** | 対照(3/3) |
| MNIST | 512 | 0.8918 | **0.8990** | 対照(3/3) |
| Fashion | 128 | **0.8027** | 0.7966 | 提案(3/3) |
| Fashion | 512 | 0.8196 | **0.8229** | 対照(2/3) |

符号がデータセットと n で入れ替わる。いずれも元のBEP(MNIST 0.9041 / Fashion 0.8246)には届かない。

---

## 5. 忠実版(-hard)と収束比較

### 5.1 60エポック(MNIST、seed=1)

```text
proto(元のBEP)             級内 232.9 / 級間 540.8 / 比 0.431  BEST_VAL 0.8985 @epoch 59 / TEST 0.9078
randdiff n=32   soft       級内 321.1 / 級間 525.7 / 比 0.611  BEST_VAL 0.8628 @epoch 58 / TEST 0.8726
randdiff n=32   hard       級内 328.4 / 級間 524.2 / 比 0.626  BEST_VAL 0.8632 @epoch 33 / TEST 0.8724
randdiff n=128  soft       級内 310.0 / 級間 527.1 / 比 0.588  BEST_VAL 0.8825 @epoch 45 / TEST 0.8911
randdiff n=128  hard       級内 316.6 / 級間 527.5 / 比 0.600  BEST_VAL 0.8727 @epoch 35 / TEST 0.8799
randdiff n=512  soft       級内 288.0 / 級間 531.9 / 比 0.542  BEST_VAL 0.8865 @epoch 50 / TEST 0.8981
randdiff n=512  hard       級内 300.2 / 級間 529.7 / 比 0.567  BEST_VAL 0.8725 @epoch 48 / TEST 0.8877
randall  n=512  soft       級内 251.8 / 級間 537.6 / 比 0.468  BEST_VAL 0.8945 @epoch 48 / TEST 0.9056
```

級内/級間はこの版から**検証最良エポックのモデル**で測っている(初版は最終エポックで測っており、
報告する精度と別モデルの値になっていた)。

### 5.2 hard vs soft の3シード(n=512、60エポック)

```text
seed=1 soft TEST 0.8981 / hard TEST 0.8877
seed=2 soft BEST_VAL 0.8907 @epoch 50 / TEST 0.8950
seed=2 hard BEST_VAL 0.8797 @epoch 31 / TEST 0.8841
seed=3 soft BEST_VAL 0.8860 @epoch 33 / TEST 0.8964
seed=3 hard BEST_VAL 0.8730 @epoch 51 / TEST 0.8865
```

平均: soft 0.8965 / hard 0.8861。3シードすべてで soft が上。

### 5.3 収束比較(MNIST、seed=1)

```text
200エポック
proto      n=0    hard=false  級内 225.5 / 級間 541.9 / 比 0.416  BEST_VAL 0.9020 @epoch 192 / TEST 0.9106
randdiff   n=32   hard=false  級内 321.9 / 級間 526.7 / 比 0.611  BEST_VAL 0.8718 @epoch 181 / TEST 0.8780
randdiff   n=32   hard=true   級内 316.6 / 級間 528.5 / 比 0.599  BEST_VAL 0.8683 @epoch 165 / TEST 0.8767

640エポック(総指定ビット数を proto の20エポックと揃えた条件: 32 x 640 = 1024 x 20)
randdiff   n=32   hard=false  級内 328.1 / 級間 526.4 / 比 0.623  BEST_VAL 0.8767 @epoch 634 / TEST 0.8825
```

| 条件 | 20ep | 60ep | 200ep | 640ep |
|---|---:|---:|---:|---:|
| proto | 0.9055 | 0.9078 | **0.9106** | — |
| randdiff n=32 soft | 0.8665 | 0.8726 | 0.8780 | 0.8825 |
| 級内/級間 (proto) | 0.438 | 0.431 | 0.416 | — |
| 級内/級間 (n=32) | 0.630 | 0.611 | 0.611 | 0.623 |

エポックを32倍にしても追いつかず、級内/級間比は 0.61 前後で頭打ちになる。
学習量の不足ではなく、より拡散した表現へ収束している。

---

## 参考: 比較基準

- crow BEP 実装(`../REPORT.md`): MNIST 92.13 ± 0.48、Fashion 83.78 ± 0.03
  (3シード、50エポック、検証分離)。本実験は20エポックなので低めに出る
- 同じ土台の実装の忠実性は `../biaslab/main_test.go` の差分テストで担保
  (`Predict` / `Forward` / 逆伝播デルタ / `Update` 後の `H`・`W`・`WT` が crow とビット単位で一致)
