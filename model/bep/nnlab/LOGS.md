# 実験の生ログ

実行日: 2026-08-14。環境: Windows 11 / Go 1.26.5 / 12論理コアCPU(12ワーカー)。
実験時の実装はリポジトリ外の一時作業ディレクトリに置いた作業コピーで、
本フォルダの `model/` `ablation/` `bepreadout/` と機能的に同一
(移植時の変更はコメント整備・進捗表示のフラグ追加のみで、乱数消費順序は不変)。

データ: `~/.crow_dataset` の二値化済みgob(crow/dataset。1bit/画素、784次元)。
本フォルダのプログラムは `dataset.LoadMNIST` / `LoadFashionMNIST` を使うので、
初回実行時に自動でダウンロードされる。

数値はいずれも「その実行の最良エポックのテスト精度」。

---

## 1. 特徴の種類 × 読み出し(ablation)

### 1.1 幅8192、MNIST、15エポック、平均化なし

```text
### type=dense readout=linear
dataset=mnist type=dense hidden=8192 readout=linear margin=25 活性率=0.516  BEST 0.9593
### type=densebias readout=linear
dataset=mnist type=densebias hidden=8192 readout=linear margin=25 活性率=0.202  BEST 0.9551
### type=and readout=linear
dataset=mnist type=and hidden=8192 k=8 readout=linear margin=25 活性率=0.186  BEST 0.9713
### type=dense readout=proto
dataset=mnist type=dense hidden=8192 readout=proto 活性率=0.516
proto読み出し(学習なし): test acc 0.1192
### type=and readout=proto
dataset=mnist type=and hidden=8192 k=8 readout=proto 活性率=0.186
proto読み出し(学習なし): test acc 0.0828
```

proto の2件はバックボーンを学習しない条件での対照なので、偶然水準になるのが正しい。

### 1.2 幅1024、MNIST、25エポック(同一幅での特徴の質の比較)

```text
dense      hidden=1024 活性率=0.523  BEST 0.9274
densebias  hidden=1024 活性率=0.199  BEST 0.9111
and k=8    hidden=1024 活性率=0.185  BEST 0.9477
```

### 1.3 k(AND型のリテラル数)の掃引、MNIST、H=8192、25エポック

```text
k=4   活性率=0.417  BEST 0.9661
k=6   活性率=0.275  BEST 0.9704
k=8   活性率=0.186  BEST 0.9731
k=12  活性率=0.086  BEST 0.9716
k=16  活性率=0.044  BEST 0.9693
k=24  活性率=0.012  BEST 0.9515
```

### 1.4 幅の掃引、MNIST、k=8、25エポック

```text
H=2048   BEST 0.9589
H=4096   BEST 0.9691
H=8192   BEST 0.9731   (1.3 より)
H=16384  BEST 0.9757
H=32768  BEST 0.9745
```

### 1.5 Fashion-MNIST

```text
H=16384 k=6  (25ep)  BEST 0.8605
H=16384 k=8  (25ep)  BEST 0.8587
H=16384 k=12 (25ep)  BEST 0.8509
H=32768 k=4  (15ep)  BEST 0.8642
H=32768 k=6  (15ep)  BEST 0.8583
H=32768 k=8  (15ep)  BEST 0.8663

マージン掃引 (H=16384, k=6, 15ep)
margin=5    BEST 0.8514
margin=50   BEST 0.8537
margin=100  BEST 0.8512
```

### 1.6 平均化パーセプトロン(-avg)

```text
MNIST   and k=8 H=8192  15ep  avg  BEST 0.9750   (平均化なし25epで 0.9731)
Fashion and k=8 H=16384 20ep  avg  BEST 0.8809   (平均化なし25epで 0.8587)
Fashion and k=8 H=32768  3ep  avg  BEST 0.8768
```

Fashion で +2.2pt。平均化が最も効いた変更。

### 1.7 局所受容野(conv型。3値なし)、MNIST、H=8192、15エポック、平均化あり

```text
patch=5x5 match=85%  活性率=0.408  BEST 0.9690
patch=7x7 match=85%  活性率=0.303  BEST 0.9710
patch=7x7 match=90%  活性率=0.217  BEST 0.9583
patch=9x9 match=85%  活性率=0.195  BEST 0.9739
patch=9x9 match=90%  活性率=0.123  BEST 0.9649
```

3値(AND型 0.9750)に頼らず、±1重み + 整数バイアスだけで 0.9739 に到達している。

### 1.8 整数スコアによる特徴選抜(-over)、MNIST、H=8192、k=8、15エポック、平均化あり

```text
over=1   活性率=0.186  BEST 0.9750
over=4   活性率=0.307  BEST 0.9750
over=16  活性率=0.308  BEST 0.9691
```

改善せず、強くすると悪化。

### 1.9 AND型 + 平均化のシード追試(参考。3値相当の構成)

```text
and k=8 H=16384 25ep 平均化なし: seed1 0.9757 / seed2 0.9747 / seed3 0.9739  → 97.48 ± 0.09
and k=8 H=16384 20ep 平均化あり: seed1 0.9780 / seed2 0.9771 / seed3 0.9764  → 97.72 ± 0.07
Fashion 同構成       20ep 平均化あり: seed1 0.8809 / seed2 0.8746 / seed3 0.8800  → 87.85 ± 0.28
```

---

## 2. BEPの読み出し差し替え(bepreadout)

BEPを15エポック学習させた後、同じバックボーンの活性で整数線形読み出しを学習させた。

```text
### 784->512->1024 (既定構成)
BEP(プロトタイプ読み出し): 0.9001
BEP backbone + 整数線形読み出し: 0.9023 (差 +0.0022)

### 784->1024 (単層。中間ボトルネックの影響を除く)
BEP(プロトタイプ読み出し): 0.8943
BEP backbone + 整数線形読み出し: 0.9119 (差 +0.0176)

### 784->8192 (単層)
BEP(プロトタイプ読み出し): 0.8924
BEP backbone + 整数線形読み出し: 0.9397 (差 +0.0473)
```

幅が広いほど、固定プロトタイプ読み出しの取りこぼしが大きい。

---

## 3. 全層学習モデル(model)

### 3.1 誤差駆動 Hebbian による第1層の学習(MNIST、H=4096、8エポック、データ由来初期化)

凍結(第1層を学習しない)の基準値は **0.9698**。

```text
lrlog2=3  impthr=2  (既定・最も強い)  BEST 0.9234
lrlog2=6  impthr=4                     BEST 0.9487
lrlog2=8  impthr=4                     BEST 0.9620
lrlog2=10 impthr=4                     BEST 0.9689
lrlog2=8  impthr=16                    BEST 0.9592
lrlog2=12 impthr=8                     BEST 0.9700
```

更新を弱めるほど凍結に近づいて良くなる。

### 3.2 グループ内1個更新(BEP Eq.9 のマスク相当)を追加

```text
group=64  lrlog2=4 warmup=0  BEST 0.9511
group=64  lrlog2=3 warmup=2  BEST 0.9613
group=16  lrlog2=4 warmup=0  BEST 0.9507
group=256 lrlog2=3 warmup=0  BEST 0.9438
```

改善しない。

### 3.3 ランダム初期化での比較(MNIST、H=4096、10エポック)

まず matchpct=85 のまま実行したところ、全条件で 0.1028(偶然水準)になった。
ランダム重みは実画像パッチと 85% も一致しないため、どのニューロンも発火せず
実験が退化していた。しきい値を適正化して測り直した。

```text
matchpct=55  凍結 0.9702 / 誤差駆動で学習 0.8966
matchpct=60  凍結 0.9653 / 誤差駆動で学習 0.8668
matchpct=65  凍結 0.9029 / 誤差駆動で学習 0.8902
```

初期化の違いによらず、誤差駆動の学習は精度を下げる。
なおランダム初期化 + 凍結(0.9702)は、データ由来初期化 + 凍結(0.9698)と同等で、
初期化の出所は精度をほとんど左右しない。

### 3.4 競合学習(MNIST、H=4096、ランダム初期化 matchpct=55、出力10エポック)

```text
競合学習なし(凍結)  BEST 0.9689
競合学習 1エポック    BEST 0.9723
競合学習 3エポック    BEST 0.9724
```

**第1層の学習が精度を上げた唯一の条件。**

### 3.5 本番規模(H=16384、patch=9、競合学習3エポック、出力20エポック、ランダム初期化)

```text
MNIST   matchpct=55  BEST 0.9791
MNIST   matchpct=70  BEST 0.9795
Fashion matchpct=55  BEST 0.8796
Fashion matchpct=70  BEST 0.8643
```

### 3.6 最良構成のシード追試(MNIST、H=16384、patch=9、matchpct=70、VQ=3、20エポック)

```text
seed=1  BEST 0.9795
seed=2  BEST 0.9806
seed=3  BEST 0.9790
```

平均 **0.9797 ± 0.0007**。

---

## 参考: 比較基準

- crow BEP 実装(`../REPORT.md`): MNIST 92.13 ± 0.48、Fashion 83.78 ± 0.03(3シード、50エポック、検証分離)
- BEP 論文忠実版(同上): MNIST 87.99 ± 1.32、Fashion 84.61 ± 0.13
- BEP 論文報告値: Fashion 85〜87%
- Tsetlin Machine(`../tsetlinlab/REPORT.md`): MNIST 97.24 ± 0.12、Fashion 87.44 ± 0.11(3シード)
