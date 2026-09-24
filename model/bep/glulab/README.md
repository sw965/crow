# glulab: 二値の乗算ユニットへ BEP を拡張する実験

[../layer.go](../layer.go) の層は `a = sign(W x)` です。ここでは枝を2本にします。

```text
a = sign(W_u x) ⊙ sign(W_g x)
```

`±1` 同士の積なので `a` は `±1` のままで、BEPが前提とする超立方体上の最適化は壊れません。既存のcrow実装は変更せず、このフォルダだけで完結します。

## これは SwiGLU ではありません

`sign(p)·sign(q) = sign(p·q)` なので、この層が計算しているのは **2つの超平面の XNOR** です。

SwiGLU の強さは連続値ゲート（0.2倍、0.8倍と情報量を段階的に調整すること）から来ますが、`±1` の積にその成分はありません。`g = +1` なら素通し、`g = -1` なら符号反転、の2択です。

したがって増えるのは**各ユニットの表現力**（単一パーセプトロンでは表せない線形分離不可能な関数を1ユニットで表せる）であり、比較すべき対照は「ゲートの有無」ではなく**幅とパラメータ数**になります。

## 枝の目標値と、同時に適用できないという制約

他の枝を固定すれば、枝の目標値は一意に決まります。

```text
u* = a* ⊙ g
g* = a* ⊙ u
```

**この2つを同時に適用してはいけません。** 両枝を反転させると `XNOR(¬u, ¬g) = XNOR(u, g)` で出力が元に戻るためです。各式は「他方を固定したとき」に限って正しい条件付きの解です。

したがって1ニューロンにつき補正を流す枝は必ず1本に絞る必要があります（`-route`）。

| route | 振り分け |
|---|---|
| `cheap` | `argmin(\|z_u\|, \|z_g\|)`。反転させやすい方の枝を直す |
| `alt` | バッチごとに枝を交互に切り替える |
| `both` | 両枝を同時に直す。上記のとおり破綻するはずの対照 |

`cheap` は [../layer.go](../layer.go) の「直しやすい順」ソートの自然な一般化です。出力の直しやすさは `|z_u|` と `|z_g|` の両方に依存するので、`conf = min(|z_u|, |z_g|)` を確信度として使い、その枝を直します。

## 対照実験

`-arch` で切り替えます。

| arch | 構成 | 重み数 | 推論(1サンプル) |
|---|---|---:|---:|
| `plain` | 784 → 512 → 1024 の通常BEP | 925,696 | 10.4us |
| `wide` | 784 → **1024** → 1024 の通常BEP | 1,851,392 | 13.5us |
| `wider1536` | 784 → 1536 → 1024 | 2,777,088 | 16.5us |
| `wider2048` | 784 → 2048 → 1024 | 3,702,784 | 23.5us |
| `wider2560` | 784 → 2560 → 1024 | 4,628,480 | 39.1us |
| `gated` | 784 → 512 → 1024 の乗算ユニット | 1,851,392 | 22.3us |

隠れ層だけを広げることで、**`wide` と `gated` の重み数が厳密に一致**します。`gated` の優位がパラメータ増によるものか乗算構造によるものかを分離するための設計です。

`wider*` は**推論時間**を揃えた対照です。`wider2048`（23.5us）が `gated`（22.3us）とほぼ同じ推論コストになります。

## 前層への合成

2枝になると、前層への希望活性は両枝の票を整数で足してから符号を取ります。

```text
a_prev* = sign( W_uᵀ(gate ⊙ u*) + W_gᵀ(gate ⊙ g*) )
```

補正を受けなかった枝の目標値は現在の符号そのものなので、「変えなくてよい」という票として入ります。

## ノイズの分布と学習率の表現

浮動小数を学習ループから取り除けるかを測るための2つの切り替えです。
結果は [../FLOAT_REMOVAL.md](../FLOAT_REMOVAL.md) のステップ3・5の根拠になっています。

| フラグ | 内容 |
|---|---|
| `-noisekind norm` | `randx.IntNorm` によるガウスノイズ（既定、標準偏差が float32） |
| `-noisekind uniform` | 整数一様ノイズ。半幅は `scale × √3 × √fanIn` でガウスと標準偏差を揃える |
| `-lr 0.1` | `rng.Float32() > lr` による確率的更新（既定） |
| `-lrnum 1 -lrden 10` | `rng.IntN(den) >= num` による有理数 Bernoulli。`lrden > 0` のときこちらを使う |

## 逆伝播ゲート

既定は `open`（遮断しない）です。[../gatelab](../gatelab) の実測で、`|z|` によるゲートは精度をほとんど動かさず、廃止した方がわずかに良かったためです（MNIST +0.23pt / Fashion +0.43pt）。`-gate abs` で論文どおりの条件にもできます。

## 診断

エポックごとに、枝の符号一致率と出力の+1率を出力します。一致率が1に近づくと `sign(z_u) == sign(z_g)` となり、出力が定数に張り付いて乗算の意味が消えます。勾配を持たないBEPにはこの縮退を止める力が無いため、必ず確認してください。健全な値は0.4〜0.5前後です。

## 実行例

```powershell
go run ./model/bep/glulab -arch plain
go run ./model/bep/glulab -arch wide
go run ./model/bep/glulab -arch wider2048
go run ./model/bep/glulab -arch gated -route cheap -noise 0.25
go run ./model/bep/glulab -arch plain -noisekind uniform
go run ./model/bep/glulab -arch plain -lrnum 1 -lrden 10
go test -run "^$" -bench BenchmarkPredict -benchtime 200000x ./model/bep/glulab/
```

主要フラグは `-dataset mnist|fashion`、`-arch`、`-route`、`-gate open|abs`、`-noisekind norm|uniform`、`-noise`、`-lr` / `-lrnum` / `-lrden`、`-gsize`、`-margin`、`-epochs`、`-seed`、`-batch`、`-valratio`、`-threads` です。

## 境界

- `-arch plain` は1枝なので [../layer.go](../layer.go) 相当の条件です。[../gatelab](../gatelab) の `-gate open` と同一の値（MNIST 90.64 ± 0.08）を出すことを確認済みで、これが本ラボの土台の妥当性確認になっています。
- `gated` は `-noise 0.5` で崩壊します。枝が2本あるとノイズはどちらに乗っても出力を反転させるため実質的に倍になり、さらに `argmin` による振り分けがノイズで不安定化して、同じニューロンがサンプルごとに別の枝へ配線されます。`-noise 0.25` 以下で使ってください。
- 検証したのは20エポック、MNISTとFashion-MNISTの2層MLPのみです。
- 結果は [REPORT.md](REPORT.md)、実行条件は [LOGS.md](LOGS.md) にあります。
