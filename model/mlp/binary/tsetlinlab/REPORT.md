# 調査・実験報告: Tsetlin Machine — 万能近似・ビット/整数完結・BEP同等以上の精度を満たす手法

- 日付: 2026-08-14
- 依頼: 「crow/model/mlp/binary のコード・BEP論文(../papers/2512.04189v2.pdf)・
  UNIVERSAL_APPROXIMATION.md・REPORT.md・PROTOTYPES.md を読んだ上で、
  (1) 万能近似定理を満たす
  (2) BEPの核である、学習から推論まで高速なビット演算及び整数を中心としたアルゴリズムである
  (3) 現行binaryモデルのコード及び論文のモデルと同等以上の精度が実際に出る
  の3条件を満たす手法・モデルを探す。UNIVERSAL_APPROXIMATION.mdは参考資料であり、手段は問わない」
- 手順: 文献調査(候補の網羅と数値の裏取り) → 最有力候補をGoでフルスクラッチ実装 →
  **BEP実装と同一の二値化済みデータ**で精度を実測 → 3シードで再現性確認
- 実験環境: Windows 11 / Go 1.26.5 / 12論理コアCPU(並列12ワーカー)
- 実験コード: 同フォルダ `main.go`。生ログ: `LOGS.md`

## 1. 結論

**Tsetlin Machine (TM)、具体的には整数重み付きマルチクラスTM**(発展形として畳み込みTM /
Coalesced TM / Fuzzy-Pattern TM)が3条件をすべて満たす。

| 条件 | 判定 | 根拠 |
|---|---|---|
| 1. 万能近似 | ✅ | §5。minterm節による二値入力上の任意写像の厳密表現(構成的証明)+ UNIVERSAL_APPROXIMATION.md §4 と同じ「モデル族」の議論がそのまま接続できる |
| 2. ビット演算・整数のみ | ✅ | §3・§4。節評価はワードAND/ゼロ判定、状態更新は8bit整数カウンタ(ビットスライスで64リテラル同時)、確率判定は整数PRNG比較。学習・推論経路に浮動小数点ゼロ |
| 3. 同等以上の精度(実測) | ✅ | §6。同一データで MNIST 97.24±0.12%(BEP実装 92.13±0.48%)、Fashion-MNIST 87.44±0.11%(BEP実装 83.78±0.03%、論文報告 85〜87%) |

## 2. Tsetlin Machine とは(アルゴリズムの全体像)

TM (Granmo, 2018, arXiv:1804.01508) は、命題論理の連言節(AND節)の集合で分類・回帰を行う
学習機械。ニューラルネットではないが、「多数の単純なユニットの投票 + 誤り駆動の局所更新」
という構造はBEPと同族で、以下の対応がある(§8)。

### 2.1 モデル(推論)

- 入力: 二値ベクトル x ∈ {0,1}^F。**リテラル**は各入力ビットとその否定の計 2F 個
  (x_1, …, x_F, ¬x_1, …, ¬x_F)。
- **節 (clause)**: リテラルの部分集合の連言。節 j は各リテラル k について
  「包含/除外」を持ち、包含した全リテラルが真のときに限り発火(=1)する。
- 各クラス c は節の集まりを持つ。偶数番の節は正極性(+)、奇数番は負極性(−)。
- クラススコア: `sum_c = Σ_{正極性} w_j·発火_j − Σ_{負極性} w_j·発火_j`(w_j は整数重み)。
- 予測: `argmax_c sum_c`。

推論に必要な演算は、
節発火 = `(includeMask AND NOT literals) == 0`(ワードごとのAND+ゼロ判定)、
スコア = 整数加減算、予測 = 整数比較、だけである。

### 2.2 学習

各節×各リテラルに **Tsetlin Automaton (TA)** と呼ばれる小さな整数カウンタ(本実装では
8bit、0..255)を置く。**カウンタが128以上(最上位ビット=1)ならそのリテラルを包含**する。
学習はこのカウンタの±1だけで進む。

サンプル (x, y) ごとに:

1. 正解クラス y と、無作為に選んだ負例クラス q の2クラスだけを更新する。
2. クラスごとに投票和 sum を計算し、`clip(sum, -T, T)` にクリップする。
   T は**投票マージン**(BEPのマージン r·K に相当する役割)。
3. 各節は確率的にフィードバックを受ける:
   - 正解クラス: 確率 `(T − clip(sum))/2T`。正極性節に Type I、負極性節に Type II
   - 負例クラス: 確率 `(T + clip(sum))/2T`。正極性節に Type II、負極性節に Type I
   - sum が既に ±T に達していれば確率0(=学習済みサンプルは更新しない。
     BEPの「マージン達成なら更新不要」と同じ枠組み)
4. **Type I フィードバック**(パターン獲得。頻出パターンを節に刻む):
   - 節が発火: 真リテラルのカウンタ +1(boost true positive)、
     偽リテラルのカウンタを確率 1/s で −1
   - 節が不発火: 全リテラルのカウンタを確率 1/s で −1(パターンの忘却・単純化)
5. **Type II フィードバック**(誤発火の抑制。判別に必要なリテラルを足す):
   - 節が発火した場合のみ: 偽リテラルのカウンタ +1
     (これにより将来そのリテラルが包含されれば、同じ入力では発火しなくなる)
6. **整数重み**(Abeyrathna et al. 2021, IEEE Access): Type I で発火時 w+1、
   Type II で発火時 w−1(下限1)。少ない節数で精度を稼ぐ。

s は**感度**(specificity)。大きいほど節が特殊化する。文献では実数(例 s=10, 15)だが、
本実装は s = 2^k に制限して「乱数ワード k 個のAND」で確率 1/s のビットマスクを
正確に生成する(浮動小数点なしで厳密)。

### 2.3 本実装と文献の標準的なTMとの差分

TM論文群と本実装(main.go)を照合する際に注意する差分の一覧。
いずれも「整数・ビット演算のみ」を厳密に守るため、または実装簡素化のための変更で、
結果(§6.1)は文献の期待レンジ内に収まっている。

| 項目 | 文献の標準形 | 本実装 |
|---|---|---|
| 感度 s | 実数(例: 10, 15) | **2^k に制限**(8, 16)。確率 1/s を乱数ワード k 個のANDで厳密に生成するため |
| Type Ia の真リテラル強化 | 確率 (s−1)/s(boost true positive オプションで確率1) | **常に確率1**(boost 常時ON。画像系タスクの慣行) |
| TA状態 | 状態数 2N の有限オートマトン(N は自由パラメータ) | **8bit固定(0..255)、包含しきい値128、初期値127**(境界のすぐ下・除外側) |
| 整数重み | Abeyrathna et al. 2021 の整数重みTM | 同論文に従い Type Ia で +1、Type II で −1(下限1)。上限 2^20 |
| 負例クラス | 毎サンプル1クラスを無作為抽選(マルチクラスTMの標準実装と同じ) | 同じ |
| 空節の扱い | 学習時は発火扱い・推論時は不発火(標準実装の慣行) | 同じ |
| literal budget / clause drop 等の追加機構 | 実装によってはあり | **無し** |
| 乱数 | 実装依存(C実装は高速な整数PRNG) | xorshift64*(整数のみ。確率判定は剰余比較) |

### 2.4 ハイパーパラメータ

| 記号 | 意味 | 本実験の値 |
|---|---|---|
| 節数/クラス | モデル容量 | MNIST 512、Fashion 2048/4096 |
| T | 投票マージン(クリップ幅) | 4×節数(自動設定) |
| s | 感度(1/s が忘却確率) | MNIST 8、Fashion 16 |
| 重み | 整数節重みの有無 | 有り |
| TA状態幅 | カウンタのビット数 | 8bit (0..255、包含しきい値128) |

## 3. 条件2の検証: 学習から推論までビット演算・整数のみ

本実装(main.go)の全経路の演算を列挙する:

| 処理 | 演算 |
|---|---|
| 節発火判定 | `include &^ literals` のワードAND + ゼロ判定(26ワード/節) |
| 投票和 | 整数加減算(int64) |
| 予測 | 整数比較のargmax |
| TA状態更新 | ビットスライス(8プレーン)へのマスク付きリップルキャリー加減算。64リテラル同時、飽和付き |
| フィードバック確率 (T±sum)/2T | 整数PRNG(xorshift64*)の剰余と整数比較 |
| 確率1/sのマスク | 乱数ワード k 個のAND(s=2^k で厳密) |
| 負例クラスの抽選・シャッフル | 整数PRNG(Fisher-Yates) |
| 整数重みの更新 | int32 の ±1 |

- 学習・推論の経路に float は一切現れない(唯一の float は「精度の表示」のみで、
  UNIVERSAL_APPROXIMATION.md §12.2 の「評価レポートは中核ではない」の区分と同じ)。
- UNIVERSAL_APPROXIMATION.md §6 の表でBEP側が「整数化案」として挙げていた項目
  (確率のuint64しきい値比較、整数ノイズ、整数絶対値等)に相当するものが、
  TMでは最初から整数で定義されている。
- ハードウェア実装の裏付け: 全デジタルTM推論ASICで MNIST 8.6nJ/フレーム・60.3kフレーム/秒
  (65nm, arXiv:2501.19347)。FPGA・マイコン実装も多数ある。

### 3.1 ビットスライスTAという実装技法

TA状態(0..255)を「8枚のビットプレーン」に分解して持つ。プレーン p のワード w は、
64個のリテラルの状態の第 p ビットを並べたもの。この表現だと:

- **包含マスク = プレーン7そのもの**(状態≥128 ⇔ 最上位ビット=1)。
  節評価に必要なマスクが常に手元にある
- ±1 はマスク付きリップルキャリー(±1する桁上がり/桁借りをANDで伝播)で
  **64リテラル同時**に適用できる。飽和(0/255)は桁あふれビットへの一括OR/ANDで処理
- 実装は `incMasked` / `decMasked`(main.go)。参照実装(リテラルごとの素朴なuint8演算)
  との完全一致を `main_test.go` のテストとFuzzで検証済み

## 4. 実装の設計メモ

- **並列化**: サンプルごとに「フェーズ1: 2クラス分の発火判定+部分投票和(節レンジ並列)」
  「フェーズ2: フィードバック(節レンジ並列)」の2バリア。各ワーカーは自分の節レンジしか
  書かないため競合しない。`go test -race` で検証済み。
  ワーカーRNGはエポック先頭にメインRNGからシードするため、
  **同一シード・同一設定なら結果はワーカー数によらず決定的**
  (実際に、実験時の作業コピー(一時作業ディレクトリ上の同一実装)と本フォルダ版で
  BEST が小数4桁まで一致。LOGS.md §6)。
- **空節の扱い**: 学習時は発火扱い(Type Iでリテラル獲得を始めるため)、推論時は不発火。
  文献の標準的な扱いに従った。
- **データ**: crow/dataset の二値化済みgob(1bit/画素、784次元)。
  BEP実装の実験(親フォルダの ../REPORT.md)と同一入力なので、入力表現の差による有利不利はない。
  ただし現行bitsxではこのgobを読めないため、旧形式互換の型で直接デコードした(§9)。
- **メモリ**: 節あたり 8プレーン×26ワード×8B = 1,664B + 重み。
  Fashion 4096節/クラス×10クラスで約68MB。

## 5. 条件1の検証: 万能近似性

### 5.0 前置き — 「ニューラルネットではないのに万能近似?」への答え

TMはニューラルネットではない(命題論理+学習オートマトンのモデル)が、
万能近似性とは矛盾しない。万能近似性は**モデル族の性質**であって、NN固有の性質ではない
(UNIVERSAL_APPROXIMATION.md §1 の定義もモデル族に対するもの。多項式族についての
ワイエルシュトラスの近似定理、フーリエ級数などが同じ性質を持つ)。

TMで効いているのは次の2点だけである。

1. **ANDの投票はブール完全**: 任意のブール関数はDNF(ANDのOR)で書けるという
   論理学の基本事実。TMの節集合は重み付きDNFなので、二値入力上の表現力の底が
   最初からブール完全になっている。
   具体例(XOR): 節「x₁ ∧ ¬x₂」(+票)、「¬x₁ ∧ x₂」(+票)、「x₁ ∧ x₂」(−票)、
   「¬x₁ ∧ ¬x₂」(−票)の4節で、投票和の符号は厳密にXOR。近似ではなく厳密表現
2. **入出力を可変精度にできる**: UNIVERSAL_APPROXIMATION.md §4.1/§4.3 の
   固定小数点セル分割・多ビット出力の議論がそのまま接続できる

別の見方として、節は「包含リテラル数をしきい値とするしきい値ゲート」なので、
TM全体は**二値重みの2層しきい値回路の部分クラス**に埋め込める。つまり
UNIVERSAL_APPROXIMATION.md §10 が引用する Yayla et al. のBNN万能近似定理の構成と
同じ部品でできており、「NNでないから満たさないのでは」という直感は当たらない。

下記 5.1 の構成が実際に厳密表現になる事は、`main_test.go` の
`TestMintermConstructionRepresentsArbitraryMapping` が自動検証している
(ランダムな784bitコード集合に任意のラベルを割り当て、minterm節を直接構成し、
全コードで投票和が厳密に一致する事、コード外入力で全投票和が0になる事を確認)。

### 5.1 構成

UNIVERSAL_APPROXIMATION.md と同じ「モデル族」(ε ごとに入力精度・出力精度・幅を
増やせる)の意味で成立する。構成は同文書 §4 とほぼ同型で、TMではむしろ簡潔になる。

1. **二値入力上の任意写像の厳密表現**: 入力符号 p ∈ C ⊆ {0,1}^d ごとに、
   そのパターンの真リテラル全て(d個)を包含する **minterm節** を1つ置く。
   この節は x = p のとき、かつそのときに限り発火する
   (1ビットでも違えば、違ったビットに対応するリテラルが偽になるため)。
   同文書 §4.2 の検出ニューロン `h_p(x) = sign(p^T x − (d−1))` と同じ役割を、
   しきい値もバイアスも無しで節1個が厳密に果たす。
   出力ビット k の正解 y_(p,k) が 1 のセルの minterm節を正極性、
   0 のセルの minterm節を負極性に割り当てれば、入力 p ∈ C での投票和は
   +w または −w となり、符号は厳密に y_(p,k) に一致する。
   二値重みしきい値回路で必要だった整数バイアス S_k すら不要。
   節数は出力ビットあたり |C|(有効コード数)で、同文書 §4.2 の検出ニューロン数と同じ。
2. **C 外の挙動**: x ∉ C では発火節が無く投票和が 0 になる。
   同文書 §4.2 の C 限定構成と同様「C 外は符号規約による既定動作」であり、
   入力経路が C 外を生成しないことの検証という同じ注意がそのまま当てはまる。
3. **可変精度入力**: 同文書 §4.1/§5.3 の固定小数点化・温度計/二進符号化がそのまま使える。
   TMの標準前処理はまさに温度計符号化(thermometer encoding)で、
   BEP論文が引用する Bacellar et al. 2024 の distributive thermometer とも共通。
4. **可変精度出力**: 同文書 §4.3/§5.4 の多ビット固定小数点ヘッド
   (出力ビットごとに節バンクを持ち、投票和の符号を出力ビットとする)で構成できる。
   もう一つの経路として**回帰TM**(Abeyrathna et al. 2020)は投票和(整数)を
   そのまま値として使う。整数の値域スケーリングは固定小数点の暗黙尺度で扱える。
5. **結論**: コンパクト集合上の連続関数 f: K → R^r と任意の ε > 0 に対し、
   同文書 §4 と同一の議論(一様連続性によるセル分割 + セルごとの固定小数点出力の
   厳密表現)で `sup_x ||F(x) − f(x)||_∞ < ε` を達成するTMが存在する。
   同文書 §4.4 の「固定ビット幅では万能近似にならない」もそのまま当てはまる
   (節数・入力符号長・出力ビット数・投票和のアキュムレータ幅は ε に応じて増える。
   投票和は節数に対して高々線形なので、アキュムレータ幅は対数的に増えれば足りる)。

**学習可能性との分離**(同文書 §7 と同じ注意): 上記は表現能力の存在証明であり、
TMのフィードバック学習がそのパラメータへ必ず収束することは保証しない。
ただしTMには基本演算子についての収束証明の系列がある:

- IDENTITY / NOT 演算子: arXiv:2007.14268(ノイズあり設定を含む)
- XOR 演算子: Jiao et al.
- AND / OR 演算子: arXiv:2109.09488(これで1bit/2bit基本演算子の収束解析が完備)
- 一般化(多ビット連言、確率的概念学習の枠組み): AAAI 2025 / arXiv:2310.02005。
  一般の場合の完全な収束証明は未解決(BEPも同様に未解決)

## 6. 条件3の検証: 精度

### 6.1 実測(本実験、同一データ・同一二値化)

| データセット | TM実測(3シード平均) | TM実測(ベスト構成) | crow BEP実装 | BEP論文忠実版 | BEP論文報告 |
|---|---:|---:|---:|---:|---:|
| MNIST | **97.24 ± 0.12** (512節, 20ep) | 97.33 | 92.13 ± 0.48 (50ep) | 87.99 ± 1.32 | — |
| Fashion-MNIST | **87.44 ± 0.11** (2048節, 30ep) | 87.78 (4096節, 40ep) | 83.78 ± 0.03 | 84.61 ± 0.13 | 85〜87% |

(crow BEP実装・論文忠実版・論文報告の数値は、**親フォルダの ../REPORT.md
(BEP検証レポート。本ファイルとは別物)**による。
3シードは 1, 2, 3。個々の値は MNIST: 97.33/97.28/97.10、Fashion: 87.42/87.34/87.56)

特筆事項:

- MNIST は **1エポック目(93.27%)の時点で crow BEP の50エポック後(92.13%)を上回った**。
  学習は約2秒/エポック(512節、12スレッドCPU)
- Fashion-MNIST は3エポック目で crow BEP(83.78%)を上回り、
  全シードで論文報告の上限(87%)も超えた。4096節/40エポックで87.78%、まだ緩やかに上昇中
- ばらつき(±0.11〜0.12)は crow BEP実装(±0.48)より小さい
- 入力は crow の固定二値化のまま。TM文献の Fashion 90%級は適応的ガウスしきい値
  (1bit/画素だが局所適応)を使っており、入力表現を文献に合わせれば上積みの余地がある

### 6.2 文献値(発展形の上限)

| モデル | MNIST | Fashion-MNIST | 出典 |
|---|---:|---:|---|
| 素の(非畳み込み)TM | 98.57 | 90.09 | CTM論文 / CoTM論文 Table 5 |
| 重み付き畳み込みTM / CoTM | 99.33 / 99.4(peak) | 91.18〜91.83 | arXiv:2108.07594 Table 2/3/5 |
| Fuzzy-Pattern TM (2025) | 98.56(20節) | **94.68**(8000節)、93.19(20節) | arXiv:2508.08350 |

- CoTM論文 Table 3(Fashion, 畳み込み+重み): 50節 82.33〜86.79 → 8000節 91.18。
  学習時間は V100 GPU で 21〜87秒/エポック
- CIFAR-10: TM Composites で 82.8%(2024, arXiv:2406.00704。生画素+画像処理
  スペシャリスト)。BEP論文の CIFAR-10 〜84% は AlexNet特徴量が前提なので
  直接比較はできない(生画素からのBEPは論文に報告がない)
- 時系列(BEP論文のRNN実験に相当する領域): TMの逐次拡張は研究途上で、
  UCRベンチマークでのBEP-TT(平均81%級)との直接比較データは見つからなかった。
  ここはBEPが現状優位な領域として残る

## 7. 他候補と棄却理由

網羅的に調べた候補と3条件への適合:

| 候補 | (1)万能近似 | (2)ビット/整数学習 | (3)精度 | 判定 |
|---|---|---|---|---|
| **Tsetlin Machine系** | ✅ | ✅ | ✅ MNIST 97〜99 / Fashion 87〜94.7 | **採用** |
| DWN(Differentiable Weightless NN, ICML2024)/ difflogic(論理ゲート網)/ LUTNet 等のLUT系 | ✅ | ❌ 学習がfloat勾配(STE/soft-logic) | ✅ MNIST 98%+ | 推論のみビット。BEPの核「学習もビット」を満たさない |
| NITI (arXiv:2009.13108) / NITRO-D (arXiv:2407.11698) / PocketNN (arXiv:2201.02863) | ✅ | △ 整数のみだが int8乗算中心 | ✅ MNIST 96.98〜99、Fashion 87.7 (PocketNN) | 次点。「整数中心」は満たすが、1bit重み・XNOR/popcountというBEPの利点(メモリ・演算器)を失う。ビット演算ではなく乗算器が要る |
| 超次元計算(HDC/VSA) | △(符号化依存) | ✅ 整数カウンタ | ❌ MNIST 概ね90前後(モデルを大きくしても95未満が相場) | 精度不足 |
| 進化計算・焼きなましによるBNN直接最適化 | ✅ | ✅ | ❌ 深い構成にスケールしない(BEP論文 §2 でも指摘) | 棄却 |
| BEP拡張(UNIVERSAL_APPROXIMATION.mdの路線: 整数バイアス+可変精度) | ✅(要実装) | ✅ | 未知(現状 Fashion 83.8〜84.6) | 自前路線として引き続き有効。ただし実証済み精度は現時点でTMが明確に上 |

## 8. BEPとTMの構造対応(設計思想の近さ)

| BEP (crow/binary) | TM | 備考 |
|---|---|---|
| 整数隠れ重み H (int8) | TA状態カウンタ (8bit) | どちらも「シナプス慣性」で破滅的忘却を抑える |
| 可視重み W = sign(H) | 包含フラグ = 状態の最上位ビット | どちらも整数状態の符号/MSBが可視構造 |
| ロジット = プロトタイプとの一致数(popcount) | クラス投票和(整数和) | どちらも整数スコアのargmax |
| マージン r·K による更新判定 | 投票マージン T(sum=±T で更新確率0) | どちらも「十分学べたサンプルは触らない」 |
| ゲート(|z|が小さいニューロンだけ更新) | フィードバック確率が (T∓sum)/2T で減衰 | 選択的更新の確率版/決定版 |
| Sign集約 + 確率的lr(実効的に疎な更新) | 確率1/sのランダムマスク更新 | どちらも更新の時間的分散 |
| Hebbian外積 ΔH = a*·aᵀ | Type I の真リテラル強化 | 共起の強化 |
| 誤差の逆伝播(多層) | **無し**(TMは実質1層の節集合) | TMは深さを畳み込み・Composite・逐次拡張で補う |

最後の行が本質的な違いで、TMは「深い表現学習」を放棄する代わりに
節という解釈可能な単位の物量と投票で精度を出す。MNIST/Fashion級ではそれで
BEPを上回るが、深い階層表現が本当に必要なタスクでどうなるかは開かれた問題。

## 9. 副産物: 発見したリポジトリの問題(データセットgob非互換)

`dataset.LoadMNIST` / `LoadFashionMNIST` は現行HEADでは動作しない。

- 公開されている v0.1.0-test のgobは、bitsx.Matrix が公開フィールド
  `{Rows int; Cols int; Data []uint64}`・`Matrices = []*Matrix` だった時代
  (omw コミット 4c0e3e7「enforce bitsx.Matrix tail-bit invariant via unexported
  data field」より前)にエンコードされたもの
- 現行 bitsx.Matrix は非公開フィールド + GobEncode/GobDecode
  (omw/mathx/bitsx/matrix_gob.go)に変わったため、デコードすると
  `gob: decoding into local type *bitsx.Matrices, received remote type Matrices`
  で失敗する
- さらに `loadWithRecovery` がこれを「キャッシュ破損」と誤認して**正常なローカル
  キャッシュを削除して再ダウンロード**し、当然また失敗する(キャッシュ喪失の実害)
- 本実験プログラムは旧形式互換の型(`oldMatrix`)で直接デコードして回避した
- 修正候補: (a) crow/dataset に旧形式互換のデコード層を置き現行Matrixへ変換
  (b) gobを現行形式で再エンコードして新リリースを公開
  (c) bitsx.GobDecode に旧形式フォールバックを追加

**解決(2026-08-14)**: (b) を採用した。なお (c) は gob の仕様上不可能
(型ネゴシエーション段階で失敗し、GobDecode 自体が呼ばれない)。

- 旧gobを現行bitsx形式で再エンコードし(全8ファイル、再読込での完全一致・
  件数・形状・ラベル範囲を検証済み)、リリース v0.2.0-test として公開した
- `crow/dataset/mnist.go` の `defaultBaseURL` は v0.2.0-test に更新済み
- `dataset.LoadMNIST` / `LoadFashionMNIST` が v0.2.0-test を正しくダウンロード・
  デコードできることを実機(新規キャッシュからの実ダウンロード)で確認済み。
  **`dataset.LoadMNIST`/`LoadFashionMNIST` のシグネチャは変更していない。**

**本実験プログラム(`oldMatrix`)側は未修正のまま残っている。**
これは、上記バグにより当時 `dataset.LoadMNIST` が使えず、それを迂回して
生gobを独自形式で直接読むために書かれたもの。バグが直った今、
**正しい直し方は「旧形式ファイルを別途用意して動かし続ける」ことではなく、
`oldMatrix`/`loadDataset` を削除して `dataset.LoadMNIST`/`LoadFashionMNIST`
の呼び出しに置き換えること**(シグネチャ不変なのでそのまま呼べるはず)。
`packSamples` が触る `img.Rows`/`img.Cols`/`img.Data[w]` は現行 `bitsx.Matrix`
では非公開フィールドのため、公開API `img.Rows()`/`img.Cols()`/`img.Word(w)`
への置き換えが必要。**この置き換え自体は未実施・未検証**(詳細・暫定の
回避策は README.md「データの入手」を参照)。

## 10. crow への統合スケッチ(実装する場合の指針)

- `bitsx.Matrix` はそのまま節の包含マスク・入力リテラル表現に使える
  (節評価 `(include &^ literals) == 0` は既存カーネルのワード演算と同族)
- TA状態は本実験のビットスライス(8プレーン)でも、素朴な `[]int8`(BEPのHと同型)
  でもよい。ビットスライスは速いが、H と同じ int8 配列の方がコードは単純
- 回帰: 投票和(整数)をそのまま値にする回帰TMは、PROTOTYPES.md B-10c の
  「線形読み出し = 1本のベクトルとの内積」の知見と自然に整合する
  (重み付き投票和は最初から線形読み出しの形をしている)
- 入力の多値化には既存の `NewThermometerMatrices` の温度計符号がそのまま流用できる
- 分類は本実験の構成で十分。少ない節数で精度が欲しい場合は
  Coalesced TM(節をクラス間で共有し、クラス×節の整数重み行列を持つ)や
  Fuzzy-Pattern TM(発火を「失敗リテラル数」による段階評価にする。popcountで実装可能)
  が文献上有効

## 11. 制約・注意(この報告の限界)

- 実測は3シード(crow側 REPORT.md の慣行に合わせた)。単一実装・単一マシン
- ハイパーパラメータ探索は粗い(T=4×節数、s∈{8,16} を文献推奨値から選んだだけ)。
  詰めれば上積みがあり得る
- s を 2^k に制限した(厳密な整数化のため)。文献の s=10, 15 とは厳密には異なるが、
  結果は文献の期待レンジ内であり実害は観測されなかった
- Fashion の文献値(90%級)との差は、主に入力二値化の差(固定しきい値 vs 適応的
  ガウスしきい値)と非畳み込み構成によるもので、本実験は「BEPと同一入力での比較」を
  優先してあえて揃えなかった
- CIFAR-10・時系列(RNN相当)は実測していない。文献値の整理のみ(§6.2)
- 万能近似は「モデル族」の性質であり、固定サイズの1モデルの性質ではない。
  また表現能力の存在証明と学習可能性は別問題(§5)。
  この区別は UNIVERSAL_APPROXIMATION.md §7・§9 の完了条件と同じ扱いを推奨する

## 12. 根拠資料

### Tsetlin Machine 本体・発展形

- O.-C. Granmo, "The Tsetlin Machine – A Game Theoretic Bandit Driven Approach to
  Optimal Pattern Recognition with Propositional Logic" — arXiv:1804.01508
- O.-C. Granmo et al., "The Convolutional Tsetlin Machine" — arXiv:1905.09688
  (素のTM: MNIST 98.57 / K-MNIST 92.03 / Fashion 90.09。CTM: 99.4 / 96.31 / 91.5)
- S. Glimsdal, O.-C. Granmo, "Coalesced Multi-Output Tsetlin Machines with Clause
  Sharing" — arXiv:2108.07594(Table 2/3/4: 節数50〜8000のスケーリング、Table 5: 他手法比較)
- K. D. Abeyrathna, O.-C. Granmo, M. Goodwin, "Extending the Tsetlin Machine With
  Integer-Weighted Clauses for Increased Interpretability" — IEEE Access 9, 2021
- K. D. Abeyrathna et al., "The Regression Tsetlin Machine – A Novel Approach to
  Interpretable Non-Linear Regression" — Phil. Trans. Royal Society A 378, 2020
- A. Wheeldon et al. 系の高速化: "Massively Parallel and Asynchronous Tsetlin Machine
  Architecture" — ICML 2021 / arXiv:2009.04861
- "The Weighted Tsetlin Machine: Compressed Representations with Weighted Clauses"
  — arXiv:1911.12607(MNIST 98.63、節数1/4)
- "Fuzzy-Pattern Tsetlin Machine" — arXiv:2508.08350(Fashion 94.68%、
  CoTM比最大316倍の学習高速化を主張)
- "TMComposites: Plug-and-Play Collaboration Between Specialized Tsetlin Machines"
  — arXiv:2309.04801、および "An Optimized Toolbox for Advanced Image Processing with
  Tsetlin Machine Composites" — arXiv:2406.00704(CIFAR-10 82.8%)
- ハードウェア: "An All-digital 8.6-nJ/Frame 65-nm Tsetlin Machine Image
  Classification Accelerator" — arXiv:2501.19347

### 収束解析

- "On the Convergence of Tsetlin Machines for the IDENTITY- and NOT Operators"
  — arXiv:2007.14268
- "On the Convergence of Tsetlin Machines for the AND and the OR Operators"
  — arXiv:2109.09488
- "Generalized Convergence Analysis of Tsetlin Automaton Based Algorithms:
  A Probabilistic Approach to Concept Learning" — AAAI 2025 / arXiv:2310.02005

### 比較候補(整数のみ学習・LUT系ほか)

- NITI: "Training Integer Neural Networks Using Integer-only Arithmetic"
  — arXiv:2009.13108
- NITRO-D: "Native Integer-only Training of Deep Convolutional Neural Networks"
  — arXiv:2407.11698
- PocketNN: "Integer-only Training and Inference of Neural Networks via Direct
  Feedback Alignment and Pocket Activations in Pure C++" — arXiv:2201.02863
  (MNIST 96.98 / Fashion 87.7)
- DWN: Bacellar et al., "Differentiable Weightless Neural Networks" — ICML 2024
  (BEP論文が温度計符号で引用している系列)

### BEP側の基準値

- 親フォルダの ../REPORT.md — BEP検証レポート(crow実装 MNIST 92.13±0.48、
  Fashion 83.78±0.03、論文忠実版 84.61±0.13、論文報告 85〜87%)
- L. Colombo et al., "BEP: A Binary Error Propagation Algorithm for Binary Neural
  Networks Training" — ICLR 2026 / arXiv:2512.04189(../papers/ に同梱)
