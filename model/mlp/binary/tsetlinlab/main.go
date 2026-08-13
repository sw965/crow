// tsetlinlab: ビット並列 Tsetlin Machine (TM) の検証実験プログラム。
//
// 目的: crow/model/mlp/binary (BEP) と同じ二値化済み MNIST / Fashion-MNIST を入力に、
// 学習から推論まで「ビット演算 + 整数カウンタ + 整数PRNG比較」だけで動く
// Tsetlin Machine の精度を実測する。詳細は同フォルダの REPORT.md を参照。
//
// これは実験用の独立したプログラム(データ収集プログラム相当)であり、
// binaryパッケージのライブラリコードからは参照されない。
//
// 実装方式:
//   - TA状態はビットスライス(8プレーン、符号なし0..255、プレーン7=最上位ビットが包含フラグ)
//   - インクリメント/デクリメントはマスク付きリップルキャリー(64リテラル同時)
//   - 節評価は (include &^ literals) == 0 のワード演算
//   - 全ての確率判定は整数PRNG(xorshift64*)の剰余/AND比較
//
// 浮動小数点は学習・推論の経路に一切使わない(精度の表示のみfloat)。
//
// 実行例(要 ~/.crow_dataset の二値化済みgob):
//
//	go run . -dataset mnist   -clauses 512  -epochs 20            # MNIST 97.3%前後
//	go run . -dataset fashion -clauses 2048 -slog2 4 -epochs 30   # Fashion 87.4%前後
//	go run . -dataset fashion -clauses 4096 -slog2 4 -epochs 40   # Fashion 87.8%前後
package main

import (
	"bytes"
	"encoding/gob"
	"flag"
	"fmt"
	"log"
	"math/bits"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"
)

const (
	numFeatures = 784             // 28x28 の1bit画素
	wordsPerFtr = 13              // ceil(784/64)
	numLitWords = 2 * wordsPerFtr // 正リテラル13語 + 否定リテラル13語
	numPlanes   = 8               // TA状態 0..255
	numClasses  = 10
)

// validMask[w]: 有効ビット(=実在するリテラル)のマスク。
// 末尾語(各13語ブロックの最終語)は 784 % 64 = 16 ビットのみ有効。
var validMask [numLitWords]uint64

func init() {
	tail := uint64(1)<<(numFeatures%64) - 1
	for w := range validMask {
		if (w+1)%wordsPerFtr == 0 {
			validMask[w] = tail
		} else {
			validMask[w] = ^uint64(0)
		}
	}
}

// ---------------------------------------------------------------------------
// 整数PRNG (xorshift64*)
// ---------------------------------------------------------------------------

type rng struct{ s uint64 }

func newRng(seed uint64) *rng {
	if seed == 0 {
		seed = 0x9E3779B97F4A7C15
	}
	return &rng{s: seed}
}

func (r *rng) next() uint64 {
	x := r.s
	x ^= x >> 12
	x ^= x << 25
	x ^= x >> 27
	r.s = x
	return x * 0x2545F4914F6CDD1D
}

// intn は [0, n) の一様整数(整数演算のみ。剰余バイアスは実験用途では無視できる)。
func (r *rng) intn(n uint64) uint64 { return r.next() % n }

// ---------------------------------------------------------------------------
// 節 (clause)
// ---------------------------------------------------------------------------

// clause は1つの節。TA状態はビットスライスで保持する。
// planes[p][w] は、リテラル群のTA状態(0..255)の第pビットを64リテラル分並べたもの。
// 状態128以上(=plane7が1)で「包含」。
type clause struct {
	planes     [numPlanes][numLitWords]uint64
	weight     int32 // 整数重み(重み無しモードでは常に1)
	includeCnt int32 // 包含リテラル数(plane7のpopcount)のキャッシュ
}

// 初期状態: 全リテラル 127 (包含境界のすぐ下、除外側)。
func (c *clause) init() {
	for p := range 7 {
		for w := range numLitWords {
			c.planes[p][w] = validMask[w]
		}
	}
	// plane7 = 0 (全て除外)
	c.weight = 1
	c.includeCnt = 0
}

// fires: 節の発火判定。包含リテラルが全て真なら true。
// lit は本サンプルの「真リテラル」ビット列(正リテラル=x、否定リテラル=~x)。
func (c *clause) fires(lit *[numLitWords]uint64) bool {
	inc := &c.planes[7]
	for w := range numLitWords {
		if inc[w]&^lit[w] != 0 {
			return false
		}
	}
	return true
}

// incMasked: mで指定したリテラルのTA状態を+1(255で飽和)。
// リップルキャリーで64リテラルを同時に処理する。
func (c *clause) incMasked(m *[numLitWords]uint64) {
	for w := range numLitWords {
		carry := m[w]
		if carry == 0 {
			continue
		}
		for p := range numPlanes {
			t := c.planes[p][w]
			c.planes[p][w] = t ^ carry
			carry &= t
		}
		if carry != 0 { // 255からの桁上がりが出たビット: 全プレーン1に戻して飽和
			for p := range numPlanes {
				c.planes[p][w] |= carry
			}
		}
	}
}

// decMasked: mで指定したリテラルのTA状態を-1(0で飽和)。
func (c *clause) decMasked(m *[numLitWords]uint64) {
	for w := range numLitWords {
		borrow := m[w]
		if borrow == 0 {
			continue
		}
		for p := range numPlanes {
			t := c.planes[p][w]
			c.planes[p][w] = t ^ borrow
			borrow = ^t & borrow
		}
		if borrow != 0 { // 0からの桁借りが出たビット: 全プレーン0に戻して飽和
			for p := range numPlanes {
				c.planes[p][w] &^= borrow
			}
		}
	}
}

func (c *clause) recount() {
	var n int
	for w := range numLitWords {
		n += bits.OnesCount64(c.planes[7][w])
	}
	c.includeCnt = int32(n)
}

// ---------------------------------------------------------------------------
// マルチクラスTM
// ---------------------------------------------------------------------------

type tm struct {
	clausesPerClass int
	T               int64 // 投票マージン
	sLog2           int   // 感度 s = 2^sLog2 (確率 1/s を乱数ワードのANDで生成)
	weighted        bool
	clauses         [numClasses][]clause
}

func newTM(clausesPerClass int, T int64, sLog2 int, weighted bool) *tm {
	m := &tm{clausesPerClass: clausesPerClass, T: T, sLog2: sLog2, weighted: weighted}
	for cls := range numClasses {
		m.clauses[cls] = make([]clause, clausesPerClass)
		for j := range m.clauses[cls] {
			m.clauses[cls][j].init()
		}
	}
	return m
}

// randMask1s は各ビットが確率 1/2^sLog2 で1になるマスクを生成(整数演算のみ)。
func randMask1s(r *rng, sLog2 int, out *[numLitWords]uint64) {
	for w := range numLitWords {
		v := r.next()
		for range sLog2 - 1 {
			v &= r.next()
		}
		out[w] = v & validMask[w]
	}
}

// classSumInfer は推論時の投票和(空節は発火しない)。
// 偶数節が正極性(+weight)、奇数節が負極性(-weight)。
func (m *tm) classSumInfer(cls int, lit *[numLitWords]uint64) int64 {
	var sum int64
	cs := m.clauses[cls]
	for j := range cs {
		c := &cs[j]
		if c.includeCnt == 0 || !c.fires(lit) {
			continue
		}
		w := int64(c.weight)
		if j&1 == 1 {
			sum -= w
		} else {
			sum += w
		}
	}
	return sum
}

func clamp(v, lo, hi int64) int64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// ---------------------------------------------------------------------------
// データ読み込み
// ---------------------------------------------------------------------------

type packedSample struct {
	lit    [numLitWords]uint64 // 真リテラル: [x(13語), ~x(13語)]
	notLit [numLitWords]uint64 // 偽リテラル: [~x, x]
	label  int
}

// oldMatrix / oldMatrices は、旧データセットgob(リリース v0.1.0-test)が作られた当時の
// bitsx.Matrix の公開フィールド形式(Rows/Cols/Data)に合わせたデコード専用の型。
// 現行の bitsx.Matrix は非公開フィールド+GobEncode/GobDecodeに変わったため、
// 旧gobは現行型ではデコードできない(詳細は REPORT.md §9)。
//
// このデコード専用の型・loadDataset は、当時 dataset.LoadMNIST 自体が上記のバグで
// 使えなかったために、それを迂回して生gobを直接読む目的で書かれたもの。
//
// 未修正(2026-08-14): バグは crow/dataset 側で解消済みで、dataset.LoadMNIST /
// LoadFashionMNIST は現行形式(リリース v0.2.0-test)を正しく読める(実機で確認済み)。
// この2関数のシグネチャは変わっていないため、本来は oldMatrix/loadDataset を丸ごと
// 削除し、dataset.LoadMNIST / dataset.LoadFashionMNIST の呼び出しに置き換えるべき
// (旧形式ファイルを別途用意する対応ではなく、こちらが正しい直し方)。
// packSamples が直接触っている img.Rows/img.Cols/img.Data[w] は現行 bitsx.Matrix では
// 非公開のため、公開API img.Rows()/img.Cols()/img.Word(w) に置き換える必要がある。
// この置き換えは未実施・未検証(README.md「データの入手」を参照)。
// 未対応のままなので、現行形式に置き換わった ~/.crow_dataset をそのまま指定すると
// loadGob[oldMatrices] のデコードに失敗する。動かす場合は旧形式ファイルを
// 別ディレクトリに用意し -datadir で指定すること。
type oldMatrix struct {
	Rows int
	Cols int
	Data []uint64
}

type oldMatrices []*oldMatrix

func loadGob[T any](path string) (T, error) {
	var zero T
	raw, err := os.ReadFile(path)
	if err != nil {
		return zero, err
	}
	var data T
	if err := gob.NewDecoder(bytes.NewReader(raw)).Decode(&data); err != nil {
		return zero, err
	}
	return data, nil
}

func loadDataset(dir, name string) (trainImgs, testImgs oldMatrices, trainLabels, testLabels []int, err error) {
	prefix := "mnist"
	if name == "fashion" {
		prefix = "fashion_mnist"
	}
	trainImgs, err = loadGob[oldMatrices](filepath.Join(dir, prefix+"_train_flat_binary_imgs.gob"))
	if err != nil {
		return
	}
	trainLabels, err = loadGob[[]int](filepath.Join(dir, prefix+"_train_int_labels.gob"))
	if err != nil {
		return
	}
	testImgs, err = loadGob[oldMatrices](filepath.Join(dir, prefix+"_test_flat_binary_imgs.gob"))
	if err != nil {
		return
	}
	testLabels, err = loadGob[[]int](filepath.Join(dir, prefix+"_test_int_labels.gob"))
	return
}

func packSamples(imgs oldMatrices, labels []int) ([]packedSample, error) {
	if len(imgs) != len(labels) {
		return nil, fmt.Errorf("長さが不一致: len(imgs) = %d, len(labels) = %d", len(imgs), len(labels))
	}
	out := make([]packedSample, len(imgs))
	for i, img := range imgs {
		if img.Rows*img.Cols != numFeatures {
			return nil, fmt.Errorf("想定外の形状: %dx%d", img.Rows, img.Cols)
		}
		if len(img.Data) != wordsPerFtr {
			return nil, fmt.Errorf("想定外のワード数: %d", len(img.Data))
		}
		for w := range wordsPerFtr {
			x := img.Data[w] & validMask[w]
			out[i].lit[w] = x
			out[i].lit[wordsPerFtr+w] = ^x & validMask[wordsPerFtr+w]
			out[i].notLit[w] = ^x & validMask[w]
			out[i].notLit[wordsPerFtr+w] = x
		}
		out[i].label = labels[i]
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// 学習・評価
// ---------------------------------------------------------------------------

// trainEpoch は1エポック学習する(サンプル順はrngでシャッフル)。
//
// 各サンプルについて、正解クラスと無作為に選んだ負例クラスの2クラスだけを更新する。
//   - フェーズ1: 2クラス分の節発火判定と投票和(節レンジ並列)
//   - フェーズ2: フィードバック(節レンジ並列。各節の更新は独立)
//
// ワーカーは節レンジを所有するため、TA状態への書き込みは競合しない。
// ワーカーRNGのシードはエポックの先頭でメインRNGから引くため、
// 同じシード・同じ設定なら結果は決定的に再現される。
func (m *tm) trainEpoch(samples []packedSample, r *rng, workers int) {
	n := len(samples)
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	// Fisher-Yates (整数のみ)
	for i := n - 1; i > 0; i-- {
		j := int(r.intn(uint64(i + 1)))
		order[i], order[j] = order[j], order[i]
	}

	type classTask struct {
		cls      int
		isTarget bool
		fires    []bool
		sum      int64
	}
	tasks := [2]classTask{}
	tasks[0].fires = make([]bool, m.clausesPerClass)
	tasks[1].fires = make([]bool, m.clausesPerClass)

	// ワーカーごとの節レンジ分割
	chunk := (m.clausesPerClass + workers - 1) / workers
	workerRngs := make([]*rng, workers)
	for i := range workerRngs {
		workerRngs[i] = newRng(r.next())
	}

	partialSums := make([][2]int64, workers)
	var wg sync.WaitGroup

	for _, idx := range order {
		s := &samples[idx]
		y := s.label
		q := int(r.intn(numClasses - 1))
		if q >= y {
			q++
		}
		tasks[0].cls, tasks[0].isTarget = y, true
		tasks[1].cls, tasks[1].isTarget = q, false

		// フェーズ1: 発火判定と投票和(節レンジ並列)
		// 学習時は空節(includeCnt==0)も発火扱いにする(Type Iでリテラル獲得を始めるため)。
		for wk := range workers {
			wg.Add(1)
			go func(wk int) {
				defer wg.Done()
				lo := wk * chunk
				hi := min(lo+chunk, m.clausesPerClass)
				for ti := range tasks {
					t := &tasks[ti]
					cs := m.clauses[t.cls]
					var sum int64
					for j := lo; j < hi; j++ {
						c := &cs[j]
						var f bool
						if c.includeCnt == 0 {
							f = true
						} else {
							f = c.fires(&s.lit)
						}
						t.fires[j] = f
						if !f {
							continue
						}
						w := int64(c.weight)
						if j&1 == 1 {
							sum -= w
						} else {
							sum += w
						}
					}
					partialSums[wk][ti] = sum
				}
			}(wk)
		}
		wg.Wait()

		for ti := range tasks {
			var sum int64
			for wk := range workers {
				sum += partialSums[wk][ti]
			}
			tasks[ti].sum = sum
		}

		// フェーズ2: フィードバック(節レンジ並列、各節は独立)
		for wk := range workers {
			wg.Add(1)
			go func(wk int) {
				defer wg.Done()
				lo := wk * chunk
				hi := min(lo+chunk, m.clausesPerClass)
				wr := workerRngs[wk]
				var rmask [numLitWords]uint64
				for ti := range tasks {
					t := &tasks[ti]
					sum := clamp(t.sum, -m.T, m.T)
					// フィードバック確率 = (T - sum)/2T (正解クラス)、(T + sum)/2T (負例クラス)。
					// 整数PRNGの剰余比較で判定する。
					var pNum uint64
					if t.isTarget {
						pNum = uint64(m.T - sum)
					} else {
						pNum = uint64(m.T + sum)
					}
					den := uint64(2 * m.T)
					cs := m.clauses[t.cls]
					for j := lo; j < hi; j++ {
						if wr.intn(den) >= pNum {
							continue
						}
						c := &cs[j]
						positive := j&1 == 0
						typeI := positive == t.isTarget
						if typeI {
							if t.fires[j] {
								// Type Ia: 真リテラルを強化(boost true positive)、
								// 偽リテラルを確率1/sで弱化
								c.incMasked(&s.lit)
								randMask1s(wr, m.sLog2, &rmask)
								for w := range rmask {
									rmask[w] &= s.notLit[w]
								}
								c.decMasked(&rmask)
								if m.weighted && c.weight < 1<<20 {
									c.weight++
								}
							} else {
								// Type Ib: 全リテラルを確率1/sで弱化
								randMask1s(wr, m.sLog2, &rmask)
								c.decMasked(&rmask)
							}
							c.recount()
						} else if t.fires[j] {
							// Type II: 発火した誤り節に、偽リテラルの包含を促す
							// (発火節では偽リテラルは必ず除外側なので、包含済みを避ける処理は不要)
							c.incMasked(&s.notLit)
							c.recount()
							if m.weighted && c.weight > 1 {
								c.weight--
							}
						}
					}
				}
			}(wk)
		}
		wg.Wait()
	}
}

// accuracy は推論精度。推論経路は整数演算のみ(比率の算出だけfloat)。
func (m *tm) accuracy(samples []packedSample, workers int) float64 {
	n := len(samples)
	correct := make([]int, workers)
	var wg sync.WaitGroup
	chunk := (n + workers - 1) / workers
	for wk := range workers {
		wg.Add(1)
		go func(wk int) {
			defer wg.Done()
			lo := wk * chunk
			hi := min(lo+chunk, n)
			for i := lo; i < hi; i++ {
				s := &samples[i]
				bestCls, bestSum := 0, int64(-1<<62)
				for cls := range numClasses {
					sum := m.classSumInfer(cls, &s.lit)
					if sum > bestSum {
						bestSum = sum
						bestCls = cls
					}
				}
				if bestCls == s.label {
					correct[wk]++
				}
			}
		}(wk)
	}
	wg.Wait()
	total := 0
	for _, c := range correct {
		total += c
	}
	return float64(total) / float64(n)
}

func defaultDataDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ".crow_dataset"
	}
	return filepath.Join(home, ".crow_dataset")
}

func main() {
	var (
		dsName   = flag.String("dataset", "mnist", "mnist | fashion")
		dataDir  = flag.String("datadir", defaultDataDir(), "二値化済みgobのディレクトリ")
		clausesN = flag.Int("clauses", 512, "クラスあたりの節数")
		tFlag    = flag.Int64("T", 0, "投票マージン (0 = 自動: 重み有り 4×節数)")
		sLog2    = flag.Int("slog2", 3, "感度 s = 2^slog2")
		epochs   = flag.Int("epochs", 10, "エポック数")
		weighted = flag.Bool("weighted", true, "整数節重みを使う")
		seed     = flag.Uint64("seed", 1, "乱数シード")
		trainSub = flag.Int("trainsub", 0, "学習サンプル数の上限 (0 = 全部)")
		threads  = flag.Int("threads", 0, "ワーカー数 (0 = NumCPU)")
	)
	flag.Parse()

	workers := *threads
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	if workers > 32 {
		workers = 32
	}

	T := *tFlag
	if T == 0 {
		if *weighted {
			T = int64(*clausesN) * 4
		} else {
			T = max(int64(*clausesN)/20, 10)
		}
	}

	trainImgs, testImgs, trainLabels, testLabels, err := loadDataset(*dataDir, *dsName)
	if err != nil {
		log.Fatal(err)
	}

	if *trainSub > 0 && *trainSub < len(trainImgs) {
		trainImgs = trainImgs[:*trainSub]
		trainLabels = trainLabels[:*trainSub]
	}

	train, err := packSamples(trainImgs, trainLabels)
	if err != nil {
		log.Fatal(err)
	}
	test, err := packSamples(testImgs, testLabels)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("dataset=%s train=%d test=%d clauses/class=%d T=%d s=2^%d weighted=%t workers=%d\n",
		*dsName, len(train), len(test), *clausesN, T, *sLog2, *weighted, workers)

	m := newTM(*clausesN, T, *sLog2, *weighted)
	r := newRng(*seed)

	best := 0.0
	for e := 1; e <= *epochs; e++ {
		t0 := time.Now()
		m.trainEpoch(train, r, workers)
		trainDur := time.Since(t0)
		t1 := time.Now()
		acc := m.accuracy(test, workers)
		best = max(best, acc)
		fmt.Printf("[%s] epoch %d: test acc %.4f (best %.4f) train %.1fs eval %.1fs\n",
			time.Now().Format("15:04:05"), e, acc, best, trainDur.Seconds(), time.Since(t1).Seconds())
	}
	fmt.Printf("BEST %.4f\n", best)
}
