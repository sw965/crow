// bepreadout: 既存の crow/model/mlp/binary (BEP) の読み出しだけを差し替えると
// どこまで精度が上がるかを測る診断実験。結果は ../REPORT.md §2.3 を参照。
//
// 手順:
//  1. BEPを通常どおり学習(バックボーンは固定ランダムプロトタイプの誤差で学習)
//  2. その時点の Accuracy(プロトタイプ読み出し) を基準として記録
//  3. 同じバックボーンの出力活性を取り出し、整数重みの線形読み出しを
//     マージン付きパーセプトロン則(整数のみ)で学習し、精度を比較
//
// ライブラリコード(crow/omw)は一切変更していない。公開APIのみ使用。
package main

import (
	"flag"
	"fmt"
	"log"
	"math/bits"
	"runtime"
	"sync"
	"time"

	"github.com/sw965/crow/dataset"
	"github.com/sw965/crow/model/mlp/binary"
	"github.com/sw965/omw/mathx/bitsx"
	"github.com/sw965/omw/mathx/randx"
)

const numClasses = 10

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

func (r *rng) intn(n uint64) uint64 { return r.next() % n }

// ---------------------------------------------------------------------------
// 活性キャッシュ: バックボーン出力をビットセットとして保持
// ---------------------------------------------------------------------------

type actCache struct {
	size   int
	words  int
	data   []uint64
	labels []int
}

func buildActCache(m *binary.Model, xs bitsx.Matrices, labels []int, workers int) (*actCache, error) {
	y0, err := m.Backbone.Predict(xs[0])
	if err != nil {
		return nil, err
	}
	size := y0.Rows() * y0.Cols()
	words := (size + 63) / 64

	c := &actCache{size: size, words: words, data: make([]uint64, len(xs)*words), labels: labels}
	var wg sync.WaitGroup
	var firstErr error
	var mu sync.Mutex
	chunk := (len(xs) + workers - 1) / workers
	for wk := range workers {
		wg.Add(1)
		go func(wk int) {
			defer wg.Done()
			lo := wk * chunk
			hi := min(lo+chunk, len(xs))
			for i := lo; i < hi; i++ {
				y, err := m.Backbone.Predict(xs[i])
				if err != nil {
					mu.Lock()
					if firstErr == nil {
						firstErr = err
					}
					mu.Unlock()
					return
				}
				dst := c.data[i*words : (i+1)*words]
				// bitsx.Matrix の行を、通し番号のビットセットへ詰め直す
				idx := 0
				for row := range y.Rows() {
					for col := range y.Cols() {
						b, err := y.Bit(row, col)
						if err != nil {
							mu.Lock()
							if firstErr == nil {
								firstErr = err
							}
							mu.Unlock()
							return
						}
						if b == 1 {
							dst[idx/64] |= 1 << uint(idx%64)
						}
						idx++
					}
				}
			}
		}(wk)
	}
	wg.Wait()
	return c, firstErr
}

func (c *actCache) get(i int) []uint64 { return c.data[i*c.words : (i+1)*c.words] }

// ---------------------------------------------------------------------------
// 整数線形読み出し(マージン付きパーセプトロン)
// ---------------------------------------------------------------------------

type linearReadout struct {
	size   int
	w      []int32
	margin int32
	clip   int32
}

func newLinearReadout(size int, margin int32) *linearReadout {
	return &linearReadout{size: size, w: make([]int32, numClasses*(size+1)), margin: margin, clip: 1 << 20}
}

func (l *linearReadout) logits(act []uint64, out *[numClasses]int64) {
	stride := l.size + 1
	for c := range numClasses {
		out[c] = int64(l.w[c*stride+l.size])
	}
	for wi, word := range act {
		for word != 0 {
			b := bits.TrailingZeros64(word)
			word &= word - 1
			i := wi*64 + b
			for c := range numClasses {
				out[c] += int64(l.w[c*stride+i])
			}
		}
	}
}

func (l *linearReadout) update(act []uint64, label int) bool {
	var lg [numClasses]int64
	l.logits(act, &lg)
	best, bestVal := -1, int64(-1<<62)
	for c := range numClasses {
		if c != label && lg[c] > bestVal {
			bestVal = lg[c]
			best = c
		}
	}
	if lg[label]-bestVal >= int64(l.margin) {
		return false
	}
	stride := l.size + 1
	yBase, pBase := label*stride, best*stride
	for wi, word := range act {
		for word != 0 {
			b := bits.TrailingZeros64(word)
			word &= word - 1
			i := wi*64 + b
			if v := l.w[yBase+i]; v < l.clip {
				l.w[yBase+i] = v + 1
			}
			if v := l.w[pBase+i]; v > -l.clip {
				l.w[pBase+i] = v - 1
			}
		}
	}
	l.w[yBase+l.size]++
	l.w[pBase+l.size]--
	return true
}

func (l *linearReadout) accuracy(c *actCache, workers int) float64 {
	correct := make([]int, workers)
	var wg sync.WaitGroup
	n := len(c.labels)
	chunk := (n + workers - 1) / workers
	for wk := range workers {
		wg.Add(1)
		go func(wk int) {
			defer wg.Done()
			lo := wk * chunk
			hi := min(lo+chunk, n)
			var lg [numClasses]int64
			for i := lo; i < hi; i++ {
				l.logits(c.get(i), &lg)
				best, bestVal := 0, int64(-1<<62)
				for cl := range numClasses {
					if lg[cl] > bestVal {
						bestVal = lg[cl]
						best = cl
					}
				}
				if best == c.labels[i] {
					correct[wk]++
				}
			}
		}(wk)
	}
	wg.Wait()
	total := 0
	for _, v := range correct {
		total += v
	}
	return float64(total) / float64(n)
}

func main() {
	var (
		dsName   = flag.String("dataset", "mnist", "mnist | fashion")
		h1       = flag.Int("h1", 512, "隠れ層1の幅")
		h2       = flag.Int("h2", 1024, "隠れ層2(出力)の幅")
		bepEpoch = flag.Int("bepepochs", 15, "BEPの学習エポック数")
		roEpoch  = flag.Int("roepochs", 25, "線形読み出しの学習エポック数")
		margin   = flag.Int("margin", 25, "線形読み出しのマージン")
		seed     = flag.Uint64("seed", 1, "線形読み出し側の乱数シード")
		threads  = flag.Int("threads", 0, "ワーカー数 (0 = NumCPU)")
	)
	flag.Parse()

	workers := *threads
	if workers <= 0 {
		workers = runtime.NumCPU()
	}

	var ds dataset.Binary[int]
	var err error
	if *dsName == "fashion" {
		ds, err = dataset.LoadFashionMNIST(nil)
	} else {
		ds, err = dataset.LoadMNIST(nil)
	}
	if err != nil {
		log.Fatal(err)
	}

	modelRng := randx.NewPCG()
	model := binary.Model{XRows: 1, XCols: 784}
	// h1 = 0 なら単層構成(784 -> h2)。中間のボトルネックの影響を切り分けるため。
	if *h1 > 0 {
		if err := model.AppendDenseLayer(*h1, modelRng); err != nil {
			log.Fatal(err)
		}
	}
	if err := model.AppendDenseLayer(*h2, modelRng); err != nil {
		log.Fatal(err)
	}
	if err := model.SetClassPrototypes(numClasses, modelRng); err != nil {
		log.Fatal(err)
	}
	shared := binary.NewSharedHyperparameters()
	if err := model.Backbone.SetSharedHyperparameters(&shared); err != nil {
		log.Fatal(err)
	}

	trainer := binary.NewTrainer(model, workers)
	trainer.MiniBatchSize = 1024
	trainer.Margin = 0.5

	fmt.Printf("dataset=%s BEP構成 784->%d->%d BEPエポック=%d 読み出しエポック=%d margin=%d\n",
		*dsName, *h1, *h2, *bepEpoch, *roEpoch, *margin)

	// --- 1. BEPを通常どおり学習 ---
	bestProto := 0.0
	for e := 1; e <= *bepEpoch; e++ {
		t0 := time.Now()
		if err := trainer.Train(ds.TrainInputs, ds.TrainLabels); err != nil {
			log.Fatal(err)
		}
		acc, err := model.Accuracy(ds.TestInputs, ds.TestLabels, workers)
		if err != nil {
			log.Fatal(err)
		}
		bestProto = max(bestProto, float64(acc))
		fmt.Printf("[BEP] epoch %d: proto読み出し test acc %.4f (best %.4f) %.1fs\n",
			e, acc, bestProto, time.Since(t0).Seconds())
	}

	// --- 2. 同じバックボーンの活性を取り出す ---
	t0 := time.Now()
	trainAct, err := buildActCache(&model, ds.TrainInputs, ds.TrainLabels, workers)
	if err != nil {
		log.Fatal(err)
	}
	testAct, err := buildActCache(&model, ds.TestInputs, ds.TestLabels, workers)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("活性の抽出: 次元=%d %.1fs\n", trainAct.size, time.Since(t0).Seconds())

	// --- 3. 線形読み出しを整数のみで学習 ---
	l := newLinearReadout(trainAct.size, int32(*margin))
	r := newRng(*seed)
	n := len(trainAct.labels)
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}

	bestLinear := 0.0
	for e := 1; e <= *roEpoch; e++ {
		for i := n - 1; i > 0; i-- {
			j := int(r.intn(uint64(i + 1)))
			order[i], order[j] = order[j], order[i]
		}
		for _, idx := range order {
			l.update(trainAct.get(idx), trainAct.labels[idx])
		}
		acc := l.accuracy(testAct, workers)
		bestLinear = max(bestLinear, acc)
		fmt.Printf("[線形読み出し] epoch %d: test acc %.4f (best %.4f)\n", e, acc, bestLinear)
	}

	fmt.Printf("\n=== 結果 ===\nBEP(プロトタイプ読み出し): %.4f\nBEP backbone + 整数線形読み出し: %.4f (差 %+.4f)\n",
		bestProto, bestLinear, bestLinear-bestProto)
}
