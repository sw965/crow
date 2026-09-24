// model: 全層を学習する純粋な二値ニューラルネットの実験プログラム。
//
// 満たすべき制約(依頼):
//   - 第1層から出力層まで純粋なNN構造で、全層が学習される
//   - 3値重みは導入しない(可視重みは ±1 のみ)
//   - BEPの高速性の核を崩さない(XNOR/popcount + 整数演算のみ。浮動小数点ゼロ)
//   - 実用的な精度
//   - 万能近似定理を満たすモデル族であること
//
// 構造:
//
//	第1層: 局所受容野(p x p パッチ)を持つ二値ニューロン。
//	       パッチ内の各画素に ±1 の重み。見ない画素は「重み0」ではなく結線が無い
//	       (=畳み込み層と同じ。3値ではない)。
//	       ニューロンごとに学習可能な整数バイアス(しきい値)を持つ。
//	出力層: クラスごとの整数重み。二値活性に対する条件付き加算のみ(乗算不要)。
//
// 第1層の学習則は2種類あり、実験で優劣を測った(結論は ../REPORT.md §4)。
//
//	競合学習 (-vq N)   : 同一受容野位置のニューロン群のうち、そのパッチに最も一致する
//	                     1個(勝者)だけを入力へ近づける。**これが有効だった手法。**
//	誤差駆動 (既定)     : 出力重みを通した希望活性の逆射影 → ゲート → Hebbian外積。
//	                     BEPと同じ構造だが、**実験では一貫して精度を下げた**(対照として残す)。
//
// 最良の設定(../REPORT.md §3):
//
//	go run . -hidden 16384 -patch 9 -randinit -matchpct 70 -vq 3 -freeze1 -epochs 20
//	  → MNIST 97.97 ± 0.07% (3シード)
//	go run . -dataset fashion -hidden 16384 -patch 9 -randinit -matchpct 55 -vq 3 -freeze1 -epochs 20
//	  → Fashion-MNIST 87.96%
//
// -freeze1 は「誤差駆動の第1層更新を止める」フラグであり、-vq による競合学習は
// その前段で実行される。したがって上記設定でも第1層は学習されている。
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
	"github.com/sw965/omw/mathx/bitsx"
)

const (
	side        = 28
	numFeatures = side * side
	wordsPerImg = 13
	numClasses  = 10
)

var validMask [wordsPerImg]uint64

func init() {
	tail := uint64(1)<<(numFeatures%64) - 1
	for w := range validMask {
		if w == wordsPerImg-1 {
			validMask[w] = tail
		} else {
			validMask[w] = ^uint64(0)
		}
	}
}

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

type sample struct {
	x     [wordsPerImg]uint64
	label int
}

func packSamples(imgs bitsx.Matrices, labels []int) []sample {
	out := make([]sample, len(imgs))
	for i, img := range imgs {
		for w := range wordsPerImg {
			raw, err := img.Word(w)
			if err != nil {
				log.Fatal(err)
			}
			out[i].x[w] = raw & validMask[w]
		}
		out[i].label = labels[i]
	}
	return out
}

// ---------------------------------------------------------------------------
// 第1層: 局所受容野を持つ学習可能な二値ニューロン層
// ---------------------------------------------------------------------------

type convLayer struct {
	size, patch, area int

	r0, c0   []int32  // パッチ左上位置
	wlo, whi []int32  // このニューロンが触るワード範囲(順伝播の高速化)
	w, mask  []uint64 // size * wordsPerImg
	h        []int8   // size * area。整数隠れ重み(BEPのHと同じ役割)
	bias     []int32  // size。学習可能な整数しきい値
	biasMax  int32
	perPos   int // 同一受容野位置あたりのニューロン数(競合学習の単位)
}

func newConvLayer(size, patch, matchPct, hInit int, train []sample, randInit bool, r *rng) *convLayer {
	area := patch * patch
	l := &convLayer{
		size: size, patch: patch, area: area,
		r0: make([]int32, size), c0: make([]int32, size),
		wlo: make([]int32, size), whi: make([]int32, size),
		w:    make([]uint64, size*wordsPerImg),
		mask: make([]uint64, size*wordsPerImg),
		h:    make([]int8, size*area),
		bias: make([]int32, size),
		// バイアスは活性前値の範囲を超えない範囲で動かす(オーバーフロー防止)
		biasMax: int32(area),
	}

	thr := (area*matchPct + 99) / 100
	initBias := int32(area - 2*thr)

	// 競合学習のため、ニューロンを受容野位置ごとのグループに分ける。
	// 同じ位置に perPos 個のニューロンを置き、その中で勝者だけが更新される。
	nPos := side - patch + 1
	numPos := nPos * nPos
	l.perPos = max(1, size/numPos)
	for i := range size {
		posIdx := (i / l.perPos) % numPos
		r0 := posIdx / nPos
		c0 := posIdx % nPos
		l.r0[i], l.c0[i] = int32(r0), int32(c0)
		var ref *sample
		var randWords [wordsPerImg]uint64
		if randInit {
			// 参照サンプルを使わず、重みを無作為に初期化する
			for w := range wordsPerImg {
				randWords[w] = r.next()
			}
			ref = &sample{x: randWords}
		} else {
			ref = &train[r.intn(uint64(len(train)))]
		}

		base := i * wordsPerImg
		lo, hi := wordsPerImg-1, 0
		for dr := range patch {
			for dc := range patch {
				idx := (r0+dr)*side + (c0 + dc)
				w, b := idx/64, uint(idx%64)
				l.mask[base+w] |= 1 << b
				bit := ref.x[w] >> b & 1
				if bit == 1 {
					l.w[base+w] |= 1 << b
					l.h[i*area+dr*patch+dc] = int8(hInit)
				} else {
					l.h[i*area+dr*patch+dc] = int8(-hInit)
				}
				lo = min(lo, w)
				hi = max(hi, w)
			}
		}
		l.wlo[i], l.whi[i] = int32(lo), int32(hi)
		l.bias[i] = initBias
	}
	return l
}

// forward は活性前値 z と活性ビットセットを求める。
// z_i = (一致数 - 不一致数) + bias_i = area - 2*不一致数 + bias_i
func (l *convLayer) forward(s *sample, z []int32, act []uint64) {
	clear(act)
	for i := range l.size {
		base := i * wordsPerImg
		mismatch := 0
		for w := int(l.wlo[i]); w <= int(l.whi[i]); w++ {
			mismatch += bits.OnesCount64((s.x[w] ^ l.w[base+w]) & l.mask[base+w])
		}
		zi := int32(l.area-2*mismatch) + l.bias[i]
		z[i] = zi
		if zi >= 0 {
			act[i/64] |= 1 << uint(i%64)
		}
	}
}

// update は、ニューロン i を希望活性 d(+1 なら発火させたい、-1 なら消したい)へ
// 近づける。BEP Eq.8 と同じ Hebbian 外積(入力の±1表現 × 希望活性)。
func (l *convLayer) update(i int, d int8, s *sample, updateBias bool) {
	base := i * wordsPerImg
	hBase := i * l.area
	r0, c0 := int(l.r0[i]), int(l.c0[i])
	var flip [wordsPerImg]uint64
	touched := false

	for dr := range l.patch {
		rowBase := (r0+dr)*side + c0
		for dc := range l.patch {
			idx := rowBase + dc
			w, b := idx/64, uint(idx%64)
			xBit := int8(s.x[w] >> b & 1)
			// 入力の±1表現: 1 -> +1, 0 -> -1
			xPM := 2*xBit - 1
			j := hBase + dr*l.patch + dc

			old := l.h[j]
			nv := int32(old) + int32(d*xPM)
			nv = max(-128, min(nv, 127))
			l.h[j] = int8(nv)

			if (old >= 0) != (nv >= 0) {
				flip[w] |= 1 << b
				touched = true
			}
		}
	}
	if touched {
		for w := int(l.wlo[i]); w <= int(l.whi[i]); w++ {
			l.w[base+w] ^= flip[w]
		}
	}
	// しきい値も同方向へ動かす(発火させたいなら閾値を下げる=biasを上げる)
	if updateBias {
		l.bias[i] = max(-l.biasMax, min(l.bias[i]+int32(d), l.biasMax))
	}
}

// competeEpoch は、第1層を競合学習(ベクトル量子化)で1エポック学習する。
//
// 同じ受容野位置のニューロン群のうち、その位置のパッチに最も一致する1個(勝者)だけを
// 入力へ近づける。勝者のみ更新するため、テンプレートが平均化で潰れず、
// 位置ごとに異なるパターンへ特化していく(古典的な競合学習/自己組織化と同じ)。
// 全て整数・ビット演算(一致数の比較と ±1 のカウンタ更新)。
func (l *convLayer) competeEpoch(samples []sample, order []int, r *rng) {
	nPos := side - l.patch + 1
	numPos := nPos * nPos
	for _, idx := range order {
		s := &samples[idx]
		for pos := range numPos {
			start := pos * l.perPos
			end := min(start+l.perPos, l.size)
			if start >= l.size {
				break
			}
			bestIdx, bestMatch := -1, -1
			for i := start; i < end; i++ {
				base := i * wordsPerImg
				mismatch := 0
				for w := int(l.wlo[i]); w <= int(l.whi[i]); w++ {
					mismatch += bits.OnesCount64((s.x[w] ^ l.w[base+w]) & l.mask[base+w])
				}
				match := l.area - mismatch
				if match > bestMatch {
					bestMatch, bestIdx = match, i
				}
			}
			if bestIdx >= 0 {
				l.update(bestIdx, 1, s, false)
			}
		}
	}
}

// ---------------------------------------------------------------------------
// 出力層: 整数重み + マージン付き平均化パーセプトロン
// ---------------------------------------------------------------------------

type outputLayer struct {
	size         int
	w            []int32
	u            []int64
	avgW         []int32
	count        int64
	margin, clip int32
	useAverage   bool
}

func newOutputLayer(size int, margin int32, useAverage bool) *outputLayer {
	o := &outputLayer{size: size, w: make([]int32, numClasses*(size+1)),
		margin: margin, clip: 1 << 20, useAverage: useAverage}
	if useAverage {
		o.u = make([]int64, numClasses*(size+1))
		o.avgW = make([]int32, numClasses*(size+1))
		o.count = 1
	}
	return o
}

func (o *outputLayer) logitsWith(w []int32, act []uint64, out *[numClasses]int64) {
	stride := o.size + 1
	for c := range numClasses {
		out[c] = int64(w[c*stride+o.size])
	}
	for wi, word := range act {
		for word != 0 {
			b := bits.TrailingZeros64(word)
			word &= word - 1
			i := wi*64 + b
			for c := range numClasses {
				out[c] += int64(w[c*stride+i])
			}
		}
	}
}

func (o *outputLayer) materializeAverage() {
	if !o.useAverage {
		return
	}
	for i, w := range o.w {
		o.avgW[i] = w - int32(o.u[i]/o.count)
	}
}

func (o *outputLayer) evalWeights() []int32 {
	if o.useAverage {
		return o.avgW
	}
	return o.w
}

// train は出力層を更新し、(更新したか, 正解クラス, 最良の誤クラス) を返す。
func (o *outputLayer) train(act []uint64, label int) (bool, int) {
	var lg [numClasses]int64
	o.logitsWith(o.w, act, &lg)
	best, bestVal := -1, int64(-1<<62)
	for c := range numClasses {
		if c != label && lg[c] > bestVal {
			bestVal = lg[c]
			best = c
		}
	}
	if lg[label]-bestVal >= int64(o.margin) {
		if o.useAverage {
			o.count++
		}
		return false, best
	}

	stride := o.size + 1
	yBase, pBase := label*stride, best*stride
	for wi, word := range act {
		for word != 0 {
			b := bits.TrailingZeros64(word)
			word &= word - 1
			i := wi*64 + b
			if v := o.w[yBase+i]; v < o.clip {
				o.w[yBase+i] = v + 1
			}
			if v := o.w[pBase+i]; v > -o.clip {
				o.w[pBase+i] = v - 1
			}
			if o.useAverage {
				o.u[yBase+i] += o.count
				o.u[pBase+i] -= o.count
			}
		}
	}
	o.w[yBase+o.size]++
	o.w[pBase+o.size]--
	if o.useAverage {
		o.u[yBase+o.size] += o.count
		o.u[pBase+o.size] -= o.count
		o.count++
	}
	return true, best
}

func (o *outputLayer) predict(act []uint64) int {
	var lg [numClasses]int64
	o.logitsWith(o.evalWeights(), act, &lg)
	best, bestVal := 0, int64(-1<<62)
	for c := range numClasses {
		if lg[c] > bestVal {
			bestVal = lg[c]
			best = c
		}
	}
	return best
}

// ---------------------------------------------------------------------------

type model struct {
	l1 *convLayer
	o  *outputLayer

	gate       int32 // |z| <= gate のニューロンだけ第1層を更新(BEPのゲート)
	lrLog2     int   // 更新確率 1/2^lrLog2
	impThr     int32 // 希望活性の強さ(出力重み差)の下限
	groupSize  int   // グループ内で1個だけ更新(BEPのマスク/TMのgroupと同じ多様性保持)
	updateBias bool
	warmup     int // 出力層だけを学習するエポック数
}

func (m *model) accuracy(samples []sample, workers int) float64 {
	correct := make([]int, workers)
	var wg sync.WaitGroup
	chunk := (len(samples) + workers - 1) / workers
	words := (m.l1.size + 63) / 64
	for wk := range workers {
		wg.Add(1)
		go func(wk int) {
			defer wg.Done()
			lo := wk * chunk
			hi := min(lo+chunk, len(samples))
			z := make([]int32, m.l1.size)
			act := make([]uint64, words)
			for i := lo; i < hi; i++ {
				m.l1.forward(&samples[i], z, act)
				if m.o.predict(act) == samples[i].label {
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
	return float64(total) / float64(len(samples))
}

// trainEpoch は1エポック学習する。
// 第1層の学習を止めたい場合は freezeL1 = true(対照実験用)。
func (m *model) trainEpoch(samples []sample, r *rng, freezeL1 bool) {
	n := len(samples)
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	for i := n - 1; i > 0; i-- {
		j := int(r.intn(uint64(i + 1)))
		order[i], order[j] = order[j], order[i]
	}

	words := (m.l1.size + 63) / 64
	z := make([]int32, m.l1.size)
	act := make([]uint64, words)
	stride := m.o.size + 1

	for _, idx := range order {
		s := &samples[idx]
		m.l1.forward(s, z, act)
		updated, wrong := m.o.train(act, s.label)
		if !updated || freezeL1 || wrong < 0 {
			continue
		}

		// 希望活性の逆射影: 出力重みの差が、そのニューロンを立てるべきか寝かせるべきかを決める。
		// これはBEPの a* = sign(W^T (g ⊙ a*)) と同じ構造で、
		// 固定プロトタイプの代わりに学習中の出力重みを使う。
		//
		// グループ内で「最も直しやすい(|z|が最小の)1個」だけを更新する。
		// これはBEP Eq.9のマスク・TMのgroup機構と同じで、特徴の多様性が
		// 平均化で潰れるのを防ぐ。
		yBase, pBase := s.label*stride, wrong*stride
		for gStart := 0; gStart < m.l1.size; gStart += m.groupSize {
			gEnd := min(gStart+m.groupSize, m.l1.size)
			bestIdx, bestAbsZ := -1, int32(1<<30)
			var bestWant int8
			for i := gStart; i < gEnd; i++ {
				diff := m.o.w[yBase+i] - m.o.w[pBase+i]
				if diff > -m.impThr && diff < m.impThr {
					continue // 出力にほとんど効かないニューロンは触らない
				}
				zi := z[i]
				if zi < 0 {
					zi = -zi
				}
				if zi > m.gate {
					continue // ゲート: 境界付近(反転させやすい)のみ
				}
				cur := int8(-1)
				if act[i/64]>>uint(i%64)&1 == 1 {
					cur = 1
				}
				want := int8(-1)
				if diff > 0 {
					want = 1
				}
				if cur == want {
					continue
				}
				if zi < bestAbsZ {
					bestAbsZ, bestIdx, bestWant = zi, i, want
				}
			}
			if bestIdx < 0 {
				continue
			}
			// 確率的な学習率(整数PRNGの上位ビット比較)
			if r.next()>>(64-uint(m.lrLog2)) != 0 {
				continue
			}
			m.l1.update(bestIdx, bestWant, s, m.updateBias)
		}
	}
}

func main() {
	var (
		dsName   = flag.String("dataset", "mnist", "mnist | fashion")
		hSize    = flag.Int("hidden", 8192, "第1層の幅")
		patch    = flag.Int("patch", 9, "受容野の辺長")
		matchPct = flag.Int("matchpct", 85, "初期しきい値に対応する一致率(%)")
		epochs   = flag.Int("epochs", 20, "エポック数")
		margin   = flag.Int("margin", 25, "出力層のマージン")
		gate     = flag.Int("gate", 8, "第1層のゲート閾値 |z| <= gate")
		lrLog2   = flag.Int("lrlog2", 3, "第1層の更新確率 1/2^lrlog2")
		impThr   = flag.Int("impthr", 2, "希望活性の重要度しきい値")
		groupSz  = flag.Int("group", 64, "第1層のグループ幅(この中で1個だけ更新)")
		hInit    = flag.Int("hinit", 4, "隠れ重みHの初期絶対値(シナプス慣性)")
		noBias   = flag.Bool("nobias", false, "第1層のバイアスを更新しない")
		warmup   = flag.Int("warmup", 0, "出力層だけを学習するエポック数")
		vqEpochs = flag.Int("vq", 0, "第1層の競合学習(教師なし)のエポック数")
		randInit = flag.Bool("randinit", false, "第1層を学習データのパッチではなく無作為に初期化")
		freezeL1 = flag.Bool("freeze1", false, "第1層を学習しない(対照実験)")
		noAvg    = flag.Bool("noavg", false, "平均化パーセプトロンを使わない")
		seed     = flag.Uint64("seed", 1, "乱数シード")
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
	train := packSamples(ds.TrainInputs, ds.TrainLabels)
	test := packSamples(ds.TestInputs, ds.TestLabels)

	r := newRng(*seed)
	m := &model{
		l1:         newConvLayer(*hSize, *patch, *matchPct, *hInit, train, *randInit, r),
		o:          newOutputLayer(*hSize, int32(*margin), !*noAvg),
		gate:       int32(*gate),
		lrLog2:     *lrLog2,
		impThr:     int32(*impThr),
		groupSize:  *groupSz,
		updateBias: !*noBias,
		warmup:     *warmup,
	}

	fmt.Printf("dataset=%s hidden=%d patch=%dx%d matchpct=%d margin=%d gate=%d lrlog2=%d impthr=%d group=%d hinit=%d nobias=%t warmup=%d vq=%d randinit=%t freeze1=%t avg=%t\n",
		*dsName, *hSize, *patch, *patch, *matchPct, *margin, *gate, *lrLog2, *impThr,
		*groupSz, *hInit, *noBias, *warmup, *vqEpochs, *randInit, *freezeL1, !*noAvg)

	// 第1層の競合学習(教師なし事前学習)
	if *vqEpochs > 0 {
		order := make([]int, len(train))
		for i := range order {
			order[i] = i
		}
		for e := 1; e <= *vqEpochs; e++ {
			t0 := time.Now()
			for i := len(order) - 1; i > 0; i-- {
				j := int(r.intn(uint64(i + 1)))
				order[i], order[j] = order[j], order[i]
			}
			m.l1.competeEpoch(train, order, r)
			fmt.Printf("[競合学習] epoch %d 完了 %.1fs\n", e, time.Since(t0).Seconds())
		}
	}

	best := 0.0
	for e := 1; e <= *epochs; e++ {
		t0 := time.Now()
		m.trainEpoch(train, r, *freezeL1 || e <= m.warmup)
		m.o.materializeAverage()
		acc := m.accuracy(test, workers)
		best = max(best, acc)
		fmt.Printf("epoch %d: test acc %.4f (best %.4f) %.1fs\n", e, acc, best, time.Since(t0).Seconds())
	}
	fmt.Printf("BEST %.4f\n", best)
}
