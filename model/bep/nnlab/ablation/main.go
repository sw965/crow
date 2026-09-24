// ablation: 「どの特徴の作り方」と「どの読み出し」が精度を決めるのかを切り分ける対照実験。
//
// 隠れ層を固定(学習しない)にして読み出しだけを揃えることで、
// 特徴の種類そのものの寄与を比較できるようにしている。
// 結論と数値は ../REPORT.md §2 を参照。全経路が整数・ビット演算のみ。
//
// 隠れ層の種類 (-type):
//
//	dense     : 全784入力に対するランダム二値重み、しきい値0(BEPのDense層と同じ形)
//	densebias : 上記 + ニューロンごとのランダム整数バイアス(しきい値を散らす)
//	and       : ランダムなk個のリテラルのAND(TMの節と同型。3値相当の「無視」を含む)
//	conv      : 局所受容野(p x p)の二値ニューロン + 整数バイアス(3値なし)
//
// 読み出し (-readout):
//
//	proto  : 固定ランダム±1プロトタイプとの一致数(BEPの出力と同じ形)
//	linear : 整数重みをマージン付きパーセプトロン則で学習(-avg で平均化)
//
// 隠れ層が固定なので活性を一度だけ計算してビットセットで持ち、
// 読み出しの学習を高速に回す設計になっている。
package main

import (
	"cmp"
	"flag"
	"fmt"
	"log"
	"math/bits"
	"runtime"
	"slices"
	"sync"
	"time"

	"github.com/sw965/crow/dataset"
	"github.com/sw965/omw/mathx/bitsx"
)

const (
	numFeatures = 784
	wordsPerImg = 13 // ceil(784/64)
	numLitWords = 2 * wordsPerImg
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

func (r *rng) intn(n uint64) uint64 { return r.next() % n }

// ---------------------------------------------------------------------------
// データ
// ---------------------------------------------------------------------------

type sample struct {
	x     [wordsPerImg]uint64 // 画素ビット
	lit   [numLitWords]uint64 // [x, ~x] (AND型で使う)
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
			v := raw & validMask[w]
			out[i].x[w] = v
			out[i].lit[w] = v
			out[i].lit[wordsPerImg+w] = ^v & validMask[w]
		}
		out[i].label = labels[i]
	}
	return out
}

// ---------------------------------------------------------------------------
// 隠れ層
// ---------------------------------------------------------------------------

type hiddenLayer struct {
	kind string
	size int

	// dense / densebias
	w    []uint64 // size * wordsPerImg
	bias []int32  // size

	// conv
	patch int
	mask  []uint64 // size * wordsPerImg。結線されている画素のマスク

	// and
	k         int
	entryWord []uint8  // size * k
	entryMask []uint64 // size * k
}

// newDenseHidden は、ランダム二値重み + 整数バイアスの隠れ層を作る。
// biasSigma > 0 なら、バイアスを [-biasSigma*sigma, 0] の一様整数から引く
// (sigma = sqrt(784) = 28 は、ランダム重み下での活性前値の標準偏差)。
func newDenseHidden(size int, biasSigmaX10 int, r *rng) *hiddenLayer {
	h := &hiddenLayer{kind: "dense", size: size}
	h.w = make([]uint64, size*wordsPerImg)
	h.bias = make([]int32, size)
	for i := range size {
		for w := range wordsPerImg {
			h.w[i*wordsPerImg+w] = r.next() & validMask[w]
		}
		if biasSigmaX10 > 0 {
			h.kind = "densebias"
			// sigma = 28。biasSigmaX10 は sigma の 1/10 単位
			span := uint64(28*biasSigmaX10/10) + 1
			h.bias[i] = -int32(r.intn(span))
		}
	}
	return h
}

// newDenseMatchHidden は、学習集合から選んだ全入力パターンを二値重みにし、
// 指定した一致率をしきい値にする全結合層を作る。局所受容野や重み0は使わず、
// 全ニューロンが784入力すべてに±1で結線される。
func newDenseMatchHidden(size, matchPct int, train []sample, r *rng) *hiddenLayer {
	h := &hiddenLayer{kind: "densematch", size: size}
	h.w = make([]uint64, size*wordsPerImg)
	h.bias = make([]int32, size)
	threshold := (numFeatures*matchPct + 99) / 100
	bias := int32(numFeatures - 2*threshold)
	for i := range size {
		ref := &train[r.intn(uint64(len(train)))]
		copy(h.w[i*wordsPerImg:(i+1)*wordsPerImg], ref.x[:])
		h.bias[i] = bias
	}
	return h
}

// newAndHidden は、ランダムなk個のリテラルのANDを特徴とする隠れ層を作る。
// 各節は、無作為に選んだ学習サンプルの、無作為なk画素の値を参照パターンとする
// (完全に無作為なパターンだと発火しない節ばかりになるため)。
func newAndHidden(size, k int, train []sample, r *rng) *hiddenLayer {
	h := &hiddenLayer{kind: "and", size: size, k: k}
	h.entryWord = make([]uint8, size*k)
	h.entryMask = make([]uint64, size*k)

	for i := range size {
		ref := &train[r.intn(uint64(len(train)))]
		// k個の異なる画素を選び、参照サンプルの値に応じて正/否定リテラルを含める
		var words [numLitWords]uint64
		for range k {
			p := int(r.intn(numFeatures))
			w, b := p/64, uint(p%64)
			if ref.x[w]>>b&1 == 1 {
				words[w] |= 1 << b
			} else {
				words[wordsPerImg+w] |= 1 << b
			}
		}
		// 非ゼロのワードだけを最大k個のエントリへ詰める
		e := 0
		for w := range numLitWords {
			if words[w] == 0 || e >= k {
				continue
			}
			h.entryWord[i*k+e] = uint8(w)
			h.entryMask[i*k+e] = words[w]
			e++
		}
		// 残りは mask=0 (常に真) の無害なエントリ
	}
	return h
}

// newConvHidden は、局所受容野(畳み込み型)の二値ニューロンからなる隠れ層を作る。
//
// 各ニューロンは 28x28 画像上の p x p パッチだけに結線され、そのパッチ内の各画素に
// ±1 の二値重みを持つ(3値ではない。見ない画素は重み0ではなく「結線が無い」)。
// 活性前値は z = 2*一致数 - p^2 + bias で、bias は整数のしきい値。
//
// 重みは、無作為に選んだ学習サンプルの同じ位置のパッチから取る(意味のある
// 部分パターン検出器にするため)。しきい値は matchPct で指定した一致率に対応する整数。
func newConvHidden(size, p, matchPct int, train []sample, r *rng) *hiddenLayer {
	h := &hiddenLayer{kind: "conv", size: size, patch: p}
	h.w = make([]uint64, size*wordsPerImg)
	h.mask = make([]uint64, size*wordsPerImg)
	h.bias = make([]int32, size)

	const side = 28
	area := p * p
	// 一致数 >= area*matchPct/100 で発火 ⇔ z = 2*一致数 - area + bias >= 0
	// ⇔ bias = area - 2*ceil(area*matchPct/100)
	thr := (area*matchPct + 99) / 100
	bias := int32(area - 2*thr)

	for i := range size {
		r0 := int(r.intn(uint64(side - p + 1)))
		c0 := int(r.intn(uint64(side - p + 1)))
		ref := &train[r.intn(uint64(len(train)))]

		base := i * wordsPerImg
		for dr := range p {
			for dc := range p {
				idx := (r0+dr)*side + (c0 + dc)
				w, b := idx/64, uint(idx%64)
				h.mask[base+w] |= 1 << b
				if ref.x[w]>>b&1 == 1 {
					h.w[base+w] |= 1 << b
				}
			}
		}
		h.bias[i] = bias
	}
	return h
}

// selectDiscriminative は、over倍の候補節を生成し、クラス識別力の高い順に size個だけ残す。
// 判定は全て整数演算(クラス別発火数のカウントとスコアの整数比較)。
//
// スコア: 節jについてクラス別発火数 n_c、総発火数 n とし、
//
//	score_j = max_c (n_c * numClasses) - n
//
// これは「あるクラスが、一様な期待値(n/numClasses)よりどれだけ多く発火したか」の
// 整数版で、正なら偏りがある。発火が極端に稀/多い節は識別に使いにくいので除外する。
func selectDiscriminative(size, k, over int, train []sample, r *rng, workers int) *hiddenLayer {
	cand := newAndHidden(size*over, k, train, r)

	// スコアリング用の部分集合(全件使う必要はない)
	n := min(len(train), 10000)
	counts := make([]int32, cand.size*numClasses)
	var wg sync.WaitGroup
	chunk := (cand.size + workers - 1) / workers
	for wk := range workers {
		wg.Add(1)
		go func(wk int) {
			defer wg.Done()
			lo := wk * chunk
			hi := min(lo+chunk, cand.size)
			for i := lo; i < hi; i++ {
				base := i * k
				for si := range n {
					s := &train[si]
					fired := true
					for j := range k {
						m := cand.entryMask[base+j]
						if s.lit[cand.entryWord[base+j]]&m != m {
							fired = false
							break
						}
					}
					if fired {
						counts[i*numClasses+s.label]++
					}
				}
			}
		}(wk)
	}
	wg.Wait()

	type scored struct {
		idx   int
		score int32
	}
	all := make([]scored, 0, cand.size)
	minFire := int32(n / 100)    // 1%未満しか発火しない節は捨てる
	maxFire := int32(n * 9 / 10) // 90%超で発火する節は捨てる
	for i := range cand.size {
		var total, best int32
		for c := range numClasses {
			v := counts[i*numClasses+c]
			total += v
			best = max(best, v)
		}
		if total < minFire || total > maxFire {
			continue
		}
		all = append(all, scored{idx: i, score: best*numClasses - total})
	}
	slices.SortFunc(all, func(a, b scored) int { return cmp.Compare(b.score, a.score) })

	keep := min(size, len(all))
	h := &hiddenLayer{kind: "andsel", size: keep, k: k}
	h.entryWord = make([]uint8, keep*k)
	h.entryMask = make([]uint64, keep*k)
	for i := range keep {
		src := all[i].idx * k
		copy(h.entryWord[i*k:(i+1)*k], cand.entryWord[src:src+k])
		copy(h.entryMask[i*k:(i+1)*k], cand.entryMask[src:src+k])
	}
	return h
}

// activate は、サンプルの活性をビットセット(out)へ書き込む。
func (h *hiddenLayer) activate(s *sample, out []uint64) {
	clear(out)
	switch h.kind {
	case "and", "andsel":
		k := h.k
		for i := range h.size {
			base := i * k
			fired := true
			for j := range k {
				m := h.entryMask[base+j]
				if s.lit[h.entryWord[base+j]]&m != m {
					fired = false
					break
				}
			}
			if fired {
				out[i/64] |= 1 << uint(i%64)
			}
		}
	case "conv":
		for i := range h.size {
			base := i * wordsPerImg
			area, mismatch := 0, 0
			for w := range wordsPerImg {
				m := h.mask[base+w]
				area += bits.OnesCount64(m)
				mismatch += bits.OnesCount64((s.x[w] ^ h.w[base+w]) & m)
			}
			// z = 2*一致数 - area + bias = area - 2*mismatch + bias
			if int32(area-2*mismatch)+h.bias[i] >= 0 {
				out[i/64] |= 1 << uint(i%64)
			}
		}
	default: // dense / densebias
		for i := range h.size {
			base := i * wordsPerImg
			mismatch := 0
			for w := range wordsPerImg {
				mismatch += bits.OnesCount64((s.x[w] ^ h.w[base+w]) & validMask[w])
			}
			// z = 784 - 2*mismatch + bias
			if int32(numFeatures-2*mismatch)+h.bias[i] >= 0 {
				out[i/64] |= 1 << uint(i%64)
			}
		}
	}
}

// ---------------------------------------------------------------------------
// 活性のキャッシュ(隠れ層が固定なので一度だけ計算する)
// ---------------------------------------------------------------------------

type actCache struct {
	words  int // 1サンプルあたりのビットセット語数
	data   []uint64
	labels []int
}

func buildActCache(h *hiddenLayer, samples []sample, workers int) *actCache {
	words := (h.size + 63) / 64
	c := &actCache{words: words, data: make([]uint64, len(samples)*words), labels: make([]int, len(samples))}
	var wg sync.WaitGroup
	chunk := (len(samples) + workers - 1) / workers
	for wk := range workers {
		wg.Add(1)
		go func(wk int) {
			defer wg.Done()
			lo := wk * chunk
			hi := min(lo+chunk, len(samples))
			for i := lo; i < hi; i++ {
				h.activate(&samples[i], c.data[i*words:(i+1)*words])
				c.labels[i] = samples[i].label
			}
		}(wk)
	}
	wg.Wait()
	return c
}

func (c *actCache) get(i int) []uint64 { return c.data[i*c.words : (i+1)*c.words] }

func (c *actCache) activeCount(i int) int {
	n := 0
	for _, w := range c.get(i) {
		n += bits.OnesCount64(w)
	}
	return n
}

// ---------------------------------------------------------------------------
// 読み出し: 整数重みの線形読み出し(マージン付きパーセプトロン則)
// ---------------------------------------------------------------------------

type linearReadout struct {
	size   int
	w      []int32 // numClasses * (size+1)、末尾は定数バイアス
	margin int32
	clip   int32

	// 平均化パーセプトロン(Daumé のトリック)。全て整数演算。
	// 平均重み = w - u/count で、評価時に整数除算で求める。
	averaged bool
	u        []int64
	count    int64
	avgW     []int32 // 評価用に materialize した平均重み
}

func newLinearReadout(size int, margin int32, averaged bool) *linearReadout {
	l := &linearReadout{size: size, w: make([]int32, numClasses*(size+1)), margin: margin, clip: 1 << 20, averaged: averaged}
	if averaged {
		l.u = make([]int64, numClasses*(size+1))
		l.avgW = make([]int32, numClasses*(size+1))
		l.count = 1
	}
	return l
}

// materializeAverage は、平均重みを整数除算で計算して avgW へ書き出す。
func (l *linearReadout) materializeAverage() {
	if !l.averaged {
		return
	}
	for i, w := range l.w {
		l.avgW[i] = w - int32(l.u[i]/l.count)
	}
}

// evalWeights は、推論に使う重み(平均化ありなら平均重み)を返す。
func (l *linearReadout) evalWeights() []int32 {
	if l.averaged {
		return l.avgW
	}
	return l.w
}

func (l *linearReadout) logitsWith(w []int32, act []uint64, out *[numClasses]int64) {
	stride := l.size + 1
	for c := range numClasses {
		out[c] = int64(w[c*stride+l.size]) // 定数バイアス
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

func (l *linearReadout) logits(act []uint64, out *[numClasses]int64) {
	l.logitsWith(l.w, act, out)
}

// update は、マージン未達なら正解クラスを+1、最良の誤クラスを-1する(整数のみ)。
// 更新したら true を返す。
func (l *linearReadout) update(act []uint64, label int) bool {
	var lg [numClasses]int64
	l.logits(act, &lg)

	best, bestVal := -1, int64(-1<<62)
	for c := range numClasses {
		if c == label {
			continue
		}
		if lg[c] > bestVal {
			bestVal = lg[c]
			best = c
		}
	}
	if lg[label]-bestVal >= int64(l.margin) {
		if l.averaged {
			l.count++
		}
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
			if l.averaged {
				l.u[yBase+i] += l.count
				l.u[pBase+i] -= l.count
			}
		}
	}
	l.w[yBase+l.size]++
	l.w[pBase+l.size]--
	if l.averaged {
		l.u[yBase+l.size] += l.count
		l.u[pBase+l.size] -= l.count
		l.count++
	}
	return true
}

func (l *linearReadout) predict(act []uint64) int {
	return l.predictWith(l.evalWeights(), act)
}

func (l *linearReadout) predictWith(w []int32, act []uint64) int {
	var lg [numClasses]int64
	l.logitsWith(w, act, &lg)
	best, bestVal := 0, int64(-1<<62)
	for c := range numClasses {
		if lg[c] > bestVal {
			bestVal = lg[c]
			best = c
		}
	}
	return best
}

func (l *linearReadout) accuracy(c *actCache, workers int) float64 {
	return l.accuracyWith(l.evalWeights(), c, workers)
}

func (l *linearReadout) accuracyWith(w []int32, c *actCache, workers int) float64 {
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
			for i := lo; i < hi; i++ {
				if l.predictWith(w, c.get(i)) == c.labels[i] {
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

// bitSlicedReadout は、学習済みの整数読み出しを符号ビットと絶対値の
// ビットプレーンへ変換した推論専用表現。各プレーンは二値であり、推論は
// AND、popcount、シフト、整数加減算だけで、元の整数ロジットを厳密に再現する。
type bitSlicedReadout struct {
	size      int
	words     int
	bitWidth  int
	sign      []uint64 // class * words。負の重みなら1
	magnitude []uint64 // class * bitWidth * words
	bias      [numClasses]int64
}

func newBitSlicedReadout(size int, w []int32) *bitSlicedReadout {
	stride := size + 1
	words := (size + 63) / 64
	maxMagnitude := uint32(0)
	for c := range numClasses {
		base := c * stride
		for i := range size {
			v := int64(w[base+i])
			if v < 0 {
				v = -v
			}
			maxMagnitude = max(maxMagnitude, uint32(v))
		}
	}
	bitWidth := max(1, bits.Len32(maxMagnitude))
	b := &bitSlicedReadout{
		size:      size,
		words:     words,
		bitWidth:  bitWidth,
		sign:      make([]uint64, numClasses*words),
		magnitude: make([]uint64, numClasses*bitWidth*words),
	}
	for c := range numClasses {
		base := c * stride
		b.bias[c] = int64(w[base+size])
		for i := range size {
			v := int64(w[base+i])
			magnitude := v
			if v < 0 {
				b.sign[c*words+i/64] |= 1 << uint(i%64)
				magnitude = -v
			}
			for bit := range bitWidth {
				if uint64(magnitude)>>uint(bit)&1 == 1 {
					idx := (c*bitWidth+bit)*words + i/64
					b.magnitude[idx] |= 1 << uint(i%64)
				}
			}
		}
	}
	return b
}

func (b *bitSlicedReadout) logits(act []uint64, out *[numClasses]int64) {
	for c := range numClasses {
		score := b.bias[c]
		signBase := c * b.words
		for bit := range b.bitWidth {
			magnitudeBase := (c*b.bitWidth + bit) * b.words
			count := 0
			negativeCount := 0
			for word := range b.words {
				activeMagnitude := act[word] & b.magnitude[magnitudeBase+word]
				count += bits.OnesCount64(activeMagnitude)
				negativeCount += bits.OnesCount64(activeMagnitude & b.sign[signBase+word])
			}
			score += int64(count-2*negativeCount) << uint(bit)
		}
		out[c] = score
	}
}

func (b *bitSlicedReadout) predict(act []uint64) int {
	var lg [numClasses]int64
	b.logits(act, &lg)
	best, bestVal := 0, int64(-1<<62)
	for c := range numClasses {
		if lg[c] > bestVal {
			bestVal = lg[c]
			best = c
		}
	}
	return best
}

func (b *bitSlicedReadout) accuracy(c *actCache, workers int) float64 {
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
			for i := lo; i < hi; i++ {
				if b.predict(c.get(i)) == c.labels[i] {
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

// ---------------------------------------------------------------------------
// 読み出し: 固定ランダムプロトタイプとの一致(BEP相当の対照)
// ---------------------------------------------------------------------------

type protoReadout struct {
	size   int
	words  int
	protos []uint64 // numClasses * words
}

func newProtoReadout(size int, r *rng) *protoReadout {
	words := (size + 63) / 64
	p := &protoReadout{size: size, words: words, protos: make([]uint64, numClasses*words)}
	for i := range p.protos {
		p.protos[i] = r.next()
	}
	return p
}

func (p *protoReadout) predict(act []uint64) int {
	best, bestMatch := 0, -1
	for c := range numClasses {
		base := c * p.words
		match := 0
		for w := range p.words {
			match += bits.OnesCount64(^(act[w] ^ p.protos[base+w]))
		}
		if match > bestMatch {
			bestMatch = match
			best = c
		}
	}
	return best
}

func (p *protoReadout) accuracy(c *actCache) float64 {
	correct := 0
	for i := range c.labels {
		if p.predict(c.get(i)) == c.labels[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(c.labels))
}

// ---------------------------------------------------------------------------

func loadPacked(name string) (train, test []sample, err error) {
	var ds dataset.Binary[int]
	if name == "fashion" {
		ds, err = dataset.LoadFashionMNIST(nil)
	} else {
		ds, err = dataset.LoadMNIST(nil)
	}
	if err != nil {
		return nil, nil, err
	}
	return packSamples(ds.TrainInputs, ds.TrainLabels),
		packSamples(ds.TestInputs, ds.TestLabels), nil
}

// splitValidation は、testをハイパーパラメータやエポック選択に使わないために、
// trainから検証集合を分離する。元のスライスは変更しない。
func splitValidation(train []sample, validationSize int, r *rng) (fit, validation []sample) {
	if validationSize <= 0 {
		return train, nil
	}
	validationSize = min(validationSize, len(train)-1)
	order := make([]int, len(train))
	for i := range order {
		order[i] = i
	}
	for i := len(order) - 1; i > 0; i-- {
		j := int(r.intn(uint64(i + 1)))
		order[i], order[j] = order[j], order[i]
	}
	validation = make([]sample, validationSize)
	fit = make([]sample, len(train)-validationSize)
	for i, idx := range order {
		if i < validationSize {
			validation[i] = train[idx]
		} else {
			fit[i-validationSize] = train[idx]
		}
	}
	return fit, validation
}

func main() {
	var (
		dsName    = flag.String("dataset", "mnist", "mnist | fashion")
		hType     = flag.String("type", "and", "dense | densebias | densematch | and | conv")
		hSize     = flag.Int("hidden", 8192, "隠れ層の幅")
		kFlag     = flag.Int("k", 8, "and型のリテラル数")
		biasSig   = flag.Int("biassigma", 20, "densebias型のバイアス範囲(sigmaの1/10単位)")
		readout   = flag.String("readout", "linear", "linear | proto")
		margin    = flag.Int("margin", 25, "線形読み出しのマージン")
		averaged  = flag.Bool("avg", false, "平均化パーセプトロン(整数のみ)")
		patch     = flag.Int("patch", 7, "conv型のパッチ辺長")
		matchPct  = flag.Int("matchpct", 85, "conv型の発火に要する一致率(%)")
		over      = flag.Int("over", 0, "and型の候補倍率。>1 なら整数スコアで識別力の高い節を選抜")
		epochs    = flag.Int("epochs", 20, "読み出しの学習エポック数")
		valSize   = flag.Int("validation", 0, "学習集合から分離する検証件数 (0 = 従来どおりtestで各epoch評価)")
		benchRuns = flag.Int("benchruns", 1, "最終test推論の反復回数")
		seed      = flag.Uint64("seed", 1, "乱数シード")
		threads   = flag.Int("threads", 0, "ワーカー数 (0 = NumCPU)")
	)
	flag.Parse()

	workers := *threads
	if workers <= 0 {
		workers = runtime.NumCPU()
	}

	train, test, err := loadPacked(*dsName)
	if err != nil {
		log.Fatal(err)
	}
	if *valSize < 0 || *valSize >= len(train) {
		log.Fatalf("validationが範囲外: %d (0 <= validation < %d であるべき)", *valSize, len(train))
	}
	if *benchRuns <= 0 {
		log.Fatalf("benchrunsが不正: %d (1以上であるべき)", *benchRuns)
	}
	if *epochs <= 0 {
		log.Fatalf("epochsが不正: %d (1以上であるべき)", *epochs)
	}
	if *hSize <= 0 {
		log.Fatalf("hiddenが不正: %d (1以上であるべき)", *hSize)
	}
	fit, validation := splitValidation(train, *valSize, newRng(*seed^0xD1B54A32D192ED03))

	r := newRng(*seed)
	hiddenStart := time.Now()

	var h *hiddenLayer
	switch *hType {
	case "and":
		if *over > 1 {
			h = selectDiscriminative(*hSize, *kFlag, *over, fit, r, workers)
		} else {
			h = newAndHidden(*hSize, *kFlag, fit, r)
		}
	case "conv":
		h = newConvHidden(*hSize, *patch, *matchPct, fit, r)
	case "densebias":
		h = newDenseHidden(*hSize, *biasSig, r)
	case "densematch":
		h = newDenseMatchHidden(*hSize, *matchPct, fit, r)
	default:
		h = newDenseHidden(*hSize, 0, r)
	}
	hiddenDuration := time.Since(hiddenStart)

	fitCacheStart := time.Now()
	trainAct := buildActCache(h, fit, workers)
	fitCacheDuration := time.Since(fitCacheStart)
	var validationAct *actCache
	validationCacheDuration := time.Duration(0)
	if len(validation) > 0 {
		validationCacheStart := time.Now()
		validationAct = buildActCache(h, validation, workers)
		validationCacheDuration = time.Since(validationCacheStart)
	}
	testCacheStart := time.Now()
	testAct := buildActCache(h, test, workers)
	testCacheDuration := time.Since(testCacheStart)

	// 活性率(特徴の疎さ)を測る
	sum := 0
	for i := range min(1000, len(fit)) {
		sum += trainAct.activeCount(i)
	}
	actRate := float64(sum) / float64(min(1000, len(fit))*h.size)

	fmt.Printf("dataset=%s type=%s hidden=%d k=%d biassigma=%d readout=%s margin=%d fit=%d validation=%d 活性率=%.3f hidden_init=%s fit_features=%s validation_features=%s test_features=%s\n",
		*dsName, h.kind, h.size, h.k, *biasSig, *readout, *margin, len(fit), len(validation), actRate,
		hiddenDuration, fitCacheDuration, validationCacheDuration, testCacheDuration)

	if *readout == "proto" {
		p := newProtoReadout(h.size, r)
		fmt.Printf("proto読み出し(学習なし): test acc %.4f\n", p.accuracy(testAct))
		return
	}

	l := newLinearReadout(h.size, int32(*margin), *averaged)
	n := len(fit)
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}

	best := 0.0
	bestEpoch := 0
	var bestWeights []int32
	for e := 1; e <= *epochs; e++ {
		for i := n - 1; i > 0; i-- {
			j := int(r.intn(uint64(i + 1)))
			order[i], order[j] = order[j], order[i]
		}
		te := time.Now()
		updates := 0
		for _, idx := range order {
			if l.update(trainAct.get(idx), trainAct.labels[idx]) {
				updates++
			}
		}
		trainDur := time.Since(te)
		l.materializeAverage()
		evalAct := testAct
		evalName := "test"
		if validationAct != nil {
			evalAct = validationAct
			evalName = "validation"
		}
		acc := l.accuracy(evalAct, workers)
		if bestEpoch == 0 || acc > best {
			best = acc
			bestEpoch = e
			bestWeights = slices.Clone(l.evalWeights())
		}
		fmt.Printf("epoch %d: %s acc %.4f (best %.4f) 更新率 %.3f train %.1fs\n",
			e, evalName, acc, best, float64(updates)/float64(n), trainDur.Seconds())
	}

	integerStart := time.Now()
	integerAccuracy := 0.0
	for range *benchRuns {
		integerAccuracy = l.accuracyWith(bestWeights, testAct, workers)
	}
	integerDuration := time.Since(integerStart)
	bitSliced := newBitSlicedReadout(h.size, bestWeights)
	bitStart := time.Now()
	bitAccuracy := 0.0
	for range *benchRuns {
		bitAccuracy = bitSliced.accuracy(testAct, workers)
	}
	bitDuration := time.Since(bitStart)
	if integerAccuracy != bitAccuracy {
		log.Fatalf("整数読み出しとビットスライス読み出しの精度が不一致: %.6f != %.6f", integerAccuracy, bitAccuracy)
	}
	selectionName := "test"
	if validationAct != nil {
		selectionName = "validation"
	}
	fmt.Printf("SELECTED epoch=%d %s=%.4f test=%.4f\n", bestEpoch, selectionName, best, integerAccuracy)
	fmt.Printf("INFERENCE integer=%s bitslice=%s runs=%d bitwidth=%d storage_bits_per_weight=%d accuracy=%.4f\n",
		integerDuration, bitDuration, *benchRuns, bitSliced.bitWidth, bitSliced.bitWidth+1, bitAccuracy)
}
