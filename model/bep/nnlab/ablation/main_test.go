package main

import (
	"math/bits"
	"testing"
)

func newTestSamples(t *testing.T, n int, r *rng) []sample {
	t.Helper()
	out := make([]sample, n)
	for i := range out {
		for w := range wordsPerImg {
			v := r.next() & validMask[w]
			out[i].x[w] = v
			out[i].lit[w] = v
			out[i].lit[wordsPerImg+w] = ^v & validMask[w]
		}
		out[i].label = i % numClasses
	}
	return out
}

// TestAndHiddenFiresOnReference は、AND型の節が参照元のサンプルでは必ず発火する事を確認する。
// 学習データを1件だけにすると、全ての節がそのサンプルを参照して作られる。
func TestAndHiddenFiresOnReference(t *testing.T) {
	r := newRng(1)
	train := newTestSamples(t, 1, r)
	h := newAndHidden(512, 8, train, r)

	act := make([]uint64, (h.size+63)/64)
	h.activate(&train[0], act)

	fired := 0
	for _, w := range act {
		fired += bits.OnesCount64(w)
	}
	if fired != h.size {
		t.Errorf("参照元サンプルで発火しない節がある: 発火 %d / %d", fired, h.size)
	}
}

// TestConvHiddenMatchesReference は、conv型の活性が素朴な参照実装と一致する事を確認する。
func TestConvHiddenMatchesReference(t *testing.T) {
	r := newRng(2)
	train := newTestSamples(t, 32, r)
	const p, matchPct = 7, 85
	h := newConvHidden(256, p, matchPct, train, r)

	act := make([]uint64, (h.size+63)/64)
	s := &train[3]
	h.activate(s, act)

	area := p * p
	thr := (area*matchPct + 99) / 100
	wantBias := int32(area - 2*thr)

	for i := range h.size {
		base := i * wordsPerImg
		// マスク内の一致数を素朴に数える
		match, total := 0, 0
		for idx := range numFeatures {
			w, b := idx/64, uint(idx%64)
			if h.mask[base+w]>>b&1 == 0 {
				continue
			}
			total++
			if s.x[w]>>b&1 == h.w[base+w]>>b&1 {
				match++
			}
		}
		if total != area {
			t.Fatalf("結線数がパッチ面積と不一致 (i = %d): got = %d, want = %d", i, total, area)
		}
		if h.bias[i] != wantBias {
			t.Fatalf("バイアスの不一致 (i = %d): got = %d, want = %d", i, h.bias[i], wantBias)
		}
		wantFire := int32(2*match-area)+h.bias[i] >= 0
		gotFire := act[i/64]>>uint(i%64)&1 == 1
		if gotFire != wantFire {
			t.Fatalf("活性の不一致 (i = %d): got = %t, want = %t (match = %d)", i, gotFire, wantFire, match)
		}
	}
}

// TestDenseHiddenThreshold は、dense型が z = 784 - 2*不一致数 + bias で判定される事を確認する。
func TestDenseHiddenThreshold(t *testing.T) {
	r := newRng(3)
	train := newTestSamples(t, 4, r)
	h := newDenseHidden(128, 0, r) // バイアス無し

	act := make([]uint64, (h.size+63)/64)
	s := &train[0]
	h.activate(s, act)

	for i := range h.size {
		base := i * wordsPerImg
		mismatch := 0
		for w := range wordsPerImg {
			mismatch += bits.OnesCount64((s.x[w] ^ h.w[base+w]) & validMask[w])
		}
		if h.bias[i] != 0 {
			t.Fatalf("バイアス無しのはずが非ゼロ (i = %d): %d", i, h.bias[i])
		}
		wantFire := numFeatures-2*mismatch >= 0
		gotFire := act[i/64]>>uint(i%64)&1 == 1
		if gotFire != wantFire {
			t.Fatalf("活性の不一致 (i = %d): got = %t, want = %t", i, gotFire, wantFire)
		}
	}
}

// TestDenseBiasIsNonPositive は、densebias型のバイアスが指定範囲に収まる事を確認する。
func TestDenseBiasIsNonPositive(t *testing.T) {
	r := newRng(4)
	const sigmaX10 = 20
	h := newDenseHidden(256, sigmaX10, r)
	if h.kind != "densebias" {
		t.Fatalf("種別の不一致: got = %s, want = densebias", h.kind)
	}
	lowest := -int32(28 * sigmaX10 / 10)
	for i := range h.size {
		if h.bias[i] > 0 || h.bias[i] < lowest {
			t.Fatalf("バイアスが範囲外 (i = %d): got = %d, want = [%d, 0]", i, h.bias[i], lowest)
		}
	}
}

func TestDenseMatchUsesFullTrainingPatterns(t *testing.T) {
	r := newRng(0xD3A5E)
	train := newTestSamples(t, 16, r)
	const matchPct = 85
	h := newDenseMatchHidden(128, matchPct, train, r)
	wantBias := int32(numFeatures - 2*((numFeatures*matchPct+99)/100))
	for i := range h.size {
		if h.bias[i] != wantBias {
			t.Fatalf("バイアスが一致率しきい値と不一致 (i=%d): got=%d want=%d", i, h.bias[i], wantBias)
		}
		weight := h.w[i*wordsPerImg : (i+1)*wordsPerImg]
		found := false
		for j := range train {
			matches := true
			for word := range wordsPerImg {
				if weight[word] != train[j].x[word] {
					matches = false
					break
				}
			}
			if matches {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("重み行が学習パターンのどれとも一致しない (i=%d)", i)
		}
	}
}

func TestLinearReadout(t *testing.T) {
	l := newLinearReadout(64, 4, false)
	act := make([]uint64, 1)
	act[0] = 0b1011

	if !l.update(act, 2) {
		t.Fatal("初期状態でマージンを満たしたと判定された")
	}
	stride := l.size + 1
	for _, i := range []int{0, 1, 3} {
		if l.w[2*stride+i] != 1 {
			t.Errorf("正解クラスの重みが+1されていない: w[2][%d] = %d", i, l.w[2*stride+i])
		}
	}
	for range 100 {
		l.update(act, 2)
	}
	if l.update(act, 2) {
		t.Error("十分学習した後もマージン未達と判定された")
	}
	if got := l.predict(act); got != 2 {
		t.Errorf("予測が不正: got = %d, want = 2", got)
	}
}

// TestLinearReadoutAveraged は、平均化重みが w - u/count と一致する事を確認する。
func TestLinearReadoutAveraged(t *testing.T) {
	l := newLinearReadout(32, 4, true)
	act := make([]uint64, 1)
	act[0] = 0b0110

	r := newRng(5)
	for range 50 {
		l.update(act, int(r.intn(numClasses)))
	}
	l.materializeAverage()
	for i := range l.w {
		want := l.w[i] - int32(l.u[i]/l.count)
		if l.avgW[i] != want {
			t.Fatalf("平均化重みの不一致 (i = %d): got = %d, want = %d", i, l.avgW[i], want)
		}
	}
}

func TestBitSlicedReadoutMatchesIntegerLogits(t *testing.T) {
	r := newRng(0xB1751CED)
	const size = 257
	l := newLinearReadout(size, 4, false)
	for i := range l.w {
		l.w[i] = int32(int(r.intn(65)) - 32)
	}
	b := newBitSlicedReadout(size, l.w)
	act := make([]uint64, (size+63)/64)
	for trial := range 100 {
		for i := range act {
			act[i] = r.next()
		}
		act[len(act)-1] &= uint64(1)<<(size%64) - 1
		var integerLogits, bitLogits [numClasses]int64
		l.logitsWith(l.w, act, &integerLogits)
		b.logits(act, &bitLogits)
		if integerLogits != bitLogits {
			t.Fatalf("ロジットが不一致 (trial %d): integer = %v, bitslice = %v", trial, integerLogits, bitLogits)
		}
	}
}

// TestFullyConnectedBinaryNetworkRepresentsTruthTable は、任意の有限二値写像を
// 全結合・二値重み・整数バイアスだけで厳密に表せる構成を全入力で検証する。
// hidden[p] は入力パターン p の一致検出器、出力は二値重みで全 hidden に全結合する。
func TestFullyConnectedBinaryNetworkRepresentsTruthTable(t *testing.T) {
	const inputBits = 5
	const outputBits = 7
	patterns := 1 << inputBits
	r := newRng(0xA11CE)
	targets := make([][]int, patterns)
	for p := range patterns {
		targets[p] = make([]int, outputBits)
		for k := range outputBits {
			targets[p][k] = 1
			if r.next()&1 == 0 {
				targets[p][k] = -1
			}
		}
	}

	for input := range patterns {
		hidden := make([]int, patterns)
		for p := range patterns {
			dot := 0
			for bit := range inputBits {
				x := -1
				if input>>bit&1 == 1 {
					x = 1
				}
				w := -1
				if p>>bit&1 == 1 {
					w = 1
				}
				dot += w * x
			}
			// sign(p^T x - (d-1)): pと完全一致するときだけ+1。
			if dot-(inputBits-1) >= 0 {
				hidden[p] = 1
			} else {
				hidden[p] = -1
			}
		}

		for k := range outputBits {
			bias := 0
			preActivation := 0
			for p := range patterns {
				weight := targets[p][k]
				bias += weight
				preActivation += weight * hidden[p]
			}
			preActivation += bias
			got := 1
			if preActivation < 0 {
				got = -1
			}
			if got != targets[input][k] {
				t.Fatalf("真理値表を再現できない: input=%d output=%d got=%d want=%d z=%d", input, k, got, targets[input][k], preActivation)
			}
			if preActivation != 2*targets[input][k] {
				t.Fatalf("活性前値が構成式と不一致: input=%d output=%d got=%d want=%d", input, k, preActivation, 2*targets[input][k])
			}
		}
	}
}

// TestActCache は、活性キャッシュが activate の結果とラベルを正しく保持する事を確認する。
func TestActCache(t *testing.T) {
	r := newRng(6)
	train := newTestSamples(t, 16, r)
	h := newAndHidden(128, 6, train, r)

	c := buildActCache(h, train, 2)
	want := make([]uint64, (h.size+63)/64)
	for i := range train {
		h.activate(&train[i], want)
		got := c.get(i)
		for w := range want {
			if got[w] != want[w] {
				t.Fatalf("キャッシュの活性が不一致 (sample %d, word %d)", i, w)
			}
		}
		if c.labels[i] != train[i].label {
			t.Fatalf("キャッシュのラベルが不一致 (sample %d)", i)
		}
	}
}
