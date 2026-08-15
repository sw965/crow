package main

import (
	"math/bits"
	"testing"
)

// bitAt は、画像の (row, col) のビットを返す。
func bitAt(s *sample, row, col int) uint64 {
	idx := row*side + col
	return s.x[idx/64] >> uint(idx%64) & 1
}

// weightAt は、ニューロン i の (row, col) に対応する可視重みのビットを返す。
func weightAt(l *convLayer, i, row, col int) uint64 {
	idx := row*side + col
	return l.w[i*wordsPerImg+idx/64] >> uint(idx%64) & 1
}

// refForward は、ビット演算を使わない素朴な参照実装。
func refForward(l *convLayer, s *sample, i int) int32 {
	r0, c0 := int(l.r0[i]), int(l.c0[i])
	mismatch := 0
	for dr := range l.patch {
		for dc := range l.patch {
			if bitAt(s, r0+dr, c0+dc) != weightAt(l, i, r0+dr, c0+dc) {
				mismatch++
			}
		}
	}
	return int32(l.area-2*mismatch) + l.bias[i]
}

func newTestSamples(t *testing.T, n int, r *rng) []sample {
	t.Helper()
	out := make([]sample, n)
	for i := range out {
		for w := range wordsPerImg {
			out[i].x[w] = r.next() & validMask[w]
		}
		out[i].label = i % numClasses
	}
	return out
}

func TestForwardMatchesReference(t *testing.T) {
	r := newRng(1)
	train := newTestSamples(t, 64, r)
	l := newConvLayer(256, 7, 85, 4, train, false, r)

	z := make([]int32, l.size)
	act := make([]uint64, (l.size+63)/64)

	for si := range 8 {
		s := &train[si]
		l.forward(s, z, act)
		for i := range l.size {
			want := refForward(l, s, i)
			if z[i] != want {
				t.Fatalf("z[%d]の不一致 (sample %d): got = %d, want = %d", i, si, z[i], want)
			}
			gotBit := act[i/64] >> uint(i%64) & 1
			wantBit := uint64(0)
			if want >= 0 {
				wantBit = 1
			}
			if gotBit != wantBit {
				t.Fatalf("活性ビット[%d]の不一致 (sample %d): got = %d, want = %d", i, si, gotBit, wantBit)
			}
		}
	}
}

// TestForwardIgnoresOutsideReceptiveField は、受容野の外の画素を変えても
// 活性前値が変わらない事(=局所受容野になっている事)を確認する。
func TestForwardIgnoresOutsideReceptiveField(t *testing.T) {
	r := newRng(2)
	train := newTestSamples(t, 16, r)
	l := newConvLayer(64, 5, 85, 4, train, false, r)

	z1 := make([]int32, l.size)
	z2 := make([]int32, l.size)
	act := make([]uint64, (l.size+63)/64)

	s := train[0]
	l.forward(&s, z1, act)

	// ニューロン0の受容野の外にある画素を1つ反転する
	r0, c0 := int(l.r0[0]), int(l.c0[0])
	flipped := false
	for row := range side {
		for col := range side {
			inside := row >= r0 && row < r0+l.patch && col >= c0 && col < c0+l.patch
			if inside {
				continue
			}
			idx := row*side + col
			s.x[idx/64] ^= 1 << uint(idx%64)
			flipped = true
			break
		}
		if flipped {
			break
		}
	}
	if !flipped {
		t.Fatal("受容野の外の画素が見つからなかった")
	}

	l.forward(&s, z2, act)
	if z1[0] != z2[0] {
		t.Errorf("受容野の外の変更で活性前値が変わった: %d -> %d", z1[0], z2[0])
	}
}

// TestUpdateMovesTowardInput は、d=+1 の更新が可視重みを入力へ近づける事
// (=一致数が減らない事)と、H の符号反転が W のビット反転と一致する事を確認する。
func TestUpdateMovesTowardInput(t *testing.T) {
	r := newRng(3)
	train := newTestSamples(t, 32, r)
	// hInit=1 にすると1回の更新で符号が変わり得るので、反転の検証がしやすい
	l := newConvLayer(32, 5, 85, 1, train, false, r)

	s := &train[0]
	const target = 0

	mismatchOf := func() int {
		r0, c0 := int(l.r0[target]), int(l.c0[target])
		n := 0
		for dr := range l.patch {
			for dc := range l.patch {
				if bitAt(s, r0+dr, c0+dc) != weightAt(l, target, r0+dr, c0+dc) {
					n++
				}
			}
		}
		return n
	}

	prev := mismatchOf()
	for range 8 {
		l.update(target, 1, s, false)

		// H の符号と W のビットが常に一致している事
		r0, c0 := int(l.r0[target]), int(l.c0[target])
		for dr := range l.patch {
			for dc := range l.patch {
				h := l.h[target*l.area+dr*l.patch+dc]
				wBit := weightAt(l, target, r0+dr, c0+dc)
				wantBit := uint64(0)
				if h >= 0 {
					wantBit = 1
				}
				if wBit != wantBit {
					t.Fatalf("Hの符号とWのビットが不一致: h = %d, wBit = %d", h, wBit)
				}
			}
		}

		cur := mismatchOf()
		if cur > prev {
			t.Fatalf("d=+1 の更新で不一致数が増えた: %d -> %d", prev, cur)
		}
		prev = cur
	}
	if prev != 0 {
		t.Errorf("十分な更新回数の後も入力と完全一致しなかった: 不一致数 = %d", prev)
	}
}

// TestUpdateBiasToggle は、updateBias の指定どおりにバイアスが動く/動かない事を確認する。
func TestUpdateBiasToggle(t *testing.T) {
	r := newRng(4)
	train := newTestSamples(t, 16, r)
	l := newConvLayer(16, 5, 85, 4, train, false, r)

	before := l.bias[0]
	l.update(0, 1, &train[0], false)
	if l.bias[0] != before {
		t.Errorf("updateBias=false なのにバイアスが変化した: %d -> %d", before, l.bias[0])
	}
	l.update(0, 1, &train[0], true)
	if l.bias[0] != before+1 {
		t.Errorf("updateBias=true でバイアスが+1されていない: %d -> %d", before, l.bias[0])
	}
}

// TestCompeteEpochSharpensWinner は、競合学習が勝者を入力へ特化させる事を確認する。
// 同一の入力を繰り返し与えると、その位置の勝者の一致数が単調に上がる。
func TestCompeteEpochSharpensWinner(t *testing.T) {
	r := newRng(5)
	train := newTestSamples(t, 8, r)
	l := newConvLayer(2048, 5, 85, 4, train, true, r)

	fixed := []sample{train[0]}
	order := []int{0}

	bestMatchAt := func(pos int) int {
		start := pos * l.perPos
		end := min(start+l.perPos, l.size)
		best := -1
		for i := start; i < end; i++ {
			base := i * wordsPerImg
			mismatch := 0
			for w := int(l.wlo[i]); w <= int(l.whi[i]); w++ {
				mismatch += bits.OnesCount64((fixed[0].x[w] ^ l.w[base+w]) & l.mask[base+w])
			}
			best = max(best, l.area-mismatch)
		}
		return best
	}

	const pos = 0
	prev := bestMatchAt(pos)
	for range 10 {
		l.competeEpoch(fixed, order, r)
		cur := bestMatchAt(pos)
		if cur < prev {
			t.Fatalf("競合学習で勝者の一致数が下がった: %d -> %d", prev, cur)
		}
		prev = cur
	}
	if prev != l.area {
		t.Errorf("繰り返し学習後も入力と完全一致しなかった: 一致数 = %d / %d", prev, l.area)
	}
}

func TestOutputLayerMarginAndPredict(t *testing.T) {
	o := newOutputLayer(64, 4, false)
	act := make([]uint64, 1)
	act[0] = 0b1011 // 特徴 0,1,3 が発火

	// 初期状態は全て0なので、マージン未達 → 更新される
	updated, wrong := o.train(act, 3)
	if !updated {
		t.Fatal("初期状態でマージンを満たしたと判定された")
	}
	if wrong < 0 || wrong == 3 {
		t.Fatalf("誤クラスの選択が不正: got = %d", wrong)
	}

	// 正解クラスの重みが増え、誤クラスの重みが減っている事
	stride := o.size + 1
	for _, i := range []int{0, 1, 3} {
		if o.w[3*stride+i] != 1 {
			t.Errorf("正解クラスの重みが+1されていない: w[3][%d] = %d", i, o.w[3*stride+i])
		}
		if o.w[wrong*stride+i] != -1 {
			t.Errorf("誤クラスの重みが-1されていない: w[%d][%d] = %d", wrong, i, o.w[wrong*stride+i])
		}
	}
	// 発火していない特徴は変わらない
	if o.w[3*stride+2] != 0 {
		t.Errorf("発火していない特徴の重みが変化した: %d", o.w[3*stride+2])
	}

	// 十分に学習させるとマージンを満たして更新されなくなる
	for range 100 {
		o.train(act, 3)
	}
	if updated, _ := o.train(act, 3); updated {
		t.Error("十分学習した後もマージン未達と判定された")
	}
	if got := o.predict(act); got != 3 {
		t.Errorf("予測が不正: got = %d, want = 3", got)
	}
}

// TestAveragedWeights は、平均化重みが w - u/count の整数除算と一致する事を確認する。
func TestAveragedWeights(t *testing.T) {
	o := newOutputLayer(32, 4, true)
	act := make([]uint64, 1)
	act[0] = 0b0110

	r := newRng(6)
	for range 50 {
		o.train(act, int(r.intn(numClasses)))
	}
	o.materializeAverage()
	for i := range o.w {
		want := o.w[i] - int32(o.u[i]/o.count)
		if o.avgW[i] != want {
			t.Fatalf("平均化重みの不一致 (i = %d): got = %d, want = %d", i, o.avgW[i], want)
		}
	}
	if o.evalWeights()[0] != o.avgW[0] {
		t.Error("平均化ありなのに evalWeights が生の重みを返した")
	}
}

// TestLearnsSyntheticTask は、クラスごとに異なる位置へ目印を置いた合成タスクを
// 競合学習+出力層学習で解けることを確認する(決定的: シード固定)。
func TestLearnsSyntheticTask(t *testing.T) {
	r := newRng(7)

	// クラス c は、位置 (3 + 2c, 3 + 2c) を左上とする 4x4 の塗り潰しを持つ
	newSample := func(c int) sample {
		var s sample
		// 背景ノイズ(各画素 確率 1/4)
		for w := range wordsPerImg {
			a, b := r.next(), r.next()
			s.x[w] = a & b & validMask[w]
		}
		r0, c0 := 3+2*c, 3+2*c
		for dr := range 4 {
			for dc := range 4 {
				idx := (r0+dr)*side + (c0 + dc)
				s.x[idx/64] |= 1 << uint(idx%64)
			}
		}
		s.label = c
		return s
	}

	train := make([]sample, 0, 2000)
	test := make([]sample, 0, 500)
	for i := range 2000 {
		train = append(train, newSample(i%numClasses))
	}
	for i := range 500 {
		test = append(test, newSample(i%numClasses))
	}

	m := &model{
		l1:         newConvLayer(1024, 5, 70, 4, train, true, r),
		o:          newOutputLayer(1024, 8, true),
		gate:       8,
		lrLog2:     3,
		impThr:     2,
		groupSize:  64,
		updateBias: true,
	}

	order := make([]int, len(train))
	for i := range order {
		order[i] = i
	}
	m.l1.competeEpoch(train, order, r)

	for range 5 {
		m.trainEpoch(train, r, true) // 第1層は競合学習済みなので誤差駆動更新は止める
	}
	m.o.materializeAverage()

	acc := m.accuracy(test, 2)
	if acc < 0.9 {
		t.Errorf("合成タスクの精度が低すぎる: got = %.4f, want >= 0.9", acc)
	}
}
