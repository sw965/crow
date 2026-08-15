package main

import (
	"math/rand/v2"
	"testing"

	"github.com/sw965/omw/mathx/bitsx"
)

func newTestModel(t *testing.T, seed uint64) *model {
	t.Helper()
	rng := rand.New(rand.NewPCG(seed, seed+1))
	m := &model{xRows: 1, xCols: 128}
	if err := m.appendLayer(64, rng); err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	protos := make(bitsx.Matrices, numClasses)
	for i := range protos {
		p, err := bitsx.NewRandMatrix(1, 64, 0, rng)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		protos[i] = p
	}
	m.protos = protos
	return m
}

// TestRebuildVisibleMatchesHiddenSign は、W の各ビットが H の符号と一致する事と、
// 転置 WT が同期している事を確認する。
func TestRebuildVisibleMatchesHiddenSign(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	m := newTestModel(t, 1)
	l := m.layers[0]

	for i := range l.h {
		l.h[i] = int8(rng.IntN(255) - 128)
	}
	if err := rebuildVisible(l); err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}

	wCols := l.w.Cols()
	for row := range l.w.Rows() {
		for col := range wCols {
			bit, err := l.w.Bit(row, col)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			want := uint64(0)
			if l.h[row*wCols+col] >= 0 {
				want = 1
			}
			if bit != want {
				t.Fatalf("Wのビットが H の符号と不一致 (row = %d, col = %d)", row, col)
			}
			bitT, err := l.wt.Bit(col, row)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			if bitT != bit {
				t.Fatalf("転置が同期していない (row = %d, col = %d)", row, col)
			}
		}
	}
}

// TestSWAMeanKeepsSign は、平均集約が総和の符号を保つ事を確認する。
// 整数除算は0方向へ切り捨てるため、素朴に sum/count の符号を使うと
// 平均の絶対値が1未満の負の重みが sign(0)=+1 で +1 に化けてしまう。
func TestSWAMeanKeepsSign(t *testing.T) {
	dst := newTestModel(t, 3)
	s := newSWA(dst)

	// 平均が -1 未満(絶対値1未満)になる組み合わせを作る:
	// 3個のうち2個が -1、1個が +1 → 総和 -1、平均 -1/3 → 整数除算で 0
	models := make([]*model, 3)
	for k := range models {
		models[k] = newTestModel(t, uint64(10+k))
		for i := range models[k].layers[0].h {
			if k < 2 {
				models[k].layers[0].h[i] = -1
			} else {
				models[k].layers[0].h[i] = 1
			}
		}
	}
	for _, m := range models {
		if err := s.add(m); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
	}
	if err := s.writeAverage(dst); err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	for i, v := range dst.layers[0].h {
		if v >= 0 {
			t.Fatalf("総和が負なのに平均が非負になった (i = %d): got = %d", i, v)
		}
	}
}

// TestSWAVoteBreaksTiesBySum は、多数決の同点を総和の符号で解く事を確認する。
// 同点を無条件に +1 へ倒すと、モデル数が偶数のときだけ大量の重みが誤って +1 になる。
func TestSWAVoteBreaksTiesBySum(t *testing.T) {
	dst := newTestModel(t, 4)
	s := newSWA(dst)
	s.vote = true

	// 2個のうち1個が -10、1個が +1 → 票は同点、総和は -9
	a := newTestModel(t, 20)
	b := newTestModel(t, 21)
	for i := range a.layers[0].h {
		a.layers[0].h[i] = -10
		b.layers[0].h[i] = 1
	}
	if err := s.add(a); err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	if err := s.add(b); err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	if err := s.writeAverage(dst); err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	for i, v := range dst.layers[0].h {
		if v >= 0 {
			t.Fatalf("同点だが総和が負なのに +1 になった (i = %d): got = %d", i, v)
		}
	}
}

// TestSWAVoteMajority は、同点でない場合に多数決どおりになる事を確認する。
func TestSWAVoteMajority(t *testing.T) {
	dst := newTestModel(t, 5)
	s := newSWA(dst)
	s.vote = true

	// 3個のうち2個が +1(総和は負)→ 多数決では +1 が勝つ
	for k := range 3 {
		m := newTestModel(t, uint64(30+k))
		for i := range m.layers[0].h {
			if k < 2 {
				m.layers[0].h[i] = 1
			} else {
				m.layers[0].h[i] = -100
			}
		}
		if err := s.add(m); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
	}
	if err := s.writeAverage(dst); err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	for i, v := range dst.layers[0].h {
		if v < 0 {
			t.Fatalf("多数決では +1 のはずが負になった (i = %d): got = %d", i, v)
		}
	}
}

// TestSWASingleModelIsIdentity は、1個だけ平均したら元のモデルと同じ可視重みになる事を確認する。
func TestSWASingleModelIsIdentity(t *testing.T) {
	src := newTestModel(t, 6)
	rng := rand.New(rand.NewPCG(7, 8))
	for i := range src.layers[0].h {
		src.layers[0].h[i] = int8(rng.IntN(255) - 128)
	}
	if err := rebuildVisible(src.layers[0]); err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}

	s := newSWA(src)
	if err := s.add(src); err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	dst := cloneModel(src)
	if err := s.writeAverage(dst); err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	if !dst.layers[0].w.Equal(src.layers[0].w) {
		t.Error("1個だけ平均したのに可視重みが変わった")
	}
}

// TestWeightAgreement は、一致率が同一モデルで1.0、反転モデルで0.0になる事を確認する。
func TestWeightAgreement(t *testing.T) {
	a := newTestModel(t, 9)
	b := cloneModel(a)

	got, err := weightAgreement(a, b)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	if got != 1.0 {
		t.Errorf("同一モデルの一致率が1.0でない: got = %f", got)
	}

	// 全ビット反転
	for i := range b.layers[0].h {
		b.layers[0].h[i] = -a.layers[0].h[i] - 1
	}
	if err := rebuildVisible(b.layers[0]); err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	got, err = weightAgreement(a, b)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	if got != 0.0 {
		t.Errorf("反転モデルの一致率が0.0でない: got = %f", got)
	}
}

// TestCloneModelIsIndependent は、複製が元と独立である事を確認する。
func TestCloneModelIsIndependent(t *testing.T) {
	src := newTestModel(t, 11)
	dst := cloneModel(src)

	dst.layers[0].h[0] = src.layers[0].h[0] + 1
	if err := rebuildVisible(dst.layers[0]); err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	if dst.layers[0].h[0] == src.layers[0].h[0] {
		t.Error("複製の変更が元へ波及している")
	}
	if len(src.protos) != len(dst.protos) {
		t.Error("プロトタイプが共有されていない")
	}
}
