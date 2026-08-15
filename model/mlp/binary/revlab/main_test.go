package main

import (
	"math/rand/v2"
	"testing"

	"github.com/sw965/crow/model/mlp/binary"
	"github.com/sw965/omw/mathx/bitsx"
)

func testMatrix(t *testing.T, rows, cols int, rng *rand.Rand) *bitsx.Matrix {
	t.Helper()
	m, err := bitsx.NewRandMatrix(rows, cols, 0, rng)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestPMMulTruthTable(t *testing.T) {
	a, err := bitsx.NewZerosMatrix(1, 4)
	if err != nil {
		t.Fatal(err)
	}
	b, err := bitsx.NewZerosMatrix(1, 4)
	if err != nil {
		t.Fatal(err)
	}
	// (-1,-1), (-1,+1), (+1,-1), (+1,+1)
	if err := a.Set(0, 2); err != nil {
		t.Fatal(err)
	}
	if err := a.Set(0, 3); err != nil {
		t.Fatal(err)
	}
	if err := b.Set(0, 1); err != nil {
		t.Fatal(err)
	}
	if err := b.Set(0, 3); err != nil {
		t.Fatal(err)
	}
	got, err := pmMul(a, b)
	if err != nil {
		t.Fatal(err)
	}
	want := []uint64{1, 0, 0, 1}
	for col, bit := range want {
		gotBit, err := got.Bit(0, col)
		if err != nil {
			t.Fatal(err)
		}
		if gotBit != bit {
			t.Fatalf("col=%d: got=%d want=%d", col, gotBit, bit)
		}
	}
}

func TestPadStatePreservesInput(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	x := testMatrix(t, 2, 784, rng)
	padded, err := padState(x, 1024)
	if err != nil {
		t.Fatal(err)
	}
	for r := range x.Rows() {
		for c := range x.Cols() {
			got, err := padded.Bit(r, c)
			if err != nil {
				t.Fatal(err)
			}
			want, err := x.Bit(r, c)
			if err != nil {
				t.Fatal(err)
			}
			if got != want {
				t.Fatalf("入力部分が変化した: row=%d col=%d", r, c)
			}
		}
		for c := x.Cols(); c < padded.Cols(); c++ {
			got, err := padded.Bit(r, c)
			if err != nil {
				t.Fatal(err)
			}
			if got != 0 {
				t.Fatalf("padding が -1 でない: row=%d col=%d", r, c)
			}
		}
	}
}

func TestSplitJoinRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	x := testMatrix(t, 3, 1024, rng)
	left, right, err := splitHalves(x)
	if err != nil {
		t.Fatal(err)
	}
	got, err := joinHalves(left, right)
	if err != nil {
		t.Fatal(err)
	}
	if !got.Equal(x) {
		t.Fatal("split と join で元に戻らない")
	}
}

func TestCouplingRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	block, err := newCoupling(1024, true, true, 0.1, rng)
	if err != nil {
		t.Fatal(err)
	}
	for i := range block.f.bias {
		block.f.bias[i] = int32(i%9 - 4)
	}
	for range 10 {
		x := testMatrix(t, 1, 1024, rng)
		y, err := block.predict(x)
		if err != nil {
			t.Fatal(err)
		}
		got, err := block.inverse(y)
		if err != nil {
			t.Fatal(err)
		}
		if !got.Equal(x) {
			t.Fatal("可逆ブロックの逆変換で元に戻らない")
		}
	}
}

func TestCouplingBackwardReturnsExactPreimage(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	block, err := newCoupling(1024, true, true, 0.1, rng)
	if err != nil {
		t.Fatal(err)
	}
	x := testMatrix(t, 1, 1024, rng)
	_, trace, err := block.forward(x, 0, rng)
	if err != nil {
		t.Fatal(err)
	}
	target := testMatrix(t, 1, 1024, rng)
	preimage, err := block.backward(trace, target, block.f.newDelta())
	if err != nil {
		t.Fatal(err)
	}
	got, err := block.predict(preimage)
	if err != nil {
		t.Fatal(err)
	}
	if !got.Equal(target) {
		t.Fatal("backward が返した希望入力を順伝播しても希望出力にならない")
	}
}

func TestFixedCouplingDoesNotCollectDelta(t *testing.T) {
	rng := rand.New(rand.NewPCG(17, 18))
	block, err := newCoupling(128, false, true, 0.1, rng)
	if err != nil {
		t.Fatal(err)
	}
	x := testMatrix(t, 1, 128, rng)
	_, trace, err := block.forward(x, 0, rng)
	if err != nil {
		t.Fatal(err)
	}
	target := testMatrix(t, 1, 128, rng)
	dl := block.f.newDelta()
	if _, err := block.backward(trace, target, dl); err != nil {
		t.Fatal(err)
	}
	for i, value := range dl.w {
		if value != 0 {
			t.Fatalf("固定ブロックの重みdeltaが非ゼロ: index=%d value=%d", i, value)
		}
	}
	for i, value := range dl.b {
		if value != 0 {
			t.Fatalf("固定ブロックのバイアスdeltaが非ゼロ: index=%d value=%d", i, value)
		}
	}
}

func TestDenseUpdateMovesOnlySelectedParameterKind(t *testing.T) {
	for _, tc := range []struct {
		name       string
		biasChoice float32
		wantBias   bool
	}{
		{name: "バイアスのみ", biasChoice: 1, wantBias: true},
		{name: "重みのみ", biasChoice: 0, wantBias: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewPCG(19, 20))
			layer, err := newDense(2, 64, true, tc.biasChoice, rng)
			if err != nil {
				t.Fatal(err)
			}
			beforeH := append([]int8(nil), layer.h...)
			dl := layer.newDelta()
			for i := range dl.w {
				dl.w[i] = 1
			}
			for i := range dl.b {
				dl.b[i] = 1
			}
			if err := layer.update(dl, 1, rng); err != nil {
				t.Fatal(err)
			}
			biasMoved := false
			for _, value := range layer.bias {
				biasMoved = biasMoved || value != 0
			}
			weightMoved := false
			for i, value := range layer.h {
				weightMoved = weightMoved || value != beforeH[i]
			}
			if biasMoved != tc.wantBias || weightMoved == tc.wantBias {
				t.Fatalf("biasMoved=%t weightMoved=%t wantBias=%t", biasMoved, weightMoved, tc.wantBias)
			}
		})
	}
}

func TestBackboneRoundTrip(t *testing.T) {
	m, err := newModel("reversible", 784, 1024, 128, 4, true, true, 0.1, 9)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(10, 11))
	for range 5 {
		x := testMatrix(t, 1, 784, rng)
		state, err := m.backbone(x)
		if err != nil {
			t.Fatal(err)
		}
		restored, err := m.inverseBackbone(state)
		if err != nil {
			t.Fatal(err)
		}
		padded, err := padState(x, 1024)
		if err != nil {
			t.Fatal(err)
		}
		if !restored.Equal(padded) {
			t.Fatal("複数ブロックのバックボーンが可逆でない")
		}
	}
}

func TestSatisfiesUpdateCriterionMatchesCrow(t *testing.T) {
	rng := rand.New(rand.NewPCG(12, 13))
	prototypes := make(bitsx.Matrices, numClasses)
	for i := range prototypes {
		prototypes[i] = testMatrix(t, 1, 128, rng)
	}
	for _, margin := range []float32{0, 0.1, 0.5, 1} {
		for range 50 {
			y := testMatrix(t, 1, 128, rng)
			label := rng.IntN(numClasses)
			got, err := satisfiesUpdateCriterion(y, label, prototypes, margin)
			if err != nil {
				t.Fatal(err)
			}
			want, err := binary.SatisfiesUpdateCriterion(y, label, prototypes, margin)
			if err != nil {
				t.Fatal(err)
			}
			if got != want {
				t.Fatalf("更新判定が不一致: margin=%g got=%t want=%t", margin, got, want)
			}
		}
	}
}

func TestModesProduceExpectedShape(t *testing.T) {
	rng := rand.New(rand.NewPCG(14, 15))
	x := testMatrix(t, 1, 784, rng)
	for _, mode := range []string{"dense", "project", "reversible"} {
		m, err := newModel(mode, 784, 1024, 128, 2, true, false, 0, 16)
		if err != nil {
			t.Fatalf("mode=%s: %v", mode, err)
		}
		y, err := m.predict(x)
		if err != nil {
			t.Fatalf("mode=%s: %v", mode, err)
		}
		if y.Rows() != 1 || y.Cols() != 128 {
			t.Fatalf("mode=%s: shape=(%d,%d)", mode, y.Rows(), y.Cols())
		}
	}
}
