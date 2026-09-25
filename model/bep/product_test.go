package bep_test

import (
	"math/rand/v2"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/sw965/crow/model/bep"
	"github.com/sw965/omw/mathx/bitsx"
)

func setRandomBias(d *bep.Dense, rng *rand.Rand) {
	for j := range d.Bias {
		d.Bias[j] = int32(rng.IntN(21) - 10)
	}
}

func newTestProductDense(t *testing.T, outs, fanIn, numBranches int, rng *rand.Rand) *bep.ProductDense {
	t.Helper()
	p, err := bep.NewProductDense(outs, fanIn, numBranches, rng)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	for _, d := range p.Branches {
		setRandomBias(d, rng)
	}
	return p
}

func naiveZ(t *testing.T, d *bep.Dense, x *bitsx.Matrix, r, j int) int {
	t.Helper()
	fanIn := d.W.Cols()
	match := 0
	for c := range fanIn {
		if mustBit(t, x, r, c) == mustBit(t, d.W, j, c) {
			match++
		}
	}
	return 2*match - fanIn + int(d.Bias[j])
}

func signBitOf(z int) uint64 {
	if z >= 0 {
		return 1
	}
	return 0
}

func absOf(v int) int {
	return max(v, -v)
}

func TestNewProductDense(t *testing.T) {
	t.Run("正常_既定値", func(t *testing.T) {
		p, err := bep.NewProductDense(10, 20, 2, rand.New(rand.NewPCG(1, 2)))
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if len(p.Branches) != 2 {
			t.Fatalf("枝の数の不一致: got = %d, want = 2", len(p.Branches))
		}
		// 入力数 20 → isqrt 4 × 1/2 = 2
		if p.MaxUpdateAbsZ != 2 {
			t.Errorf("MaxUpdateAbsZ の不一致: got = %d, want = 2", p.MaxUpdateAbsZ)
		}
		for b, d := range p.Branches {
			if d.W.Rows() != 10 || d.W.Cols() != 20 {
				t.Errorf("Branches[%d] の形状の不一致: got = (%d, %d), want = (10, 20)", b, d.W.Rows(), d.W.Cols())
			}
			assertDenseConsistent(t, d)
		}
		// 入力数 784 → isqrt 28 × 1/8 = 3
		wide, err := bep.NewProductDense(10, 784, 2, rand.New(rand.NewPCG(1, 2)))
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if wide.MarginAbsZ != 3 {
			t.Errorf("MarginAbsZ の不一致: got = %d, want = 3", wide.MarginAbsZ)
		}
		if p.Branches[0].W.Equal(p.Branches[1].W) {
			t.Error("枝の重みが同じ値で初期化されている")
		}
		if err := p.Validate(); err != nil {
			t.Errorf("予期せぬエラー: %v", err)
		}
	})

	t.Run("異常_枝の数が2未満", func(t *testing.T) {
		for _, n := range []int{-1, 0, 1} {
			if _, err := bep.NewProductDense(10, 20, n, rand.New(rand.NewPCG(1, 2))); err == nil {
				t.Errorf("numBranches = %d でエラーにならない", n)
			}
		}
	})
}

func TestProductDensePredict(t *testing.T) {
	const xRows, fanIn, outs = 3, 70, 90
	for _, numBranches := range []int{2, 3} {
		rng := rand.New(rand.NewPCG(11, 12))
		p := newTestProductDense(t, outs, fanIn, numBranches, rng)
		x, err := bitsx.NewRandMatrix(xRows, fanIn, rng)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}

		y, err := p.Predict(x)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		branchYs := make([]*bitsx.Matrix, numBranches)
		for b, d := range p.Branches {
			branchYs[b], err = d.Predict(x)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
		}
		for r := range xRows {
			for j := range outs {
				// ±1 の積は、符号ビットの XNOR を順に畳み込んだもの(-1 の個数が偶数なら +1)
				want := uint64(1)
				for _, by := range branchYs {
					want = 1 ^ want ^ mustBit(t, by, r, j)
				}
				if got := mustBit(t, y, r, j); got != want {
					t.Fatalf("枝 %d 本: 出力が枝の符号の積と不一致 (r = %d, j = %d): got = %d, want = %d", numBranches, r, j, got, want)
				}
			}
		}

		forwardY, _, err := p.Forward(x)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if !forwardY.Equal(y) {
			t.Errorf("枝 %d 本: Forward が Predict と一致しない", numBranches)
		}
	}
}

func TestProductDenseSingleBranchMatchesDense(t *testing.T) {
	const xRows, fanIn, outs = 2, 70, 90
	d, err := bep.NewDense(outs, fanIn, rand.New(rand.NewPCG(61, 62)))
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	branch, err := bep.NewDense(outs, fanIn, rand.New(rand.NewPCG(61, 62)))
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	setRandomBias(d, rand.New(rand.NewPCG(63, 64)))
	setRandomBias(branch, rand.New(rand.NewPCG(63, 64)))
	// NewProductDense は1本を受け付けないので直接組み立てる
	p := &bep.ProductDense{Branches: []*bep.Dense{branch}, MaxUpdateAbsZ: d.MaxUpdateAbsZ, MarginAbsZ: d.MarginAbsZ}

	dataRNG := rand.New(rand.NewPCG(65, 66))
	for step := range uint64(5) {
		x, err := bitsx.NewRandMatrix(xRows, fanIn, dataRNG)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		tgt, err := bitsx.NewRandMatrix(xRows, outs, dataRNG)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}

		yD, bwD, err := d.Forward(x)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		yP, bwP, err := p.Forward(x)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if !yP.Equal(yD) {
			t.Fatalf("step %d: Forward の出力が Dense と不一致", step)
		}

		recD, err := d.NewBatchRecord(1, xRows)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		recP, err := p.NewBatchRecord(1, xRows)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		inD, err := bwD(tgt, recD, 0)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		inP, err := bwP(tgt, recP, 0)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if !inP.Equal(inD) {
			t.Fatalf("step %d: 前層への目標が Dense と不一致", step)
		}

		deltasD, err := d.BatchDeltas(recD)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		deltasP, err := p.BatchDeltas(recP)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if len(deltasP) != len(deltasD) {
			t.Fatalf("step %d: Deltas の数が不一致: got = %d, want = %d", step, len(deltasP), len(deltasD))
		}
		for k := range deltasD {
			if !slices.Equal(deltasP[k], deltasD[k]) {
				t.Fatalf("step %d: Deltas[%d] が Dense と不一致", step, k)
			}
		}

		deltasD.Sign()
		deltasP.Sign()
		if err := d.Update(deltasD, 1, rand.New(rand.NewPCG(step, 2))); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if err := p.Update(deltasP, 1, rand.New(rand.NewPCG(step, 2))); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if !branch.W.Equal(d.W) || !branch.WT.Equal(d.WT) || !slices.Equal(branch.H, d.H) || !slices.Equal(branch.Bias, d.Bias) {
			t.Fatalf("step %d: Update 後の重みかバイアスが Dense と不一致", step)
		}
	}
}

type productExpectation struct {
	// 出力が食い違っていないニューロンは -1
	cheapest []int
	minAbsZ  []int
	want     []uint64
	targets  [][]uint64
}

func expectProduct(t *testing.T, p *bep.ProductDense, x, tgt *bitsx.Matrix) productExpectation {
	t.Helper()
	rows, outs := x.Rows(), p.Branches[0].W.Rows()
	e := productExpectation{
		cheapest: make([]int, rows*outs),
		minAbsZ:  make([]int, rows*outs),
		want:     make([]uint64, rows*outs),
		targets:  make([][]uint64, len(p.Branches)),
	}
	for b := range p.Branches {
		e.targets[b] = make([]uint64, rows*outs)
	}
	for r := range rows {
		for j := range outs {
			i := r*outs + j
			zs := make([]int, len(p.Branches))
			y := uint64(1)
			for b, d := range p.Branches {
				zs[b] = naiveZ(t, d, x, r, j)
				y = 1 ^ y ^ signBitOf(zs[b])
				e.targets[b][i] = signBitOf(zs[b])
			}
			e.cheapest[i] = -1
			if y == mustBit(t, tgt, r, j) {
				continue
			}
			cheapest := 0
			for b, z := range zs {
				if absOf(z) < absOf(zs[cheapest]) {
					cheapest = b
				}
			}
			e.cheapest[i] = cheapest
			e.minAbsZ[i] = absOf(zs[cheapest])
			e.want[i] = 1 - signBitOf(zs[cheapest])
			e.targets[cheapest][i] = e.want[i]
		}
	}
	return e
}

func TestProductDenseBackward(t *testing.T) {
	const fanIn, outs = 70, 90
	newSample := func(t *testing.T, xRows, maxUpdateAbsZ, marginAbsZ int) (*bep.ProductDense, *bitsx.Matrix, *bitsx.Matrix, *bitsx.Matrix, bep.Deltas) {
		t.Helper()
		rng := rand.New(rand.NewPCG(21, 22))
		p := newTestProductDense(t, outs, fanIn, 2, rng)
		p.MaxUpdateAbsZ = maxUpdateAbsZ
		p.MarginAbsZ = marginAbsZ
		x, err := bitsx.NewRandMatrix(xRows, fanIn, rng)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		tgt, err := bitsx.NewRandMatrix(xRows, outs, rng)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		_, bw, err := p.Forward(x)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		rec, err := p.NewBatchRecord(1, xRows)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		inTarget, err := bw(tgt, rec, 0)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		deltas, err := p.BatchDeltas(rec)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		return p, x, tgt, inTarget, deltas
	}

	t.Run("正常_食い違ったニューロンは最も|z|の小さい枝だけを直す", func(t *testing.T) {
		const xRows = 2
		// 上限が |z| の最大値なら、食い違ったニューロンは全て選ばれる。正解のニューロンを押さないようマージンは無効にする
		p, x, tgt, _, deltas := newSample(t, xRows, 2*fanIn, 0)
		e := expectProduct(t, p, x, tgt)

		want := make(bep.Deltas, 0, 4)
		for range p.Branches {
			want = append(want, make(bep.Delta, outs*fanIn), make(bep.Delta, outs))
		}
		mismatches := 0
		for r := range xRows {
			for j := range outs {
				i := r*outs + j
				b := e.cheapest[i]
				if b < 0 {
					continue
				}
				mismatches++
				want[2*b+1][j] += int16(2*int(e.want[i]) - 1)
				for c := range fanIn {
					v := int16(-1)
					if mustBit(t, x, r, c) == e.want[i] {
						v = 1
					}
					want[2*b][j*fanIn+c] += v
				}
			}
		}
		if mismatches == 0 {
			t.Fatal("食い違ったニューロンがなく、テストになっていない")
		}
		for k := range want {
			if !slices.Equal(deltas[k], want[k]) {
				t.Fatalf("Deltas[%d] が素朴な計算と一致しない", k)
			}
		}
	})

	t.Run("正常_前層への目標は各枝の希望出力への票の合計の符号", func(t *testing.T) {
		const xRows = 2
		// 直すニューロンが一部だけでも、票は食い違った全てのニューロンの向きで数え、マージンで押すニューロンは票を変えない
		p, x, tgt, inTarget, _ := newSample(t, xRows, 8, 6)
		e := expectProduct(t, p, x, tgt)
		for r := range xRows {
			for c := range fanIn {
				votes := 0
				for b, d := range p.Branches {
					for j := range outs {
						if mustBit(t, d.W, j, c) == e.targets[b][r*outs+j] {
							votes++
						} else {
							votes--
						}
					}
				}
				if got, want := mustBit(t, inTarget, r, c), signBitOf(votes); got != want {
					t.Fatalf("前層への目標の不一致 (r = %d, c = %d): got = %d, want = %d (票 = %d)", r, c, got, want, votes)
				}
			}
		}
	})

	t.Run("正常_枝の|z|の最小値がMaxUpdateAbsZ以下の食い違いだけを直す", func(t *testing.T) {
		const maxUpdateAbsZ = 8
		p, x, tgt, _, deltas := newSample(t, 1, maxUpdateAbsZ, 0)
		e := expectProduct(t, p, x, tgt)
		selected, skipped := 0, 0
		for j := range outs {
			touched := -1
			for b := range p.Branches {
				if deltas[2*b+1][j] != 0 {
					if touched >= 0 {
						t.Fatalf("j = %d: 2本の枝が両方とも直されている", j)
					}
					touched = b
				}
			}
			wantTouched := -1
			if e.cheapest[j] >= 0 && e.minAbsZ[j] <= maxUpdateAbsZ {
				wantTouched = e.cheapest[j]
				selected++
			} else if e.cheapest[j] >= 0 {
				skipped++
			}
			if touched != wantTouched {
				t.Fatalf("j = %d (最小の|z| = %d): 直した枝の不一致: got = %d, want = %d", j, e.minAbsZ[j], touched, wantTouched)
			}
		}
		if selected == 0 || skipped == 0 {
			t.Fatalf("しきい値の両側に食い違いがなく、テストになっていない: selected = %d, skipped = %d", selected, skipped)
		}
	})

	t.Run("正常_出力が正解でも枝の|z|の最小値がMarginAbsZ未満なら最も|z|の小さい枝を今の符号の向きへ押す", func(t *testing.T) {
		const maxUpdateAbsZ, marginAbsZ = 8, 6
		p, x, tgt, _, deltas := newSample(t, 1, maxUpdateAbsZ, marginAbsZ)
		pushed, kept := 0, 0
		for j := range outs {
			zs := make([]int, len(p.Branches))
			y := uint64(1)
			cheapest := 0
			for b, d := range p.Branches {
				zs[b] = naiveZ(t, d, x, 0, j)
				y = 1 ^ y ^ signBitOf(zs[b])
				if absOf(zs[b]) < absOf(zs[cheapest]) {
					cheapest = b
				}
			}
			if y != mustBit(t, tgt, 0, j) {
				continue
			}
			for b := range p.Branches {
				var want int16
				if b == cheapest && absOf(zs[b]) < marginAbsZ {
					want = int16(2*int(signBitOf(zs[b])) - 1)
				}
				if got := deltas[2*b+1][j]; got != want {
					t.Fatalf("j = %d, 枝 %d (z = %v): バイアスのデルタの不一致: got = %d, want = %d", j, b, zs, got, want)
				}
			}
			if absOf(zs[cheapest]) < marginAbsZ {
				pushed++
			} else {
				kept++
			}
		}
		if pushed == 0 || kept == 0 {
			t.Fatalf("マージンの両側に正解のニューロンがなく、テストになっていない: pushed = %d, kept = %d", pushed, kept)
		}
	})

	t.Run("正常_複数サンプルの記録は1サンプルずつの合計と一致", func(t *testing.T) {
		const xRows, n = 2, 5
		rng := rand.New(rand.NewPCG(31, 32))
		p := newTestProductDense(t, outs, fanIn, 2, rng)
		batch, err := p.NewBatchRecord(n, xRows)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		var want bep.Deltas
		for s := range n {
			x, err := bitsx.NewRandMatrix(xRows, fanIn, rng)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			tgt, err := bitsx.NewRandMatrix(xRows, outs, rng)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			_, bw, err := p.Forward(x)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			single, err := p.NewBatchRecord(1, xRows)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			if _, err := bw(tgt, single, 0); err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			singleDeltas, err := p.BatchDeltas(single)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			if want == nil {
				for _, dl := range singleDeltas {
					want = append(want, slices.Clone(dl))
				}
			} else {
				for k, dl := range singleDeltas {
					for i, v := range dl {
						want[k][i] += v
					}
				}
			}
			if _, err := bw(tgt, batch, s); err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
		}
		got, err := p.BatchDeltas(batch)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		for k := range want {
			if !slices.Equal(got[k], want[k]) {
				t.Fatalf("Deltas[%d] が1サンプルずつの合計と一致しない", k)
			}
		}

		if err := batch.Clear(); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		cleared, err := p.BatchDeltas(batch)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		for k, dl := range cleared {
			if slices.ContainsFunc(dl, func(v int16) bool { return v != 0 }) {
				t.Fatalf("Clear後も Deltas[%d] が非ゼロ", k)
			}
		}
	})

	t.Run("異常_別の型のBatchRecord", func(t *testing.T) {
		rng := rand.New(rand.NewPCG(41, 42))
		p := newTestProductDense(t, outs, fanIn, 2, rng)
		x, err := bitsx.NewRandMatrix(1, fanIn, rng)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		y, bw, err := p.Forward(x)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if _, err := bw(y, otherRecord{}, 0); err == nil {
			t.Error("backward がエラーにならない")
		}
		if _, err := p.BatchDeltas(otherRecord{}); err == nil {
			t.Error("BatchDeltas がエラーにならない")
		}
	})
}

func TestProductDenseUpdate(t *testing.T) {
	t.Run("正常_枝ごとのデルタをその枝だけに反映", func(t *testing.T) {
		const fanIn, outs = 70, 90
		rng := rand.New(rand.NewPCG(71, 72))
		p := newTestProductDense(t, outs, fanIn, 2, rng)
		beforeH0 := slices.Clone(p.Branches[0].H)
		beforeBias0 := slices.Clone(p.Branches[0].Bias)
		before := slices.Clone(p.Branches[1].H)
		beforeBias := slices.Clone(p.Branches[1].Bias)

		deltas := bep.Deltas{make(bep.Delta, outs*fanIn), make(bep.Delta, outs), make(bep.Delta, outs*fanIn), make(bep.Delta, outs)}
		for i := range deltas[0] {
			deltas[0][i] = 1
		}
		for j := range deltas[1] {
			deltas[1][j] = 1
		}
		// 更新確率 1 で、枝0の全要素が +1 される
		if err := p.Update(deltas, 0, rng); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		for _, d := range p.Branches {
			assertDenseConsistent(t, d)
		}
		if !slices.Equal(p.Branches[1].H, before) || !slices.Equal(p.Branches[1].Bias, beforeBias) {
			t.Error("デルタが 0 の枝1 が変わった")
		}
		for i, h := range p.Branches[0].H {
			if want := beforeH0[i] + 1; h != want {
				t.Fatalf("枝0 の H[%d] の不一致: got = %d, want = %d", i, h, want)
			}
		}
		for j, b := range p.Branches[0].Bias {
			if want := beforeBias0[j] + 1; b != want {
				t.Fatalf("枝0 の Bias[%d] の不一致: got = %d, want = %d", j, b, want)
			}
		}
	})

	t.Run("異常_deltasの数が枝の数と合わない", func(t *testing.T) {
		p := newTestProductDense(t, 10, 20, 2, rand.New(rand.NewPCG(1, 2)))
		if err := p.Update(bep.Deltas{make(bep.Delta, 200), make(bep.Delta, 10)}, 0, rand.New(rand.NewPCG(1, 2))); err == nil {
			t.Error("エラーを期待したが、nilが返された")
		}
	})
}

func TestProductDenseValidate(t *testing.T) {
	tests := []struct {
		name    string
		modify  func(p *bep.ProductDense)
		wantErr string
	}{
		{name: "正常_既定値", modify: func(*bep.ProductDense) {}},
		{name: "異常_枝が1本", modify: func(p *bep.ProductDense) { p.Branches = p.Branches[:1] }, wantErr: "Branches"},
		{name: "異常_枝がnil", modify: func(p *bep.ProductDense) { p.Branches[1] = nil }, wantErr: "nil"},
		{
			name: "異常_枝の形状が不一致",
			modify: func(p *bep.ProductDense) {
				d, err := bep.NewDense(10, 21, rand.New(rand.NewPCG(3, 4)))
				if err != nil {
					panic(err)
				}
				p.Branches[1] = d
			},
			wantErr: "形状",
		},
		{name: "異常_MaxUpdateAbsZが0", modify: func(p *bep.ProductDense) { p.MaxUpdateAbsZ = 0 }, wantErr: "MaxUpdateAbsZ"},
		{name: "異常_MaxUpdateAbsZが|z|の最大値を超える", modify: func(p *bep.ProductDense) { p.MaxUpdateAbsZ = 41 }, wantErr: "MaxUpdateAbsZ"},
		{name: "異常_MarginAbsZが負", modify: func(p *bep.ProductDense) { p.MarginAbsZ = -1 }, wantErr: "MarginAbsZ"},
		{name: "異常_MarginAbsZが|z|の最大値を超える", modify: func(p *bep.ProductDense) { p.MarginAbsZ = 41 }, wantErr: "MarginAbsZ"},
		{name: "異常_枝のMarginAbsZが負", modify: func(p *bep.ProductDense) { p.Branches[1].MarginAbsZ = -1 }, wantErr: "MarginAbsZ"},
		{name: "異常_枝のBiasが範囲外", modify: func(p *bep.ProductDense) { p.Branches[0].Bias[3] = 21 }, wantErr: "Bias"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := bep.NewProductDense(10, 20, 2, rand.New(rand.NewPCG(1, 2)))
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			tt.modify(p)
			err = p.Validate()
			if tt.wantErr == "" {
				if err != nil {
					t.Errorf("予期せぬエラー: %v", err)
				}
				return
			}
			if err == nil {
				t.Fatal("エラーを期待したが、nilが返された")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("エラーに %q が含まれない: %v", tt.wantErr, err)
			}
		})
	}
}

func newTestProductModel(t *testing.T) (bep.Model, *rand.Rand) {
	t.Helper()
	rng := rand.New(rand.NewPCG(81, 82))
	model := bep.Model{XRows: 1, XCols: 64}
	if err := model.AppendProductDenseLayer(48, 2, rng); err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	if err := model.AppendDenseLayer(32, rng); err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	if err := model.SetClassPrototypes(4, rng); err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	return model, rng
}

func TestModelAppendProductDenseLayer(t *testing.T) {
	t.Run("正常_前の層の出力数を入力数にする", func(t *testing.T) {
		model, _ := newTestProductModel(t)
		p, ok := model.Backbone[0].(*bep.ProductDense)
		if !ok {
			t.Fatal("先頭の層が *bep.ProductDense ではない")
		}
		if cols := p.Branches[0].W.Cols(); cols != 64 {
			t.Errorf("積の層の入力数の不一致: got = %d, want = 64", cols)
		}
		if cols := model.Backbone[1].(*bep.Dense).W.Cols(); cols != 48 {
			t.Errorf("後ろの層の入力数の不一致: got = %d, want = 48", cols)
		}
		yRows, yCols, err := model.Backbone.OutputShape(model.XRows, model.XCols)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if yRows != 1 || yCols != 32 {
			t.Errorf("出力形状の不一致: got = (%d, %d), want = (1, 32)", yRows, yCols)
		}
	})

	t.Run("異常_XRowsとXColsが未設定", func(t *testing.T) {
		var model bep.Model
		if err := model.AppendProductDenseLayer(8, 2, rand.New(rand.NewPCG(1, 2))); err == nil {
			t.Error("エラーを期待したが、nilが返された")
		}
	})

	t.Run("異常_枝の数が2未満", func(t *testing.T) {
		model := bep.Model{XRows: 1, XCols: 64}
		if err := model.AppendProductDenseLayer(8, 1, rand.New(rand.NewPCG(1, 2))); err == nil {
			t.Error("エラーを期待したが、nilが返された")
		}
		if len(model.Backbone) != 0 {
			t.Errorf("エラーなのに層が追加された: len(Backbone) = %d", len(model.Backbone))
		}
	})
}

func TestSequenceSettersWithProductDense(t *testing.T) {
	t.Run("正常_SetMaxUpdateAbsZScaleは積の層にも設定", func(t *testing.T) {
		model, _ := newTestProductModel(t)
		if err := model.Backbone.SetMaxUpdateAbsZScale(1, 1); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		// 積の層は入力数 64 → isqrt 8、通常の層は入力数 48 → isqrt 6
		if p := model.Backbone[0].(*bep.ProductDense); p.MaxUpdateAbsZ != 8 {
			t.Errorf("積の層の MaxUpdateAbsZ の不一致: got = %d, want = 8", p.MaxUpdateAbsZ)
		}
		if d := model.Backbone[1].(*bep.Dense); d.MaxUpdateAbsZ != 6 {
			t.Errorf("通常の層の MaxUpdateAbsZ の不一致: got = %d, want = 6", d.MaxUpdateAbsZ)
		}
	})

	t.Run("正常_SetMarginAbsZScaleは積の層にも設定", func(t *testing.T) {
		model, _ := newTestProductModel(t)
		if err := model.Backbone.SetMarginAbsZScale(1, 2); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		// 積の層は入力数 64 → isqrt 8 × 1/2 = 4、通常の層は入力数 48 → isqrt 6 × 1/2 = 3
		if p := model.Backbone[0].(*bep.ProductDense); p.MarginAbsZ != 4 {
			t.Errorf("積の層の MarginAbsZ の不一致: got = %d, want = 4", p.MarginAbsZ)
		}
		if d := model.Backbone[1].(*bep.Dense); d.MarginAbsZ != 3 {
			t.Errorf("通常の層の MarginAbsZ の不一致: got = %d, want = 3", d.MarginAbsZ)
		}
	})

	t.Run("異常_積の層で倍率が大きすぎるなら全層を変えない", func(t *testing.T) {
		model, _ := newTestProductModel(t)
		p := model.Backbone[0].(*bep.ProductDense)
		dense := model.Backbone[1].(*bep.Dense)
		beforeP, beforeD := p.MarginAbsZ, dense.MarginAbsZ
		if err := model.Backbone.SetMarginAbsZScale(100, 1); err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
		if p.MarginAbsZ != beforeP || dense.MarginAbsZ != beforeD {
			t.Errorf("エラーなのに MarginAbsZ が変わった: got = (%d, %d), want = (%d, %d)", p.MarginAbsZ, dense.MarginAbsZ, beforeP, beforeD)
		}
	})
}

func TestModelWithProductDense(t *testing.T) {
	t.Run("正常_保存と読み込みで出力と設定が変わらない", func(t *testing.T) {
		model, rng := newTestProductModel(t)
		p := model.Backbone[0].(*bep.ProductDense)
		p.MaxUpdateAbsZ = 3
		p.Branches[1].Bias[0] = -7
		x := newTestInput(t, rng)
		want, err := model.PredictLogits(x)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}

		path := filepath.Join(t.TempDir(), "model.gob")
		if err := model.Save(path); err != nil {
			t.Fatalf("保存失敗: %v", err)
		}
		loaded, err := bep.LoadModel(path)
		if err != nil {
			t.Fatalf("読み込み失敗: %v", err)
		}
		got, err := loaded.PredictLogits(x)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if !slices.Equal(got, want) {
			t.Errorf("logitsの不一致: got = %v, want = %v", got, want)
		}
		lp, ok := loaded.Backbone[0].(*bep.ProductDense)
		if !ok {
			t.Fatal("読み込んだ先頭の層が *bep.ProductDense ではない")
		}
		if lp.MaxUpdateAbsZ != 3 || lp.Branches[1].Bias[0] != -7 {
			t.Errorf("層の設定が保存されていない: got = (MaxUpdateAbsZ %d, Bias[0] %d), want = (3, -7)", lp.MaxUpdateAbsZ, lp.Branches[1].Bias[0])
		}
	})

	t.Run("正常_学習できる", func(t *testing.T) {
		model, rng := newTestProductModel(t)
		const n = 64
		xs := make(bitsx.Matrices, n)
		labels := make([]int, n)
		for i := range n {
			xs[i] = newTestInput(t, rng)
			labels[i] = rng.IntN(4)
		}
		before := slices.Clone(model.Backbone[0].(*bep.ProductDense).Branches[0].H)

		trainer := newTestTrainer(t, &model)
		trainer.MiniBatchSize = 16
		for range 3 {
			if err := trainer.Train(xs, labels); err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
		}
		p := model.Backbone[0].(*bep.ProductDense)
		for _, d := range p.Branches {
			assertDenseConsistent(t, d)
		}
		if slices.Equal(p.Branches[0].H, before) {
			t.Error("学習しても積の層の枝0 が変わらない")
		}
	})

	t.Run("異常_Trainer.Validateは積の層の不正を層番号付きで返す", func(t *testing.T) {
		model, _ := newTestProductModel(t)
		model.Backbone[0].(*bep.ProductDense).MaxUpdateAbsZ = 0
		err := newTestTrainer(t, &model).Validate()
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
		if !strings.Contains(err.Error(), "layer 0") || !strings.Contains(err.Error(), "MaxUpdateAbsZ") {
			t.Errorf("エラーに層番号か MaxUpdateAbsZ が含まれない: %v", err)
		}
	})
}
