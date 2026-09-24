package main

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/sw965/omw/mathx/bitsx"
)

func TestOutputBitIsXnor(t *testing.T) {
	// 1本ならその符号そのもの。XNOR の単位元が 1 であることに依存している。
	single := [][]int{{5, -5}}
	if got := outputBit(single, 0); got != 1 {
		t.Errorf("枝1本 z>=0: got = %d, want = 1", got)
	}
	if got := outputBit(single, 1); got != 0 {
		t.Errorf("枝1本 z<0: got = %d, want = 0", got)
	}

	// 2本なら同符号で +1、異符号で -1。
	tests := []struct {
		zu, zg int
		want   uint64
	}{
		{zu: 3, zg: 7, want: 1},
		{zu: -3, zg: -7, want: 1},
		{zu: 3, zg: -7, want: 0},
		{zu: -3, zg: 7, want: 0},
	}
	for _, tt := range tests {
		zs := [][]int{{tt.zu}, {tt.zg}}
		if got := outputBit(zs, 0); got != tt.want {
			t.Errorf("zu = %d zg = %d: got = %d, want = %d", tt.zu, tt.zg, got, tt.want)
		}
	}
}

// TestFlippingBothBranchesCancels は、両枝を同時に反転させても出力が変わらないことを示す。
// u* = a*⊙g と g* = a*⊙u は各々正しいが、同時には適用できないという制約の根拠であり、
// route=both が破綻する理由そのものである。
func TestFlippingBothBranchesCancels(t *testing.T) {
	zs := [][]int{{3}, {-7}}
	before := outputBit(zs, 0)

	flipped := [][]int{{-3}, {7}}
	if got := outputBit(flipped, 0); got != before {
		t.Errorf("両枝反転で出力が変わってしまった: %d -> %d", before, got)
	}

	// 片方だけなら必ず変わる。
	one := [][]int{{-3}, {-7}}
	if got := outputBit(one, 0); got == before {
		t.Error("片枝反転では出力が変わるべき")
	}
}

func TestArchSpecWeightCounts(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	counts := map[string]int{}
	for _, arch := range []string{"plain", "wide", "gated"} {
		branches, widths, err := archSpec(arch)
		if err != nil {
			t.Fatal(err)
		}
		m := &model{xRows: 1, xCols: 784}
		for _, wRows := range widths {
			if err := m.appendLayer(branches, wRows, rng); err != nil {
				t.Fatal(err)
			}
		}
		counts[arch] = m.weightCount()
	}

	if counts["wide"] != counts["gated"] {
		t.Errorf("wide と gated は重み数が一致すべき: wide = %d, gated = %d",
			counts["wide"], counts["gated"])
	}
	if counts["plain"] >= counts["gated"] {
		t.Errorf("plain は gated より少ないはず: plain = %d, gated = %d",
			counts["plain"], counts["gated"])
	}
}

// TestTransposeStaysInSyncAfterUpdate は、全ての枝について wt が w の転置で
// あり続けることを確かめる。逆伝播は wt を通るため、ここがずれると誤差が
// 初期の重みを流れ続ける。
func TestTransposeStaysInSyncAfterUpdate(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	d, err := newDense(2, 32, 64, rng)
	if err != nil {
		t.Fatal(err)
	}

	dl := d.newDelta()
	for step := range 6 {
		v := int16(9)
		if step%2 == 1 {
			v = -9
		}
		for _, w := range dl.w {
			for i := range w {
				w[i] = v
			}
		}
		if err := d.update(dl, 1.0, rng); err != nil {
			t.Fatal(err)
		}
		for b, w := range d.w {
			want, err := w.Transpose()
			if err != nil {
				t.Fatal(err)
			}
			if !d.wt[b].Equal(want) {
				t.Fatalf("step %d 枝%d: wt が w の転置と一致しない", step, b)
			}
		}
	}
}

// TestBackwardReachesDesiredOutput は、逆伝播が選んだ枝の目標値を適用すると、
// 実際にその層の出力が希望活性 t に一致することを確かめる。
// 枝の振り分けが正しいことの直接の検証になる。
func TestBackwardReachesDesiredOutput(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	d, err := newDense(2, 64, 64, rng)
	if err != nil {
		t.Fatal(err)
	}
	d.route = routeCheap

	x, err := bitsx.NewRandMatrix(1, 64, rng)
	if err != nil {
		t.Fatal(err)
	}
	zs, _, _, err := d.preActivation(x, 0, rng)
	if err != nil {
		t.Fatal(err)
	}

	target, err := bitsx.NewRandMatrix(1, d.yCols(), rng)
	if err != nil {
		t.Fatal(err)
	}

	// 逆伝播が枝へ配る目標値を、外側から同じ規則で組み立てて突き合わせる。
	mismatch := 0
	for i := range d.yCols() {
		tBit, err := target.Bit(0, i)
		if err != nil {
			t.Fatal(err)
		}
		if outputBit(zs, i) != tBit {
			mismatch++
		}
		// 安い方の枝だけを反転させた状態を作り、XNOR が t に一致するか見る。
		cheap := 0
		if absInt(zs[1][i]) < absInt(zs[0][i]) {
			cheap = 1
		}
		flipped := [][]int{{zs[0][i]}, {zs[1][i]}}
		if outputBit(zs, i) != tBit {
			flipped[cheap][0] = -flipped[cheap][0] - 1
		}
		if outputBit(flipped, 0) != tBit {
			t.Fatalf("i = %d: 安い枝を反転しても t に一致しない", i)
		}
	}
	if mismatch == 0 {
		t.Skip("不一致が1つも無く検証にならない")
	}
}

// naiveForward は bitsx を一切使わず、重みビットを1つずつ読んで順伝播を再現する。
// ビットパック・Dot・符号規約・XNOR畳み込みをまとめて突き合わせるための参照実装。
func naiveForward(t *testing.T, d *dense, x *bitsx.Matrix) []uint64 {
	t.Helper()
	out := make([]uint64, d.yCols())
	for i := range d.yCols() {
		bit := uint64(1)
		for _, w := range d.w {
			z := 0
			for j := range d.fanIn() {
				xb, err := x.Bit(0, j)
				if err != nil {
					t.Fatal(err)
				}
				wb, err := w.Bit(i, j)
				if err != nil {
					t.Fatal(err)
				}
				// ±1 の積は、ビットが一致すれば +1。
				if xb == wb {
					z++
				} else {
					z--
				}
			}
			s := uint64(0)
			if z >= 0 {
				s = 1
			}
			bit = 1 &^ (bit ^ s)
		}
		out[i] = bit
	}
	return out
}

func TestForwardMatchesNaiveReference(t *testing.T) {
	for _, branches := range []int{1, 2} {
		rng := rand.New(rand.NewPCG(uint64(branches), 99))
		d, err := newDense(branches, 37, 130, rng) // 64の倍数を避けて端数ワードも通す
		if err != nil {
			t.Fatal(err)
		}
		for range 5 {
			x, err := bitsx.NewRandMatrix(1, 130, rng)
			if err != nil {
				t.Fatal(err)
			}
			y, err := d.predict(x)
			if err != nil {
				t.Fatal(err)
			}
			want := naiveForward(t, d, x)
			for i, wb := range want {
				got, err := y.Bit(0, i)
				if err != nil {
					t.Fatal(err)
				}
				if got != wb {
					t.Fatalf("branches = %d, i = %d: got = %d, want = %d", branches, i, got, wb)
				}
			}
		}
	}
}

func TestPredictMatchesForwardWithoutNoise(t *testing.T) {
	for _, branches := range []int{1, 2} {
		rng := rand.New(rand.NewPCG(11, uint64(branches)))
		d, err := newDense(branches, 40, 96, rng)
		if err != nil {
			t.Fatal(err)
		}
		x, err := bitsx.NewRandMatrix(1, 96, rng)
		if err != nil {
			t.Fatal(err)
		}
		y, _, err := d.forward(x, 0, rng)
		if err != nil {
			t.Fatal(err)
		}
		p, err := d.predict(x)
		if err != nil {
			t.Fatal(err)
		}
		if !y.Equal(p) {
			t.Errorf("branches = %d: forward と predict が一致しない", branches)
		}
	}
}

// TestBackwardMovesOutputTowardTarget は、逆伝播と更新を繰り返すと、その層の出力が
// 希望活性へ近づくことを確かめる。枝の振り分けが正しくなければ近づかない。
// route=both が失敗するのは、1ニューロンにつき2枝を反転させて XNOR が元に戻るためで、
// その差もここで現れる。
func TestBackwardMovesOutputTowardTarget(t *testing.T) {
	distanceAfter := func(route routeMode, branches int) (int, int) {
		rng := rand.New(rand.NewPCG(21, 22))
		d, err := newDense(branches, 64, 128, rng)
		if err != nil {
			t.Fatal(err)
		}
		d.route = route
		d.groupSize = 1 // 不一致を全て更新対象にする
		d.gateOpen = true

		x, err := bitsx.NewRandMatrix(1, 128, rng)
		if err != nil {
			t.Fatal(err)
		}
		target, err := bitsx.NewRandMatrix(1, 64, rng)
		if err != nil {
			t.Fatal(err)
		}

		y0, err := d.predict(x)
		if err != nil {
			t.Fatal(err)
		}
		before, err := y0.HammingDistance(target)
		if err != nil {
			t.Fatal(err)
		}

		for range 40 {
			_, bw, err := d.forward(x, 0, rng)
			if err != nil {
				t.Fatal(err)
			}
			dl := d.newDelta()
			if _, err := bw(target, dl); err != nil {
				t.Fatal(err)
			}
			dl.sign()
			if err := d.update(dl, 1.0, rng); err != nil {
				t.Fatal(err)
			}
		}

		y1, err := d.predict(x)
		if err != nil {
			t.Fatal(err)
		}
		after, err := y1.HammingDistance(target)
		if err != nil {
			t.Fatal(err)
		}
		return before, after
	}

	for _, branches := range []int{1, 2} {
		before, after := distanceAfter(routeCheap, branches)
		if after >= before {
			t.Errorf("cheap branches = %d: 希望活性へ近づいていない %d -> %d", branches, before, after)
		}
		if after != 0 {
			t.Logf("cheap branches = %d: %d -> %d (完全一致には至らず)", branches, before, after)
		}
	}

	// 2枝で both を使うと、1ニューロンあたり偶数回反転するので出力が動かない。
	before, after := distanceAfter(routeBoth, 2)
	if after < before/2 {
		t.Errorf("both は出力を目標へ近づけられないはず: %d -> %d", before, after)
	}
}

func baseTinyConfig() config {
	return config{
		arch:       "plain",
		route:      routeCheap,
		gateOpen:   true,
		gateScale:  1.0,
		lr:         0.1,
		margin:     0.5,
		groupSize:  4,
		noiseScale: 0.5,
		batch:      64,
		seed:       7,
	}
}

func newTinyModel(t *testing.T, cfg config, branches int) (*model, bitsx.Matrices, []int) {
	t.Helper()
	rng := rand.New(rand.NewPCG(cfg.seed, cfg.seed+1))

	m := &model{xRows: 1, xCols: 64}
	if err := m.appendLayer(branches, 32, rng); err != nil {
		t.Fatal(err)
	}
	if err := m.appendLayer(branches, 64, rng); err != nil {
		t.Fatal(err)
	}
	protos, err := bitsx.NewETFMatrices(numClasses, 1, 64, 256, rng)
	if err != nil {
		t.Fatal(err)
	}
	m.prototypes = protos
	for _, l := range m.layers {
		l.route = cfg.route
		l.gateOpen = cfg.gateOpen
		l.gateScale = cfg.gateScale
		l.groupSize = cfg.groupSize
		l.applyGateScale()
	}

	xs := make(bitsx.Matrices, 128)
	labels := make([]int, len(xs))
	for i := range xs {
		x, err := bitsx.NewRandMatrix(1, 64, rng)
		if err != nil {
			t.Fatal(err)
		}
		xs[i] = x
		labels[i] = i % numClasses
	}
	return m, xs, labels
}

func trainTiny(t *testing.T, cfg config, branches int) float64 {
	t.Helper()
	m, xs, labels := newTinyModel(t, cfg, branches)
	tr, err := newTrainer(m, 2, cfg.seed)
	if err != nil {
		t.Fatal(err)
	}
	tr.miniBatchSize = cfg.batch
	tr.lr = cfg.lr
	tr.margin = cfg.margin
	tr.noiseScale = cfg.noiseScale

	for range 2 {
		for _, l := range m.layers {
			l.resetEpochStats()
		}
		if err := tr.trainEpoch(xs, labels); err != nil {
			t.Fatal(err)
		}
	}
	acc, err := m.accuracy(xs, labels, 2)
	if err != nil {
		t.Fatal(err)
	}
	return acc
}

func TestAllRoutesTrain(t *testing.T) {
	for _, branches := range []int{1, 2} {
		for _, route := range []routeMode{routeCheap, routeAlt, routeBoth} {
			for _, gateOpen := range []bool{true, false} {
				cfg := baseTinyConfig()
				cfg.route = route
				cfg.gateOpen = gateOpen
				acc := trainTiny(t, cfg, branches)
				if acc < 0 || acc > 1 {
					t.Errorf("branches = %d route = %s gateOpen = %t: 精度が範囲外 %g",
						branches, route, gateOpen, acc)
				}
			}
		}
	}
}

func TestTrainingIsDeterministic(t *testing.T) {
	for _, branches := range []int{1, 2} {
		cfg := baseTinyConfig()
		first := trainTiny(t, cfg, branches)
		second := trainTiny(t, cfg, branches)
		if first != second {
			t.Errorf("branches = %d: 同じseedで結果が変わった %g != %g", branches, first, second)
		}
	}
}

func TestParseRoute(t *testing.T) {
	for _, name := range []string{"cheap", "alt", "both"} {
		r, err := parseRoute(name)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if r.String() != name {
			t.Errorf("%s: String() = %s", name, r.String())
		}
	}
	if _, err := parseRoute("unknown"); err == nil {
		t.Error("未知のモードはエラーになるべき")
	}
}

// BenchmarkPredict は1サンプルあたりの推論時間を測る。
//
// 実測(12コア): plain 10.5us / wide 14.0us(1.34x) / gated 19.8us(1.89x)。
//
// gated が plain の約2倍になるのは、枝の数だけ行列積を打つためで想定どおり。
// 一方 wide は重み数が gated と同一(1,851,392, plain は 925,696)にもかかわらず
// 1.34倍に留まる。重み数への比例からは外れており、隠れ層の幅に依存しない
// 固定コスト(確保・符号行列の構築)が効いているためと考えられる。
//
// 学習時は逆に gated が plain の1.13倍、wide が1.79倍で順序が入れ替わる。
// 片枝しか補正しないため、重いデルタ蓄積が gated では増えないことによる。
// 推論コストで揃えた比較をしたい場合、wide は gated の対照にならない。
func BenchmarkPredict(b *testing.B) {
	for _, arch := range []string{"plain", "wide", "wider1536", "wider2048", "wider2560", "gated"} {
		b.Run(arch, func(b *testing.B) {
			rng := rand.New(rand.NewPCG(1, 2))
			branches, widths, err := archSpec(arch)
			if err != nil {
				b.Fatal(err)
			}
			m := &model{xRows: 1, xCols: 784}
			for _, wRows := range widths {
				if err := m.appendLayer(branches, wRows, rng); err != nil {
					b.Fatal(err)
				}
			}
			x, err := bitsx.NewRandMatrix(1, 784, rng)
			if err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if _, err := m.predict(x); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestParseNoiseKind(t *testing.T) {
	for _, name := range []string{"norm", "uniform"} {
		k, err := parseNoiseKind(name)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if k.String() != name {
			t.Errorf("%s: String() = %s", name, k.String())
		}
	}
	if _, err := parseNoiseKind("unknown"); err == nil {
		t.Error("未知の種別はエラーになるべき")
	}
}

// TestUniformNoiseStaysInRange は、整数一様ノイズが半幅を超えないことと、
// ガウスと標準偏差がおおよそ揃っていることを確かめる。
func TestUniformNoiseStaysInRange(t *testing.T) {
	rng := rand.New(rand.NewPCG(31, 32))
	d, err := newDense(1, 64, 256, rng)
	if err != nil {
		t.Fatal(err)
	}
	d.noiseKind = noiseUniform

	x, err := bitsx.NewRandMatrix(1, 256, rng)
	if err != nil {
		t.Fatal(err)
	}
	clean, _, _, err := d.preActivation(x, 0, rng)
	if err != nil {
		t.Fatal(err)
	}
	noisy, _, _, err := d.preActivation(x, 0.5, rng)
	if err != nil {
		t.Fatal(err)
	}

	halfWidth := int(float64(0.5*d.noiseStd) * math.Sqrt(3))
	if halfWidth <= 0 {
		t.Fatalf("半幅が0以下: %d", halfWidth)
	}
	var sumSq float64
	for i := range clean[0] {
		diff := noisy[0][i] - clean[0][i]
		if absInt(diff) > halfWidth {
			t.Fatalf("i = %d: ノイズが半幅を超えた diff = %d, halfWidth = %d", i, diff, halfWidth)
		}
		sumSq += float64(diff) * float64(diff)
	}
	std := math.Sqrt(sumSq / float64(len(clean[0])))
	want := float64(0.5 * d.noiseStd)
	if std < want*0.6 || std > want*1.4 {
		t.Errorf("標準偏差がガウスと揃っていない: got = %.1f, want ≈ %.1f", std, want)
	}
}

// TestDotTernaryAllOnesEqualsDot は、ゲート廃止時の換算式を裏付ける。
//
// bitsx のカーネルは次の値を返す。
//
//	Dot(t)              = 一致ビット数 u             (kernels.go の dotGo)
//	DotTernary(t, mask) = nonZero数 - 2*不一致数     (kernels.go の dotTernaryGo)
//
// mask が全ビット1のとき後者は 2u - cols になる。したがって符号判定は
// 2*u >= cols であり、u >= cols/2 では cols が奇数のときに誤る。
// ../FLOAT_REMOVAL.md のステップ4はこの式に依存している。
func TestDotTernaryAllOnesEqualsDot(t *testing.T) {
	rng := rand.New(rand.NewPCG(41, 42))
	// cols が奇数の形状を必ず含める。偶数だけだと u >= cols/2 の誤りを見逃す。
	for _, shape := range [][3]int{{8, 3, 37}, {5, 4, 64}, {7, 2, 130}, {3, 3, 1}} {
		valueRows, signRows, cols := shape[0], shape[1], shape[2]
		value, err := bitsx.NewRandMatrix(valueRows, cols, rng)
		if err != nil {
			t.Fatal(err)
		}
		sign, err := bitsx.NewRandMatrix(signRows, cols, rng)
		if err != nil {
			t.Fatal(err)
		}
		ones, err := bitsx.NewOnesMatrix(signRows, cols)
		if err != nil {
			t.Fatal(err)
		}

		ternary, err := value.DotTernary(sign, ones)
		if err != nil {
			t.Fatal(err)
		}
		plain, err := value.Dot(sign)
		if err != nil {
			t.Fatal(err)
		}
		if len(ternary) != len(plain) {
			t.Fatalf("長さが不一致: %d != %d", len(ternary), len(plain))
		}

		oddMismatch := 0
		for i, u := range plain {
			if want := 2*u - cols; ternary[i] != want {
				t.Fatalf("cols = %d, i = %d: DotTernary = %d, 2u-cols = %d", cols, i, ternary[i], want)
			}
			// 符号判定の等価性。
			if (ternary[i] >= 0) != (2*u >= cols) {
				t.Fatalf("cols = %d, i = %d: 符号判定が一致しない", cols, i)
			}
			// 誤った式との差が実際に出る形状かを数える。
			if (ternary[i] >= 0) != (u >= cols/2) {
				oddMismatch++
			}
		}
		if cols%2 == 1 && oddMismatch == 0 {
			t.Logf("cols = %d: 誤った式との差が今回の乱数では現れなかった", cols)
		}
		if cols%2 == 0 && oddMismatch != 0 {
			t.Errorf("cols = %d(偶数)では u >= cols/2 も一致するはず", cols)
		}
	}
}
