package pucb_test

import (
	"math"
	"testing"

	"github.com/sw965/crow/pucb"
	"github.com/sw965/omw/mathx/randx"
	"github.com/sw965/omw/slicesx"
)

func TestCalculator(t *testing.T) {
	t.Run("正常_初期状態のQは0", func(t *testing.T) {
		c := &pucb.Calculator{}
		if got := c.Q(); got != 0.0 {
			t.Errorf("Qの不一致: got = %f, want = 0.0", got)
		}
		if got := c.Visits(); got != 0 {
			t.Errorf("Visitsの不一致: got = %d, want = 0", got)
		}
	})

	t.Run("正常_QはWの平均", func(t *testing.T) {
		c := &pucb.Calculator{}
		// 1.0 と 0.0 を観測すると、Q = 0.5
		if err := c.AddW(1.0); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		c.IncrementVisits()
		if err := c.AddW(0.0); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		c.IncrementVisits()

		if got := c.Q(); math.Abs(float64(got)-0.5) > 0.0001 {
			t.Errorf("Qの不一致: got = %f, want = 0.5", got)
		}
	})

	t.Run("正常_pendingはVirtualValueとして扱われる", func(t *testing.T) {
		c := &pucb.Calculator{VirtualValue: 0.5}
		c.IncrementPending()
		c.IncrementPending()

		// 未観測が2つある場合、Visits = 2, W = VirtualValue * 2
		if got := c.Visits(); got != 2 {
			t.Errorf("Visitsの不一致: got = %d, want = 2", got)
		}
		if got := c.W(); math.Abs(float64(got)-1.0) > 0.0001 {
			t.Errorf("Wの不一致: got = %f, want = 1.0", got)
		}
		if got := c.Q(); math.Abs(float64(got)-0.5) > 0.0001 {
			t.Errorf("Qの不一致: got = %f, want = 0.5", got)
		}

		// 解放すると元に戻る
		if err := c.DecrementPending(); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if err := c.DecrementPending(); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if got := c.Visits(); got != 0 {
			t.Errorf("Visitsの不一致: got = %d, want = 0", got)
		}
	})

	t.Run("異常_pendingのunderflow", func(t *testing.T) {
		c := &pucb.Calculator{}
		if err := c.DecrementPending(); err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
	})

	t.Run("異常_AddWにNaNとInf", func(t *testing.T) {
		c := &pucb.Calculator{}
		if err := c.AddW(float32(math.NaN())); err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
		if err := c.AddW(float32(math.Inf(1))); err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
	})
}

func TestNewAlphaGoFunc(t *testing.T) {
	f := pucb.NewAlphaGoFunc(1.0)
	// u = q + c*p*sqrt(sumVisits)/(1+selfVisits) = 0.5 + 1*1*sqrt(4)/(1+1) = 1.5
	got := f(0.5, 1.0, 4, 1)
	want := float32(1.5)
	if math.Abs(float64(got-want)) > 0.0001 {
		t.Errorf("値の不一致: got = %f, want = %f", got, want)
	}
}

func TestNewAlphaZeroFunc(t *testing.T) {
	t.Run("正常", func(t *testing.T) {
		f, err := pucb.NewAlphaZeroFunc(1.25, 19652.0)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if f == nil {
			t.Fatal("Funcがnil")
		}
		// 探索が進むほど、探索項の係数cが大きくなる(単調増加)
		u1 := f(0.0, 1.0, 100, 0)
		u2 := f(0.0, 1.0, 400, 0)
		if u2 <= u1 {
			t.Errorf("sumVisitsの増加でuが増えていない: u1 = %f, u2 = %f", u1, u2)
		}
	})

	t.Run("異常_cInitがNaN", func(t *testing.T) {
		_, err := pucb.NewAlphaZeroFunc(float32(math.NaN()), 19652.0)
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
	})

	t.Run("異常_cBaseが0以下", func(t *testing.T) {
		_, err := pucb.NewAlphaZeroFunc(1.25, 0.0)
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
	})
}

func TestCalculatorU_Error(t *testing.T) {
	alphaGo := pucb.NewAlphaGoFunc(1.0)

	tests := []struct {
		name      string
		c         *pucb.Calculator
		sumVisits int
	}{
		{
			name:      "異常_sumVisitsが負",
			c:         &pucb.Calculator{Func: alphaGo},
			sumVisits: -1,
		},
		{
			name: "異常_Pが負",
			c:    &pucb.Calculator{Func: alphaGo, P: -0.5},
		},
		{
			name: "異常_Funcがnil",
			c:    &pucb.Calculator{},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := tc.c.U(tc.sumVisits); err == nil {
				t.Fatal("エラーを期待したが、nilが返された")
			}
		})
	}
}

func TestVirtualSelectorVisitRatioByKey(t *testing.T) {
	t.Run("正常_訪問数に比例", func(t *testing.T) {
		s := pucb.VirtualSelector[string]{
			"a": &pucb.Calculator{},
			"b": &pucb.Calculator{},
		}
		// aを3回、bを1回訪問
		for range 3 {
			s["a"].IncrementVisits()
		}
		s["b"].IncrementVisits()

		ratios := s.VisitRatioByKey()
		if math.Abs(float64(ratios["a"])-0.75) > 0.0001 {
			t.Errorf("aの比率の不一致: got = %f, want = 0.75", ratios["a"])
		}
		if math.Abs(float64(ratios["b"])-0.25) > 0.0001 {
			t.Errorf("bの比率の不一致: got = %f, want = 0.25", ratios["b"])
		}
	})

	t.Run("準正常_訪問数が全て0なら一様", func(t *testing.T) {
		s := pucb.VirtualSelector[string]{
			"a": &pucb.Calculator{},
			"b": &pucb.Calculator{},
		}
		ratios := s.VisitRatioByKey()
		for k, r := range ratios {
			if math.Abs(float64(r)-0.5) > 0.0001 {
				t.Errorf("%s の比率の不一致: got = %f, want = 0.5", k, r)
			}
		}
	})

	t.Run("準正常_空", func(t *testing.T) {
		s := pucb.VirtualSelector[string]{}
		ratios := s.VisitRatioByKey()
		if len(ratios) != 0 {
			t.Errorf("空のセレクタから空でない結果: %v", ratios)
		}
	})
}

func TestVirtualSelectorMaxKeysAndSelect(t *testing.T) {
	alphaGo := pucb.NewAlphaGoFunc(1.0)

	t.Run("正常_Pが高い行動が選ばれる", func(t *testing.T) {
		s := pucb.VirtualSelector[string]{
			"high": &pucb.Calculator{Func: alphaGo, P: 0.9},
			"low":  &pucb.Calculator{Func: alphaGo, P: 0.1},
		}
		// 訪問実績を作る(sumVisits > 0 で探索項が効く状態)
		s["high"].IncrementVisits()
		s["low"].IncrementVisits()

		ks, err := s.MaxKeys()
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if len(ks) != 1 || ks[0] != "high" {
			t.Errorf("MaxKeysの不一致: got = %v, want = [high]", ks)
		}
	})

	t.Run("統計_同率なら一様に選ばれる", func(t *testing.T) {
		s := pucb.VirtualSelector[string]{
			"a": &pucb.Calculator{Func: alphaGo, P: 0.5},
			"b": &pucb.Calculator{Func: alphaGo, P: 0.5},
		}

		rng := randx.NewPCG()
		n := 10000
		got := make([]string, n)
		for i := range n {
			k, err := s.Select(rng)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			got[i] = k
		}

		counts := slicesx.Counts(got)
		const eps = 0.03
		for _, k := range []string{"a", "b"} {
			ratio := float64(counts[k]) / float64(n)
			if math.Abs(ratio-0.5) > eps {
				t.Errorf("%s の選択比率の不一致: got = %.3f, want = 0.5(±%.3f)", k, ratio, eps)
			}
		}
	})

	t.Run("異常_Funcがnilの場合はSelectがエラー", func(t *testing.T) {
		s := pucb.VirtualSelector[string]{
			"a": &pucb.Calculator{},
		}
		rng := randx.NewPCG()
		if _, err := s.Select(rng); err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
	})
}
