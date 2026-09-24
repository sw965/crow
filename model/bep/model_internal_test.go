package bep

import (
	"math/rand/v2"
	"testing"

	"github.com/sw965/omw/mathx/bitsx"
)

func TestArgmax(t *testing.T) {
	tests := []struct {
		name   string
		logits []int
		want   int
	}{
		{name: "正常_1要素", logits: []int{5}, want: 0},
		{name: "正常_最大が1つ", logits: []int{1, 9, 3}, want: 1},
		{name: "正常_同点なら後ろの添字", logits: []int{7, 2, 7, 7, 1}, want: 3},
		{name: "正常_負の値", logits: []int{-5, -1, -3}, want: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := argmax(tt.logits); got != tt.want {
				t.Errorf("argmax(%v) = %d, want = %d", tt.logits, got, tt.want)
			}
		})
	}
}

func TestValueFromOnesCount(t *testing.T) {
	uniform := []float32{0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
	tests := []struct {
		name      string
		ones      int
		totalBits int
		values    []float32
		want      float32
	}{
		// 総ビット数 1000・11レベルなら、レベル i の温度計はちょうど 100·i ビットが 1
		{name: "正常_全て0ならValuesの先頭", ones: 0, totalBits: 1000, values: uniform, want: 0},
		{name: "正常_全て1ならValuesの末尾", ones: 1000, totalBits: 1000, values: uniform, want: 1.0},
		{name: "正常_温度計のレベル3そのもの", ones: 300, totalBits: 1000, values: uniform, want: 0.3},
		{name: "正常_レベル3と4の中間", ones: 350, totalBits: 1000, values: uniform, want: 0.35},
		{name: "正常_等間隔でないValuesも補間", ones: 50, totalBits: 100, values: []float32{0, 1, 10}, want: 1},
		{name: "正常_等間隔でないValuesの区間内", ones: 75, totalBits: 100, values: []float32{0, 1, 10}, want: 5.5},
		{name: "正常_Valuesが1つ", ones: 30, totalBits: 100, values: []float32{7}, want: 7},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := valueFromOnesCount(tt.ones, tt.totalBits, tt.values)
			if diff := got - tt.want; diff > 1e-5 || diff < -1e-5 {
				t.Errorf("got = %f, want = %f", got, tt.want)
			}
		})
	}
}

func TestValidateThermometer(t *testing.T) {
	thermo, err := bitsx.NewThermometerMatrices(5, 1, 40)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	if err := validateThermometer(thermo); err != nil {
		t.Errorf("温度計なのにエラー: %v", err)
	}

	rng := rand.New(rand.NewPCG(1, 2))
	etf, err := bitsx.NewETFMatrices(5, 1, 40, 100, rng)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	if err := validateThermometer(etf); err == nil {
		t.Error("ETF なのにエラーにならない")
	}

	if err := validateThermometer(thermo[:1]); err == nil {
		t.Error("プロトタイプが1つなのにエラーにならない")
	}
}
