package ucb

import (
	"fmt"
	"math"
	"math/rand"
	"golang.org/x/exp/maps"
	omwmath "github.com/sw965/omw/math"
	omwrand "github.com/sw965/omw/math/rand"
)

type Func func(float32, float32, int, int) float32

func NewStandardFunc(c float32) Func {
	return func(v, p float32, total, n int) float32 {
		return v + c*p*math.Sqrt(math.Log(float32(total+1))/float32(n+1))
	}
}

func NewAlphaGoFunc(c float32) Func {
	return func(v, p float32, total, n int) float32 {
		return v + c*p*math.Sqrt(float32(total))/float32(n+1)
	}
}

type Calculator struct {
	Func       Func
	TotalValue float32
	P          float32
	Trial      int
}

func (c *Calculator) AverageValue() float32 {
	return float32(c.TotalValue) / float32(c.Trial+1)
}

func (c *Calculator) Calculation(totalTrial int) float32 {
	v := c.AverageValue()
	return c.Func(v, c.P, totalTrial, c.Trial)
}

type Manager[KS ~[]K, K comparable]map[K]*Calculator

func (m Manager[KS, K]) TotalValue() float32 {
	t := 0.0
	for _, v := range m {
		t += v.TotalValue
	}
	return t
}

func (m Manager[KS, K]) TotalTrial() int {
	t := 0
	for _, v := range m {
		t += v.Trial
	}
	return t
}

func (m Manager[KS, K]) AverageValue() float32 {
	totalTrial := m.TotalTrial()
	if totalTrial == 0 {
		return 0.0
	}
	totalValue := m.TotalValue()
	return float32(totalValue) / float32(totalTrial)
}

func (m Manager[KS, K]) Max() float32 {
	total := m.TotalTrial()
	ucbs := make([]float32, 0, len(m))
	for _, v := range m {
		ucbs = append(ucbs, v.Calculation(total))
	}
	return omwmath.Max(ucbs...)
}

func (m Manager[KS, K]) MaxKeys() KS {
	max := m.Max()
	total := m.TotalTrial()
	ks := make(KS, 0, len(m))
	for k, v := range m {
		if v.Calculation(total) == max {
			ks = append(ks, k)
		}
	}
	return ks
}

func (m Manager[KS, K]) MaxTrialKeys() KS {
	max := m.MaxTrial()
	ks := make(KS, 0, len(m))
	for k, v := range m {
		if v.Trial == max {
			ks = append(ks, k)
		}
	}
	return ks
}

func (m Manager[KS, K]) MaxTrial() int {
	trials := make([]int, 0, len(m))
	for _, v := range m {
		trials = append(trials, v.Trial)
	}
	return omwmath.Max(trials...)
}

func (m Manager[KS, K]) TrialPercentByKey() map[K]float32 {
	total := m.TotalTrial()
	ps := map[K]float32{}
	for k, v := range m {
		ps[k] = float32(v.Trial) / float32(total)
	}
	return ps
}

func (m Manager[KS, K]) SelectKeyByTrialPercentAboveFractionOfMax(t float32, r *rand.Rand) (K, error) {
	if t > 1.0 || t < 0.0 {
		var k K
		return k, fmt.Errorf("閾値は0.0 <= t <= 1.0 でなければならない")
	}
	ps := m.TrialPercentByKey()
	max := omwmath.Max(maps.Values(ps)...)
	n := len(ps)
	options := make(KS, 0, n)
	ws := make([]float32, 0, n)

	for a, p := range ps {
		if p >= max * t {
			options = append(options, a)
			ws = append(ws, p)
		}
	}

	idx := omwrand.IntByWeight(ws, r, 0.0)
	return options[idx], nil
}

type Managers[KS ~[]K, K comparable] []Manager[KS, K]