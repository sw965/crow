package ucb

import (
	"fmt"
	"math"
	"math/rand"
	"golang.org/x/exp/maps"
	omwmath "github.com/sw965/omw/math"
	omwrand "github.com/sw965/omw/math/rand"
)

type Func func(float64, float64, int, int) float64

func NewStandardFunc(c float64) Func {
	return func(v, p float64, total, n int) float64 {
		return v + c*p*math.Sqrt(math.Log(float64(total+1))/float64(n+1))
	}
}

func NewAlphaGoFunc(c float64) Func {
	return func(v, p float64, total, n int) float64 {
		return v + c*p*math.Sqrt(float64(total))/float64(n+1)
	}
}

type Calculator struct {
	Func       Func
	TotalValue float64
	P          float64
	Trial      int
}

func (c *Calculator) AverageValue() float64 {
	return float64(c.TotalValue) / float64(c.Trial+1)
}

func (c *Calculator) Calculation(totalTrial int) float64 {
	v := c.AverageValue()
	return c.Func(v, c.P, totalTrial, c.Trial)
}

type Manager[KS ~[]K, K comparable]map[K]*Calculator

func (m Manager[KS, K]) TotalValue() float64 {
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

func (m Manager[KS, K]) AverageValue() float64 {
	totalTrial := m.TotalTrial()
	if totalTrial == 0 {
		return 0.0
	}
	totalValue := m.TotalValue()
	return float64(totalValue) / float64(totalTrial)
}

func (m Manager[KS, K]) Max() float64 {
	total := m.TotalTrial()
	ucbs := make([]float64, 0, len(m))
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

func (m Manager[KS, K]) TrialPercentByKey() map[K]float64 {
	total := m.TotalTrial()
	ps := map[K]float64{}
	for k, v := range m {
		ps[k] = float64(v.Trial) / float64(total)
	}
	return ps
}

func (m Manager[KS, K]) SelectKeyByTrialPercentAboveFractionOfMax(t float64, r *rand.Rand) (K, error) {
	if t > 1.0 || t < 0.0 {
		var k K
		return k, fmt.Errorf("閾値は0.0 <= t <= 1.0 でなければならない")
	}
	ps := m.TrialPercentByKey()
	max := omwmath.Max(maps.Values(ps)...)
	n := len(ps)
	options := make(KS, 0, n)
	ws := make([]float64, 0, n)

	for a, p := range ps {
		if p >= max * t {
			options = append(options, a)
			ws = append(ws, p)
		}
	}

	idx, err := omwrand.IntByWeight(ws, r)
	if err != nil {
		var k K
		return k, err
	}
	return options[idx], nil
}

type Managers[KS ~[]K, K comparable] []Manager[KS, K]