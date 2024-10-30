package ucb

import (
	omwmath "github.com/sw965/omw/math"
	omwrand "github.com/sw965/omw/math/rand"
	"math"
	"math/rand"
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

type Manager[KS ~[]K, K comparable] map[K]*Calculator

func (m Manager[KS, K]) TotalValues() []float64 {
	vs := make([]float64, 0, len(m))
	for _, v := range m {
		vs = append(vs, v.TotalValue)
	}
	return vs
}

func (m Manager[KS, K]) TotalValue() float64 {
	return omwmath.Sum(m.TotalValues()...)
}

func (m Manager[KS, K]) Trials() []int {
	trials := make([]int, 0, len(m))
	for _, v := range m {
		trials = append(trials, v.Trial)
	}
	return trials
}

func (m Manager[KS, K]) TotalTrial() int {
	return omwmath.Sum(m.Trials()...)
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

func (m Manager[KS, K]) RandMaxKey(r *rand.Rand) K {
	return omwrand.Choice(m.MaxKeys(), r)
}

func (m Manager[KS, K]) MaxTrialKeys() KS {
	max := omwmath.Max(m.Trials()...)
	ks := make(KS, 0, len(m))
	for k, v := range m {
		if v.Trial == max {
			ks = append(ks, k)
		}
	}
	return ks
}

func (m Manager[KS, K]) RandMaxTrialKey(r *rand.Rand) K {
	return omwrand.Choice(m.MaxTrialKeys(), r)
}

func (m Manager[KS, K]) TrialPercents() map[K]float64 {
	total := m.TotalTrial()
	ps := map[K]float64{}
	for k, v := range m {
		ps[k] = float64(v.Trial) / float64(total)
	}
	return ps
}

type Managers[KS ~[]K, K comparable] []Manager[KS, K]

func (ms Managers[KS, K]) RandMaxKeys(r *rand.Rand) KS {
	ks := make(KS, len(ms))
	for i, m := range ms {
		ks[i] = m.RandMaxKey(r)
	}
	return ks
}

func (ms Managers[KS, K]) RandMaxTrialKeys(r *rand.Rand) KS {
	ks := make(KS, len(ms))
	for i, m := range ms {
		ks[i] = m.RandMaxTrialKey(r)
	}
	return ks
}

func (ms Managers[KS, K]) AverageValues() []float64 {
	vs := make([]float64, len(ms))
	for i, m := range ms {
		vs[i] = m.AverageValue()
	}
	return vs
}
