package ucb

import (
	"math"
	"math/rand"
	omath "github.com/sw965/omw/math"
	orand "github.com/sw965/omw/rand"
)

type Func func(float64, float64, int, int) float64

func New1Func(c float64) Func {
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
	ret := make([]float64, 0, len(m))
	for _, v := range m {
		ret = append(ret, v.TotalValue)
	}
	return ret
}

func (m Manager[KS, K]) TotalValue() float64 {
	return omath.Sum(m.TotalValues()...)
}

func (m Manager[KS, K]) Trials() []int {
	ret := make([]int, 0, len(m))
	for _, v := range m {
		ret = append(ret, v.Trial)
	}
	return ret
}

func (m Manager[KS, K]) TotalTrial() int {
	return omath.Sum(m.Trials()...)
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
	return omath.Max(ucbs...)
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

func (m Manager[KS, K]) ActionByMax(r *rand.Rand) K {
	return orand.Choice(m.MaxKeys(), r)
}

func (m Manager[KS, K]) MaxTrialKeys() KS {
	max := omath.Max(m.Trials()...)
	ks := make(KS, 0, len(m))
	for k, v := range m {
		if v.Trial == max {
			ks = append(ks, k)
		}
	}
	return ks
}

func (m Manager[KS, K]) ActionByMaxTrial(r *rand.Rand) K {
	return orand.Choice(m.MaxTrialKeys(), r)
}

func (m Manager[KS, K]) TrialPercents() map[K]float64 {
	total := m.TotalTrial()
	ret := map[K]float64{}
	for k, v := range m {
		ret[k] = float64(v.Trial) / float64(total)
	}
	return ret
}

type Managers[KS ~[]K, K comparable] []Manager[KS, K]

func (ms Managers[KS, K]) JointActionByMax(r *rand.Rand) KS {
	ret := make(KS, len(ms))
	for i, m := range ms {
		ret[i] = m.ActionByMax(r)
	}
	return ret
}

func (ms Managers[KS, K]) JointActionByMaxTrial(r *rand.Rand) KS {
	ret := make(KS, len(ms))
	for i, m := range ms {
		ret[i] = m.ActionByMaxTrial(r)
	}
	return ret
}