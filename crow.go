package crow

import (
	"golang.org/x/exp/constraints"
	"math"
	"github.com/sw965/omw"
)

func NumericalGradient[X constraints.Float](xs []X, f func([]X) X) []X {
	h := X(0.0001)
	n := len(xs)
	grad := make([]X, n)
	for i := 0; i < n; i++ {
		tmp := xs[i]
		xs[i] = tmp + h
		y1 := f(xs)

		xs[i] = tmp - h
		y2 := f(xs)

		grad[i] = (y1 - y2) / (h * 2)
		xs[i] = tmp
	}
	return grad
}

func PolicyUpperConfidenceBound(c, p, v float64, n, a int) float64 {
	fn := float64(n)
	return v + (c * p * math.Sqrt(fn) / float64(a+1))
}

type UtilPUCB struct {
	AccumReward float64
	Trial       int
	P float64
}

func (p *UtilPUCB) AverageReward() float64 {
	return float64(p.AccumReward) / float64(p.Trial+1)
}

func (p *UtilPUCB) Get(totalTrial int, c float64) float64 {
	v := p.AverageReward()
	return PolicyUpperConfidenceBound(c, p.P, v, totalTrial, p.Trial)
}

type PUCBMapManager[KS ~[]K, K comparable] map[K]*UtilPUCB

func (m PUCBMapManager[KS, K]) Trials() []int {
	y := make([]int, 0, len(m))
	for _, v := range m {
		y = append(y, v.Trial)
	}
	return y
}
func (m PUCBMapManager[KS, K]) Max(c float64) float64 {
	total := omw.Sum(m.Trials()...)
	y := make([]float64, 0, len(m))
	for _, v := range m {
		y = append(y, v.Get(total, c))
	}
	return omw.Max(y...)
}

func (m PUCBMapManager[KS, K]) MaxKeys(c float64) KS {
	max := m.Max(c)
	total := omw.Sum(m.Trials()...)
	ks := make([]K, 0, len(m))
	for k, v := range m {
		if v.Get(total, c) == max {
			ks = append(ks, k)
		}
	}
	return ks
}

func (m PUCBMapManager[KS, K]) MaxTrialKeys() KS {
	max := omw.Max(m.Trials()...)
	ks := make([]K, 0, len(m))
	for k, v := range m {
		if v.Trial == max {
			ks = append(ks, k)
		}
	}
	return ks
}

func (m PUCBMapManager[KS, K]) TrialPercents() map[K]float64 {
	total := omw.Sum(m.Trials()...)
	y := map[K]float64{}
	for k, v := range m {
		y[k] = float64(v.Trial) / float64(total)
	}
	return y
}

type PUCBMapManagers[KS ~[]K, K comparable] []PUCBMapManager[KS, K]

type ActionPolicY[A comparable] map[A]float64
type ActionPolicyFunc[S any, A comparable] func(*S) ActionPolicY[A]

type ActionPolicYs[A comparable] []ActionPolicY[A]
type ActionPoliciesFunc[S any, A comparable] func(*S) ActionPolicYs[A]