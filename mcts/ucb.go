package mcts

import (
	omathw "github.com/sw965/omw/math"
	cwmath "github.com/sw965/crow/math"
)

type PUCB struct {
	AccumReward float64
	Trial       int
	P float64
}

func (p *PUCB) AverageReward() float64 {
	return float64(p.AccumReward) / float64(p.Trial+1)
}

func (p *PUCB) Get(c float64, totalTrial int) float64 {
	v := p.AverageReward()
	return cwmath.PolicyUpperConfidenceBound(c, v, p.P, totalTrial, p.Trial)
}

type PUCBMapManager[K comparable] map[K]*PUCB

func (m PUCBMapManager[K]) Trials() []int {
	y := make([]int, 0, len(m))
	for _, v := range m {
		y = append(y, v.Trial)
	}
	return y
}
func (m PUCBMapManager[K]) Max(c float64) float64 {
	total := omathw.Sum(m.Trials()...)
	y := make([]float64, 0, len(m))
	for _, v := range m {
		y = append(y, v.Get(c, total))
	}
	return omathw.Max(y...)
}

func (m PUCBMapManager[K]) MaxKeys(c float64) []K {
	max := m.Max(c)
	total := omathw.Sum(m.Trials()...)
	ks := make([]K, 0, len(m))
	for k, v := range m {
		if v.Get(c, total) == max {
			ks = append(ks, k)
		}
	}
	return ks
}

func (m PUCBMapManager[K]) MaxTrialKeys() []K {
	max := omathw.Max(m.Trials()...)
	ks := make([]K, 0, len(m))
	for k, v := range m {
		if v.Trial == max {
			ks = append(ks, k)
		}
	}
	return ks
}