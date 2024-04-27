package pucb

import (
	"math"
	"math/rand"
	"github.com/sw965/omw"
)

func Calculation(v, p, c float64, totalN, actionN int) float64 {
	total := float64(totalN)
	n := float64(actionN+1)
	return v + (p * c * math.Sqrt(total) / n)
}

type Calculator struct {
	P float64
	TotalValue float64
	Trial       int
}

func (c *Calculator) AverageValue() float64 {
	return float64(c.TotalValue) / float64(c.Trial+1)
}

func (c *Calculator) Calculation(totalTrial int, C float64) float64 {
	v := c.AverageValue()
	return Calculation(v, c.P, C, totalTrial, c.Trial)
}

type Manager[KS ~[]K, K comparable] map[K]*Calculator

func (m Manager[KS, K]) Trials() []int {
	trials := make([]int, 0, len(m))
	for _, v := range m {
		trials = append(trials, v.Trial)
	}
	return trials
}

func (m Manager[KS, K]) TotalTrial() int {
	return omw.Sum(m.Trials()...)
}

func (m Manager[KS, K]) Max(c float64) float64 {
	total := m.TotalTrial()
	pucbs := make([]float64, 0, len(m))
	for _, v := range m {
		pucbs = append(pucbs, v.Calculation(total, c))
	}
	return omw.Max(pucbs...)
}

func (m Manager[KS, K]) MaxKeys(c float64) KS {
	max := m.Max(c)
	total := m.TotalTrial()
	ks := make(KS, 0, len(m))
	for k, v := range m {
		if v.Calculation(total, c) == max {
			ks = append(ks, k)
		}
	}
	return ks
}

func (m Manager[KS, K]) MaxTrialKeys() KS {
	max := omw.Max(m.Trials()...)
	ks := make(KS, 0, len(m))
	for k, v := range m {
		if v.Trial == max {
			ks = append(ks, k)
		}
	}
	return ks
}

func (m Manager[KS, K]) TrialPercents() map[K]float64 {
	total := m.TotalTrial()
	percents := map[K]float64{}
	for k, v := range m {
		percents[k] = float64(v.Trial) / float64(total)
	}
	return percents
}

type Managers[KS ~[]K, K comparable] []Manager[KS, K]

func (ms Managers[KS, K]) MaxKeys(c float64, r *rand.Rand) KS {
	ks := make(KS, len(ms))
	for playerI, m := range ms {
		ks[playerI] = omw.RandChoice(m.MaxKeys(c), r)
	}
	return ks
}

func (ms Managers[KS, K]) MaxTrialKeys(r *rand.Rand) KS {
	ks := make(KS, len(ms))
	for playerI, m := range ms {
		ks[playerI] = omw.RandChoice(m.MaxTrialKeys(), r)
	}
	return ks
}