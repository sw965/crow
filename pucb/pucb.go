package pucb

import (
	"math"
	"github.com/sw965/omw"
)

func Get(p, v float64, n, a int, X float64) float64 {
	return v + (p * X * math.Sqrt(float64(n)) / float64(a+1))
}

type Manager struct {
	P      float64
	AccumReward float64
	Trial       int
}

func (m *Manager) AverageReward() float64 {
	return float64(m.AccumReward) / float64(m.Trial+1)
}

func (m *Manager) Get(totalTrial int, X float64) float64 {
	v := m.AverageReward()
	return Get(v, m.P, totalTrial, m.Trial, X)
}

type ManagerByKey[K comparable] map[K]*Manager

func (mbk ManagerByKey[K]) TotalTrial() int {
	y := 0
	for _, v := range mbk {
		y += v.Trial
	}
	return y
}
func (mbk ManagerByKey[K]) Max(X float64) float64 {
	total := mbk.TotalTrial()
	ys := make([]float64, 0, len(mbk))
	for _, v := range mbk {
		y := v.Get(total, X)
		ys = append(ys, y)
	}
	return omw.Max(ys...)
}

func (mbk ManagerByKey[K]) MaxKeys(X float64) []K {
	max := mbk.Max(X)
	total := mbk.TotalTrial()
	ks := make([]K, 0, len(mbk))
	for k, v := range mbk {
		y := v.Get(total, X)
		if y == max {
			ks = append(ks, k)
		}
	}
	return ks
}

func (mbk ManagerByKey[K]) MaxTrial(X float64) int {
	trials := make([]int, 0, len(mbk))
	for _, v := range mbk {
		trials = append(trials, v.Trial)
	}
	return omw.Max(trials...)
}

func (mbk ManagerByKey[K]) MaxTrialKeys(X float64) []K {
	max := mbk.MaxTrial(X)
	ks := make([]K, 0, len(mbk))
	for k, v := range mbk {
		if v.Trial == max {
			ks = append(ks, k)
		}
	}
	return ks
}