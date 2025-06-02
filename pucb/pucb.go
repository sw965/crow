package pucb

import (
	"github.com/chewxy/math32"
	omath "github.com/sw965/omw/math"
)

type Func func(float32, float32, int, int) float32

func NewStandardFunc(c float32) Func {
	return func(v, p float32, total, n int) float32 {
		tf := float32(total)
	 	nf := 1.0 + float32(n)
		exploration := c*p*math32.Sqrt(2*math32.Log(tf)/nf)
	  	return v + exploration
	}
}

func NewAlphaGoFunc(c float32) Func {
	return func(v, p float32, total, n int) float32 {
		tf := float32(total)
		nf := 1.0 + float32(n)
		exploration := c*p*math32.Sqrt(tf) / nf
		return v + exploration
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

type Manager[K comparable]map[K]*Calculator

func (m Manager[K]) TotalValue() float32 {
	var t float32 = 0.0
	for _, v := range m {
		t += v.TotalValue
	}
	return t
}

func (m Manager[K]) TotalTrial() int {
	t := 0
	for _, v := range m {
		t += v.Trial
	}
	return t
}

func (m Manager[K]) AverageValue() float32 {
	totalTrial := m.TotalTrial()
	if totalTrial == 0 {
		return 0.0
	}
	totalValue := m.TotalValue()
	return float32(totalValue) / float32(totalTrial)
}

func (m Manager[K]) Max() float32 {
	total := m.TotalTrial()
	pucbs := make([]float32, 0, len(m))
	for _, v := range m {
		pucbs = append(pucbs, v.Calculation(total))
	}
	return omath.Max(pucbs...)
}

func (m Manager[K]) MaxKeys() []K {
	max := m.Max()
	total := m.TotalTrial()
	ks := make([]K, 0, len(m))
	for k, v := range m {
		if v.Calculation(total) == max {
			ks = append(ks, k)
		}
	}
	return ks
}

func (m Manager[K]) MaxTrialKeys() []K {
	max := m.MaxTrial()
	ks := make([]K, 0, len(m))
	for k, v := range m {
		if v.Trial == max {
			ks = append(ks, k)
		}
	}
	return ks
}

func (m Manager[K]) MaxTrial() int {
	trials := make([]int, 0, len(m))
	for _, v := range m {
		trials = append(trials, v.Trial)
	}
	return omath.Max(trials...)
}

func (m Manager[K]) TrialPercentByKey() map[K]float32 {
	total := m.TotalTrial()
	ps := map[K]float32{}
	for k, v := range m {
		ps[k] = float32(v.Trial) / float32(total)
	}
	return ps
}

type Managers[K comparable] []Manager[K]