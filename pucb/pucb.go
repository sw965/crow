// Package pucb provides PUCB (PUCT-like) utilities for action selection.
// Input validation for PUCB calculations is centralized in Calculator.Calculation.
//
// Package pucb は PUCB（PUCT系）選択のためのユーティリティを提供します。
// PUCB 計算の入力バリデーションは Calculator.Calculation に集約されています。
//
// https://arxiv.org/abs/1810.11755
// https://github.com/liuanji/WU-UCT

package pucb

import (
	"github.com/chewxy/math32"
	"math/rand/v2"
	"github.com/sw965/omw/mathx/randx"
	"fmt"
	"math"
)

type Func func(float32, float32, int, int) float32

func NewAlphaGoFunc(c float32) Func {
	return func(v, p float32, total, n int) float32 {
		tf := float32(total)
		nf := float32(1 + n)
		exploration := c * p * math32.Sqrt(tf) / nf
		return v + exploration
	}
}
type Calculator struct {
    Func       Func
    totalValue float32
    P          float32
    trial      int
    pending    int // 未観測の数
}

func (c *Calculator) GetTotalValue() float32 {
	return c.totalValue
}

func (c *Calculator) AddTotalValue(v float32) {
	c.totalValue += v
}

func (c *Calculator) GetTrial() int {
	return c.trial
}

func (c *Calculator) IncrementTrial() {
	c.trial += 1
}

func (c *Calculator) AverageValue() float32 {
	if c.trial == 0 {
		return 0.0
	}
	return float32(c.totalValue) / float32(c.trial)
}

func (c *Calculator) GetPending() int {
    return c.pending
}

func (c *Calculator) IncrementPending() {
    c.pending += 1
}

func (c *Calculator) DecrementPending() {
    c.pending -= 1
    if c.pending < 0 {
        panic(fmt.Sprintf("BUG: pending=%d < 0", c.pending))
    }
}

func (c *Calculator) EffectiveTrial() int {
    return c.trial + c.pending
}

func (c *Calculator) Calculation(effTotal int) float32 {
	if c.Func == nil {
		panic("BUG: pucb.Calculator.Func が nil です")
	}

	if effTotal < 0 {
		panic(fmt.Sprintf("BUG: effTotal=%d < 0 である為、PUCB計算を実行出来ません", effTotal))
	}

	if c.trial < 0 {
		panic(fmt.Sprintf("BUG: trial=%d < 0 である為、PUCB計算を実行出来ません", c.trial))
	}

    if c.pending < 0 {
        panic(fmt.Sprintf("BUG: pending=%d < 0", c.pending))
    }

	if effTotal < c.trial {
		panic(fmt.Sprintf("BUG: effTotal=%d < trial=%d である為、PUCB計算を実行出来ません", effTotal, c.trial))
	}

	// WU-UCT: N' = N + O
    effTrial := c.EffectiveTrial()
    if effTotal < effTrial {
        panic(fmt.Sprintf("BUG: effTotal=%d < effTrial=%d", effTotal, effTrial))
    }

	if tv := float64(c.totalValue); math.IsNaN(tv) || math.IsInf(tv, 0) {
		panic(fmt.Sprintf("BUG: totalValue=%v が不正です(NaN/Inf)", c.totalValue))
	}

	if p := float64(c.P); math.IsNaN(p) || math.IsInf(p, 0) || c.P < 0 {
		panic(fmt.Sprintf("BUG: P=%v が不正です（負数/NaN/Inf)", c.P))
	}

	v := c.AverageValue()
	if vf := float64(v); math.IsNaN(vf) || math.IsInf(vf, 0) {
		panic(fmt.Sprintf("BUG: AverageValue=%v が不正です(NaN/Inf)。totalValue=%v, trial=%d", v, c.totalValue, c.trial))
	}

	out := c.Func(v, c.P, effTotal, effTrial)
	if ov := float64(out); math.IsNaN(ov) || math.IsInf(ov, 0) {
		msg := fmt.Sprintf(
			"BUG: pucb.Funcが不正な値(NaN/Inf)を出力した為、処理が停止しました。v=%f, p=%f, effTotal=%d, effTrial=%d, out=%f",
			v, c.P, effTotal, effTrial, out,
		)
		panic(msg)
	}

	return out
}

type VirtualSelector[K comparable] map[K]*Calculator

func (s VirtualSelector[K]) TotalTrial() int {
	total := 0
	for _, c := range s {
		total += c.trial
	}
	return total
}

func (s VirtualSelector[K]) TotalEffectiveTrial() int {
    total := 0
    for _, c := range s {
        total += c.EffectiveTrial()
    }
    return total
}

func (s VirtualSelector[K]) TrialPercentByKey() map[K]float32 {
	n := len(s)
	if n == 0 {
		return map[K]float32{}
	}

	total := s.TotalTrial()
	m := map[K]float32{}

	if total == 0 {
		p := 1.0 / float32(n)
    	for k := range s {
        	m[k] = p
    	}
    	return m
	}

	for k, c := range s {
		m[k] = float32(c.trial) / float32(total)
	}
	return m
}

func (s VirtualSelector[K]) MaxPUCBKeys() []K {
	total := s.TotalEffectiveTrial()
	ks := make([]K, 0, len(s))

	const eps float32 = 0.0001

	var max float32
	first := true

	for k, c := range s {
		v := c.Calculation(total)

		if first {
			max = v
			ks = append(ks, k)
			first = false
			continue
		}

		// 「明確に」最大更新（eps以上 上なら max 更新して候補を入れ替え）
		if v > max+eps {
			max = v
			ks = ks[:0]
			ks = append(ks, k)
			continue
		}

		// 誤差 eps 以内なら同率扱い
		if math32.Abs(v-max) <= eps {
			ks = append(ks, k)
		}
	}

	return ks
}

func (s VirtualSelector[K]) Select(rng *rand.Rand) (K, error) {
	ks := s.MaxPUCBKeys()
	return randx.Choice(ks, rng)
}

// Selectors は Selectorのスライス型
// 下記のURLのDUCTアルゴリズムのような場面で使う
// https://www.terry-u16.net/entry/decoupled-uct
type VirtualSelectors[K comparable] []VirtualSelector[K]