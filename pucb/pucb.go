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
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/chewxy/math32"
	"github.com/sw965/omw/mathx/randx"
)

type Func func(float32, float32, int, int) float32

func NewAlphaGoFunc(c float32) Func {
	return func(q, p float32, sumVisits, selfVisits int) float32 {
		sum := float32(sumVisits)
		n := float32(1 + selfVisits)
		exploration := c * p * math32.Sqrt(sum) / n
		return q + exploration
	}
}

func NewAlphaZeroFunc(cinit, cbase float32) Func {
	if cbase <= 0 {
		panic(fmt.Sprintf("BUG: cbase=%v は正である必要があります", cbase))
	}
	if v := float64(cinit); math.IsNaN(v) || math.IsInf(v, 0) {
		panic(fmt.Sprintf("BUG: cinit=%v が不正です(NaN/Inf)", cinit))
	}
	if v := float64(cbase); math.IsNaN(v) || math.IsInf(v, 0) {
		panic(fmt.Sprintf("BUG: cbase=%v が不正です(NaN/Inf)", cbase))
	}

	return func(q, p float32, sumVisits, selfVisits int) float32 {
		sum := float32(sumVisits)
		n := float32(1 + selfVisits)
		c := cinit + math32.Log((sum+cbase+1.0)/cbase)
		exploration := c * p * math32.Sqrt(sum) / n
		return q + exploration
	}
}

type Calculator struct {
	Func         Func
	P            float32
	w            float32
	visits       int
	o            int
	VirtualValue float32
}

func (c *Calculator) W() float32 {
	return c.w
}

func (c *Calculator) AddW(v float32) {
	c.w += v
}

func (c *Calculator) IncrementVisits() {
	c.visits += 1
}

func (c *Calculator) O() int {
	return c.o
}

func (c *Calculator) IncrementO() {
	c.o += 1
}

func (c *Calculator) DecrementO() {
	c.o -= 1
	if c.o < 0 {
		panic(fmt.Sprintf("BUG: o=%d < 0", c.o))
	}
}

func (c *Calculator) EffectiveVisits() int {
	return c.visits + c.o
}

func (c *Calculator) EffectiveW() float32 {
	return c.w + (c.VirtualValue * float32(c.o))
}

func (c *Calculator) EffectiveQ() float32 {
	visits := c.EffectiveVisits()
	if visits == 0 {
		return 0.0
	}
	w := c.EffectiveW()
	return w / float32(visits)
}

func (c *Calculator) EffectiveU(sumEffVisits int) float32 {
	if c.Func == nil {
		panic("BUG: pucb.Calculator.Func が nil です")
	}

	if sumEffVisits < 0 {
		panic(fmt.Sprintf("BUG: sumEffVisits=%d < 0 である為、PUCB計算を実行出来ません", sumEffVisits))
	}

	if c.visits < 0 {
		panic(fmt.Sprintf("BUG: visits=%d < 0 である為、PUCB計算を実行出来ません", c.visits))
	}

	if c.o < 0 {
		panic(fmt.Sprintf("BUG: o=%d < 0", c.o))
	}

	if sumEffVisits < c.visits {
		panic(fmt.Sprintf("BUG: sumEffVisits=%d < visits=%d である為、PUCB計算を実行出来ません", sumEffVisits, c.visits))
	}

	// WU-UCT: N' = N + O
	selfEffVisits := c.EffectiveVisits()
	if sumEffVisits < selfEffVisits {
		panic(fmt.Sprintf("BUG: sumEffVisits=%d < selfEffVisits=%d", sumEffVisits, selfEffVisits))
	}

	if w := float64(c.w); math.IsNaN(w) || math.IsInf(w, 0) {
		panic(fmt.Sprintf("BUG: w=%v が不正です(NaN/Inf)", c.w))
	}

	if p := float64(c.P); math.IsNaN(p) || math.IsInf(p, 0) || c.P < 0 {
		panic(fmt.Sprintf("BUG: P=%v が不正です（負数/NaN/Inf)", c.P))
	}

	q := c.EffectiveQ()
	if qv := float64(q); math.IsNaN(qv) || math.IsInf(qv, 0) {
		panic(fmt.Sprintf("BUG: Q=%v が不正です(NaN/Inf)。w=%v, visits=%d", q, c.w, c.visits))
	}

	u := c.Func(q, c.P, sumEffVisits, selfEffVisits)
	if uv := float64(u); math.IsNaN(uv) || math.IsInf(uv, 0) {
		msg := fmt.Sprintf(
			"BUG: pucb.Funcが不正な値(NaN/Inf)を出力した為、処理が停止しました。q=%f, p=%f, sumEffVisits=%d, selfEffVisits=%d, u=%f",
			q, c.P, sumEffVisits, selfEffVisits, u,
		)
		panic(msg)
	}

	return u
}

type VirtualSelector[K comparable] map[K]*Calculator

func (s VirtualSelector[K]) SumEffectiveVisits() int {
	sum := 0
	for _, c := range s {
		sum += c.EffectiveVisits()
	}
	return sum
}

func (s VirtualSelector[K]) EffectiveVisitPercentByKey() map[K]float32 {
	n := len(s)
	if n == 0 {
		return map[K]float32{}
	}

	sum := s.SumEffectiveVisits()
	m := map[K]float32{}

	if sum == 0 {
		p := 1.0 / float32(n)
		for k := range s {
			m[k] = p
		}
		return m
	}

	for k, c := range s {
		m[k] = float32(c.visits) / float32(sum)
	}
	return m
}

func (s VirtualSelector[K]) EffectiveMaxKeys() []K {
	sum := s.SumEffectiveVisits()
	ks := make([]K, 0, len(s))

	const eps float32 = 0.0001

	var max float32
	first := true

	for k, c := range s {
		v := c.EffectiveU(sum)

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
	ks := s.EffectiveMaxKeys()
	return randx.Choice(ks, rng)
}

// Selectors は Selectorのスライス型
// 下記のURLのDUCTアルゴリズムのような場面で使う
// https://www.terry-u16.net/entry/decoupled-uct
type VirtualSelectors[K comparable] []VirtualSelector[K]
