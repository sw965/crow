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
	"github.com/sw965/omw/mathx"
	"github.com/sw965/omw/mathx/randx"
)

type Func func(float32, float32, int, int) float32

func NewAlphaGoFunc(c float32) Func {
	return func(q, p float32, sumVisits, selfVisits int) float32 {
		n := float32(1 + selfVisits)
		exploration := c * p * mathx.Sqrt(float32(sumVisits)) / n
		return q + exploration
	}
}

func NewAlphaZeroFunc(cinit, cbase float32) (Func, error) {
	if mathx.IsNaN(cinit) || mathx.IsInf(cinit, 0) {
		return nil, fmt.Errorf("cinitが不正(NaN/Inf): cinit=%.6g", cinit)
	}

	if cbase <= 0 || mathx.IsNaN(cbase) || mathx.IsInf(cbase, 0) {
		return nil, fmt.Errorf("cbaseが不正(<=0/NaN/Inf): cbase=%.6g", cbase)
	}

	return func(q, p float32, sumVisits, selfVisits int) float32 {
		n := float32(1 + selfVisits)
		c := cinit + mathx.Log((float32(sumVisits)+cbase+1.0)/cbase)
		exploration := c * p * mathx.Sqrt(float32(sumVisits)) / n
		return q + exploration
	}, nil
}

type Calculator struct {
	Func         Func
	P            float32
	w            float32
	visits       int
	o            int
	VirtualValue float32
}

func (c *Calculator) AddW(v float32) error {
	if mathx.IsNaN(v) || mathx.IsInf(v, 0) {
		return fmt.Errorf("vが不正(NaN/Inf): v=%.6g", v)
	}
	c.w += v
	return nil
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

func (c *Calculator) DecrementO() error {
	if c.o == 0 {
		return fmt.Errorf("oが不正(underflow): o=0")
	}
	c.o -= 1
	return nil
}

func (c *Calculator) Visits() int {
	return c.visits + c.o
}

func (c *Calculator) W() float32 {
	return c.w + (c.VirtualValue * float32(c.o))
}

func (c *Calculator) Q() float32 {
	visits := c.Visits()
	if visits == 0 {
		return 0.0
	}
	w := c.W()
	return w / float32(visits)
}

func (c *Calculator) ValidateForSumVisits(sumVisits int) error {
	if sumVisits < 0 {
		return fmt.Errorf("sumVisitsが不正(<0): sumVisits=%d", sumVisits)
	}

	visits := c.Visits()
	if sumVisits < visits {
		return fmt.Errorf("sumVisits=%d < visits=%d", sumVisits, visits)
	}

	if p := float64(c.P); c.P < 0 || math.IsNaN(p) || math.IsInf(p, 0) {
		return fmt.Errorf("Pが不正(負/NaN/Inf): P=%.6g", c.P)
	}

	if c.Func == nil {
		return fmt.Errorf("Funcが未初期化(nil)")
	}
	return nil
}

func (c *Calculator) U(sumVisits int) (float32, error) {
	if err := c.ValidateForSumVisits(sumVisits); err != nil {
		return 0.0, err
	}

	visits := c.Visits()
	q := c.Q()
	u := c.Func(q, c.P, sumVisits, visits)

	if v := float64(u); math.IsNaN(v) || math.IsInf(v, 0) {
		return 0.0, fmt.Errorf(
			"Funcの戻り値が不正(NaN/Inf): q=%.6g P=%.6g sumVisits=%d visits=%d u=%.6g",
			q, c.P, sumVisits, visits, u,
		)

	}
	return u, nil
}

type VirtualSelector[K comparable] map[K]*Calculator

func (s VirtualSelector[K]) SumVisits() int {
	sum := 0
	for _, c := range s {
		sum += c.Visits()
	}
	return sum
}

func (s VirtualSelector[K]) VisitPercentByKey() map[K]float32 {
	n := len(s)
	if n == 0 {
		return map[K]float32{}
	}

	sum := s.SumVisits()
	m := map[K]float32{}

	if sum == 0 {
		p := 1.0 / float32(n)
		for k := range s {
			m[k] = p
		}
		return m
	}

	for k, c := range s {
		m[k] = float32(c.Visits()) / float32(sum)
	}
	return m
}

const eps float32 = 0.0001

func (s VirtualSelector[K]) MaxKeys() ([]K, error) {
	sum := s.SumVisits()
	ks := make([]K, 0, len(s))
	var max float32
	first := true

	for k, c := range s {
		u, err := c.U(sum)
		if err != nil {
			return nil, err
		}

		if first {
			max = u
			ks = append(ks, k)
			first = false
			continue
		}

		// 「明確に」最大更新（eps以上 上なら max 更新して候補を入れ替え）
		if u > max+eps {
			max = u
			ks = ks[:0]
			ks = append(ks, k)
			continue
		}

		// 誤差 eps 以内なら同率扱い
		if float32(math.Abs(float64(u-max))) <= eps {
			ks = append(ks, k)
		}
	}
	return ks, nil
}

func (s VirtualSelector[K]) Select(rng *rand.Rand) (K, error) {
	ks, err := s.MaxKeys()
	if err != nil {
		var zero K
		return zero, err
	}
	return randx.Choice(ks, rng)
}

// Selectors は Selectorのスライス型
// 下記のURLのDUCTアルゴリズムのような場面で使う
// https://www.terry-u16.net/entry/decoupled-uct
type VirtualSelectors[K comparable] []VirtualSelector[K]