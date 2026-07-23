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

	"github.com/sw965/omw/mathx/randx"
)

func sqrt32(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

func log32(x float32) float32 {
	return float32(math.Log(float64(x)))
}

func isNaN32(x float32) bool {
	return math.IsNaN(float64(x))
}

func isInf32(x float32) bool {
	return math.IsInf(float64(x), 0)
}

type Func func(float32, float32, int, int) float32

func NewAlphaGoFunc(c float32) Func {
	return func(q, p float32, sumVisits, selfVisits int) float32 {
		n := float32(1 + selfVisits)
		exploration := c * p * sqrt32(float32(sumVisits)) / n
		return q + exploration
	}
}

func NewAlphaZeroFunc(cInit, cBase float32) (Func, error) {
	if isNaN32(cInit) || isInf32(cInit) {
		return nil, fmt.Errorf("cinitが不正(NaN/Inf): cinit=%.6g", cInit)
	}

	if cBase <= 0 || isNaN32(cBase) || isInf32(cBase) {
		return nil, fmt.Errorf("cbaseが不正(<=0/NaN/Inf): cbase=%.6g", cBase)
	}

	return func(q, p float32, sumVisits, selfVisits int) float32 {
		n := float32(1 + selfVisits)
		c := cInit + log32((float32(sumVisits)+cBase+1.0)/cBase)
		exploration := c * p * sqrt32(float32(sumVisits)) / n
		return q + exploration
	}, nil
}

type Calculator struct {
	Func   Func
	P      float32
	w      float32
	visits int
	// pending は「選択済みだが、まだ結果を観測していない」回数。
	// 並列探索で複数のワーカーが同じ行動に集中しないようにする、virtual lossの仕組みに使う。
	pending      int
	VirtualValue float32
}

func (c *Calculator) AddW(v float32) error {
	if isNaN32(v) || isInf32(v) {
		return fmt.Errorf("vが不正(NaN/Inf): v=%.6g", v)
	}
	c.w += v
	return nil
}

func (c *Calculator) IncrementVisits() {
	c.visits += 1
}

func (c *Calculator) Pending() int {
	return c.pending
}

func (c *Calculator) IncrementPending() {
	c.pending += 1
}

func (c *Calculator) DecrementPending() error {
	if c.pending == 0 {
		return fmt.Errorf("pendingが不正(underflow): pending = 0")
	}
	c.pending -= 1
	return nil
}

func (c *Calculator) Visits() int {
	return c.visits + c.pending
}

func (c *Calculator) W() float32 {
	return c.w + (c.VirtualValue * float32(c.pending))
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

	if c.P < 0 || isNaN32(c.P) || isInf32(c.P) {
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

	if isNaN32(u) || isInf32(u) {
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

func (s VirtualSelector[K]) VisitRatioByKey() map[K]float32 {
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
