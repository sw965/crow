package tensor

import (
	"fmt"
	"math"
	"math/rand"
	"golang.org/x/exp/slices"
	omwrand "github.com/sw965/omw/math/rand"
)

type D1 []float64

func NewD1Zeros(n int) D1 {
	return make(D1, n)
}

func NewD1ZerosLike(d1 D1) D1 {
	return NewD1Zeros(len(d1))
}

func NewD1Ones(n int) D1 {
	ret := make(D1, n)
	for i := range ret {
		ret[i] = 1.0
	}
	return ret
}

func NewD1OnesLike(d1 D1) D1 {
	n := len(d1)
	return NewD1Ones(n)
}

func NewD1RandUniform(n int, min, max float64, r *rand.Rand) D1 {
	ret := make(D1, n)
	for i := range ret {
		ret[i] = omwrand.Float64(min, max, r)
	}
	return ret
}

func NewD1He(n int, r *rand.Rand) D1 {
	std := math.Sqrt(2.0 / float64(n))
	he := make(D1, n)
	for i := range he {
		he[i] = r.NormFloat64() * std
	}
	return he
}

func NewD1Rademacher(n int, r *rand.Rand) D1 {
	d1 := make(D1, n)
	for i := range d1 {
		var e float64
		if omwrand.Bool(r) {
			e = 1.0
		} else {
			e = -1.0
		}
		d1[i] = e
	}
	return d1
}

func (d1 D1) AddScalar(s float64) {
	for i := range d1 {
		d1[i] += s
	}
}

func (d1 D1) Add(other D1) error {
	if len(d1) != len(other) {
		return fmt.Errorf("tensor.D1の長さが一致しないため、加算できません。")
	}

	for i := range d1 {
		d1[i] += other[i]
	}
	return nil
}

func (d1 D1) SubScalar(s float64) {
	for i := range d1 {
		d1[i] -= s
	}
}

func (d1 D1) Sub(other D1) error {
	if len(d1) != len(other) {
		return fmt.Errorf("tensor.D1 の長さが一致しないため、減算できません。")
	}

	for i := range d1 {
		d1[i] -= other[i]
	}
	return nil
}

func (d1 D1) MulScalar(s float64) {
	for i := range d1 {
		d1[i] *= s
	}
}

func (d1 D1) Mul(other D1) error {
	if len(d1) != len(other) {
		return fmt.Errorf("tensor.D1 の長さが一致しないため、乗算できません。")
	}

	for i := range d1 {
		d1[i] *= other[i]
	}
	return nil
}

func (d1 D1) DivScalar(s float64) {
	for i := range d1 {
		d1[i] /= s
	}
}

func (d1 D1) Div(other D1) error {
	if len(d1) != len(other) {
		return fmt.Errorf("tensor.D1 の長さが一致しないため、除算できません。")
	}

	for i := range d1 {
		d1[i] /= other[i]
	}
	return nil
}

func (d1 D1) Clone() D1 {
	return slices.Clone(d1)
}

func (d1 D1) Copy(src D1) {
	for i := range d1 {
		d1[i] = src[i]
	}
}

func (d1 D1) Reciprocal() D1 {
	y := make(D1, len(d1))
	for i, e := range d1 {
		y[i] = 1.0 / e
	}
	return y
}

func D1AddScalar(d1 D1, s float64) D1 {
	y := slices.Clone(d1)
	y.AddScalar(s)
	return y
}

func D1Add(a, b D1) (D1, error) {
	y := slices.Clone(a)
	err := y.Add(b)
	return y, err
}

func D1SubScalar(d1 D1, s float64) D1 {
	y := slices.Clone(d1)
	y.SubScalar(s)
	return y
}

func D1Sub(a, b D1) (D1, error) {
	y := slices.Clone(a)
	err := y.Sub(b)
	return y, err
}

func D1MulScalar(d1 D1, s float64) D1 {
	y := slices.Clone(d1)
	y.MulScalar(s)
	return y
}

func D1Mul(a, b D1) (D1, error) {
	y := slices.Clone(a)
	err := y.Mul(b)
	return y, err
}

func D1DivScalar(d1 D1, s float64) D1 {
	y := slices.Clone(d1)
	y.DivScalar(s)
	return y
}

func D1Div(a, b D1) (D1, error) {
	y := slices.Clone(a)
	err := y.Div(b)
	return y, err
}
