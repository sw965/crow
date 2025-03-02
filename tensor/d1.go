package tensor

import (
	"fmt"
	omwmath "github.com/sw965/omw/math"
	"golang.org/x/exp/slices"
)

type D1 []float64

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

func (d1 D1) Max() float64 {
	return omwmath.Max(d1...)
}

func (d1 D1) Reshape3D(dep, row, col int) (D3, error) {
    if len(d1) != dep*row*col {
        return nil, fmt.Errorf("指定したサイズ (%d, %d, %d) と入力データ長 %d が一致しません", dep, row, col, len(d1))
    }

    d3 := make(D3, dep)
    for i := 0; i < dep; i++ {
        d3[i] = make(D2, row)
        for j := 0; j < row; j++ {
            d3[i][j] = make(D1, col)
        }
    }

    idx := 0
    for i := 0; i < dep; i++ {
        for j := 0; j < row; j++ {
            for k := 0; k < col; k++ {
                d3[i][j][k] = d1[idx]
                idx++
            }
        }
    }
    return d3, nil
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
