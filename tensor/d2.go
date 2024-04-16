package tensor

import (
    "fmt"
    "math/rand"
    "golang.org/x/exp/slices"
)

type D2 []D1

func NewD2Zeros(r, c int) D2 {
	y := make(D2, r)
	for i := range y {
		y[i] = make(D1, c)
	}
	return y
}

func NewD2ZerosLike(x D2) D2 {
    y := make(D2, len(x))
    for i := range y {
        y[i] = make(D1, len(x[i]))
    }
    return y
}

func NewD2Ones(r, c int) D2 {
	y := make(D2, r)
	for i := range y {
		y[i] = NewD1Ones(c)
	}
	return y
}

func NewD2RandomUniform(r, c int, min, max float64, random *rand.Rand) D2 {
    y := make(D2, r)
    for i := range y {
        y[i] = NewD1RandomUniform(c, min, max, random)
    }
    return y
}

func NewD2He(r, c int, random *rand.Rand) D2 {
    y := make(D2, r)
    for i := range y {
        y[i] = NewD1He(c, random)
    }
    return y
}

func (d2 D2) Zeros() {
    for i := range d2 {
        d2[i].Zeros()
    }
}

func (d2 D2) Clone() D2 {
    y := make(D2, len(d2))
    for i := range y {
        y[i] = slices.Clone(d2[i])
    }
    return y
}

func (d2 D2) AddScalar(scalar float64) {
    for i := range d2 {
        d2[i].AddScalar(scalar)
    }
}

func (d2 D2) AddD1(d1 D1) error {
    for i := range d2 {
        err := d2[i].Add(d1)
        if err != nil {
            return err
        }
    }
    return nil
}

func (d2 D2) Add(other D2) error {
    if len(d2) != len(other) {
        return fmt.Errorf("tensor.D2の行数が一致しないため、加算できません。")
    }

    for i := range d2 {
        err := d2[i].Add(other[i])
        if err != nil {
            return err
        }
    }
    return nil
}

func (d2 D2) SubScalar(scalar float64) {
    for i := range d2 {
        d2[i].SubScalar(scalar)
    }
}

func (d2 D2) SubD1(d1 D1) error {
    for i := range d2 {
        err := d2[i].Sub(d1)
        if err != nil {
            return err
        }
    }
    return nil
}

func (d2 D2) Sub(other D2) error {
    if len(d2) != len(other) {
        return fmt.Errorf("tensor.D2の行数が一致しないため、減算できません")
    }
    for i := range d2 {
        err := d2[i].Sub(other[i])
        if err != nil {
            return err
        }
    }
    return nil
}

func (d2 D2) MulScalar(scalar float64) {
    for i := range d2 {
        d2[i].MulScalar(scalar)
    }
}

func (d2 D2) MulD1(d1 D1) error {
    for i := range d2 {
        err := d2[i].Mul(d1)
        if err != nil {
            return err
        }
    }
    return nil
}

func (d2 D2) Mul(other D2) error {
    if len(d2) != len(other) {
        return fmt.Errorf("tensor.D2の行数が一致しないため、乗算できません")
    }
    for i := range d2 {
        err := d2[i].Mul(other[i])
        if err != nil {
            return err
        }
    }
    return nil
}

func (d2 D2) DivScalar(scalar float64) {
    for i := range d2 {
        d2[i].DivScalar(scalar)
    }
}

func (d2 D2) DivD1(d1 D1) error {
    for i := range d2 {
        err := d2[i].Div(d1)
        if err != nil {
            return err
        }
    }
    return nil
}

func (d2 D2) Div(other D2) error {
    if len(d2) != len(other) {
        return fmt.Errorf("tensor.D2の行数が一致しないため、除算できません。")
    }

    for i := range d2 {
        err := d2[i].Div(other[i])
        if err != nil {
            return err
        }
    }
    return nil
}

func (d2 D2) Transpose() D2 {
    yr, yc := len(d2[0]), len(d2)
	y := NewD2Zeros(yr, yc)
	for i := 0; i < yr; i++ {
		for j := 0; j < yc; j++ {
			y[i][j] = d2[j][i]
		}
	}
	return y
}

func (d2 D2) DotProduct(other D2) (D2, error) {
	if len(d2[0]) != len(other) {
		return nil, fmt.Errorf("tensor.D2の列数と行数が一致しないため、内積の計算ができません。")
	}

    var err error
    t := other.Transpose()
	y := NewD2Zeros(len(d2), len(other[0]))
	for i := range d2 {
        row := d2[i]
        for j := range t {
            y[i][j], err = row.DotProduct(t[j])
            if err != nil {
                return D2{}, err
            }
        }
    }
	return y, nil
}

func D2AddScalar(d2 D2, scalar float64) D2 {
    y := d2.Clone()
    y.AddScalar(scalar)
    return y
}

func D2AddD1(d2 D2, d1 D1) (D2, error) {
    y := d2.Clone()
    err := y.AddD1(d1)
    return y, err
}

func D2Add(d2, other D2) (D2, error) {
    y := d2.Clone()
    err := y.Add(other)
    return y, err
}

func D2SubScalar(d2 D2, scalar float64) D2 {
    y := d2.Clone()
    y.SubScalar(scalar)
    return y
}

func D2SubD1(d2 D2, d1 D1) (D2, error) {
    y := d2.Clone()
    err := y.SubD1(d1)
    return y, err
}

func D2Sub(d2, other D2) (D2, error) {
    y := d2.Clone()
    err := y.Sub(other)
    return y, err
}

func D2MulScalar(d2 D2, scalar float64) D2 {
    y := d2.Clone()
    y.MulScalar(scalar)
    return y
}

func D2MulD1(d2 D2, d1 D1) (D2, error) {
    y := d2.Clone()
    err := y.MulD1(d1)
    return y, err
}


func D2Mul(d2, other D2) (D2, error) {
    y := d2.Clone()
    err := y.Mul(other)
    return y, err
}


func D2DivScalar(d2 D2, scalar float64) D2 {
    y := d2.Clone()
    y.DivScalar(scalar)
    return y
}

func D2DivD1(d2 D2, d1 D1) (D2, error) {
    y := d2.Clone()
    err := y.DivD1(d1)
    return y, err
}

func D2Div(d2, other D2) (D2, error) {
    y := d2.Clone()
    err := y.Div(other)
    return y, err
}