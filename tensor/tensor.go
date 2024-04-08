package tensor

import (
    "math"
    "fmt"
    "github.com/sw965/omw"
)

type D1 []float64

func (d1 D1) AddScalar(scalar float64) D1 {
    y := make(D1, len(d1))
    for i := range y {
        y[i] = d1[i] + scalar
    }
    return y
}

func (d1 D1) SubScalar(scalar float64) D1 {
    y := make(D1, len(d1))
    for i := range y {
        y[i] = d1[i] - scalar
    }
    return y
}

func (d1 D1) MulScalar(scalar float64) D1 {
    y := make(D1, len(d1))
    for i := range y {
        y[i] = d1[i] * scalar
    }
    return y
}

func (d1 D1) DivScalar(scalar float64) D1 {
    y := make(D1, len(d1))
    for i := range y {
        y[i] = d1[i] / scalar
    }
    return y
}

func (d1 D1) Add(other D1) (D1, error) {
    if len(d1) != len(other) {
        return D1{}, fmt.Errorf("tensor.D1の長さが一致しないため、加算できません。")
    }

    y := make(D1, len(other))
    for i := range y {
        y[i] = d1[i] + other[i]
    }
    return y, nil
}

func (d1 D1) Sub(other D1) (D1, error) {
    if len(d1) != len(other) {
        return D1{}, fmt.Errorf("tensor.D1 の長さが一致しないため、減算できません。")
    }

    y := make(D1, len(d1))
    for i := range y {
        y[i] = d1[i] - other[i]
    }
    return y, nil
}

func (d1 D1) Mul(other D1) (D1, error) {
    if len(d1) != len(other) {
        return D1{}, fmt.Errorf("tensor.D1 の長さが一致しないため、乗算できません。")
    }

    y := make(D1, len(d1))
    for i := range y {
        y[i] = d1[i] * other[i]
    }
    return y, nil
}

func (d1 D1) Div(other D1) (D1, error) {
    if len(d1) != len(other) {
        return D1{}, fmt.Errorf("tensor.D1 の長さが一致しないため、除算できません。")
    }

    y := make(D1, len(d1))
    for i := range y {
        y[i] = d1[i] / other[i]
    }
    return y, nil
}

func (d1 D1) DotProduct(other D1) (float64, error) {
    mul, err := d1.Mul(other)
    return omw.Sum(mul...), err
}

func (d1 D1) Dev() D1 {
    m := omw.Mean(d1...)
    return d1.SubScalar(m)
}

func (d1 D1) Var() float64 {
    n := len(d1)
    sum := 0.0
    dev := d1.Dev()
    for i := range d1 {
        di := dev[i]
        sum += di * di
    }
    return sum / float64(n)
}

func (d1 D1) Std() float64 {
    return math.Sqrt(d1.Var())
}

func (d1 D1) Standardize() D1 {
    dev := d1.Dev()
    std := d1.Std()
    return dev.DivScalar(std)
}

type D2 []D1

func (d2 D2) AddScalar(scalar float64) D2 {
    y := make(D2, len(d2))
    for i := range y {
        y[i] = d2[i].AddScalar(scalar)
    }
    return y
}

func (d2 D2) SubScalar(scalar float64) D2 {
    y := make(D2, len(d2))
    for i := range y {
        y[i] = d2[i].SubScalar(scalar)
    }
    return y
}

func (d2 D2) MulScalar(scalar float64) D2 {
    y := make(D2, len(d2))
    for i := range d2 {
        y[i] = d2[i].MulScalar(scalar)
    }
    return y
}

func (d2 D2) DivScalar(scalar float64) D2 {
    y := make(D2, len(d2))
    for i := range d2 {
        y[i] = d2[i].DivScalar(scalar)
    }
    return y
}

func (d2 D2) AddD1(d1 D1) (D2, error) {
    var err error
    y := make(D2, len(d2))
    for i := range d2 {
        y[i], err = d2[i].Add(d1)
        if err != nil {
            return D2{}, err
        }
    }
    return y, nil
}


func (d2 D2) SubD1(d1 D1) (D2, error) {
    var err error
    y := make(D2, len(d2))
    for i := range d2 {
        y[i], err = d2[i].Sub(d1)
        if err != nil {
            return D2{}, err
        }
    }
    return y, nil
}

func (d2 D2) MulD1(d1 D1) (D2, error) {
    var err error
    y := make(D2, len(d2))
    for i := range d2 {
        y[i], err = d2[i].Mul(d1)
        if err != nil {
            return D2{}, err
        }
    }
    return y, nil
}

func (d2 D2) DivD1(d1 D1) (D2, error) {
    var err error
    y := make(D2, len(d2))
    for i := range d2 {
        y[i], err = d2[i].Div(d1)
        if err != nil {
            return D2{}, err
        }
    }
    return y, nil
}

func (d2 D2) Add(other D2) (D2, error) {
    if len(d2) != len(other) {
        return D2{}, fmt.Errorf("tensor.D2の行数が一致しないため、加算できません。")
    }
    var err error
    y := make(D2, len(d2))
    for i := range d2 {
        y[i], err = d2[i].Add(other[i])
        if err != nil {
            return D2{}, err
        }
    }
    return y, nil
}

func (d2 D2) Sub(other D2) (D2, error) {
    if len(d2) != len(other) {
        return D2{}, fmt.Errorf("tensor.D2の行数が一致しないため、減算できません。")
    }

    var err error
    y := make(D2, len(d2))
    for i := range d2 {
        y[i], err = d2[i].Sub(other[i])
        if err != nil {
            return D2{}, err
        }
    }
    return y, nil
}

func (d2 D2) Mul(other D2) (D2, error) {
    if len(d2) != len(other) {
        return D2{}, fmt.Errorf("tensor.D2の行数が一致しないため、乗算できません。")
    }

    var err error
    y := make(D2, len(d2))
    for i := range d2 {
        y[i], err = d2[i].Mul(other[i])
        if err != nil {
            return D2{}, err
        }
    }
    return y, nil
}

func (d2 D2) Div(other D2) (D2, error) {
    if len(d2) != len(other) {
        return D2{}, fmt.Errorf("tensor.D2の行数が一致しないため、除算できません。")
    }

    var err error
    y := make(D2, len(d2))
    for i := range d2 {
        y[i], err = d2[i].Div(other[i])
        if err != nil {
            return D2{}, err
        }
    }
    return y, nil
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