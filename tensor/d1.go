package tensor

import (
    "fmt"
    "golang.org/x/exp/slices"
    "github.com/sw965/omw"
)

type D1 []float64

func (d1 D1) AddScalar(scalar float64) {
    for i := range d1 {
        d1[i] += scalar
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

func (d1 D1) SubScalar(scalar float64) {
    for i := range d1 {
        d1[i] -= scalar
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

func (d1 D1) MulScalar(scalar float64) {
    for i := range d1 {
        d1[i] *= scalar
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

func (d1 D1) DivScalar(scalar float64) {
    for i := range d1 {
        d1[i] /= scalar
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

func (d1 D1) MapFunc(f func(float64)float64) D1 {
    return omw.MapFunc[D1](d1, f)
}

func (d1 D1) Clone() D1 {
    return slices.Clone(d1)
}

func (d1 D1) Copy(other D1) {
    for i := range d1 {
        d1[i] = other[i]
    }
}

func (d1 D1) Zeros() {
    for i := range d1 {
        d1[i] = 0
    }
}

func (d1 D1) Max() float64 {
    return omw.Max(d1...)
}

func D1AddScalar(d1 D1, other float64) D1 {
    y := slices.Clone(d1)
    y.AddScalar(other)
    return y
}

func D1Add(d1, other D1) (D1, error) {
    y := slices.Clone(d1)
    err := y.Add(other)
    return y, err
}

func D1SubScalar(d1 D1, scalar float64) D1 {
    y := slices.Clone(d1)
    y.SubScalar(scalar)
    return y
}

func D1Sub(d1, other D1) (D1, error) {
    y := slices.Clone(d1)
    err := y.Sub(other)
    return y, err
}

func D1MulScalar(d1 D1, scalar float64) D1 {
    y := slices.Clone(d1)
    y.MulScalar(scalar)
    return y
}

func D1Mul(d1, other D1) (D1, error) {
    y := slices.Clone(d1)
    err := y.Mul(other)
    return y, err
}

func D1DivScalar(d1 D1, scalar float64) D1 {
    y := slices.Clone(d1)
    y.DivScalar(scalar)
    return y
}

func D1Div(d1, other D1) (D1, error) {
    y := slices.Clone(d1)
    err := y.Div(other)
    return y, err
}