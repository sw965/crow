package tensor

import (
    "fmt"
    "github.com/sw965/omw/fn"
    omwmath "github.com/sw965/omw/math"
    omwslices "github.com/sw965/omw/slices"
)

type D1 []float64

func (d1 D1) Add(other D1) (D1, error) {
    return fn.Zip[D1, D1, D1](d1, other, omwmath.Add[float64])
}

func (d1 D1) Sub(other D1) (D1, error) {
    return fn.Zip[D1, D1, D1](d1, other, omwmath.Sub[float64])
}

func (d1 D1) Mul(other D1) (D1, error) {
    return fn.Zip[D1, D1, D1](d1, other, omwmath.Mul[float64])
}

func (d1 D1) Div(other D1) (D1, error) {
    return fn.Zip[D1, D1, D1](d1, other, omwmath.Div[float64])
}

func (d1 D1) DotProduct(other D1) (float64, error) {
    mul, err := d1.Mul(other)
    return omwmath.Sum(mul...), err
}

type D2 []D1

func NewD2(xss [][]float64) (D2, error) {
    c := len(xss[0])
    d2 := make(D2, len(xss))
    for i, xs := range xss {
        n := len(xs)
        if n != c {
            return D2{}, fmt.Errorf("tensor.D2 の 列数 は 全て同じでなければならない")
        }
        
        d1:= make(D1, len(xs))
        for j, x := range xs {
            d1[j] = x
        }
        d2[i] = d1
    }
    return d2, nil
}

func NewD2Zeros(r, c int) D2 {
    return omwslices.Zeros2D[D2, D1](r, c)
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
	transposed := other.Transpose()
	y := make(D2, len(d2))
    var err error
	for i := range y {
		y[i] = make(D1, len(transposed))
		for j := range y[i] {
			y[i][j], err = d2[i].DotProduct(transposed[j])
            if err != nil {
                return D2{}, err
            }
		}
	}
	return y, nil
}