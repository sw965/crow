package tensor

import (
    "fmt"
    omwmath "github.com/sw965/omw/math"
    "math/rand"
)

type D1 []float64

func NewD1Random(r int, min, max float64, random *rand.Rand) D1 {
    y := make(D1, r)
    for i := range y {
        y[i] = random.Float64()*(max-min) + min
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
    return omwmath.Sum(mul...), err
}

type D2 []D1

func NewD2(xss [][]float64) (D2, error) {
    r := len(xss)
    if r == 0 {
        return nil, fmt.Errorf("tensor.D2を生成する場合、要素を持たなければなりません。 (len(d2) != 0 でなければならない)")
    }

    c := len(xss[0])
    for i, xs := range xss {
        if len(xs) == 0 {
            errMsg := fmt.Sprintf(
                "tensor.D2を生成する場合、全ての行は、最低でも一つの要素を持たなければなりません。。第%d行目の要素数が0です。", i+1,
            )
            return D2{}, fmt.Errorf(errMsg)
        }
        if len(xs) != c {
            errMsg := fmt.Sprintf(
                "tensor.D2を生成する場合、全ての行は、同じ列数でなければなりません。第1行目の列数: %d, 第%d行目の列数: %d",
                c, i+1, len(xs),
            )
            return D2{}, fmt.Errorf(errMsg)
        }
    }
    
    y := NewD2Zeros(r, c)
    for i := range y {
        for j := range y {
            y[i][j] = xss[i][j]
        }
    }
    return y, nil
}

func NewD2Zeros(r, c int) D2 {
    y := make(D2, r)
    for i := range y {
        y[i] = make(D1, c)
    }
    return y
}

func NewD2Random(r, c int, min, max float64, random *rand.Rand) D2 {
    y := make(D2, r)
    for i := range y {
        y[i] = NewD1Random(c, min, max, random)
    }
    return y
}

func (d2 D2) Shape() D2Shape {
    return D2Shape{Row:len(d2), Col:len(d2[0])}
}

func (d2 D2) Add1DByRow(d1 D1) (D2, error) {
    y := NewD2Zeros(len(d2), len(d1))

    for i := range y {
        row := y[i]
        if len(row) != len(d1) {
            errMsg := fmt.Sprintf(
                "tensor.D2 の %d 行目の列数 の長さ (%d) と tensor.D1 の長さ (%d) が一致しないため、1次元ベクトルを2次元テンソルの各行に加算することができません。",
                i+1, len(row), len(d1),
            )
            return D2{}, fmt.Errorf(errMsg)
        }

        for j := range y[i] {
            y[i][j] = d1[j]
        }
    }
    return y, nil
}

func (d2 D2) Mul(other D2) (D2, error) {
    if len(d2) != len(other) {
        return D2{}, fmt.Errorf("tensor.D2の行数が一致しないため、掛け算できません。")
    }

    y := NewD2Zeros(len(d2), len(d2[0]))
    for i := range d2 {
        if len(d2[i]) != len(other[i]) {
            errMsg := fmt.Sprintf("tensor.D2の 第%d行目 の列数が一致しないため、掛け算できません。", i+1)
            return D2{}, fmt.Errorf(errMsg)
        }
        for j := range d2[i] {
            y[i][j] = d2[i][j] * other[i][j]
        }
    }
    return y, nil
}

func (d2 D2) SumByRow() D1 {
    y := make(D1, len(d2))
    for i := range y {
        y[i] = omwmath.Sum(d2[i]...)
    }
    return y
}

func (d2 D2) SumByCol() D1 {
    y := make(D1, len(d2))
    for i := range d2 {
        for j := range d2[i] {
            y[j] += d2[i][j]
        }
    }
    return y
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

	y := NewD2Zeros(len(d2), len(other[0]))
	for i := range y {
		for j := range y[i] {
			sum := 0.0
			for k := range d2[i] {
				sum += d2[i][k] * other[k][j]
			}
			y[i][j] = sum
		}
	}

	return y, nil
}

type D2Shape struct {
    Row int
    Col int
}