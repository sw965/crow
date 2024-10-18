package tensor

import (
	"fmt"
	"golang.org/x/exp/slices"
)

type D2 []D1

func (d2 D2) AddScalar(s float64) {
	for i := range d2 {
		d2[i].AddScalar(s)
	}
}

func (d2 D2) AddD1Row(d1 D1) error {
	for i := range d2 {
		err := d2[i].Add(d1)
		if err != nil {
			return err
		}
	}
	return nil
}

func (d2 D2) AddD1Col(d1 D1) error {
	if len(d2) != len(d1) {
		return fmt.Errorf("tensor.D2の行数とtensor.D1の要素数が一致しないため、加算できません。")
	}
	for i := range d2 {
		d2[i].AddScalar(d1[i])
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

func (d2 D2) SubScalar(s float64) {
	for i := range d2 {
		d2[i].SubScalar(s)
	}
}

func (d2 D2) SubD1Row(d1 D1) error {
	for i := range d2 {
		err := d2[i].Sub(d1)
		if err != nil {
			return err
		}
	}
	return nil
}

func (d2 D2) SubD1Col(d1 D1) error {
	if len(d2) != len(d1) {
		return fmt.Errorf("tensor.D2の行数とtensor.D1の要素数が一致しないため、減算できません。")
	}
	for i := range d2 {
		d2[i].SubScalar(d1[i])
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

func (d2 D2) MulScalar(s float64) {
	for i := range d2 {
		d2[i].MulScalar(s)
	}
}

func (d2 D2) MulD1Row(d1 D1) error {
	for i := range d2 {
		err := d2[i].Mul(d1)
		if err != nil {
			return err
		}
	}
	return nil
}

func (d2 D2) MulD1Col(d1 D1) error {
	if len(d2) != len(d1) {
		return fmt.Errorf("tensor.D2の行数とtensor.D1の要素数が一致しないため、乗算できません。")
	}
	for i := range d2 {
		d2[i].MulScalar(d1[i])
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

func (d2 D2) DivScalar(s float64) {
	for i := range d2 {
		d2[i].DivScalar(s)
	}
}

func (d2 D2) DivD1Row(d1 D1) error {
	for i := range d2 {
		err := d2[i].Div(d1)
		if err != nil {
			return err
		}
	}
	return nil
}

func (d2 D2) DivD1Col(d1 D1) error {
	if len(d2) != len(d1) {
		return fmt.Errorf("tensor.D2の行数とtensor.D1の要素数が一致しないため、除算できません。")
	}
	for i := range d2 {
		d2[i].DivScalar(d1[i])
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

func (d2 D2) DotProduct(other D2) D2 {
	y := make(D2, len(d2))
	for i := range y {
		y[i] = make(D1, len(other[0]))
		for j := range y[i] {
			for k := range d2[i] {
				y[i][j] += d2[i][k] * other[k][j]
			}
		}
	}
	return y
}

func (d2 D2) Copy(src D2) {
	for i := range d2 {
		d2[i].Copy(src[i])
	}
}

func (d2 D2) Clone() D2 {
	y := make(D2, len(d2))
	for i := range y {
		y[i] = slices.Clone(d2[i])
	}
	return y
}

func (d2 D2) MaxRow() D1 {
	max := make(D1, len(d2))
	for i := range d2 {
		max[i] = d2[i].Max()
	}
	return max
}

func (d2 D2) MapFunc(f func(float64) float64) D2 {
	y := make(D2, len(d2))
	for i := range d2 {
		y[i] = d2[i].MapFunc(f)
	}
	return y
}

func D2AddScalar(d2 D2, s float64) D2 {
	y := d2.Clone()
	y.AddScalar(s)
	return y
}

func D2AddD1Row(d2 D2, d1 D1) (D2, error) {
	y := d2.Clone()
	err := y.AddD1Row(d1)
	return y, err
}

func D2AddD1Col(d2 D2, d1 D1) (D2, error) {
	y := d2.Clone()
	err := y.AddD1Col(d1)
	return y, err
}

func D2Add(a, b D2) (D2, error) {
	y := a.Clone()
	err := y.Add(b)
	return y, err
}

func D2SubScalar(d2 D2, s float64) D2 {
	y := d2.Clone()
	y.SubScalar(s)
	return y
}

func D2SubD1Row(d2 D2, d1 D1) (D2, error) {
	y := d2.Clone()
	err := y.SubD1Row(d1)
	return y, err
}

func D2SubD1Col(d2 D2, d1 D1) (D2, error) {
	y := d2.Clone()
	err := y.SubD1Col(d1)
	return y, err
}

func D2Sub(a, b D2) (D2, error) {
	y := a.Clone()
	err := y.Sub(b)
	return y, err
}

func D2MulScalar(d2 D2, s float64) D2 {
	y := d2.Clone()
	y.MulScalar(s)
	return y
}

func D2MulD1Row(d2 D2, d1 D1) (D2, error) {
	y := d2.Clone()
	err := y.MulD1Row(d1)
	return y, err
}

func D2MulD1Col(d2 D2, d1 D1) (D2, error) {
	y := d2.Clone()
	err := y.MulD1Col(d1)
	return y, err
}

func D2Mul(a, b D2) (D2, error) {
	y := a.Clone()
	err := y.Mul(b)
	return y, err
}

func D2DivScalar(d2 D2, s float64) D2 {
	y := d2.Clone()
	y.DivScalar(s)
	return y
}

func D2DivD1Row(d2 D2, d1 D1) (D2, error) {
	y := d2.Clone()
	err := y.DivD1Row(d1)
	return y, err
}

func D2DivD1Col(d2 D2, d1 D1) (D2, error) {
	y := d2.Clone()
	err := y.DivD1Col(d1)
	return y, err
}

func D2Div(a, b D2) (D2, error) {
	y := a.Clone()
	err := y.Div(b)
	return y, err
}
