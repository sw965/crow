package tensor

import (
	"fmt"
)

type D3 []D2

func (d3 D3) AddScalar(s float64) {
	for i := range d3 {
		d3[i].AddScalar(s)
	}
}

func (d3 D3) Add(other D3) error {
	if len(d3) != len(d3) {
		return fmt.Errorf("tensor.D3の行数が一致しないため、加算できません")
	}
	for i := range d3 {
		err := d3[i].Add(other[i])
		if err != nil {
			return err
		}
	}
	return nil
}

func (d3 D3) SubScalar(s float64) {
	for i := range d3 {
		d3[i].SubScalar(s)
	}
}

func (d3 D3) Sub(other D3) error {
	if len(d3) != len(d3) {
		return fmt.Errorf("tensor.D3の行数が一致しないため、減算できません")
	}
	for i := range d3 {
		err := d3[i].Sub(other[i])
		if err != nil {
			return err
		}
	}
	return nil
}

func (d3 D3) MulScalar(s float64) {
	for i := range d3 {
		d3[i].MulScalar(s)
	}
}

func (d3 D3) Mul(other D3) error {
	if len(d3) != len(d3) {
		return fmt.Errorf("tensor.D3の行数が一致しないため、乗算できません")
	}
	for i := range d3 {
		err := d3[i].Mul(other[i])
		if err != nil {
			return err
		}
	}
	return nil
}

func (d3 D3) DivScalar(s float64) {
	for i := range d3 {
		d3[i].DivScalar(s)
	}
}

func (d3 D3) Div(other D3) error {
	if len(d3) != len(d3) {
		return fmt.Errorf("tensor.D3の行数が一致しないため、除算できません")
	}
	for i := range d3 {
		err := d3[i].Div(other[i])
		if err != nil {
			return err
		}
	}
	return nil
}

func (d3 D3) Copy(src D3) {
	for i := range d3 {
		d3[i].Copy(src[i])
	}
}

func (d3 D3) Clone() D3 {
	y := make(D3, len(d3))
	for i := range y {
		y[i] = d3[i].Clone()
	}
	return y
}

func (d3 D3) MaxRow() D2 {
	max := make(D2, len(d3))
	for i := range d3 {
		max[i] = d3[i].MaxRow()
	}
	return max
}

func (d3 D3) MapFunc(f func(float64) float64) D3 {
	y := make(D3, len(d3))
	for i := range d3 {
		y[i] = d3[i].MapFunc(f)
	}
	return y
}

func D3AddScalar(d3 D3, s float64) D3 {
	y := d3.Clone()
	y.AddScalar(s)
	return y
}

func D3Add(a, b D3) (D3, error) {
	y := a.Clone()
	err := y.Add(b)
	return y, err
}

func D3SubScalar(d3 D3, s float64) D3 {
	y := d3.Clone()
	y.SubScalar(s)
	return y
}

func D3Sub(a, b D3) (D3, error) {
	y := a.Clone()
	err := y.Sub(b)
	return y, err
}

func D3MulScalar(d3 D3, s float64) D3 {
	y := d3.Clone()
	y.MulScalar(s)
	return y
}

func D3Mul(a, b D3) (D3, error) {
	y := a.Clone()
	err := y.Mul(b)
	return y, err
}

func D3DivScalar(d3 D3, s float64) D3 {
	y := d3.Clone()
	y.DivScalar(s)
	return y
}

func D3Div(a, b D3) (D3, error) {
	y := a.Clone()
	err := y.Div(b)
	return y, err
}
