package tensor

import (
	"fmt"
	"github.com/sw965/omw"
)

type D3 []D2

func NewD3ZerosLike(d3 D3) D3 {
	return omw.MapFunc[D3](d3, NewD2ZerosLike)
}

func (d3 D3) Zeros() {
	for i := range d3 {
		d3[i].Zeros()
	}
}

func (d3 D3) Clone() D3 {
	y := make(D3, len(d3))
	for i := range y {
		y[i] = d3[i].Clone()
	}
	return y
}

func (d3 D3) AddScalar(scalar float64) {
	for i := range d3 {
		d3[i].AddScalar(scalar)
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

func (d3 D3) SubScalar(scalar float64) {
	for i := range d3 {
		d3[i].SubScalar(scalar)
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

func (d3 D3) MulScalar(scalar float64) {
	for i := range d3 {
		d3[i].MulScalar(scalar)
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

func (d3 D3) DivScalar(scalar float64) {
	for i := range d3 {
		d3[i].DivScalar(scalar)
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

func D3Add(d3, other D3) (D3, error) {
	y := d3.Clone()
	err := y.Add(other)
	return y, err
}

func D3Sub(d3, other D3) (D3, error) {
	y := d3.Clone()
	err := y.Sub(other)
	return y, err
}

func D3Mul(d3, other D3) (D3, error) {
	y := d3.Clone()
	err := y.Mul(other)
	return y, err
}

func D3Div(d3, other D3) (D3, error) {
	y := d3.Clone()
	err := y.Div(other)
	return y, err
}