package tensor

import (
	"fmt"
	"math/rand"
	"github.com/sw965/omw/fn"
)

type D3 []D2

func NewD3Zeros(r, c, d int) D3 {
	ret := make(D3, r)
	for i := range ret {
		ret[i] = NewD2Zeros(c, d)
	}
	return ret
}

func NewD3ZerosLike(d3 D3) D3 {
	return fn.Map[D3](d3, NewD2ZerosLike)
}

func NewD3Ones(r, c, d int) D3 {
	ret := make(D3, r)
	for i := range ret {
		ret[i] = NewD2Ones(c, d)
	}
	return ret
}

func NewD3OnesLike(d3 D3) D3 {
	return fn.Map[D3](d3, NewD2OnesLike)
}

func NewD3RandUniform(d, r, c int, min, max float32, rng *rand.Rand) D3 {
	ret := make(D3, d)
	for i := range ret {
		ret[i] = NewD2RandUniform(r, c, min, max, rng)
	}
	return ret
}

func NewD3He(d, r, c int, rng *rand.Rand) D3 {
	he := make(D3, d)
	for i := range he {
		he[i] = NewD2He(r, c, rng)
	}
	return he
}

func (d3 D3) AddScalar(s float32) {
	for i := range d3 {
		d3[i].AddScalar(s)
	}
}

func (d3 D3) AddD1Depth(d1 D1) error {
	if len(d3) != len(d1) {
		return fmt.Errorf("tensor.D3の奥数とtensor.D1の要素数が一致しないため、加算できません。")
	}
	for i := range d3 {
		d3[i].AddScalar(d1[i])
	}
	return nil
}

func (d3 D3) AddD1Row(d1 D1) error {
	for i := range d3 {
		err := d3[i].AddD1Row(d1)
		if err != nil {
			return err
		}
	}
	return nil
}

func (d3 D3) AddD1Col(d1 D1) error {
	for i := range d3 {
		err := d3[i].AddD1Col(d1)
		if err != nil {
			return err
		}
	}
	return nil
}

func (d3 D3) AddD2(d2 D2) error {
	for i := range d3 {
		err := d3[i].Add(d2)
		if err != nil {
			return err
		}
	}
	return nil
}

func (d3 D3) Add(d2 D3) error {
	for i := range d3 {
		err := d3[i].Add(d2[i])
		if err != nil {
			return err
		}
	}
	return nil
}

func (d3 D3) SubScalar(s float32) {
	for i := range d3 {
		d3[i].SubScalar(s)
	}
}

func (d3 D3) SubD1Depth(d1 D1) error {
	if len(d3) != len(d1) {
		return fmt.Errorf("tensor.D3の奥数とtensor.D1の要素数が一致しないため、減算できません。")
	}
	for i := range d3 {
		d3[i].SubScalar(d1[i])
	}
	return nil
}

func (d3 D3) SubD1Row(d1 D1) error {
	for i := range d3 {
		err := d3[i].SubD1Row(d1)
		if err != nil {
			return err
		}
	}
	return nil
}

func (d3 D3) SubD1Col(d1 D1) error {
	for i := range d3 {
		err := d3[i].SubD1Col(d1)
		if err != nil {
			return err
		}
	}
	return nil
}

func (d3 D3) SubD2(d2 D2) error {
	for i := range d3 {
		err := d3[i].Sub(d2)
		if err != nil {
			return err
		}
	}
	return nil
}

func (d3 D3) Sub(d2 D3) error {
	if len(d3) != len(d2) {
		return fmt.Errorf("tensor.D3の行数が一致しないため、減算できません")
	}
	for i := range d3 {
		err := d3[i].Sub(d2[i])
		if err != nil {
			return err
		}
	}
	return nil
}

func (d3 D3) MulScalar(s float32) {
	for i := range d3 {
		d3[i].MulScalar(s)
	}
}

func (d3 D3) MulD1Depth(d1 D1) error {
	if len(d3) != len(d1) {
		return fmt.Errorf("tensor.D3の奥数とtensor.D1の要素数が一致しないため、乗算できません。")
	}
	for i := range d3 {
		d3[i].MulScalar(d1[i])
	}
	return nil
}

func (d3 D3) MulD1Row(d1 D1) error {
	for i := range d3 {
		err := d3[i].MulD1Row(d1)
		if err != nil {
			return err
		}
	}
	return nil
}

func (d3 D3) MulD1Col(d1 D1) error {
	for i := range d3 {
		err := d3[i].MulD1Col(d1)
		if err != nil {
			return err
		}
	}
	return nil
}

func (d3 D3) MulD2(d2 D2) error {
	for i := range d3 {
		err := d3[i].Mul(d2)
		if err != nil {
			return err
		}
	}
	return nil
}

func (d3 D3) Mul(d2 D3) error {
	if len(d3) != len(d2) {
		return fmt.Errorf("tensor.D3の行数が一致しないため、乗算できません")
	}
	for i := range d3 {
		err := d3[i].Mul(d2[i])
		if err != nil {
			return err
		}
	}
	return nil
}

func (d3 D3) DivScalar(s float32) {
	for i := range d3 {
		d3[i].DivScalar(s)
	}
}

func (d3 D3) DivD1Depth(d1 D1) error {
	if len(d3) != len(d1) {
		return fmt.Errorf("tensor.D3の奥数とtensor.D1の要素数が一致しないため、除算できません。")
	}
	for i := range d3 {
		d3[i].DivScalar(d1[i])
	}
	return nil
}

func (d3 D3) DivD1Row(d1 D1) error {
	for i := range d3 {
		err := d3[i].DivD1Row(d1)
		if err != nil {
			return err
		}
	}
	return nil
}

func (d3 D3) DivD1Col(d1 D1) error {
	for i := range d3 {
		err := d3[i].DivD1Col(d1)
		if err != nil {
			return err
		}
	}
	return nil
}

func (d3 D3) DivD2(d2 D2) error {
	for i := range d3 {
		err := d3[i].Div(d2)
		if err != nil {
			return err
		}
	}
	return nil
}

func (d3 D3) Div(d2 D3) error {
	if len(d3) != len(d2) {
		return fmt.Errorf("tensor.D3の行数が一致しないため、除算できません")
	}
	for i := range d3 {
		err := d3[i].Div(d2[i])
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

func (d3 D3) Size() int {
	size := 0
	for _, d2 := range d3 {
		size += d2.Size()
	}
	return size
}

func (d3 D3) Flatten() D1 {
	flat := make(D1, 0, d3.Size())
	for _, d2 := range d3 {
		flat = append(flat, d2.Flatten()...)
	}
	return flat
}

func (d3 D3) Reciprocal() D3 {
	y := make(D3, len(d3))
	for i, d2 := range d3 {
		y[i] = d2.Reciprocal()
	}
	return y
}

func D3AddScalar(d3 D3, s float32) D3 {
	y := d3.Clone()
	y.AddScalar(s)
	return y
}

func D3AddD1Depth(d3 D3, d1 D1) (D3, error) {
	y := d3.Clone()
	err := y.AddD1Depth(d1)
	return y, err
}

func D3AddD1Row(d3 D3, d1 D1) (D3, error) {
	y := d3.Clone()
	err := y.AddD1Row(d1)
	return y, err
}

func D3AddD1Col(d3 D3, d1 D1) (D3, error) {
	y := d3.Clone()
	err := y.AddD1Col(d1)
	return y, err
}

func D3AddD2(d3 D3, d2 D2) (D3, error) {
	y := d3.Clone()
	err := y.AddD2(d2)
	return y, err
}

func D3Add(a, b D3) (D3, error) {
	y := a.Clone()
	err := y.Add(b)
	return y, err
}

func D3SubScalar(d3 D3, s float32) D3 {
	y := d3.Clone()
	y.SubScalar(s)
	return y
}

func D3SubD1Depth(d3 D3, d1 D1) (D3, error) {
	y := d3.Clone()
	err := y.SubD1Depth(d1)
	return y, err
}

func D3SubD1Row(d3 D3, d1 D1) (D3, error) {
	y := d3.Clone()
	err := y.SubD1Row(d1)
	return y, err
}

func D3SubD1Col(d3 D3, d1 D1) (D3, error) {
	y := d3.Clone()
	err := y.SubD1Col(d1)
	return y, err
}

func D3SubD2(d3 D3, d2 D2) (D3, error) {
	y := d3.Clone()
	err := y.SubD2(d2)
	return y, err
}

func D3Sub(a, b D3) (D3, error) {
	y := a.Clone()
	err := y.Sub(b)
	return y, err
}

func D3MulScalar(d3 D3, s float32) D3 {
	y := d3.Clone()
	y.MulScalar(s)
	return y
}

func D3MulD1Depth(d3 D3, d1 D1) (D3, error) {
	y := d3.Clone()
	err := y.MulD1Depth(d1)
	return y, err
}

func D3MulD1Row(d3 D3, d1 D1) (D3, error) {
	y := d3.Clone()
	err := y.MulD1Row(d1)
	return y, err
}

func D3MulD1Col(d3 D3, d1 D1) (D3, error) {
	y := d3.Clone()
	err := y.MulD1Col(d1)
	return y, err
}

func D3MulD2(d3 D3, d2 D2) (D3, error) {
	y := d3.Clone()
	err := y.MulD2(d2)
	return y, err
}

func D3Mul(a, b D3) (D3, error) {
	y := a.Clone()
	err := y.Mul(b)
	return y, err
}

func D3DivScalar(d3 D3, s float32) D3 {
	y := d3.Clone()
	y.DivScalar(s)
	return y
}

func D3DivD1Depth(d3 D3, d1 D1) (D3, error) {
	y := d3.Clone()
	err := y.DivD1Depth(d1)
	return y, err
}

func D3DivD1Row(d3 D3, d1 D1) (D3, error) {
	y := d3.Clone()
	err := y.DivD1Row(d1)
	return y, err
}

func D3DivD1Col(d3 D3, d1 D1) (D3, error) {
	y := d3.Clone()
	err := y.DivD1Col(d1)
	return y, err
}

func D3DivD2(d3 D3, d2 D2) (D3, error) {
	y := d3.Clone()
	err := y.DivD2(d2)
	return y, err
}

func D3Div(a, b D3) (D3, error) {
	y := a.Clone()
	err := y.Div(b)
	return y, err
}