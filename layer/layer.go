package layer

import (
	"fmt"

	"github.com/sw965/crow/tensor"
	"github.com/sw965/crow/mlfuncs"
	"github.com/sw965/omw"
)

type VarKey int

const (
	KEY_W VarKey = iota
	KEY_B
	KEY_ALPHA
)

type ScalarVarMap map[VarKey]float64

type ScalarTrainVarManager struct {
	Param ScalarVarMap
	Grad ScalarVarMap
}

func (v *ScalarTrainVarManager) AddGrad(dParam float64) {
	v.Grad += dParam
}

func (v *ScalarTrainVarManager) DivGrad(n float64) {
	for key, _ := range v.Grad {
		v.Grad[key] /= n
	} 
}

type ScalarTrainVarManagers []*ScalarTrainVarManager

type D1VarMap map[VarKey]tensor.D1

type D1TrainVarManager struct {
	Param D1VarMap
	Grad D1VarMap
}

func (v *D1TrainVarManager) AddGrad(dParam tensor.D1) {
	for key, _ := range v.Grad {
		grad := v.Grad[key]
		for i := range grad {
			grad[i] = dParam[i]
		}
	}
}

func (v *D1TrainVarManager) DivGrad(n float64) {
	for key, _ := range v.Grad {
		grad := v.Grad[key]
		for i := range g {
			grad[i] /= n
		}
	}
}

type D1TrainVarManagers []*D1TrainVarManager

type D2VarMap map[VarKey]tensor.D1

type D2TrainVarManager struct {
	Param D2VarMap
	Grad D2VarMap
}

func (v *D2TrainVarManager) AddGrad(dParam tensor.D2) {
	for key, _ := range v.Grad {
		grad := v.Grad[k]
		for i := range gradk {
			dParami := dParam[i]
			gradi := grad[i]
			for j := range gradki {
				gradi[j] = dParami[j]
			}
		}
	}
}

func (v *D2TrainVarManager) DivGrad(n float64) {
	for key, _ := range pg.Gr {
		grad := v.Grad[key]
		for i := range grad {
			for j := range grad[i] {
				grad[i][j] /= n
			}
		}
	}
}

type D2TrainVarManagers []*D2TrainVarManager

type D1Forward func(tensor.D1, D1Backwards) (tensor.D1, D1Backwards, error)
type D1Forwards []D1Forward

func (fs D1Forwards) Run(x tensor.D1) (tensor.D1, D1Backwards, error) {
	bs := make(D1Backwards, 0, len(fs))
	var err error
	for _, f := range fs {
		x, bs, err = f(x, bs)
		if err != nil {
			return tensor.D1{}, D1Backwards{}, err
		}
	}
	return x, bs, err
}

type D1Backward func(tensor.D1) (tensor.D1, error)
type D1Backwards []D1Backward

func (bs D1Backwards) Run(chain tensor.D1) (tensor.D1, error) {
	var err error
	bs = omw.Reverse(bs)
	for _, b := range bs {
		chain, err = b(chain)
		if err != nil {
			return tensor.D1{}, err
		}
	}
	return chain, nil
}

func NewAffineForward(d2v *D2TrainVarManager, d1v D1TrainVarManager) Forward {
	return func(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
		w := d2v.Param[KEY_W]
		b := d1v.Param[KEY_B]

		dot, err := tensor.D2{x}.DotProduct(w)
		if err != nil {
			return tensor.D1{}, Backwards{}, err
		}
		y, err := dot[0].Add(b[0])

		var backward Backward
		backward = func(chain tensor.D1) (tensor.D1, error) {
			// ∂L/∂x
			dx, err := tensor.D2{chain}.DotProduct(w.Transpose())
			if err != nil {
				return tensor.D1{}, err
			}
		
			// ∂L/∂w
			dw, err := tensor.D2{x}.Transpose().DotProduct(tensor.D2{chain})
			if err != nil {
				return tensor.D1{}, err
			}
			d2v.AddGrad(dw)

			// ∂L/∂b
			db := chain
			d1v.AddGrad(db)
			return dx[0], err
		}
		backwards = append(backwards, backward)
		return y, backwards, err
	}
}

func ReLUForward(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
	y := mlfuncs.ReLU(x)
	var backward Backward
	backward = func(chain tensor.D1) (tensor.D1, error) {
		dydx := mlfuncs.ReLUDerivative(x)
		// ∂L/∂x
		dx, err := dydx.Mul(chain)
		return dx, err
	}
	backwards = append(backwards, backward)
	return y, backwards, nil
}

func NewPReLUForward(sv *ScalarTrainVarManager) D1Forward {
	return func(x tensor.D1, backwards D1Backwards) (tensor.D1, D1Backwards, error) {
		alpha := sv.Param[KEY_ALPHA]
		y := mlfuncs.PReLU(x, alpha)
		var backward Backward
		backward = func(chain tensor.D1) (tensor.D1, error) {
			dydx, dydAlpha := mlfuncs.PReLUDerivative(x, alpha)

			// ∂L/∂alpha
			dAlpha := omw.Sum(chain.MulScalar(dydAlpha)...)
			sv.AddGrad(dAlpha)

			// ∂L/∂x
			dx, err := dydx.Mul(chain)
			return dx, err
		}
		backwards = append(backwards, backward)
		return y, backwards, nil
	}
}

func SigmoidForward(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
	y := mlfuncs.Sigmoid(x)
	var backward Backward
	backward = func(chain tensor.D1) (tensor.D1, error) {
		dydx := mlfuncs.SigmoidGrad(y)
		// ∂L/∂x
		dx, err := dydx.Mul(chain)
		return dx, err
	}
	backwards = append(backwards, backward)
	return y, backwards, nil
}

func TanhForward(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
	y := mlfuncs.Tanh(x)
	var backward Backward
	backward = func(chain tensor.D1) (tensor.D1, error) {
		dydx := mlfuncs.TanhGrad(y)
		// ∂L/∂x
		dx, err := dydx.Mul(chain)
		return dx, err
	}
	backwards = append(backwards, backward)
	return y, backwards, nil
}

func MeanSquaredErrorForward(y, t tensor.D1, backwards Backwards) (float64, Backwards, error) {
	z, err := mlfuncs.MeanSquaredError(y, t)
	var backward Backward
	backward = func(chain tensor.D1) (tensor.D1, error) {
		dy, err := mlfuncs.MeanSquaredErrorDerivative(y, t)
		if err != nil {
			return tensor.D1{}, err
		}
		return dy.Mul(chain)
	}
	backwards = append(backwards, backward)
	return z, backwards, err
}