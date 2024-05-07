package layer1d

import (
	"math/rand"
	"github.com/sw965/omw"
	"github.com/sw965/crow/mlfuncs/1d"
	"github.com/sw965/crow/tensor"
)

type Forward func(tensor.D1, Backwards) (tensor.D1, Backwards, error)
type Forwards []Forward

func (fs Forwards) Propagate(x tensor.D1) (tensor.D1, Backwards, error) {
	bs := make(Backwards, 0, len(fs))
	var err error
	for _, f := range fs {
		x, bs, err = f(x, bs)
		if err != nil {
			return tensor.D1{}, Backwards{}, err
		}
	}
	y := x
	return y, bs, nil
}

type Backward func(tensor.D1) (tensor.D1, error)
type Backwards []Backward

func (bs Backwards) Propagate(chain tensor.D1) (tensor.D1, error) {
	bs = omw.ReverseElement(bs)
	var err error
	for _, b := range bs {
		chain, err = b(chain)
		if err != nil {
			return tensor.D1{}, err
		}
	}
	return chain, nil
}

func NewAffineForward(w tensor.D2, b tensor.D1, gradW tensor.D2, gradB tensor.D1) Forward {
	return func(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
		dot := tensor.D2{x}.DotProduct(w)
		y, err := tensor.D1Add(dot[0], b)

		var backward Backward
		backward = func(chain tensor.D1) (tensor.D1, error) {
			// ∂L/∂x
			dx := tensor.D2{chain}.DotProduct(w.Transpose())
		
			// ∂L/∂w
			dw := tensor.D2{x}.Transpose().DotProduct(tensor.D2{chain})
			gradW.Copy(dw)

			// ∂L/∂b
			db := chain
			gradB.Copy(db)
			return dx[0], err
		}
		backwards = append(backwards, backward)
		return y, backwards, err
	}
}

func NewReLUForward() Forward {
	return func(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
		y := mlfuncs1d.ReLU(x)
		var backward Backward
		backward = func(chain tensor.D1) (tensor.D1, error) {
			dydx := mlfuncs1d.ReLUDerivative(x)
			// ∂L/∂x
			dx, err := tensor.D1Mul(dydx, chain)
			return dx, err
		}
		backwards = append(backwards, backward)
		return y, backwards, nil
	}
}

func NewLeakyReLUForward(alpha float64) Forward {
	return func(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
		y := mlfuncs1d.LeakyReLU(x, alpha)
		var backward Backward
		backward = func(chain tensor.D1) (tensor.D1, error) {
			dydx := mlfuncs1d.LeakyReLUDerivative(x, alpha)
			dx, err := tensor.D1Mul(dydx, chain)
			return dx, err
		}
		backwards = append(backwards, backward)
		return y, backwards, nil
	}
}

func NewParamReLUForward(alpha, gradAlpha *float64) Forward {
	return func(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
		y := mlfuncs1d.LeakyReLU(x, *alpha)
		var backward Backward
		backward = func(chain tensor.D1) (tensor.D1, error) {
			dydx, dydVectorizedAlpha := mlfuncs1d.ParamReLUDerivative(x, *alpha)

			// ∂L/∂dVectorizedAlpha
			dVectorizedAlpha, err := tensor.D1Mul(dydVectorizedAlpha, chain)
			if err != nil {
				return tensor.D1{}, err
			}
			// ∂L/∂alpha
			*gradAlpha = omw.Sum(dVectorizedAlpha...)

			// ∂L/∂x
			dx, err := tensor.D1Mul(dydx, chain)
			return dx, err
		}
		backwards = append(backwards, backward)
		return y, backwards, nil
	}
}

func NewRandReLUForward(min, max float64, isTrain *bool, r *rand.Rand) Forward {
	return func(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
		y, noise := mlfuncs1d.RandReLU(x, min, max, *isTrain, r)
		var backward Backward
		backward = func(chain tensor.D1) (tensor.D1, error) {
			dydx := mlfuncs1d.LeakyReLUDerivative(x, noise)
			// ∂L/∂x
			dx, err := tensor.D1Mul(dydx, chain)
			return dx, err
		}
		backwards = append(backwards, backward)
		return y, backwards, nil
	}
}

func NewParamRandReLUForward(alpha *float64, min, max float64, gradAlpha *float64, isTrain *bool, r *rand.Rand) Forward {
	return func(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
		y, noise := mlfuncs1d.ParamRandReLU(x, *alpha, min, max, *isTrain, r)
		var backward Backward
		backward = func(chain tensor.D1) (tensor.D1, error) {
			dydx, dydVectorizedAlpha := mlfuncs1d.ParamRandReLUDerivative(x, *alpha, noise)

			// ∂L/∂dVectorizedAlpha
			dVectorizedAlpha, err := tensor.D1Mul(dydVectorizedAlpha, chain)
			if err != nil {
				return tensor.D1{}, err
			}
			// ∂L/∂alpha
			*gradAlpha = omw.Sum(dVectorizedAlpha...)

			// ∂L/∂x
			dx, err := tensor.D1Mul(dydx, chain)
			return dx, err
		}
		backwards = append(backwards, backward)
		return y, backwards, nil
	}
}

func NewSigmoidForward() Forward {
	return func(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
		y := mlfuncs1d.Sigmoid(x)
		var backward Backward
		backward = func(chain tensor.D1) (tensor.D1, error) {
			dydx := mlfuncs1d.SigmoidGrad(y)
			// ∂L/∂x
			dx, err := tensor.D1Mul(dydx, chain)
			return dx, err
		}
		backwards = append(backwards, backward)
		return y, backwards, nil
	}
}

func NewTanhForward() Forward {
	return func(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
		y := mlfuncs1d.Tanh(x)
		var backward Backward
		backward = func(chain tensor.D1) (tensor.D1, error) {
			dydx := mlfuncs1d.TanhGrad(y)
			// ∂L/∂x
			dx, err := tensor.D1Mul(dydx, chain)
			return dx, err
		}
		backwards = append(backwards, backward)
		return y, backwards, nil
	}
}

func NewDropoutForward(p float64, isTrain *bool, r *rand.Rand) Forward {
	return func(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
		y, mask := mlfuncs1d.Dropout(x, p, *isTrain, r)
		var backward Backward
		backward = func(chain tensor.D1) (tensor.D1, error) {
			dx, err := tensor.D1Mul(mask, chain)
			return dx, err
		}
		backwards = append(backwards, backward)
		return y, backwards, nil
	}
}

type YLossForward func(tensor.D1, tensor.D1) (float64, YLossBackward, error)
type YLossBackward func() (tensor.D1, error)

func NewMeanSquaredErrorForward() YLossForward {
	return func(y, t tensor.D1) (float64, YLossBackward, error) {
		loss, err := mlfuncs1d.MeanSquaredError(y, t)
		var backward YLossBackward
		backward = func() (tensor.D1, error) {
			dLdy, err := mlfuncs1d.MeanSquaredErrorDerivative(y, t)
			if err != nil {
				return tensor.D1{}, err
			}
			return dLdy, err
		}
		return loss, backward, err
	}
}