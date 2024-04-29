package layer

import (
	"math/rand"
	"github.com/sw965/omw"
	"github.com/sw965/crow/mlfuncs"
	"github.com/sw965/crow/tensor"
)

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
	y := x
	return y, bs, nil
}

type D1Backward func(tensor.D1) (tensor.D1, error)
type D1Backwards []D1Backward

type D1BackPropagator struct {
	backwards D1Backwards
	lossBackward D1LossBackward
}

func NewD1BackPropagator(backwards D1Backwards, lossBackward D1LossBackward) D1BackPropagator {
	return D1BackPropagator{backwards:backwards, lossBackward:lossBackward}
}

func (bp *D1BackPropagator) Run() (tensor.D1, error) {
	chain, err := bp.lossBackward()
	if err != nil {
		return tensor.D1{}, err
	}
	bs := omw.Reverse(bp.backwards)
	for _, b := range bs {
		chain, err = b(chain)
		if err != nil {
			return tensor.D1{}, err
		}
	}
	return chain, nil
}

func NewD1AffineForward(w tensor.D2, b tensor.D1, gradW tensor.D2, gradB tensor.D1) D1Forward {
	return func(x tensor.D1, backwards D1Backwards) (tensor.D1, D1Backwards, error) {
		dot, err := tensor.D2{x}.DotProduct(w)
		if err != nil {
			return tensor.D1{}, D1Backwards{}, err
		}
		y, err := tensor.D1Add(dot[0], b)

		var backward D1Backward
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

func NewD1ReLUForward() D1Forward {
	return func(x tensor.D1, backwards D1Backwards) (tensor.D1, D1Backwards, error) {
		y := mlfuncs.D1ReLU(x)
		var backward D1Backward
		backward = func(chain tensor.D1) (tensor.D1, error) {
			dydx := mlfuncs.D1ReLUDerivative(x)
			// ∂L/∂x
			dx, err := tensor.D1Mul(dydx, chain)
			return dx, err
		}
		backwards = append(backwards, backward)
		return y, backwards, nil
	}
}

func NewD1LeakyReLUForward(alpha float64) D1Forward {
	return func(x tensor.D1, backwards D1Backwards) (tensor.D1, D1Backwards, error) {
		y := mlfuncs.D1LeakyReLU(x, alpha)
		var backward D1Backward
		backward = func(chain tensor.D1) (tensor.D1, error) {
			dydx := mlfuncs.D1LeakyReLUDerivative(x, alpha)
			dx, err := tensor.D1Mul(dydx, chain)
			return dx, err
		}
		backwards = append(backwards, backward)
		return y, backwards, nil
	}
}

func NewD1ParamReLUForward(alpha, gradAlpha *float64) D1Forward {
	return func(x tensor.D1, backwards D1Backwards) (tensor.D1, D1Backwards, error) {
		y := mlfuncs.D1LeakyReLU(x, *alpha)
		var backward D1Backward
		backward = func(chain tensor.D1) (tensor.D1, error) {
			dydx, dydVectorizedAlpha := mlfuncs.D1ParamReLUDerivative(x, *alpha)

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

func NewD1RandReLUForward(min, max float64, isTrain *bool, r *rand.Rand) D1Forward {
	return func(x tensor.D1, backwards D1Backwards) (tensor.D1, D1Backwards, error) {
		y, noise := mlfuncs.D1RandReLU(x, min, max, *isTrain, r)
		var backward D1Backward
		backward = func(chain tensor.D1) (tensor.D1, error) {
			dydx := mlfuncs.D1LeakyReLUDerivative(x, noise)
			// ∂L/∂x
			dx, err := tensor.D1Mul(dydx, chain)
			return dx, err
		}
		backwards = append(backwards, backward)
		return y, backwards, nil
	}
}

func NewD1ParamRandReLUForward(alpha *float64, min, max float64, gradAlpha *float64, isTrain *bool, r *rand.Rand) D1Forward {
	return func(x tensor.D1, backwards D1Backwards) (tensor.D1, D1Backwards, error) {
		y, noise := mlfuncs.D1ParamRandReLU(x, *alpha, min, max, *isTrain, r)
		var backward D1Backward
		backward = func(chain tensor.D1) (tensor.D1, error) {
			dydx, dydVectorizedAlpha := mlfuncs.D1ParamRandReLUDerivative(x, *alpha, noise)

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

func NewD1SigmoidForward() D1Forward {
	return func(x tensor.D1, backwards D1Backwards) (tensor.D1, D1Backwards, error) {
		y := mlfuncs.D1Sigmoid(x)
		var backward D1Backward
		backward = func(chain tensor.D1) (tensor.D1, error) {
			dydx := mlfuncs.D1SigmoidGrad(y)
			// ∂L/∂x
			dx, err := tensor.D1Mul(dydx, chain)
			return dx, err
		}
		backwards = append(backwards, backward)
		return y, backwards, nil
	}
}

func NewD1TanhForward() D1Forward {
	return func(x tensor.D1, backwards D1Backwards) (tensor.D1, D1Backwards, error) {
		y := mlfuncs.D1Tanh(x)
		var backward D1Backward
		backward = func(chain tensor.D1) (tensor.D1, error) {
			dydx := mlfuncs.D1TanhGrad(y)
			// ∂L/∂x
			dx, err := tensor.D1Mul(dydx, chain)
			return dx, err
		}
		backwards = append(backwards, backward)
		return y, backwards, nil
	}
}

func NewD1DropoutForward(p float64, isTrain *bool, r *rand.Rand) D1Forward {
	return func(x tensor.D1, backwards D1Backwards) (tensor.D1, D1Backwards, error) {
		y, mask := mlfuncs.D1Dropout(x, p, *isTrain, r)
		var backward D1Backward
		backward = func(chain tensor.D1) (tensor.D1, error) {
			dx, err := tensor.D1Mul(mask, chain)
			return dx, err
		}
		backwards = append(backwards, backward)
		return y, backwards, nil
	}
}

type D1LossForward func(tensor.D1, tensor.D1) (float64, D1LossBackward, error)
type D1LossBackward func() (tensor.D1, error)

func NewD1MeanSquaredErrorForward() D1LossForward {
	return func(y, t tensor.D1) (float64, D1LossBackward, error) {
		loss, err := mlfuncs.D1MeanSquaredError(y, t)
		var backward D1LossBackward
		backward = func() (tensor.D1, error) {
			dLdy, err := mlfuncs.D1MeanSquaredErrorDerivative(y, t)
			if err != nil {
				return tensor.D1{}, err
			}
			return dLdy, err
		}
		return loss, backward, err
	}
}