package layer1d

import (
	"fmt"
	"github.com/sw965/crow/ml/1d"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/omw/fn"
	omwmath "github.com/sw965/omw/math"
	omwslices "github.com/sw965/omw/slices"
	"math/rand"
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
	bs = omwslices.Reverse(bs)
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
		y := ml1d.ReLU(x)
		var backward Backward
		backward = func(chain tensor.D1) (tensor.D1, error) {
			dydx := ml1d.ReLUDerivative(x)
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
		y := ml1d.LeakyReLU(x, alpha)
		var backward Backward
		backward = func(chain tensor.D1) (tensor.D1, error) {
			dydx := ml1d.LeakyReLUDerivative(x, alpha)
			dx, err := tensor.D1Mul(dydx, chain)
			return dx, err
		}
		backwards = append(backwards, backward)
		return y, backwards, nil
	}
}

func NewParamReLUForward(alpha, gradAlpha *float64) Forward {
	return func(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
		y := ml1d.LeakyReLU(x, *alpha)
		var backward Backward
		backward = func(chain tensor.D1) (tensor.D1, error) {
			dydx, dydVectorizedAlpha := ml1d.ParamReLUDerivative(x, *alpha)

			// ∂L/∂dVectorizedAlpha
			dVectorizedAlpha, err := tensor.D1Mul(dydVectorizedAlpha, chain)
			if err != nil {
				return tensor.D1{}, err
			}
			// ∂L/∂alpha
			*gradAlpha = omwmath.Sum(dVectorizedAlpha...)

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
		y, noise := ml1d.RandReLU(x, min, max, *isTrain, r)
		var backward Backward
		backward = func(chain tensor.D1) (tensor.D1, error) {
			dydx := ml1d.LeakyReLUDerivative(x, noise)
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
		y, noise := ml1d.ParamRandReLU(x, *alpha, min, max, *isTrain, r)
		var backward Backward
		backward = func(chain tensor.D1) (tensor.D1, error) {
			dydx, dydVectorizedAlpha := ml1d.ParamRandReLUDerivative(x, *alpha, noise)

			// ∂L/∂dVectorizedAlpha
			dVectorizedAlpha, err := tensor.D1Mul(dydVectorizedAlpha, chain)
			if err != nil {
				return tensor.D1{}, err
			}
			// ∂L/∂alpha
			*gradAlpha = omwmath.Sum(dVectorizedAlpha...)

			// ∂L/∂x
			dx, err := tensor.D1Mul(dydx, chain)
			return dx, err
		}
		backwards = append(backwards, backward)
		return y, backwards, nil
	}
}

func SigmoidForward(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
	y := ml1d.Sigmoid(x)
	var backward Backward
	backward = func(chain tensor.D1) (tensor.D1, error) {
		dydx := ml1d.SigmoidGrad(y)
		// ∂L/∂x
		dx, err := tensor.D1Mul(dydx, chain)
		return dx, err
	}
	backwards = append(backwards, backward)
	return y, backwards, nil
}

func TanhForward(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
	y := ml1d.Tanh(x)
	var backward Backward
	backward = func(chain tensor.D1) (tensor.D1, error) {
		dydx := ml1d.TanhGrad(y)
		// ∂L/∂x
		dx, err := tensor.D1Mul(dydx, chain)
		return dx, err
	}
	backwards = append(backwards, backward)
	return y, backwards, nil
}

func SoftmaxForward(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
    y := ml1d.Softmax(x)
    var backward Backward
    backward = func(chain tensor.D1) (tensor.D1, error) {
		return ml1d.SoftmaxDerivative(y, chain)
    }
    backwards = append(backwards, backward)
    return y, backwards, nil
}

func SoftmaxForwardForCrossEntropy(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
    y := ml1d.Softmax(x)
    var backward Backward
    backward = func(chain tensor.D1) (tensor.D1, error) {
		return chain, nil
    }
    backwards = append(backwards, backward)
    return y, backwards, nil
}

func NewDropoutForward(p float64, isTrain *bool, r *rand.Rand) Forward {
	return func(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
		y, mask := ml1d.Dropout(x, p, *isTrain, r)
		var backward Backward
		backward = func(chain tensor.D1) (tensor.D1, error) {
			dx, err := tensor.D1Mul(mask, chain)
			return dx, err
		}
		backwards = append(backwards, backward)
		return y, backwards, nil
	}
}

func NewLinearSumForward(w tensor.D1, b *float64, gradW tensor.D1, gradB *float64) Forward {
	return func(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
		y, err := ml1d.LinearSum(x, w, *b)
		if err != nil {
			return tensor.D1{}, Backwards{}, err
		}

		var backward Backward
		backward = func(chain tensor.D1) (tensor.D1, error) {
			if len(chain) != 1 {
				return tensor.D1{}, fmt.Errorf("LinearSumForward len(chain) != 1")
			}

			dydx, dydw, err := ml1d.LinearSumDerivative(x, w)
			if err != nil {
				return tensor.D1{}, err
			}
			dx := tensor.D1MulScalar(dydx, chain[0])
			dw := tensor.D1MulScalar(dydw, chain[0])
			gradW.Copy(dw)
			*gradB = chain[0]
			return dx, err
		}
		backwards = append(backwards, backward)
		return tensor.D1{y}, backwards, nil
	}
}

func IdentityForward(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
	y := fn.Identity[tensor.D1](x)
	var backward Backward
	backward = func(chain tensor.D1) (tensor.D1, error) {
		return chain, nil
	}
	backwards = append(backwards, backward)
	return y, backwards, nil
}