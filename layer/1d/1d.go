package layer1d

import (
	"fmt"
	"github.com/sw965/crow/ml/1d"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/omw/fn"
	omwmath "github.com/sw965/omw/math"
	omwslices "github.com/sw965/omw/slices"
	"math"
	"math/rand"
)

type GradBuffer  struct {
	D1 tensor.D1
	D2 tensor.D2
	D3 tensor.D3
}

func NewGradBuffer(c int) GradBuffer {
	return GradBuffer{
		D1:make(tensor.D1, 0, c),
		D2:make(tensor.D2, 0, c),
		D3:make(tensor.D3, 0, c),
	}
}

func (g *GradBuffer) NewZerosLike() GradBuffer {
	return GradBuffer{
		D1:tensor.NewD1ZerosLike(g.D1),
		D2:tensor.NewD2ZerosLike(g.D2),
		D3:tensor.NewD3ZerosLike(g.D3),
	}
}

func (g *GradBuffer) Add(other *GradBuffer) error {
	err := g.D1.Add(other.D1)
	if err != nil {
		return err
	}
	err = g.D2.Add(other.D2)
	if err != nil {
		return err
	}
	err = g.D3.Add(other.D3)
	if err != nil {
		return err
	}
	return nil
}

func (g *GradBuffer) Reverse() {
	g.D1 = omwslices.Reverse(g.D1)
	g.D2 = omwslices.Reverse(g.D2)
	g.D3 = omwslices.Reverse(g.D3)
}

func (g *GradBuffer) ComputeL2Norm() float64 {
	sqSum := 0.0
	for _, e := range g.D1 {
		sqSum += (e * e)
	}

	for _, ei := range g.D2 {
		for _, eij := range ei {
			sqSum += (eij * eij)
		}
	}

	for _, ei := range g.D3 {
		for _, eij := range ei {
			for _, eijk := range eij {
				sqSum += (eijk * eijk)
			}
		}
	}
	return math.Sqrt(sqSum)
}

func (g *GradBuffer) ClipUsingL2Norm(maxNorm float64) {
	norm := g.ComputeL2Norm()
	scale := maxNorm / norm
	if scale < 1.0 {
		g.D1.MulScalar(scale)
		g.D2.MulScalar(scale)
		g.D3.MulScalar(scale)
	}
}

type GradBuffers []GradBuffer

func (gs GradBuffers) Total() GradBuffer {
	total := gs[0].NewZerosLike()
	for _, g := range gs {
		total.Add(&g)
	}
	return total
}

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

type Backward func(tensor.D1, *GradBuffer) (tensor.D1, *GradBuffer, error)
type Backwards []Backward

func (bs Backwards) Propagate(chain tensor.D1) (tensor.D1, GradBuffer, error) {
	bs = omwslices.Reverse(bs)
	n := len(bs)
	gb := &GradBuffer{
		D1:make(tensor.D1, 0, n),
		D2:make(tensor.D2, 0, n),
		D3:make(tensor.D3, 0, n),
	}
	var err error
	for _, b := range bs {
		chain, gb, err = b(chain, gb)
		if err != nil {
			return tensor.D1{}, GradBuffer{}, err
		}
	}
	gb.Reverse()
	return chain, *gb, nil
}

func NewAffineForward(w tensor.D2, b tensor.D1) Forward {
	return func(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
		dot := tensor.D2{x}.DotProduct(w)
		y, err := tensor.D1Add(dot[0], b)

		var backward Backward
		backward = func(chain tensor.D1, gb *GradBuffer) (tensor.D1, *GradBuffer, error) {
			// ∂L/∂x
			dx := tensor.D2{chain}.DotProduct(w.Transpose())

			// ∂L/∂w
			dw := tensor.D2{x}.Transpose().DotProduct(tensor.D2{chain})
			gb.D3 = append(gb.D3, dw)

			// ∂L/∂b
			db := chain
			gb.D2 = append(gb.D2, db)
			return dx[0], gb, err
		}

		backwards = append(backwards, backward)
		return y, backwards, err
	}
}

func NewReLUForward() Forward {
	return func(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
		y := ml1d.ReLU(x)
		var backward Backward
		backward = func(chain tensor.D1, gb *GradBuffer) (tensor.D1, *GradBuffer, error) {
			dydx := ml1d.ReLUDerivative(x)
			// ∂L/∂x
			dx, err := tensor.D1Mul(dydx, chain)
			return dx, gb, err
		}
		backwards = append(backwards, backward)
		return y, backwards, nil
	}
}

func NewLeakyReLUForward(alpha float64) Forward {
	return func(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
		y := ml1d.LeakyReLU(x, alpha)
		var backward Backward
		backward = func(chain tensor.D1, gb *GradBuffer) (tensor.D1, *GradBuffer, error) {
			dydx := ml1d.LeakyReLUDerivative(x, alpha)
			dx, err := tensor.D1Mul(dydx, chain)
			return dx, gb, err
		}
		backwards = append(backwards, backward)
		return y, backwards, nil
	}
}


func NewParamReLUForward(alpha *float64) Forward {
	return func(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
		y := ml1d.LeakyReLU(x, *alpha)
		var backward Backward
		backward = func(chain tensor.D1, gb *GradBuffer) (tensor.D1, *GradBuffer, error) {
			dydx, dydVectorizedAlpha := ml1d.ParamReLUDerivative(x, *alpha)

			// ∂L/∂dVectorizedAlpha
			dVectorizedAlpha, err := tensor.D1Mul(dydVectorizedAlpha, chain)
			if err != nil {
				return tensor.D1{}, &GradBuffer{}, err
			}

			// ∂L/∂alpha
			dAlpha := omwmath.Sum(dVectorizedAlpha...)
			gb.D1 = append(gb.D1, dAlpha)

			// ∂L/∂x
			dx, err := tensor.D1Mul(dydx, chain)
			return dx, gb, err
		}
		backwards = append(backwards, backward)
		return y, backwards, nil
	}
}

func NewRandReLUForward(min, max float64, isTrain *bool, r *rand.Rand) Forward {
	return func(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
		y, noise := ml1d.RandReLU(x, min, max, *isTrain, r)
		var backward Backward
		backward = func(chain tensor.D1, gb *GradBuffer) (tensor.D1, *GradBuffer, error) {
			dydx := ml1d.LeakyReLUDerivative(x, noise)
			// ∂L/∂x
			dx, err := tensor.D1Mul(dydx, chain)
			return dx, gb, err
		}
		backwards = append(backwards, backward)
		return y, backwards, nil
	}
}

func NewParamRandReLUForward(alpha *float64, min, max float64, isTrain *bool, r *rand.Rand) Forward {
	return func(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
		y, noise := ml1d.ParamRandReLU(x, *alpha, min, max, *isTrain, r)
		var backward Backward
		backward = func(chain tensor.D1, gb *GradBuffer) (tensor.D1, *GradBuffer, error) {
			dydx, dydVectorizedAlpha := ml1d.ParamRandReLUDerivative(x, *alpha, noise)

			// ∂L/∂dVectorizedAlpha
			dVectorizedAlpha, err := tensor.D1Mul(dydVectorizedAlpha, chain)
			if err != nil {
				return tensor.D1{}, &GradBuffer{}, err
			}
			// ∂L/∂alpha
			dAlpha := omwmath.Sum(dVectorizedAlpha...)
			gb.D1 = append(gb.D1, dAlpha)

			// ∂L/∂x
			dx, err := tensor.D1Mul(dydx, chain)
			return dx, gb, err
		}
		backwards = append(backwards, backward)
		return y, backwards, nil
	}
}

func SigmoidForward(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
	y := ml1d.Sigmoid(x)
	var backward Backward
	backward = func(chain tensor.D1, gb *GradBuffer) (tensor.D1, *GradBuffer, error) {
		dydx := ml1d.SigmoidGrad(y)
		// ∂L/∂x
		dx, err := tensor.D1Mul(dydx, chain)
		return dx, gb, err
	}
	backwards = append(backwards, backward)
	return y, backwards, nil
}

func TanhForward(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
	y := ml1d.Tanh(x)
	var backward Backward
	backward = func(chain tensor.D1, gb *GradBuffer) (tensor.D1, *GradBuffer, error) {
		dydx := ml1d.TanhGrad(y)
		// ∂L/∂x
		dx, err := tensor.D1Mul(dydx, chain)
		return dx, gb, err
	}
	backwards = append(backwards, backward)
	return y, backwards, nil
}

func SoftmaxForwardForCrossEntropy(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
    y := ml1d.Softmax(x)
    var backward Backward
    backward = func(chain tensor.D1, gb *GradBuffer) (tensor.D1, *GradBuffer, error) {
		return chain, gb, nil
    }
    backwards = append(backwards, backward)
    return y, backwards, nil
}

func NewDropoutForward(p float64, isTrain *bool, r *rand.Rand) Forward {
	return func(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
		y, mask := ml1d.Dropout(x, p, *isTrain, r)
		var backward Backward
		backward = func(chain tensor.D1, gb *GradBuffer) (tensor.D1, *GradBuffer, error) {
			dx, err := tensor.D1Mul(mask, chain)
			return dx, gb, err
		}
		backwards = append(backwards, backward)
		return y, backwards, nil
	}
}

func NewLinearSumForward(w tensor.D1, b *float64) Forward {
	return func(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
		y, err := ml1d.LinearSum(x, w, *b)
		if err != nil {
			return tensor.D1{}, Backwards{}, err
		}

		var backward Backward
		backward = func(chain tensor.D1, gb *GradBuffer) (tensor.D1, *GradBuffer, error) {
			if len(chain) != 1 {
				return tensor.D1{}, &GradBuffer{}, fmt.Errorf("LinearSumForward len(chain) != 1")
			}

			dydx, dydw, err := ml1d.LinearSumDerivative(x, w)
			if err != nil {
				return tensor.D1{}, &GradBuffer{}, err
			}

			//∂L/∂x
			dx := tensor.D1MulScalar(dydx, chain[0])

			//∂L/∂w
			dw := tensor.D1MulScalar(dydw, chain[0])
			gb.D2 = append(gb.D2, dw)

			//∂L/∂b
			db := chain[0]
			gb.D1 = append(gb.D1, db)
			return dx, gb, err
		}
		backwards = append(backwards, backward)
		return tensor.D1{y}, backwards, nil
	}
}

func IdentityForward(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
	y := fn.Identity[tensor.D1](x)
	var backward Backward
	backward = func(chain tensor.D1, gb *GradBuffer) (tensor.D1, *GradBuffer, error) {
		return chain, gb, nil
	}
	backwards = append(backwards, backward)
	return y, backwards, nil
}