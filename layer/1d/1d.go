package layer1d

import (
	"github.com/sw965/crow/ml/1d"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/omw/fn"
	omwslices "github.com/sw965/omw/slices"
)

type GradBuffer struct {
	Biases  tensor.D2
	Weights tensor.D3
}

func NewGradBuffer(c int) GradBuffer {
	return GradBuffer{
		Biases:make(tensor.D2, 0, c),
		Weights:make(tensor.D3, 0, c),
	}
}

func (g *GradBuffer) NewZerosLike() GradBuffer {
	return GradBuffer{
		Biases:tensor.NewD2ZerosLike(g.Biases),
		Weights:tensor.NewD3ZerosLike(g.Weights),
	}
}

func (g *GradBuffer) Add(other *GradBuffer) error {
	err := g.Biases.Add(other.Biases)
	if err != nil {
		return err
	}
	err = g.Weights.Add(other.Weights)
	if err != nil {
		return err
	}

	return nil
}

func (g *GradBuffer) Reverse() {
	g.Biases = omwslices.Reverse(g.Biases)
	g.Weights = omwslices.Reverse(g.Weights)
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
			return nil, nil, err
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
		Biases:make(tensor.D2, 0, n),
		Weights:make(tensor.D3, 0, n),
	}
	var err error
	dx := chain
	for _, b := range bs {
		dx, gb, err = b(dx, gb)
		if err != nil {
			return nil, GradBuffer{}, err
		}
	}
	gb.Reverse()
	return dx, *gb, nil
}

func NewFullyConnectedForward(w tensor.D2, b tensor.D1) Forward {
	return func(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
		u := tensor.D2{x}.DotProduct(w)
		y, err := tensor.D1Add(u[0], b)

		var backward Backward
		backward = func(chain tensor.D1, gb *GradBuffer) (tensor.D1, *GradBuffer, error) {
			// ∂L/∂x
			dx := tensor.D2{chain}.DotProduct(w.Transpose())

			// ∂L/∂w
			dw := tensor.D2{x}.Transpose().DotProduct(tensor.D2{chain})
			gb.Weights = append(gb.Weights, dw)

			// ∂L/∂b
			db := chain
			gb.Biases = append(gb.Biases, db)

			return dx[0], gb, nil
		}

		backwards = append(backwards, backward)
		return y, backwards, err
	}
}

func ReLUForward(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
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

func NewLeakyReLUForward(alpha float64) Forward {
	f := ml1d.LeakyReLU(alpha)
	fp := ml1d.LeakyReLUDerivative(alpha)
	return func(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
		y := f(x)
		var backward Backward
		backward = func(chain tensor.D1, gb *GradBuffer) (tensor.D1, *GradBuffer, error) {
			dydx := fp(x)
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

func SoftmaxForwardForCrossEntropy(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
    y := ml1d.Softmax(x)
	var backward Backward
    backward = func(chain tensor.D1, gb *GradBuffer) (tensor.D1, *GradBuffer, error) {
		return chain, gb, nil
    }
    backwards = append(backwards, backward)
    return y, backwards, nil
}

// func NewLinearSumForward(w tensor.D2, b tensor.D1) Forward {
// 	return func(x tensor.D3, backwards Backwards) (tensor.D3, Backwards, error) {
// 		if len(x) != 1 {
// 			return nil, nil, fmt.Errorf("入力値が不適")
// 		}

// 		y, err := ml2d.LinearSum(x[0], w, b)
// 		if err != nil {
// 			return nil, nil, err
// 		}

// 		var backward Backward
// 		backward = func(chain tensor.D3, gb *GradBuffer) (tensor.D3, *GradBuffer, error) {
// 			if len(chain) != 1 {
// 				return nil, nil, fmt.Errorf("LinearSumForward len(chain) != 1")
// 			}

// 			dydx, dydw, err := ml2d.LinearSumDerivative(x[0], w)
// 			if err != nil {
// 				return nil, nil, err
// 			}

// 			//∂L/∂x
// 			dx, err := tensor.D2MulD1Col(dydx, chain[0][0])
// 			if err != nil {
// 				return nil, nil, err
// 			}

// 			//∂L/∂w
// 			dw, err := tensor.D2MulD1Col(dydw, chain[0][0])
// 			if err != nil {
// 				return nil, nil, err
// 			}
// 			gb.Weights = append(gb.Weights, dw)

// 			//∂L/∂b
// 			db := chain[0][0]
// 			gb.Biases = append(gb.Biases, db)
// 			return tensor.D3{dx}, gb, err
// 		}
// 		backwards = append(backwards, backward)
// 		return tensor.D3{tensor.D2{y}}, backwards, nil
// 	}
// }

func IdentityForward(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
	y := fn.Identity[tensor.D1](x)
	var backward Backward
	backward = func(chain tensor.D1, gb *GradBuffer) (tensor.D1, *GradBuffer, error) {
		return chain, gb, nil
	}
	backwards = append(backwards, backward)
	return y, backwards, nil
}