package layer

import (
	"fmt"
	"github.com/sw965/crow/ml/1d"
	"github.com/sw965/crow/ml/2d"
	"github.com/sw965/crow/ml/3d"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/omw/fn"
	omwmath "github.com/sw965/omw/math"
	omwslices "github.com/sw965/omw/slices"
)

type GradBuffer struct {
	Biases  tensor.D2
	Weights tensor.D3
	Filters []tensor.D4
}

func NewGradBuffer(c int) GradBuffer {
	return GradBuffer{
		Biases:make(tensor.D2, 0, c),
		Weights:make(tensor.D3, 0, c),
		Filters:make([]tensor.D4, 0, c),
	}
}

func (g *GradBuffer) NewZerosLike() GradBuffer {
	d5Zeros := make([]tensor.D4, len(g.Filters))
	for i := range d5Zeros {
		d5Zeros[i] = tensor.NewD4ZerosLike(g.Filters[i])
	}
	return GradBuffer{
		Biases:tensor.NewD2ZerosLike(g.Biases),
		Weights:tensor.NewD3ZerosLike(g.Weights),
		Filters:d5Zeros,
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

	for i := range g.Filters {
		err := g.Filters[i].Add(other.Filters[i])
		if err != nil {
			return err
		}
	}

	return nil
}

func (g *GradBuffer) Reverse() {
	g.Biases = omwslices.Reverse(g.Biases)
	g.Weights = omwslices.Reverse(g.Weights)
	g.Filters = omwslices.Reverse(g.Filters)
}

type GradBuffers []GradBuffer

func (gs GradBuffers) Total() GradBuffer {
	total := gs[0].NewZerosLike()
	for _, g := range gs {
		total.Add(&g)
	}
	return total
}

type Forward func(tensor.D3, Backwards) (tensor.D3, Backwards, error)
type Forwards []Forward

func (fs Forwards) Propagate(x tensor.D3) (tensor.D3, Backwards, error) {
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

type Backward func(tensor.D3, *GradBuffer) (tensor.D3, *GradBuffer, error)
type Backwards []Backward

func (bs Backwards) Propagate(chain tensor.D1) (tensor.D3, GradBuffer, error) {
	bs = omwslices.Reverse(bs)
	n := len(bs)
	gb := &GradBuffer{
		Biases:make(tensor.D2, 0, n),
		Weights:make(tensor.D3, 0, n),
		Filters:make([]tensor.D4, 0, n),
	}
	var err error
	dx := tensor.D3{tensor.D2{chain}}
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
	return func(x tensor.D3, backwards Backwards) (tensor.D3, Backwards, error) {
		if len(x) != 1 {
			return nil, nil, fmt.Errorf("線形変換の入力値として不適です")
		}

		if len(x[0]) != 1 {
			return nil, nil, fmt.Errorf("不適です。")
		}

		u := x[0].DotProduct(w)
		y, err := tensor.D1Add(u[0], b)

		var backward Backward
		backward = func(chain tensor.D3, gb *GradBuffer) (tensor.D3, *GradBuffer, error) {
			if len(chain) != 1 {
				return nil, nil, fmt.Errorf("不適な勾配")
			}

			if len(chain[0]) != 1 {
				return nil, nil, fmt.Errorf("不適な勾配")
			}

			// ∂L/∂x
			dx := chain[0].DotProduct(w.Transpose())

			// ∂L/∂w
			dw := x[0].Transpose().DotProduct(chain[0])
			gb.Weights = append(gb.Weights, dw)

			// ∂L/∂b
			db := chain[0][0]
			gb.Biases = append(gb.Biases, db)

			return tensor.D3{dx}, gb, nil
		}

		backwards = append(backwards, backward)
		return tensor.D3{tensor.D2{y}}, backwards, err
	}
}

func NewConvForward(filter tensor.D4, b tensor.D1) Forward {
	return func(x tensor.D3, backwards Backwards) (tensor.D3, Backwards, error) {
		y := ml3d.Conv(x, filter)
		err := y.AddD1Depth(b)
		var backward Backward
		backward = func(chain tensor.D3, gb *GradBuffer) (tensor.D3, *GradBuffer, error) {
			dx, dFilter := ml3d.ConvDerivative(x, filter, chain)
			gb.Filters = append(gb.Filters, dFilter)

			cd := len(chain)
			// ∂L/∂b
			db := make(tensor.D1, cd)
			for i := 0; i < cd; i++ {
				db[i] = omwmath.Sum(chain[i].Flatten()...)
			}
			gb.Biases = append(gb.Biases, db)
			return dx, gb, nil
		}
		backwards = append(backwards, backward)
		return y, backwards, err
	}
}

func ReLUForward(x tensor.D3, backwards Backwards) (tensor.D3, Backwards, error) {
	y := ml3d.ReLU(x)
	var backward Backward
	backward = func(chain tensor.D3, gb *GradBuffer) (tensor.D3, *GradBuffer, error) {
		dydx := ml3d.ReLUDerivative(x)
		// ∂L/∂x
		dx, err := tensor.D3Mul(dydx, chain)
		return dx, gb, err
	}
	backwards = append(backwards, backward)
	return y, backwards, nil
}

func NewLeakyReLUForward(alpha float64) Forward {
	f := ml3d.LeakyReLU(alpha)
	fp := ml3d.LeakyReLUDerivative(alpha)
	return func(x tensor.D3, backwards Backwards) (tensor.D3, Backwards, error) {
		y := f(x)
		var backward Backward
		backward = func(chain tensor.D3, gb *GradBuffer) (tensor.D3, *GradBuffer, error) {
			dydx := fp(x)
			dx, err := tensor.D3Mul(dydx, chain)
			return dx, gb, err
		}
		backwards = append(backwards, backward)
		return y, backwards, nil
	}
}

func SigmoidForward(x tensor.D3, backwards Backwards) (tensor.D3, Backwards, error) {
	y := ml3d.Sigmoid(x)
	var backward Backward
	backward = func(chain tensor.D3, gb *GradBuffer) (tensor.D3, *GradBuffer, error) {
		dydx := ml3d.SigmoidGrad(y)
		// ∂L/∂x
		dx, err := tensor.D3Mul(dydx, chain)
		return dx, gb, err
	}
	backwards = append(backwards, backward)
	return y, backwards, nil
}

func SoftmaxForwardForCrossEntropy(x tensor.D3, backwards Backwards) (tensor.D3, Backwards, error) {
	if len(x) != 1 {
		return nil, nil, fmt.Errorf("入力値が不適")
	}

	xi := x[0]
	if len(xi) != 1 {
		return nil, nil, fmt.Errorf("入力値が不適")
	}

    y := ml1d.Softmax(xi[0])
 
	var backward Backward
    backward = func(chain tensor.D3, gb *GradBuffer) (tensor.D3, *GradBuffer, error) {
		return chain, gb, nil
    }
    backwards = append(backwards, backward)
    return tensor.D3{tensor.D2{y}}, backwards, nil
}

func NewLinearSumForward(w tensor.D2, b tensor.D1) Forward {
	return func(x tensor.D3, backwards Backwards) (tensor.D3, Backwards, error) {
		if len(x) != 1 {
			return nil, nil, fmt.Errorf("入力値が不適")
		}

		y, err := ml2d.LinearSum(x[0], w, b)
		if err != nil {
			return nil, nil, err
		}

		var backward Backward
		backward = func(chain tensor.D3, gb *GradBuffer) (tensor.D3, *GradBuffer, error) {
			if len(chain) != 1 {
				return nil, nil, fmt.Errorf("LinearSumForward len(chain) != 1")
			}

			dydx, dydw, err := ml2d.LinearSumDerivative(x[0], w)
			if err != nil {
				return nil, nil, err
			}

			//∂L/∂x
			dx, err := tensor.D2MulD1Col(dydx, chain[0][0])
			if err != nil {
				return nil, nil, err
			}

			//∂L/∂w
			dw, err := tensor.D2MulD1Col(dydw, chain[0][0])
			if err != nil {
				return nil, nil, err
			}
			gb.Weights = append(gb.Weights, dw)

			//∂L/∂b
			db := chain[0][0]
			gb.Biases = append(gb.Biases, db)
			return tensor.D3{dx}, gb, err
		}
		backwards = append(backwards, backward)
		return tensor.D3{tensor.D2{y}}, backwards, nil
	}
}

func FlatForward(x tensor.D3, backwards Backwards) (tensor.D3, Backwards, error) {
	y := x.Flatten()
	d, h, w := len(x), len(x[0]), len(x[0][0])
	var backward Backward
	backward = func(chain tensor.D3, gb *GradBuffer) (tensor.D3, *GradBuffer, error) {
		dx, err := chain[0][0].Reshape3D(d, h, w)
		return dx, gb, err
	}
	backwards = append(backwards, backward)
	return tensor.D3{tensor.D2{y}}, backwards, nil
}

func GAPForward(x tensor.D3, backwards Backwards) (tensor.D3, Backwards, error) {
    y := make(tensor.D1, len(x))
    d, h, w := len(x), len(x[0]), len(x[0][0])
    size := float64(h * w)
    for i := range y {
        y[i] = omwmath.Sum(x[i].Flatten()...) / size
    }

    var backward Backward
    backward = func(chain tensor.D3, gb *GradBuffer) (tensor.D3, *GradBuffer, error) {
		c := chain[0][0]
        dx := make(tensor.D3, d)
        for i := 0; i < d; i++ {
            dx[i] = make(tensor.D2, h)
            for j := 0; j < h; j++ {
                dx[i][j] = make(tensor.D1, w)
                for k := 0; k < w; k++ {
                    dx[i][j][k] = c[i] / size
                }
            }
        }
        return dx, gb, nil
    }
    backwards = append(backwards, backward)
    return tensor.D3{tensor.D2{y}}, backwards, nil
}

func IdentityForward(x tensor.D3, backwards Backwards) (tensor.D3, Backwards, error) {
	y := fn.Identity[tensor.D3](x)
	var backward Backward
	backward = func(chain tensor.D3, gb *GradBuffer) (tensor.D3, *GradBuffer, error) {
		return chain, gb, nil
	}
	backwards = append(backwards, backward)
	return y, backwards, nil
}