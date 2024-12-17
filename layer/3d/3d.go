package layer

import (
	"fmt"
	"github.com/sw965/crow/ml/1d"
	"github.com/sw965/crow/ml/2d"
	"github.com/sw965/crow/ml/3d"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/omw/fn"
	omwslices "github.com/sw965/omw/slices"
	"math"
)

type GradBuffer  struct {
	Scalars tensor.D1
	D1s     tensor.D2
	D2s     tensor.D3
	D3s     []tensor.D3
}

func NewGradBuffer(c int) GradBuffer {
	return GradBuffer{
		Scalars:make(tensor.D1, 0, c),
		D1s:make(tensor.D2, 0, c),
		D2s:make(tensor.D3, 0, c),
		D3s:make([]tensor.D3, 0, c),
	}
}

func (g *GradBuffer) NewZerosLike() GradBuffer {
	d3s := make([]tensor.D3, len(g.D3s))
	for i, d3 := range g.D3s {
		d3s[i] = tensor.NewD3ZerosLike(d3)
	}

	return GradBuffer{
		Scalars:tensor.NewD1ZerosLike(g.Scalars),
		D1s:tensor.NewD2ZerosLike(g.D1s),
		D2s:tensor.NewD3ZerosLike(g.D2s),
		D3s:d3s,
	}
}

func (g *GradBuffer) Add(other *GradBuffer) error {
	err := g.Scalars.Add(other.Scalars)
	if err != nil {
		return err
	}
	err = g.D1s.Add(other.D1s)
	if err != nil {
		return err
	}
	err = g.D2s.Add(other.D2s)
	if err != nil {
		return err
	}

	for i, d3 := range other.D3s {
		err := g.D3s[i].Add(d3)
		if err != nil {
			return err
		}
	}
	return nil
}

func (g *GradBuffer) Reverse() {
	g.Scalars = omwslices.Reverse(g.Scalars)
	g.D1s = omwslices.Reverse(g.D1s)
	g.D2s = omwslices.Reverse(g.D2s)
	g.D3s = omwslices.Reverse(g.D3s)
}

func (g *GradBuffer) ComputeL2Norm() float64 {
	sqSum := 0.0
	for _, e := range g.Scalars {
		sqSum += (e * e)
	}

	for _, ei := range g.D1s {
		for _, eij := range ei {
			sqSum += (eij * eij)
		}
	}

	for _, ei := range g.D2s {
		for _, eij := range ei {
			for _, eijk := range eij {
				sqSum += (eijk * eijk)
			}
		}
	}

	for _, ei := range g.D3s {
		for _, eij := range ei {
			for _, eijk := range eij {
				for _, eijkl := range eijk {
					sqSum += (eijkl * eijkl)
				}
			}
		}
	}
	return math.Sqrt(sqSum)
}

func (g *GradBuffer) ClipUsingL2Norm(maxNorm float64) {
	norm := g.ComputeL2Norm()
	scale := maxNorm / norm
	if scale < 1.0 {
		g.Scalars.MulScalar(scale)
		g.D1s.MulScalar(scale)
		g.D2s.MulScalar(scale)
		for i := range g.D3s {
			g.D3s[i].MulScalar(scale)
		}
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
		Scalars:make(tensor.D1, 0, n),
		D1s:make(tensor.D2, 0, n),
		D2s:make(tensor.D3, 0, n),
		D3s:make([]tensor.D3, 0, n),
	}
	var err error
	dx := tensor.D3{tensor.D2{chain}}
	for _, b := range bs {
		dx, gb, err = b(dx, gb)
		if err != nil {
			return nil, nil, err
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
			gb.D2s = append(gb.D2s, dw)

			// ∂L/∂b
			db := chain[0][0]
			gb.D1s = append(gb.D1s, db)

			return tensor.D3{dx}, gb, nil
		}

		backwards = append(backwards, backward)
		return tensor.D3{tensor.D2{y}}, backwards, err
	}
}

func NewConvolutionForward(filter tensor.D3, b tensor.D1, stride int) Forward {
	return func(x tensor.D3, backwards Backwards) (tensor.D3, Backwards, error) {
		y := ml3d.Convolution(x, filter, stride)
		err := y.AddD1Depth(b)
		var backward Backward
		backward = func(chain tensor.D3, gb *GradBuffer) (tensor.D3, *GradBuffer, error) {
			dx, dw := ml3d.ConvolutionDerivative(x, filter, chain, stride)
			gb.D3s = append(gb.D3s, dw)

			cd := len(chain)
			// ∂L/∂b
			db := make(tensor.D1, cd)
			for i := 0; i < cd; i++ {
				db[i] = omwmath.Sum(chain[i].Flatten()...)
			}
			gb.D1s = append(gb.D1s, db)
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
			gb.D2s = append(gb.D2s, dw)

			//∂L/∂b
			db := chain[0][0]
			gb.D1s = append(gb.D1s, db)
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
		dx := chain[0][0].Reshape3D(d, h, w)
		return dx
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