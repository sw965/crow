package general

import (
	"github.com/sw965/crow/tensor"
	"github.com/sw965/crow/tensor/nn"
	tmath "github.com/sw965/crow/tensor/math"
	"slices"
)

type ForwardD3 func(tensor.D3) (tensor.D3, BackwardD3, error)
type ForwardsD3 []ForwardD3

func (fs ForwardsD3) Propagate(x tensor.D3) (tensor.D3, BackwardsD3, error) {
	var err error
	var backward BackwardD3
	backwards := make(BackwardsD3, len(fs))
	for i, f := range fs {
		x, backward, err = f(x)
		if err != nil {
			return tensor.D3{}, nil, err
		}
		backwards[i] = backward
	}
	y := x
	slices.Reverse(backwards)
	return y, backwards, nil
}

type BackwardD3 func(tensor.D3, *GradBuffer) (tensor.D3, error)
type BackwardsD3 []BackwardD3

func (bs BackwardsD3) Propagate(chain tensor.D3) (tensor.D3, GradBuffer, error) {
	n := len(bs)
	grad := GradBuffer{
		Filters:make(tensor.D4Slice, 0, n),
		Weights:make(tensor.D2Slice, 0, n),
		Gammas: make(tensor.D1Slice, 0, n),
		Biases: make(tensor.D1Slice, 0, n),
	}
	var err error

	for _, b := range bs {
		chain, err = b(chain, &grad)
		if err != nil {
			return tensor.D3{}, GradBuffer{}, err
		}
	}

	slices.Reverse(grad.Filters)
	slices.Reverse(grad.Weights)
	slices.Reverse(grad.Gammas)
	slices.Reverse(grad.Biases)
	dx := chain
	return dx, grad, nil
}

func NewConvForward(filter tensor.D4, stride int) ForwardD3 {
	return func(x tensor.D3) (tensor.D3, BackwardD3, error) {
		y, col, colFilter := nn.Conv2DWithColVar(x, filter, stride)
		var backward BackwardD3
		backward = func(chain tensor.D3, grad *GradBuffer) (tensor.D3, error) {
			chain2d := chain.Transpose120().ToD1().Reshape2D(-1, filter.Batches)
			dFilter := col.TransDotNoTrans(chain2d).Transpose().ToD1().Reshape4D(filter.Batches, filter.Channels, filter.Rows, filter.Cols)
			grad.Filters = append(grad.Filters, dFilter)

			dCol := chain2d.NoTransDotTrans(colFilter)
			dx := nn.Col2Im(dCol, x, filter.Rows, filter.Cols, stride)
			return dx, nil
		}
		return y, backward, nil
	}
}

type ForwardD1 func(tensor.D1) (tensor.D1, BackwardD1, error)
type ForwardsD1 []ForwardD1

func (fs ForwardsD1) Propagate(x tensor.D1) (tensor.D1, BackwardsD1, error) {
	var err error
	var backward BackwardD1
	backwards := make(BackwardsD1, len(fs))
	for i, f := range fs {
		x, backward, err = f(x)
		if err != nil {
			return tensor.D1{}, nil, err
		}
		backwards[i] = backward
	}
	y := x
	slices.Reverse(backwards)
	return y, backwards, nil
}

type BackwardD1 func(tensor.D1, *GradBuffer) (tensor.D1, error)
type BackwardsD1 []BackwardD1

func (bs BackwardsD1) Propagate(chain tensor.D1) (tensor.D1, GradBuffer, error) {
	n := len(bs)
	grad := GradBuffer{
		Filters: make(tensor.D4Slice, 0, n),
		Weights:make(tensor.D2Slice, 0, n),
		Gammas: make(tensor.D1Slice, 0, n),
		Biases: make(tensor.D1Slice, 0, n),
	}
	var err error

	for _, b := range bs {
		chain, err = b(chain, &grad)
		if err != nil {
			return tensor.D1{}, GradBuffer{}, err
		}
	}

	slices.Reverse(grad.Filters)
	slices.Reverse(grad.Weights)
	slices.Reverse(grad.Gammas)
	slices.Reverse(grad.Biases)
	dx := chain
	return dx, grad, nil
}

func NewDotForwardD1(w tensor.D2) ForwardD1 {
	return func(x tensor.D1) (tensor.D1, BackwardD1, error) {
		y := x.DotNoTrans2D(w)
		var backward BackwardD1
		backward = func(chain tensor.D1, grad *GradBuffer) (tensor.D1, error) {
			dx := chain.DotTrans2D(w)
			dw := x.Outer(chain)
			grad.Weights = append(grad.Weights, dw)
			return dx, nil
		}
		return y, backward, nil
	}
}

func NewLeakyReLUForwardD1(alpha float32) ForwardD1 {
	return func(x tensor.D1) (tensor.D1, BackwardD1, error) {
		y := nn.ReLUD1WithAlpha(x, alpha)
		var backward BackwardD1
		backward = func(chain tensor.D1, _ *GradBuffer) (tensor.D1, error) {
			grad := nn.LeakyReLUD1Derivative(x, alpha)
			dx := grad.Hadamard(chain)
			return dx, nil
		}
		return y, backward, nil
	}
}

func NewInstanceNormalizationForwardD1(gamma, beta tensor.D1) (ForwardD1, error) {
	return func(x tensor.D1) (tensor.D1, BackwardD1, error) {
		z, mean, std := tmath.StandardizeWithStats(x)
		// y = z * γ + β
		y := z.Hadamard(gamma).Axpy(1.0, beta)

		var backward BackwardD1
		backward = func(chain tensor.D1, grad *GradBuffer) (tensor.D1, error) {
			dBeta := chain.Clone()
			grad.Biases = append(grad.Biases, dBeta)

			dGamma := chain.Hadamard(z)

			grad.Gammas = append(grad.Gammas, dGamma)

			jacobianGradX := tmath.StandardizationDerivative(x, mean, std)

			gradY := chain.Hadamard(gamma)

			dx := gradY.DotTrans2D(jacobianGradX)
			//dx := jacobianGradX.DotTrans2D(gradY)
			return dx, nil
		}
		return y, backward, nil
	}, nil
}

func SoftmaxForwardForCrossEntropyLoss(x tensor.D1) (tensor.D1, BackwardD1, error) {
	y := tmath.Softmax(x)
	var backward BackwardD1
	backward = func(chain tensor.D1, _ *GradBuffer) (tensor.D1, error) {
		dx := chain
		return dx, nil
	}
	return y, backward, nil
}