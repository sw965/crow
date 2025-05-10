package general

import (
	"fmt"
	"github.com/sw965/crow/blas32/tensor/2d"
	"github.com/sw965/crow/blas32/tensors/2d"
	"github.com/sw965/crow/blas32/vector"
	"github.com/sw965/crow/blas32/vectors"
	oslices "github.com/sw965/omw/slices"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
	"math/rand"
	"slices"
	"github.com/sw965/omw/parallel"
)

type GradBuffer struct {
	Weights []blas32.General
	Gammas  []blas32.Vector
	Biases  []blas32.Vector
}

func (g *GradBuffer) NewZerosLike() GradBuffer {
	return GradBuffer{
		Weights:tensors2d.NewZerosLike(g.Weights),
		Biases:vectors.NewZerosLike(g.Biases),
	}
}

func (g *GradBuffer) Clone() GradBuffer {
	return GradBuffer{
		Weights:tensors2d.Clone(g.Weights),
		Biases:vectors.Clone(g.Biases),
	}
}

func (g *GradBuffer) Axpy(alpha float32, x *GradBuffer) {
	tensors2d.Axpy(alpha, x.Weights, g.Weights)
	vectors.Axpy(alpha, x.Biases, g.Biases)
}

func (g *GradBuffer) Scal(alpha float32) {
	tensors2d.Scal(alpha, g.Weights)
	vectors.Scal(alpha, g.Biases)
}

type GradBuffers []GradBuffer

func (gs GradBuffers) Total() GradBuffer {
	total := gs[0].Clone()
	for _, g := range gs[1:] {
		total.Axpy(1.0, &g)
	}
	return total
}

type Parameter struct {
	Weights []blas32.General
	Gammas  []blas32.Vector
	Biases  []blas32.Vector
}

func (p *Parameter) NewGradZerosLike() GradBuffer {
	return GradBuffer{
		Weights: tensors2d.NewZerosLike(p.Weights),
		Biases:  vectors.NewZerosLike(p.Biases),
	}
}

func (p *Parameter) Clone() Parameter {
	return Parameter{
		Weights: tensors2d.Clone(p.Weights),
		Biases:  vectors.Clone(p.Biases),
	}
}

func (p *Parameter) AxpyGrad(alpha float32, grad *GradBuffer) {
	tensors2d.Axpy(alpha, grad.Weights, p.Weights)
	vectors.Axpy(alpha, grad.Biases, p.Biases)
}

type Forward func(blas32.Vector) (blas32.Vector, Backward, error)
type Forwards []Forward

func (fs Forwards) Propagate(x blas32.Vector) (blas32.Vector, Backwards, error) {
	var err error
	var backward Backward
	backwards := make(Backwards, len(fs))
	for i, f := range fs {
		x, backward, err = f(x)
		if err != nil {
			return blas32.Vector{}, nil, err
		}
		backwards[i] = backward
	}
	y := x
	slices.Reverse(backwards)
	return y, backwards, nil
}

type Backward func(blas32.Vector, *GradBuffer) (blas32.Vector, error)
type Backwards []Backward

func (bs Backwards) Propagate(chain blas32.Vector) (blas32.Vector, GradBuffer, error) {
	n := len(bs)
	grad := GradBuffer{
		Weights:make([]blas32.General, 0, n),
		Biases: make([]blas32.Vector, 0, n),
	}
	var err error

	for _, b := range bs {
		chain, err = b(chain, &grad)
		if err != nil {
			return blas32.Vector{}, GradBuffer{}, err
		}
	}

	slices.Reverse(grad.Weights)
	slices.Reverse(grad.Biases)
	dx := chain
	return dx, grad, nil
}

func NewDotForward(w blas32.General) Forward {
	return func(x blas32.Vector) (blas32.Vector, Backward, error) {
		y := vector.DotNoTrans2D(x, w)
		var backward Backward
		backward = func(chain blas32.Vector, grad *GradBuffer) (blas32.Vector, error) {
			dx := vector.DotTrans2D(chain, w)
			dw := vector.Outer(x, chain)
			grad.Weights = append(grad.Weights, dw)
			return dx, nil
		}
		return y, backward, nil
	}
}

func NewLeakyReLUForward(alpha float32) Forward {
	return func(x blas32.Vector) (blas32.Vector, Backward, error) {
		y := vector.LeakyReLU(x, alpha)
		var backward Backward
		backward = func(chain blas32.Vector, _ *GradBuffer) (blas32.Vector, error) {
			grad := vector.LeakyReLUDerivative(x, alpha)
			dx, err := vector.Hadamard(grad, chain)
			return dx, err
		}
		return y, backward, nil
	}
}

func NewInstanceNormalizationForward(gamma, beta blas32.Vector) Forward {
	return func(x blas32.Vector) (blas32.Vector, Backward, error) {
		u, mean, std, err := vector.StandardizeWithStats(x)
		if err != nil {
			return blas32.Vector{}, nil, err
		}

		y, err := vector.Hadamard(u, gamma)
		if err != nil {
			return blas32.Vector{}, nil, err
		}

		blas32.Axpy(1.0, beta, y)

		var backward Backward
		backward = func(chain blas32.Vector, grad *GradBuffer) (blas32.Vector, error) {
			dBeta := vector.Clone(chain)
			grad.Biases = append(grad.Biases, dBeta)

			dGamma, err := vector.Hadamard(chain, u)
			if err != nil {
				return blas32.Vector{}, err
			}
			grad.Gammas = append(grad.Gammas, dGamma)

			jacobianGradX, err := vector.StandardizationDerivative(x, mean, std)
			if err != nil {
				return blas32.Vector{}, err
			}

			gradY, err := vector.Hadamard(chain, gamma)
    		if err != nil {
        		return blas32.Vector{}, err
    		}

    		dx := blas32.Vector{
        		N:    x.N,
        		Inc:  1,
        		Data: make([]float32, x.N),
    		}

    		blas32.Gemv(blas.NoTrans, 1.0, jacobianGradX, gradY, 0.0, dx)
			return dx, nil
		}
		return y, backward, nil
	}
}

func SoftmaxForwardForCrossEntropyLoss(x blas32.Vector) (blas32.Vector, Backward, error) {
	y := vector.Softmax(x)
	var backward Backward
	backward = func(chain blas32.Vector, _ *GradBuffer) (blas32.Vector, error) {
		dx := chain
		return dx, nil
	}
	return y, backward, nil
}

type PredictLoss struct {
	Func       func(blas32.Vector, blas32.Vector) (float32, error)
	Derivative func(blas32.Vector, blas32.Vector) (blas32.Vector, error)
}

func NewCrossEntropyLossForSoftmax() PredictLoss {
	return PredictLoss{
		Func:       vector.CrossEntropy,
		Derivative: vector.SoftmaxCrossEntropyLossDerivative,
	}
}

type Model struct {
	Parameter   Parameter
	Forwards1D  Forwards
	PredictLoss PredictLoss
}

func (m *Model) AppendDot(xn, yn int, rng *rand.Rand) {
	w := tensor2d.NewHe(xn, yn, rng)
	m.Parameter.Weights = append(m.Parameter.Weights, w)
	m.Forwards1D = append(m.Forwards1D, NewDotForward(w))
}

func (m *Model) AppendLeakyReLU(alpha float32) {
	m.Forwards1D = append(m.Forwards1D, NewLeakyReLUForward(alpha))
}

func (m *Model) AppendInstanceNormalization(xn int) {
	gamma := blas32.Vector{
		N:xn,
		Inc:1,
		Data:make([]float32, xn),
	}

	for i := range gamma.Data {
		gamma.Data[i] = 1.0
	}

	m.Parameter.Gammas = append(m.Parameter.Gammas, gamma)

	beta := vector.NewZeros(xn)
	m.Parameter.Biases = append(m.Parameter.Biases, beta)

	m.Forwards1D = append(m.Forwards1D, NewInstanceNormalizationForward(gamma, beta))
}

func (m *Model) AppendSoftmaxForCrossEntropyLoss() {
	m.Forwards1D = append(m.Forwards1D, SoftmaxForwardForCrossEntropyLoss)
}

func (m *Model) SetCrossEntropyLossForSoftmax() {
	m.PredictLoss = NewCrossEntropyLossForSoftmax()
}

func (m Model) Clone() Model {
	return Model{
		Parameter: m.Parameter.Clone(),
		Forwards1D: m.Forwards1D,
		PredictLoss:m.PredictLoss,
	}
}

func (m *Model) Predict(x blas32.Vector) (blas32.Vector, error) {
	y, _, err := m.Forwards1D.Propagate(x)
	return y, err
}

func (m *Model) Accuracy(xs, ts []blas32.Vector) (float32, error) {
	n := len(xs)
	if n != len(ts) {
		return 0.0, fmt.Errorf("バッチサイズが一致しません。")
	}

	correct := 0
	for i := range xs {
		y, err := m.Predict(xs[i])
		if err != nil {
			return 0.0, err
		}
		if oslices.MaxIndices(y.Data)[0] == oslices.MaxIndices(ts[i].Data)[0] {
			correct += 1
		}
	}
	return float32(correct) / float32(n), nil
}

func (m *Model) BackPropagate(x, t blas32.Vector) (blas32.Vector, GradBuffer, error) {
	y, backwards, err := m.Forwards1D.Propagate(x)
	if err != nil {
		return blas32.Vector{}, GradBuffer{}, err
	}
	firstChain, err := m.PredictLoss.Derivative(y, t)
	if err != nil {
		return blas32.Vector{}, GradBuffer{}, err
	}
	return backwards.Propagate(firstChain)
}

func (m *Model) ComputeGrad(xs, ts []blas32.Vector, p int) (GradBuffer, error) {	
	n := len(xs)
	if n != len(ts) {
		return GradBuffer{}, fmt.Errorf("バッチサイズが一致しません。")
	}

	gradByWorker := make(GradBuffers, p)
	for i := range gradByWorker {
		gradByWorker[i] = m.Parameter.NewGradZerosLike()
	}

	errCh := make(chan error, p)
	worker := func(workerIdx int, idxs []int) {
		for _, idx := range idxs {
			x := xs[idx]
			t := ts[idx]
			_, grad, err := m.BackPropagate(x, t)
			if err != nil {
				errCh <- err
				return
			}
			gradByWorker[workerIdx].Axpy(1.0, &grad)
		}
		errCh <- nil
	}

	for workerIdx, idxs := range parallel.DistributeIndicesEvenly(n, p) {
		go worker(workerIdx, idxs)
	}

	for i := 0; i < p; i++ {
		if err := <-errCh; err != nil {
			return GradBuffer{}, err
		}
	}

	total := gradByWorker.Total()
	total.Scal(1.0 / float32(n))
	return total, nil
}

type Optimize func(*Model, *GradBuffer) error

type Momentum struct {
	LearningRate float32
	MomentumRate float32
	velocity     GradBuffer
}

func NewMomentum(param *Parameter) Momentum {
	return Momentum{
		LearningRate:0.01,
		MomentumRate:0.9,
		velocity:param.NewGradZerosLike(),
	}
}

func (m *Momentum) Optimize(model *Model, grad *GradBuffer, c float32) error {
	l2Grad := GradBuffer{
		Weights:tensors2d.NewZerosLike(model.Parameter.Weights),
		Biases:vectors.NewZerosLike(model.Parameter.Biases),
	}
	for i := range l2Grad.Weights {
		w := model.Parameter.Weights[i]
		g := l2Grad.Weights[i]
		for j := range w.Data {
			//(c / 2.0) * w^2 の微分は c * w
			g.Data[j] = c * w.Data[j]
		}
	}

	grad.Axpy(1.0, &l2Grad)
	m.velocity.Scal(m.MomentumRate)
	m.velocity.Axpy(-m.LearningRate, grad)
	model.Parameter.AxpyGrad(1.0, &m.velocity)

	return nil
}