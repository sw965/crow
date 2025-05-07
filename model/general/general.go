package general

import (
	"fmt"
	"github.com/sw965/crow/blas32/tensor/2d"
	"github.com/sw965/crow/blas32/vector"
	"github.com/sw965/crow/blas32/tensors/2d"
	"github.com/sw965/crow/blas32/vectors"
	omath "github.com/sw965/omw/math"
	oslices "github.com/sw965/omw/slices"
	"gonum.org/v1/gonum/blas/blas32"
	"math"
	"math/rand"
	"github.com/sw965/omw/parallel"
	"github.com/sw965/crow/model/layer"
	"github.com/sw965/crow/model/layer/1d"
)

type Parameter struct {
	Weights []blas32.General
	Biases  []blas32.Vector
}

func (p *Parameter) NewGradZerosLike() layer.GradBuffer {
	return layer.GradBuffer{
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

func (p *Parameter) AxpyGrad(alpha float32, grad *layer.GradBuffer) {
	tensors2d.Axpy(alpha, grad.Weights, p.Weights)
	vectors.Axpy(alpha, grad.Biases, p.Biases)
}

type PredictLoss struct {
	Func       func(blas32.Vector, blas32.Vector) (float32, error)
	Derivative func(blas32.Vector, blas32.Vector) (blas32.Vector, error)
}

func NewCrossEntropyLossForSoftmax() PredictLoss {
	f := func(y, t blas32.Vector) (float32, error) {
		loss := float32(0.0)
		e := float32(0.0001)
		for i := range y.Data {
			ye := float64(omath.Max(y.Data[i], e))
			te := t.Data[i]
			loss += -te * float32(math.Log(ye))
		}
		return loss, nil
	}

	d := func(y, t blas32.Vector) (blas32.Vector, error) {
		dx := blas32.Vector{
			N:    y.N,
			Inc:  y.Inc,
			Data: make([]float32, y.N),
		}
		blas32.Copy(y, dx)
		blas32.Axpy(-1.0, t, dx)
		return dx, nil
	}

	return PredictLoss{
		Func:       f,
		Derivative: d,
	}
}

type Model struct {
	Parameter   Parameter
	Forwards1D  layer1d.Forwards
	PredictLoss PredictLoss
}

func (m *Model) AppendAffine(xn, yn int, rng *rand.Rand) {
	w := tensor2d.NewHe(xn, yn, rng)
	m.Parameter.Weights = append(m.Parameter.Weights, w)

	b := vector.NewZeros(yn)
	m.Parameter.Biases = append(m.Parameter.Biases, b)

	m.Forwards1D = append(m.Forwards1D, layer1d.NewAffineForward(w, b))
}

func (m *Model) AppendLeakyReLU(alpha float32) {
	m.Forwards1D = append(m.Forwards1D, layer1d.NewLeakyReLUForward(alpha))
}

func (m *Model) AppendOutputSoftmaxAndSetCrossEntropyLoss() {
	m.Forwards1D = append(m.Forwards1D, layer1d.SoftmaxForOutputForward)
	m.PredictLoss = NewCrossEntropyLossForSoftmax()
}

func (m Model) Clone() Model {
	return Model{
		Parameter: m.Parameter.Clone(),
		Forwards1D: m.Forwards1D,
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

func (m *Model) BackPropagate(x, t blas32.Vector) (blas32.Vector, layer.GradBuffer, error) {
	y, backwards, err := m.Forwards1D.Propagate(x)
	if err != nil {
		return blas32.Vector{}, layer.GradBuffer{}, err
	}
	firstChain, err := m.PredictLoss.Derivative(y, t)
	if err != nil {
		return blas32.Vector{}, layer.GradBuffer{}, err
	}
	return backwards.Propagate(firstChain)
}

func (m *Model) ComputeGrad(xs, ts []blas32.Vector, p int) (layer.GradBuffer, error) {	
	n := len(xs)
	if n != len(ts) {
		return layer.GradBuffer{}, fmt.Errorf("バッチサイズが一致しません。")
	}

	gradByWorker := make(layer.GradBuffers, p)
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
			return layer.GradBuffer{}, err
		}
	}

	avg := gradByWorker.Average()
	return avg, nil
}