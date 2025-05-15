package general

import (
	"fmt"
	"github.com/sw965/crow/tensor"
	tmath "github.com/sw965/crow/tensor/math"
	oslices "github.com/sw965/omw/slices"
	"math/rand"
	"github.com/sw965/omw/parallel"
)

type PredictLoss struct {
	Func       func(tensor.D1, tensor.D1) float32
	Derivative func(tensor.D1, tensor.D1) tensor.D1
}

func NewCrossEntropyLossForSoftmax() PredictLoss {
	return PredictLoss{
		Func:       tmath.CrossEntropy,
		Derivative: tmath.SoftmaxCrossEntropyLossDerivative,
	}
}

func NewSumSquaredLoss() PredictLoss {
	return PredictLoss{
		Func : tmath.SumSquaredLoss,
		Derivative : tmath.SumSquaredLossDerivative,
	}
}

type Model struct {
	Parameter   Parameter
	ForwardsD3  ForwardsD3
	D3ToD1      func(tensor.D3) tensor.D1
	D1ToD3      func(tensor.D1, tensor.D3) tensor.D3
	ForwardsD1  ForwardsD1
	PredictLoss PredictLoss
}

func (m *Model) SetFlat() {
	m.D3ToD1 = func(d3 tensor.D3) tensor.D1 {
		return d3.ToD1()
	}

	m.D1TOD3 = func(d1, d3 tensor.D1) tensor.D3 {
		return d1.Reshape3D(d3.Channels, d3.Rows, d3.Cols)
	}
}

func (m *Model) AppendConv2D(fb, fch, fr, fcol, stride int, rng *rand.Rand) {
	filter := tensor.NewD4Zeros(fb, fch, fr, fcol)
	for i := range filter.Data {
		filter.Data[i] = rng.Float32()
	}
	m.Parameter.Filters = append(m.Parameter.Filters, filter)
	m.ForwardsD3 = append(m.ForwardsD3, NewConvForward(filter, stride))
}

func (m *Model) AppendDot(xn, yn int, rng *rand.Rand) {
	w := tensor.NewD2He(xn, yn, rng)
	m.Parameter.Weights = append(m.Parameter.Weights, w)
	m.ForwardsD1 = append(m.ForwardsD1, NewDotForwardD1(w))
}

func (m *Model) AppendLeakyReLU(alpha float32) {
	m.ForwardsD1 = append(m.ForwardsD1, NewLeakyReLUForwardD1(alpha))
}

func (m *Model) AppendInstanceNormalization(n int) error {
	gamma := tensor.NewD1Ones(n)
	m.Parameter.Gammas = append(m.Parameter.Gammas, gamma)

	beta := tensor.NewD1Zeros(n)
	m.Parameter.Biases = append(m.Parameter.Biases, beta)

	forward, err := NewInstanceNormalizationForwardD1(gamma, beta)
	if err != nil {
		return err
	}
	m.ForwardsD1 = append(m.ForwardsD1, forward)
	return nil
}

func (m *Model) AppendSoftmaxForCrossEntropyLoss() {
	m.ForwardsD1 = append(m.ForwardsD1, SoftmaxForwardForCrossEntropyLoss)
}

func (m *Model) SetCrossEntropyLossForSoftmax() {
	m.PredictLoss = NewCrossEntropyLossForSoftmax()
}

func (m *Model) SetSumSquaredLoss() {
	m.PredictLoss = NewSumSquaredLoss()
}

func (m Model) Clone() Model {
	return Model{
		Parameter: m.Parameter.Clone(),
		ForwardsD1: m.ForwardsD1,
		PredictLoss:m.PredictLoss,
	}
}

func (m *Model) Predict(x tensor.D1) (tensor.D1, error) {
	y, _, err := m.ForwardsD1.Propagate(x)
	return y, err
}

func (m *Model) Accuracy(xs, ts tensor.D1Slice) (float32, error) {
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

func (m *Model) BackPropagate(x, t tensor.D1) (tensor.D1, GradBuffer, error) {
	y, backwards, err := m.ForwardsD1.Propagate(x)
	if err != nil {
		return tensor.D1{}, GradBuffer{}, err
	}
	firstChain := m.PredictLoss.Derivative(y, t)
	return backwards.Propagate(firstChain)
}

func (m *Model) ComputeGrad(xs, ts tensor.D1Slice, p int) (GradBuffer, error) {	
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
			gradByWorker[workerIdx].AxpyInPlace(1.0, grad)
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
	total.ScalInPlace(1.0 / float32(n))
	return total, nil
}

type Optimize func(*Model, *GradBuffer, float32) error

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
	l2Grad := model.Parameter.NewGradZerosLike()
	for i := range l2Grad.Weights {
		w := model.Parameter.Weights[i]
		g := l2Grad.Weights[i]
		for j := range w.Data {
			//(c / 2.0) * w^2 の微分は c * w
			g.Data[j] = c * w.Data[j]
		}
	}

	grad.AxpyInPlace(1.0, l2Grad)
	m.velocity.ScalInPlace(m.MomentumRate)
	m.velocity.AxpyInPlace(-m.LearningRate, *grad)
	model.Parameter.AxpyInPlaceGrad(1.0, &m.velocity)
	return nil
}