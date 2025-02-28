package linear

import (
	"fmt"
	"math/rand"
	omwrand "github.com/sw965/omw/math/rand"
	omwslices "github.com/sw965/omw/slices"
	"github.com/sw965/omw/parallel"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/crow/ml/1d"
	"github.com/sw965/crow/ml/2d"
)

type GradBuffer struct {
	Weight tensor.D2
	Bias tensor.D1
}

func (g *GradBuffer) NewZerosLike() GradBuffer {
	return GradBuffer{
		Bias:tensor.NewD1ZerosLike(g.Bias),
		Weight:tensor.NewD2ZerosLike(g.Weight),
	}
}

func (g *GradBuffer) Add(other *GradBuffer) error {
	err := g.Bias.Add(other.Bias)
	if err != nil {
		return err
	}
	err = g.Weight.Add(other.Weight)
	if err != nil {
		return err
	}
	return nil
}

type GradBuffers []GradBuffer

func (gs GradBuffers) Total() GradBuffer {
	total := gs[0].NewZerosLike()
	for _, g := range gs {
		total.Add(&g)
	}
	return total
}

type Parameter struct {
	Weight tensor.D2
	Bias tensor.D1
}

func NewInitParameter(wns []int) Parameter {
	xn := len(wns)
	w := make(tensor.D2, xn)
	b := make(tensor.D1, xn)
	for i, n := range wns {
		w[i] = tensor.NewD1Ones(n)
		b[i] = 0.0
	}
	return Parameter{Weight:w, Bias:b}
}

type Sum struct {
	Parameter Parameter

	YCalculator func(tensor.D1) tensor.D1
	YDifferentiator func(tensor.D1) tensor.D1

	YLossCalculator func(tensor.D1, tensor.D1) (float64, error)
	YLossDifferentiator func(tensor.D1, tensor.D1) (tensor.D1, error)
}

func (s *Sum) SetParameter(param *Parameter) {
	s.Parameter.Weight.Copy(param.Weight)
	s.Parameter.Bias.Copy(param.Bias)
}

func (s *Sum) SetSigmoid() {
	s.YCalculator = ml1d.Sigmoid
	s.YDifferentiator = ml1d.SigmoidGrad
}

func (s *Sum) SetSoftmaxForCrossEntropy() {
	s.YCalculator = ml1d.Softmax
	s.YDifferentiator = func(chain tensor.D1) tensor.D1 {
		return chain
	}
}

func (s *Sum) SetSumSquaredError() {
	s.YLossCalculator = ml1d.SumSquaredError
	s.YLossDifferentiator = ml1d.SumSquaredErrorDerivative
}

func (s *Sum) SetCrossEntropyError() {
	s.YLossCalculator = ml1d.CrossEntropyError
	s.YLossDifferentiator = ml1d.CrossEntropyErrorDerivative
}

func (s *Sum) Predict(x tensor.D2) (tensor.D1, error) {
	y, err := ml2d.LinearSum(x, s.Parameter.Weight, s.Parameter.Bias)
	return y, err
}

func (s *Sum) MeanLoss(xs tensor.D3, ts tensor.D2) (float64, error) {
	n := len(xs)
	if n != len(ts) {
		return 0.0, fmt.Errorf("バッチサイズが一致しません。")
	}

	sum := 0.0
	for i := range xs {
		y, err := s.Predict(xs[i])
		if err != nil {
			return 0.0, err
		}
		yLoss, err := s.YLossCalculator(y, ts[i])
		if err != nil {
			return 0.0, err
		}
		sum += yLoss
	}
	mean := sum / float64(n)
	return mean, nil
}

func (s *Sum) Accuracy(xs tensor.D3, ts tensor.D2) (float64, error) {
	n := len(xs)
	if n != len(ts) {
		return 0.0, fmt.Errorf("バッチサイズが一致しません。")
	}

	correct := 0
	for i := range xs {
		y, err := s.Predict(xs[i])
		if err != nil {
			return 0.0, err
		}
		if omwslices.MaxIndex(y) == omwslices.MaxIndex(ts[i]) {
			correct += 1
		}
	}
	return float64(correct) / float64(n), nil
}

func (s *Sum) BackPropagate(x tensor.D2, t tensor.D1) (GradBuffer, error) {
	gb := GradBuffer{}
	w := s.Parameter.Weight
	b := s.Parameter.Bias

	u, err := ml2d.LinearSum(x, w, b)
	if err != nil {
		return GradBuffer{}, err
	}

	y := s.YCalculator(u)

	dLdy, err := s.YLossDifferentiator(y, t)
	if err != nil {
		return GradBuffer{}, err
	}

	dydu := s.YDifferentiator(y)

	dLdu, err := tensor.D1Mul(dLdy, dydu)
	if err != nil {
		return GradBuffer{}, err
	}

	//∂L/∂w 
	dw, err := tensor.D2MulD1Col(x, dLdu)
	if err != nil {
		return GradBuffer{}, err
	}
	gb.Weight = dw

	//∂L/∂b
	gb.Bias = dLdu

	return gb, nil
}

func (s *Sum) ComputeGrad(xs tensor.D3, ts tensor.D2, p int) (GradBuffer, error) {
	firstGradBuffer, err := s.BackPropagate(xs[0], ts[0])
	if err != nil {
		return GradBuffer{}, err
	}

	gradBuffers := make(GradBuffers, p)
	for i := 0; i < p; i++ {
		gradBuffers[i] = firstGradBuffer.NewZerosLike()
	}

	n := len(xs)
	errCh := make(chan error, p)
	defer close(errCh)

	write := func(idxs []int, gorutineI int) {
		for _, idx := range idxs {
			//firstGradBufferで、0番目のデータの勾配は計算済みなので0にアクセスしないように、+1とする。
			x := xs[idx+1]
			t := ts[idx+1]
			gradBuffer, err := s.BackPropagate(x, t)
			if err != nil {
				errCh <- err
				return
			}
			gradBuffers[gorutineI].Add(&gradBuffer)
		}
		errCh <- nil
	}

	for gorutineI, idxs := range parallel.DistributeIndicesEvenly(n-1, p) {
		go write(idxs, gorutineI)
	}
	
	for i := 0; i < p; i++ {
		err := <- errCh
		if err != nil {
			return GradBuffer{}, err
		}
	}
	
	total := gradBuffers.Total()
	total.Add(&firstGradBuffer)
	
	nf := float64(n)
	total.Bias.DivScalar(nf)
	total.Weight.DivScalar(nf)
	return total, nil
}

func (s *Sum) TrainBySGD(xs tensor.D3, ts tensor.D2, c *MiniBatchConfig, r *rand.Rand) error {
	lr := c.LearningRate
	batchSize := c.BatchSize
	p := c.Parallel

	n := len(xs)

	if n < batchSize {
		return fmt.Errorf("データ数 < バッチサイズである為、モデルの訓練を出来ません、")
	}

	if c.Epoch <= 0 {
		return fmt.Errorf("エポック数が0以下である為、モデルの訓練を開始出来ません。")
	}

	iter := n / batchSize * c.Epoch
	for i := 0; i < iter; i++ {
		idxs := omwrand.Ints(batchSize, 0, n, r)
		miniXs := omwslices.ElementsByIndices(xs, idxs...)
		miniTs := omwslices.ElementsByIndices(ts, idxs...)
		grad, err := s.ComputeGrad(miniXs, miniTs, p)
		if err != nil {
			return err
		}

		grad.Bias.MulScalar(lr)
		grad.Weight.MulScalar(lr)
		
		s.Parameter.Bias.Sub(grad.Bias)
		s.Parameter.Weight.Sub(grad.Weight)
	}
	return nil
}

type MiniBatchConfig struct {
	LearningRate float64
	BatchSize int
	Epoch int
	Parallel int
}