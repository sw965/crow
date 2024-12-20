package model

import (
	"fmt"
	"github.com/sw965/crow/layer"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/crow/ml/1d"
	omwjson "github.com/sw965/omw/json"
	omwslices "github.com/sw965/omw/slices"
	omwrand "github.com/sw965/omw/math/rand"
	"math/rand"
	"github.com/sw965/omw/parallel"
)

type Parameter struct {
	Biases tensor.D2
	Weights tensor.D3
	Filters []tensor.D4
}

func LoadParameterJSON(path string) (Parameter, error) {
	param, err := omwjson.Load[Parameter](path)
	return param, err
}

func (p *Parameter) WriteJSON(path string) error {
	err := omwjson.Write[Parameter](p, path)
	return err
}

type Sequential struct {
	Parameter Parameter
	Forwards layer.Forwards

	YLossCalculator     func(tensor.D1, tensor.D1) (float64, error)
	YLossDifferentiator func(tensor.D1, tensor.D1) (tensor.D1, error)
}

func (s *Sequential) SetParameter(param *Parameter) {
	s.Parameter.Biases.Copy(param.Biases)
	s.Parameter.Weights.Copy(param.Weights)
	for i := range s.Parameter.Filters {
		for j := range s.Parameter.Filters[i] {
			s.Parameter.Filters[i][j].Copy(param.Filters[i][j])
		}
	}
}

func (s *Sequential) SetCrossEntropyError() {
	s.YLossCalculator = ml1d.CrossEntropyError
	s.YLossDifferentiator = ml1d.CrossEntropyErrorDerivative
}

func (s *Sequential) AppendConvLayer(r, c, d, ch int, rn *rand.Rand) {
	he := make(tensor.D4, ch)
	for i := 0; i < ch; i++ {
		he[i] = tensor.NewD3He(d, r, c, rn)
	}
	b := tensor.NewD1Zeros(ch)

	s.Parameter.Filters = append(s.Parameter.Filters, he)
	s.Parameter.Biases = append(s.Parameter.Biases, b)
	s.Forwards = append(s.Forwards, layer.NewConvForward(he, b))
}

func (s *Sequential) AppendFullyConnectedLayer(r, c int, rn *rand.Rand) {
	he := tensor.NewD2He(r, c, rn)
	b := tensor.NewD1Zeros(c)
	s.Parameter.Weights = append(s.Parameter.Weights, he)
	s.Parameter.Biases = append(s.Parameter.Biases, b)
	s.Forwards = append(s.Forwards, layer.NewFullyConnectedForward(he, b))
}

func (s *Sequential) AppendReLULayer() {
	s.Forwards = append(s.Forwards, layer.ReLUForward)
}

func (s *Sequential) AppendLeakyReLULayer(alpha float64) {
	s.Forwards = append(s.Forwards, layer.NewLeakyReLUForward(alpha))
}

func (s *Sequential) AppendFlatLayer() {
	s.Forwards = append(s.Forwards, layer.FlatForward)
}

func (s *Sequential) AppendGAPLayer() {
	s.Forwards = append(s.Forwards, layer.GAPForward)
}

func (s *Sequential) AppendSoftmaxForCrossEntropyLayer() {
	s.Forwards = append(s.Forwards, layer.SoftmaxForwardForCrossEntropy)
}

func (s *Sequential) Predict(x tensor.D3) (tensor.D1, error) {
	y, _, err := s.Forwards.Propagate(x)
	return y[0][0], err
}

func (s *Sequential) MeanLoss(x tensor.D4, t tensor.D2) (float64, error) {
	n := len(x)
	if n != len(t) {
		return 0.0, fmt.Errorf("バッチ数が一致しません。")
	}
	sum := 0.0
	for i := range x {
		y, err := s.Predict(x[i])
		if err != nil {
			return 0.0, err
		}
		yLoss, err := s.YLossCalculator(y, t[i])
		if err != nil {
			return 0.0, err
		}
		sum += yLoss
	}
	mean := sum / float64(n)
	return mean, nil
}

func (s *Sequential) Accuracy(x tensor.D4, t tensor.D2) (float64, error) {
	n := len(x)
	if n != len(t) {
		return 0.0, fmt.Errorf("バッチ数が一致しません。")
	}

	correct := 0
	for i := range x {
		y, err := s.Predict(x[i])
		if err != nil {
			return 0.0, err
		}
		if omwslices.MaxIndex(y) == omwslices.MaxIndex(t[i]) {
			correct += 1
		}
	}
	return float64(correct) / float64(n), nil
}

func (s *Sequential) BackPropagate(x tensor.D3, t tensor.D1) (layer.GradBuffer, error) {
	y, backwards, err := s.Forwards.Propagate(x)
	if err != nil {
		return layer.GradBuffer{}, err
	}

	dLdy, err := s.YLossDifferentiator(y[0][0], t)
	if err != nil {
		return layer.GradBuffer{}, err
	}

	_, gradBuffer, err := backwards.Propagate(dLdy)
	return gradBuffer, err
}

func (s *Sequential) ComputeGrad(features tensor.D4, labels tensor.D2, p int) (layer.GradBuffer, error) {
	firstGradBuffer, err := s.BackPropagate(features[0], labels[0])
	if err != nil {
		return layer.GradBuffer{}, err
	}
	
	gradBuffers := make(layer.GradBuffers, p)
	for i := 0; i < p; i++ {
		gradBuffers[i] = firstGradBuffer.NewZerosLike()
	}
	
	n := len(features)
	errCh := make(chan error, p)
	defer close(errCh)
	
	write := func(idxs []int, gorutineI int) {
		for _, idx := range idxs {
			x := features[idx]
			t := labels[idx]
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
			return layer.GradBuffer{}, err
		}
	}
	
	total := gradBuffers.Total()
	total.Add(&firstGradBuffer)
	
	nf := float64(n)
	total.Biases.DivScalar(nf)
	total.Weights.DivScalar(nf)
	for i := range total.Filters {
		total.Filters[i].DivScalar(nf)
	}
	return total, nil
}

func (s *Sequential) Train(features tensor.D4, labels tensor.D2, c *MiniBatchConfig, r *rand.Rand) error {
	lr := c.LearningRate
	batchSize := c.BatchSize
	p := c.Parallel

	n := len(features)

	if n < batchSize {
		return fmt.Errorf("データ数 < バッチサイズである為、モデルの訓練を出来ません、")
	}

	if c.Epoch <= 0 {
		return fmt.Errorf("エポック数が0以下である為、モデルの訓練を開始出来ません。")
	}

	iter := n / batchSize * c.Epoch
	for i := 0; i < iter; i++ {
		idxs := omwrand.Ints(batchSize, 0, n, r)
		xs := omwslices.ElementsByIndices(features, idxs...)
		ts := omwslices.ElementsByIndices(labels, idxs...)
		grad, err := s.ComputeGrad(xs, ts, p)
		if err != nil {
			return err
		}

		grad.Biases.MulScalar(lr)
		grad.Weights.MulScalar(lr)
		for j := range grad.Filters {
			grad.Filters[j].MulScalar(lr)
		}
		
		s.Parameter.Biases.Sub(grad.Biases)
		s.Parameter.Weights.Sub(grad.Weights)

		for j := range grad.Filters {
			s.Parameter.Filters[j].Sub(grad.Filters[j])
		}
	}
	return nil
}

type MiniBatchConfig struct {
	LearningRate float64
	BatchSize int
	Epoch int
	Parallel int
}