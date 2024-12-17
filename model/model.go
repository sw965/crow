package model

import (
	"fmt"
	"github.com/sw965/crow/layer"
	"github.com/sw965/crow/tensor"
	omwjson "github.com/sw965/omw/json"
	omwslices "github.com/sw965/omw/slices"
	omwrand "github.com/sw965/omw/math/rand"
	"math/rand"
	"github.com/sw965/omw/parallel"
)

type Parameter struct {
	D1s tensor.D2
	D2s tensor.D3
	D3s []tensor.D3
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
	s.Parameter.D1s.Copy(param.D1s)
	s.Parameter.D2s.Copy(param.D2s)
	for i := range s.Parameter.D3s {
		s.Parameter.D3s[i].Copy(param.D3s[i])
	}
}

func (s *Sequential) Predict(x tensor.D3) (tensor.D1, error) {
	y, _, err := s.Forwards.Propagate(x)
	return y[0][0], err
}

func (s *Sequential) MeanLoss(x []tensor.D3, t tensor.D2) (float64, error) {
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

func (s *Sequential) Accuracy(x []tensor.D3, t tensor.D2) (float64, error) {
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

func (s *Sequential) ComputeGrad(features []tensor.D3, labels tensor.D2, p int) (layer.GradBuffer, error) {
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
	total.D1s.DivScalar(nf)
	total.D2s.DivScalar(nf)
	for i := range total.D3s {
		total.D3s[i].DivScalar(nf)
	}
	return total, nil
}

func (s *Sequential) Train(features []tensor.D3, labels tensor.D2, mbc *MiniBatchConfig, r *rand.Rand) error {
	n := len(features)
	if n < mbc.BatchSize {
		return fmt.Errorf("データ数 < バッチサイズである為、モデルの訓練を出来ません、")
	}

	if mbc.Epoch <= 0 {
		return fmt.Errorf("エポック数が0以下である為、モデルの訓練を開始出来ません。")
	}

	iter := n / mbc.BatchSize * mbc.Epoch
	for i := 0; i < iter; i++ {
		idxs := omwrand.Ints(mbc.BatchSize, 0, n, r)
		xs := omwslices.ElementsByIndices(features, idxs...)
		ts := omwslices.ElementsByIndices(labels, idxs...)
		grad, err := s.ComputeGrad(xs, ts, mbc.Parallel)
		if err != nil {
			return err
		}

		grad.D1s.MulScalar(mbc.LearningRate)
		grad.D2s.MulScalar(mbc.LearningRate)
		for i := range grad.D3s {
			grad.D3s[i].MulScalar(mbc.LearningRate)
		}
		
		s.Parameter.D1s.Sub(grad.D1s)
		s.Parameter.D2s.Sub(grad.D2s)
		for i := range s.Parameter.D3s {
			s.Parameter.D3s[i].Sub(grad.D3s[i])
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