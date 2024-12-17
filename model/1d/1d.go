package model1d

import (
	"fmt"
	"github.com/sw965/crow/layer/1d"
	"github.com/sw965/crow/ml/1d"
	"github.com/sw965/crow/ml/2d"
	"github.com/sw965/crow/ml/3d"
	"github.com/sw965/crow/tensor"
	mlmodel "github.com/sw965/crow/ml/model"
	omwjson "github.com/sw965/omw/json"
	omwslices "github.com/sw965/omw/slices"
	"math/rand"
	"github.com/sw965/omw/parallel"
)

type Parameter struct {
	Scalars tensor.D1
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

	YLossCalculator      func(tensor.D1, tensor.D1) (float64, error)
	YLossDifferentiator func(tensor.D1, tensor.D1) (tensor.D1, error)

	ParamScalarsLossCalculator func(tensor.D1) float64
	Param1DsLossCalculator func(tensor.D2) float64
	Param2DsLossCalculator func(tensor.D3) float64
	Param3DsLossCalculator func([]tensor.D3) float64

	Param1DLossDifferentiator func(tensor.D1) tensor.D1
	Param2DLossDifferentiator func(tensor.D2) tensor.D2
	Param3DLossDifferentiator func(tensor.D3) tensor.D3

	GradMaxL2Norm float64
}

func NewSequential(param Parameter) Sequential {
	return Sequential{
		Parameter:                  param,
		ParamScalarsLossCalculator: func(_ tensor.D1) float64 { return 0.0 },
		ParamScalarsLossDifferentiator: tensor.NewD1ZerosLike,

		Param1DsLossCalculator:     func(_ tensor.D2) float64 { return 0.0 },
		Param1DsLossDifferentiator: tensor.NewD2ZerosLike,

		Param2DsLossCalculator:     func(_ tensor.D3) float64 { return 0.0 },
		Param2DsLossDifferentiator: tensor.NewD3ZerosLike,

		Param3DsLossCalculator    : func(_ []tensor.D3) float64 { return 0.0},
		Param3DsLossDifferentiator: func(d4 []tensor.D3) []tensor.D3 {
			zeros := make([]tensor.D3, len(d4))
			for i, d3 := range d4 {
				zeros[i] = tensor.NewD3ZerosLike(d3)
			}
			return zeros
		}
	}
}

func (s *Sequential) SetSumSquaredError() {
	s.YLossCalculator = ml1d.SumSquaredError
	s.YLossDifferentiator = ml1d.SumSquaredErrorDerivative
}

func (s *Sequential) SetCrossEntropyError() {
	s.YLossCalculator = ml1d.CrossEntropyError
	s.YLossDifferentiator = ml1d.CrossEntropyErrorDerivative
}

func NewLinearSum(param Parameter, output layer1d.Forward, c float64) Sequential {
	forwards := layer.Forwards{
		layer.NewLinearSumForward(param.D2s[0], param.D1s[0]),
		output,
	}

	linearSum := NewSequential(param)
	linearSum.Forwards = forwards
	linearSum.Param2DsLossCalculator = ml2d.L2Regularization(c)
	linearSum.Param2DsLossDifferentiator = ml2d.L2RegularizationDerivative(c)
	return linearSum
}

func NewIdentityLinearSum(param Parameter, c float64) Sequential {
	ls := NewLinearSum(param, layer.IdentityForward, c)
	ls.SetSumSquaredError()
	return ls
}

func NewSigmoidLinearSum(param Parameter, c float64) Sequential {
	ls := NewLinearSum(param, layer.SigmoidForward, c)
	ls.SetSumSquaredError()
	return ls
}

func NewSoftmaxLinearSum(param Parameter, c float64) Sequential {
	ls := NewLinearSum(param, layer.SoftmaxForwardForCrossEntropy, c)
	ls.SetCrossEntropyError()
	return ls
}

func (m *Sequential) SetParameter(param *Parameter) {
	m.Parameter.D1.Copy(param.D1)
	m.Parameter.D2.Copy(param.D2)
	m.Parameter.D3.Copy(param.D3)
}

func (m *Sequential) Predict(x tensor.D1) (tensor.D1, error) {
	y, _, err := m.Forwards.Propagate(x)
	return y, err
}

func (m *Sequential) MeanLoss(x, t tensor.D2) (float64, error) {
	n := len(x)
	if n != len(t) {
		return 0.0, fmt.Errorf("バッチ数が一致しません。")
	}

	sum := m.Param1DLossCalculator(m.Parameter.D1)
	sum += m.Param2DLossCalculator(m.Parameter.D2)
	sum += m.Param3DLossCalculator(m.Parameter.D3)
	sum *= float64(n)

	for i := range x {
		y, err := m.Predict(x[i])
		if err != nil {
			return 0.0, err
		}
		yLoss, err := m.YLossCalculator(y, t[i])
		if err != nil {
			return 0.0, err
		}
		sum += yLoss
	}
	mean := sum / float64(n)
	return mean, nil
}

func (m *Sequential) Accuracy(x, t tensor.D2) (float64, error) {
	n := len(x)
	if n != len(t) {
		return 0.0, fmt.Errorf("バッチ数が一致しません。")
	}

	correct := 0
	for i := range x {
		y, err := m.Predict(x[i])
		if err != nil {
			return 0.0, err
		}
		if omwslices.MaxIndex(y) == omwslices.MaxIndex(t[i]) {
			correct += 1
		}
	}
	return float64(correct) / float64(n), nil
}

func (m *Sequential) BackPropagate(x, t tensor.D1) (layer1d.GradBuffer, error) {
	y, backwards, err := m.Forwards.Propagate(x)
	if err != nil {
		return nil err
	}

	dLdy, err := m.YLossDifferentiator(y, t)
	if err != nil {
		return layer1d.GradBuffer, err
	}

	_, gradBuffer, err := backwards.Propagate(dLdy)
	return gradBuffer, err
}

func NewGradComputer(p int) mlmodel.GradComputer[*Sequential, tensor.D2, tensor.D2, tensor.D1, tensor.D1, layer1d.GradBuffer] {
	return func(m *Sequential, features, labels tensor.D2) (layer1d.GradBuffer, error) {
		firstGradBuffer, err := m.BackPropagate(features[0], labels[0])
		if err != nil {
			return layer1d.GradBuffer{}, err
		}
	
		gradBuffers := make(layer1d.GradBuffers, p)
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
				gradBuffer, err := m.BackPropagate(x, t)
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
				return layer1d.GradBuffer{}, err
			}
		}
	
		total := gradBuffers.Total()
		total.Add(&firstGradBuffer)
	
		nf := float64(n)
		total.D1.DivScalar(nf)
		total.D2.DivScalar(nf)
		total.D3.DivScalar(nf)
	
		param1DLossGrad := m.Param1DLossDifferentiator(m.Parameter.D1)
		param2DLossGrad := m.Param2DLossDifferentiator(m.Parameter.D2)
		param3DLossGrad := m.Param3DLossDifferentiator(m.Parameter.D3)
	
		total.D1.Add(param1DLossGrad)
		total.D2.Add(param2DLossGrad)
		total.D3.Add(param3DLossGrad)
	
		if m.GradMaxL2Norm > 0.0 {
			total.ClipUsingL2Norm(m.GradMaxL2Norm)    
		}
		return total, nil
	}
}

func SGD(m *Sequential, gb layer1d.GradBuffer, lr float64) error {
	grad1D := tensor.D1MulScalar(gb.D1, lr)
	grad2D := tensor.D2MulScalar(gb.D2, lr)
	grad3D := tensor.D3MulScalar(gb.D3, lr)

	err := m.Parameter.D1.Sub(grad1D)
	if err != nil {
		return err
	}

	err = m.Parameter.D2.Sub(grad2D)
	if err != nil {
		return err
	}

	err = m.Parameter.D3.Sub(grad3D)
	if err != nil {
		return err
	}
	return nil
}

type Momentum struct {
	Velocity1D tensor.D1
	Velocity2D tensor.D2
	Velocity3D tensor.D3
	Rate float64
}

func NewMomentum(model *Sequential, rate float64) Momentum {
	return Momentum{
		Velocity1D:tensor.NewD1ZerosLike(model.Parameter.D1),
		Velocity2D:tensor.NewD2ZerosLike(model.Parameter.D2),
		Velocity3D:tensor.NewD3ZerosLike(model.Parameter.D3),
		Rate:rate,
	}
}

func (m *Momentum) Optimizer(model *Sequential, gb layer1d.GradBuffer, lr float64) error {
	rate := m.Rate

	for i := range m.Velocity1D {
		v := m.Velocity1D[i]
		grad := gb.D1[i]
		m.Velocity1D[i] = rate*v - lr*grad
	}

	for i := range m.Velocity2D {
		for j := range m.Velocity2D[i] {
			v := m.Velocity2D[i][j]
			grad := gb.D2[i][j]
			m.Velocity2D[i][j] = rate*v - lr*grad
		}
	}

	for i := range m.Velocity3D {
		for j := range m.Velocity3D[i] {
			for k := range m.Velocity3D[i][j] {
				v := m.Velocity3D[i][j][k]
				grad := gb.D3[i][j][k]
				m.Velocity3D[i][j][k] = rate*v - lr*grad
			}
		}
	}

	err := model.Parameter.D1.Add(m.Velocity1D)
	if err != nil {
		return err
	}

	err = model.Parameter.D2.Add(m.Velocity2D)
	if err != nil {
		return err
	}

	return model.Parameter.D3.Add(m.Velocity3D)
}

func NewTrainer(m *Sequential, features, labels tensor.D2, p int) mlmodel.Trainer[*Sequential, tensor.D2, tensor.D2, tensor.D1, tensor.D1, layer1d.GradBuffer] {
	return mlmodel.Trainer[*Sequential, tensor.D2, tensor.D2, tensor.D1, tensor.D1, layer1d.GradBuffer]{
		Features:features,
		Labels:labels,
		GradComputer:NewGradComputer(p),
		Optimizer:SGD,
		BatchSize:16,
		Epoch:1,
	}
}