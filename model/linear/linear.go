package linear

import (
	omwjson "github.com/sw965/omw/json"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/crow/ml/1d"
	"github.com/sw965/crow/ml/2d"
	"github.com/sw965/omw/parallel"
	"runtime"
)

type Parameter struct {
	W tensor.D2
	B tensor.D1
}

func NewZerosParameter(r, c int) Parameter {
	return Parameter{
		W:tensor.NewD2Zeros(r, c),
		B:tensor.NewD1Zeros(c),
	}
}

func LoadParameterJSON(path string) (Parameter, error) {
	param, err := omwjson.Load[Parameter](path)
	return param, err
}

func (p *Parameter) WriteJSON(path string) error {
	err := omwjson.Write[Parameter](p, path)
	return err
}

type Model struct {
	Parameter Parameter
	OutputCalculator     func(tensor.D1) tensor.D1
	OutputDifferentiator func(tensor.D1, tensor.D1) (tensor.D1, error)
	YLossCalculator      func(tensor.D1, tensor.D1) (float64, error)
	YLossDifferentiator  func(tensor.D1, tensor.D1) (tensor.D1, error)
}

func NewSigmoidModel(r, c int) Model {
	param := NewZerosParameter(r, c)
	model := Model{
		Parameter:param,
		OutputCalculator:ml1d.Sigmoid,
		OutputDifferentiator:func(y, chain tensor.D1) (tensor.D1, error) {
			dydx := ml1d.SigmoidGrad(y)
			// ∂L/∂x
			dx, err := tensor.D1Mul(dydx, chain)
			return dx, err
		},
		YLossCalculator:ml1d.SumSquaredError,
		YLossDifferentiator:ml1d.SumSquaredErrorDerivative,
	}
	return model
}

func NewSoftmaxModel(r, c int) Model {
	param := NewZerosParameter(r, c)
	model := Model{
		Parameter:param,
		OutputCalculator:ml1d.Softmax,
		OutputDifferentiator:func(_, chain tensor.D1) (tensor.D1, error) {
			return chain, nil
		},
		YLossCalculator:ml1d.CrossEntropyError,
		YLossDifferentiator:ml1d.CrossEntropyErrorDerivative,
	}
	return model
}

func (m *Model) SetParameter(param *Parameter) {
	m.Parameter.W.Copy(param.W)
	m.Parameter.B.Copy(param.B)
}

func (m *Model) Predict(x tensor.D2) (tensor.D1, error) {
	u, err := ml2d.LinearSum(x, m.Parameter.W, m.Parameter.B)
	if err != nil {
		return nil, err
	}
	return m.OutputCalculator(u), err
}

func (m *Model) ComputeGrad(xs tensor.D3, ts tensor.D2, p int) (tensor.D2, tensor.D1, error) {
	n := len(xs)
	gradWs := make(tensor.D3, p)
	for i := range gradWs {
		gradWs[i] = tensor.NewD2ZerosLike(m.Parameter.W)
	}

	gradBs := make(tensor.D2, p)
	for i := range gradBs {
		gradBs[i] = tensor.NewD1ZerosLike(m.Parameter.B)
	}

	errCh := make(chan error, p)
	defer close(errCh)

	for gorutineI, idxs := range parallel.DistributeIndicesEvenly(n, p) {
		go func(idxs []int, gorutineI int) {
			for _, idx := range idxs {
				x := xs[idx]
				t := ts[idx]

				y, err := m.Predict(x)
				if err != nil {
					errCh <- err
					return
				}
		
				dLdy, err := m.YLossDifferentiator(y, t)
				if err != nil {
					errCh <- err
					return
				}
		
				dLdu, err := m.OutputDifferentiator(y, dLdy)
				if err != nil {
					errCh <- err
					return
				}
		
				_, dudw, err := ml2d.LinearSumDerivative(x, m.Parameter.W)
				if err != nil {
					errCh <- err
					return
				}
		
				//∂L/∂w
				dw, err := tensor.D2MulD1Row(dudw, dLdu)
				if err != nil {
					errCh <- err
				}

				err = gradWs[gorutineI].Add(dw)
				if err != nil {
					errCh <- err
					return
				}

				//∂L/∂b
				db := dLdu
				err = gradBs[gorutineI].Add(db)
				if err != nil {
					errCh <- err
					return
				}
			}
			errCh <- nil
		}(idxs, gorutineI)
	}

	for i := 0; i < p; i++ {
		err := <- errCh
		if err != nil {
			return nil, nil, err
		}
	}

	gradW := tensor.NewD2ZerosLike(m.Parameter.W)
	for _, e := range gradWs {
		gradW.Add(e)
	}
	gradW.DivScalar(float64(n))

	gradB := tensor.NewD1ZerosLike(m.Parameter.B)
	for _, e := range gradBs {
		gradB.Add(e)
	}
	gradB.DivScalar(float64(n))

	return gradW, gradB, nil
}

func (m *Model) Train(xs tensor.D3, ts tensor.D2, c *MiniBatchConfig) error {
	n := len(xs)
	lr := c.LearningRate
	p := c.Parallel
	iterNum := (n / c.BatchSize) * c.Epoch

	for i := 0; i < iterNum; i++ {
		gradW, gradB, err := m.ComputeGrad(xs, ts, p)
		if err != nil {
			return err
		}

		gradW.MulScalar(lr)
		gradB.MulScalar(lr)

		m.Parameter.W.Sub(gradW)
		m.Parameter.B.Sub(gradB)
	}
	return nil
}

type MiniBatchConfig struct {
	LearningRate float64
	BatchSize int
	Epoch int
	Parallel int
}

func NewDefaultMiniBatchConfig() MiniBatchConfig {
	return MiniBatchConfig{
		LearningRate:0.001,
		BatchSize:16,
		Epoch:1,
		Parallel:runtime.NumCPU(),
	}
}