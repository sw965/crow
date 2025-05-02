package mlp

import (
	"gonum.org/v1/gonum/blas/blas32"
	"github.com/sw965/crow/blas32/vector"
	"github.com/sw965/crow/blas32/tensor/2d"
	"math/rand"
	cmath "github.com/sw965/crow/math"
)

type Forward func(blas32.Vector) (blas32.Vector, error)
type Forwards []Forward

func (fs Forwards) Propagate(x blas32.Vector) (blas32.Vector, error) {
	var err error
	for _, f := range fs {
		x, err = f(x)
		if err != nil {
			return blas32.Vector{}, err
		}
	}
	return x, nil
}

type GradBuffer struct {
	Weights []blas32.General
	Biases  []blas32.Vector
}

type GradBuffers []GradBuffer

func (gs GradBuffers) TotalAxpy(alpha float32) GradBuffer {
	total := gs[0]
	for _, grad := range gs[1:] {
		for i := range grad.Weights {
			tensor2d.Axpy(alpha, grad.Weights[i], total.Weights[i])
		}
		for i := range grad.Biases {
			blas32.Axpy(alpha, grad.Biases[i], total.Biases[i])
		}
	}
	return total
}

type Parameter struct {
	Weights []blas32.General
	Biases []blas32.Vector

	WeightsLen int
	BiasesLen int
}

func (p *Parameter) Clone() Parameter {
	ws := make([]blas32.General, p.WeightsLen)
	for i := range ws {
		ws[i] = tensor2d.Clone(p.Weights[i])
	}

	bs := make([]blas32.Vector, p.BiasesLen)
	for i := range bs {
		bs[i] = vector.Clone(p.Biases[i])
	}
	return Parameter{
		Weights:ws,
		Biases:bs,
		WeightsLen:p.WeightsLen,
		BiasesLen:p.BiasesLen,
	}
}

func (p *Parameter) NewGradSameShapeZeros() GradBuffer {
	grad := GradBuffer{
		Weights:make([]blas32.General, p.WeightsLen),
		Biases:make([]blas32.Vector, p.BiasesLen),
	}
	for i := range p.Weights {
		grad.Weights[i] = tensor2d.NewZerosLike(p.Weights[i])
	}

	for i := range p.Biases {
		grad.Biases[i] = vector.NewZerosLike(p.Biases[i])
	}
	return grad
}

type Model struct {
	Parameter Parameter
	Forwards  Forwards
	LossFunc func(*Model) (float32, error)
}

func (m Model) AppendAffine(xn, yn int, rng *rand.Rand) {
	w := tensor2d.NewHe(xn, yn, rng)
	b := vector.NewZeros(yn)
	forward := func(x blas32.Vector) (blas32.Vector, error) {
		return vector.Affine(x, w, b), nil
	}
	m.Forwards = append(m.Forwards, forward)
}

func (m Model) Clone() Model {
	return Model{
		Parameter:m.Parameter.Clone(),
		Forwards:m.Forwards,
		LossFunc:m.LossFunc,
	}
}

func (m *Model) Predict(x blas32.Vector) (blas32.Vector, error) {
	return m.Forwards.Propagate(x)
}

func (m *Model) EstimateGradBySPSA(c float32, rng *rand.Rand, p int) (GradBuffer, error) {
	grads := make(GradBuffers, p)
	for i := 0; i < p; i++ {
		grads[i] = m.Parameter.NewGradSameShapeZeros()
	}
	errCh := make(chan error, p)

	worker := func(goroutineI int) {
		deltaWs := make([]blas32.General, m.Parameter.WeightsLen)
		for i := range deltaWs {
			w := m.Parameter.Weights[i]
			deltaWs[i] = tensor2d.NewRademacherLike(w, rng)
		}

		deltaBs := make([]blas32.Vector, m.Parameter.BiasesLen)
		for i := range deltaBs {
			b := m.Parameter.Biases[i]
			deltaBs[i] = vector.NewRademacherLike(b, rng)
		}

		plusModel := m.Clone()
		minusModel := m.Clone()

		for i := range plusModel.Parameter.Weights {
			tensor2d.Axpy(c, deltaWs[i], plusModel.Parameter.Weights[i])
			tensor2d.Axpy(-c, deltaWs[i], minusModel.Parameter.Weights[i])
		}

		for i := range plusModel.Parameter.Biases {
			blas32.Axpy(c, deltaBs[i], plusModel.Parameter.Biases[i])
			blas32.Axpy(-c, deltaBs[i], minusModel.Parameter.Biases[i])
		}

		plusLoss, err := m.LossFunc(&plusModel)
		if err != nil {
			errCh <- err
			return
		}

		minusLoss, err := m.LossFunc(&minusModel)
		if err != nil {
			errCh <- err
			return
		}

		grad := grads[goroutineI]

		for i := range grad.Weights {
			gradW := grad.Weights[i]
			deltaW := deltaWs[i]
			for j := range gradW.Data {
				gradW.Data[j] = cmath.CentralDifference(plusLoss, minusLoss, c*deltaW.Data[j])
			}
		}

		for i := range grad.Biases {
			gradB := grad.Biases[i]
			deltaB := deltaBs[i]
			for j := range gradB.Data {
				gradB.Data[j] = cmath.CentralDifference(plusLoss, minusLoss, c*deltaB.Data[j])
			}
		}
		errCh <- nil
	}

	for i := 0; i < p; i++ {
		go worker(i)
	}

	for i := 0; i < p; i++ {
		if err := <- errCh; err != nil {
			return GradBuffer{}, err
		}
	}

	total := grads.TotalAxpy(float32(p))
	return total, nil
}

type Optimizer func(*Model) error

