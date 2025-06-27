package linear

import (
	"fmt"
	cmath "github.com/sw965/crow/math"
	crand "github.com/sw965/crow/math/rand"
	omath "github.com/sw965/omw/math"
	"github.com/sw965/omw/parallel"
	oslices "github.com/sw965/omw/slices"
	"math/rand"
)

type Model struct {
	Parameter   Parameter
	OutputLayer OutputLayer
	BiasIndices []int
}

func (m *Model) U(input Input) []float32 {
	u := make([]float32, len(m.BiasIndices))
	for row, entries := range input {
		for _, entry := range entries {
			u[row] += entry.X * m.Parameter.Weight[entry.WeightIndex]
		}
	}

	for i, idx := range m.BiasIndices {
		u[i] += m.Parameter.Bias[idx]
	}
	return u
}

func (m *Model) Predict(input Input) []float32 {
	u := m.U(input)
	return m.OutputLayer.Func(u)
}

func (m *Model) BackPropagate(input Input, t []float32, lossLayer PredictLossLayer) GradBuffer {
	u := m.U(input)
	y := m.OutputLayer.Func(u)

	dLdy := lossLayer.Derivative(y, t)
	dLdu := make([]float32, len(dLdy))

	if m.OutputLayer.Derivative == nil {
		/*
			出力層の導関数が定義されていない場合、連鎖律をそのまま通す。
			Softmax関数はこの設計。
		*/
		for i := range dLdu {
			dLdu[i] = dLdy[i]
		}
	} else {
		dydu := m.OutputLayer.Derivative(y)
		for i := range dLdu {
			//dL/du = dL/dy * dy/du
			dLdu[i] = dLdy[i] * dydu[i]
		}
	}

	dLdw := make([]float32, len(m.Parameter.Weight))

	for row, entries := range input {
		for _, entry := range entries {
			//u = (x*w) + b
			//du/dw = x
			//dL/dw = dL/du * du/dw = dL/du * x
			dLdw[entry.WeightIndex] += dLdu[row] * entry.X
		}
	}

	dLdb := make([]float32, len(m.Parameter.Bias))
	for i, idx := range m.BiasIndices {
		//u = (x * w) + b
		//du / db = 1
		//dL/db = dL/du * du/db = dL/db * 1 = dL/db
		dLdb[idx] += dLdu[i]
	}

	return GradBuffer{
		Weight: dLdw,
		Bias:   dLdb,
	}
}

func (m Model) Clone() Model {
	m.Parameter = m.Parameter.Clone()
	return m
}

func (m *Model) Accuracy(inputs Inputs, ts [][]float32, p int) (float32, error) {
	n := len(inputs)
	if n != len(ts) {
		return 0.0, fmt.Errorf("バッチサイズが一致しません。")
	}
	correctByWorker := make([]int, p)

	err := parallel.For(p, n, func(workerId, idx int) error {
		input := inputs[idx]
		t := ts[idx]
		y := m.Predict(input)
		if oslices.MaxIndices(y)[0] == oslices.MaxIndices(t)[0] {
			correctByWorker[workerId] += 1
		}
		return nil
	})

	if err != nil {
		return 0.0, err
	}

	totalCorrect := omath.Sum(correctByWorker...)
	return float32(totalCorrect) / float32(n), nil
}

func (m Model) ComputeGrad(inputs Inputs, ts [][]float32, lossLayer PredictLossLayer, p int) (GradBuffer, error) {
	n := len(inputs)
	if n != len(ts) {
		return GradBuffer{}, fmt.Errorf("バッチサイズが一致しません。")
	}

	gradByWorker := make(GradBuffers, p)
	for i := 0; i < p; i++ {
		gradByWorker[i] = m.Parameter.NewGradBufferZerosLike()
	}

	err := parallel.For(p, n, func(workerId, idx int) error {
		input := inputs[idx]
		t := ts[idx]
		grad := m.BackPropagate(input, t, lossLayer)
		gradByWorker[workerId].Axpy(1.0, grad)
		return nil
	})

	if err != nil {
		return GradBuffer{}, err
	}

	total := gradByWorker.Total()
	total.Scal(1.0 / float32(n))
	return total, nil
}

func (m Model) EstimateGradBySPSA(c float32, lossFunc func(Model, int) (float32, error), rngByWorker []*rand.Rand) (GradBuffer, error) {
	p := len(rngByWorker)
	gradByWorker := make(GradBuffers, p)
	for i := 0; i < p; i++ {
		gradByWorker[i] = m.Parameter.NewGradBufferZerosLike()
	}
	n := p

	err := parallel.For(p, n, func(workerId, _ int) error {
		rng := rngByWorker[workerId]
		deltaW := make([]float32, len(m.Parameter.Weight))
		for i := range deltaW {
			deltaW[i] = crand.Rademacher(rng)
		}

		deltaB := make([]float32, len(m.Parameter.Bias))
		for i := range deltaB {
			deltaB[i] = crand.Rademacher(rng)
		}

		delta := Parameter{Weight: deltaW, Bias: deltaB}

		perturbation := delta.Clone()
		perturbation.Scal(c)

		plusModel := m.Clone()
		plusModel.Parameter.Axpy(1.0, perturbation)

		minusModel := m.Clone()
		minusModel.Parameter.Axpy(-1.0, perturbation)

		plusLoss, err := lossFunc(plusModel, workerId)
		if err != nil {
			return err
		}

		minusLoss, err := lossFunc(minusModel, workerId)
		if err != nil {
			return err
		}

		grad := gradByWorker[workerId]
		for i := range grad.Weight {
			grad.Weight[i] += cmath.CentralDifference(plusLoss, minusLoss, perturbation.Weight[i])
		}

		for i := range grad.Bias {
			grad.Bias[i] += cmath.CentralDifference(plusLoss, minusLoss, perturbation.Bias[i])
		}
		return nil
	})

	if err != nil {
		return GradBuffer{}, err
	}

	total := gradByWorker.Total()
	total.Scal(1.0 / float32(p))
	return total, nil
}

func (m Model) PartialDifferentiation(lossFunc func(Model, int) (float32, error), p int) (GradBuffer, error) {
	const h float32 = 1e-4
	gradByWorker := make(GradBuffers, p)
	for i := 0; i < p; i++ {
		gradByWorker[i] = m.Parameter.NewGradBufferZerosLike()
	}
	n := p

	err := parallel.For(p, n, func(workerId, _ int) error {
		cm := m.Clone()
		param := cm.Parameter

		// Weightの偏微分
		for i := range param.Weight {
			tmp := param.Weight[i]

			// プラス方向への微小変化
			param.Weight[i] = tmp + h
			plusLoss, err := lossFunc(cm, workerId)
			if err != nil {
				return err
			}

			// マイナス方向への微小変化
			param.Weight[i] = tmp - h
			minusLoss, err := lossFunc(cm, workerId)
			if err != nil {
				return err
			}

			// 微分
			gradByWorker[workerId].Weight[i] = cmath.CentralDifference(plusLoss, minusLoss, h)

			// 元に戻す
			param.Weight[i] = tmp
		}

		// Biasの偏微分
		for i := range param.Bias {
			tmp := param.Bias[i]

			// プラス方向への微小変化
			param.Bias[i] = tmp + h
			plusLoss, err := lossFunc(cm, workerId)
			if err != nil {
				return err
			}

			// マイナス方向への微小変化
			param.Bias[i] = tmp - h
			minusLoss, err := lossFunc(cm, workerId)
			if err != nil {
				return err
			}

			// 微分
			gradByWorker[workerId].Bias[i] = cmath.CentralDifference(plusLoss, minusLoss, h)

			// 元に戻す
			param.Bias[i] = tmp	
		}
		return nil
	})

	if err != nil {
		return GradBuffer{}, err
	}

	total := gradByWorker.Total()
	total.Scal(1.0 / float32(p))
	return total, nil
}