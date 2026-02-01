package linear

import (
	"fmt"
	cmath "github.com/sw965/crow/mathx"
	crand "github.com/sw965/crow/mathx/randx"
	omath "github.com/sw965/omw/mathx"
	"github.com/sw965/omw/parallel"
	"github.com/sw965/omw/slicesx"
	"math/rand/v2"
)

type Model struct {
	Parameter   Parameter
	OutputLayer OutputLayer
	BiasIndices []int
}

func (m *Model) Logits(input Input) ([]float32, error) {
	outSize := len(m.BiasIndices)
	if len(input) != outSize {
		return nil, fmt.Errorf("入力データの次元数(%d)がモデルの出力次元数(%d)と異なります", len(input), len(m.BiasIndices))
	}

	u := make([]float32, outSize)
	for row, entries := range input {
		for _, entry := range entries {
			// entry.WeightIndexが範囲外アクセスで、パニックが起きる可能性があるが、パフォーマンス重視の為、ここはチェックしない
			u[row] += entry.X * m.Parameter.Weight[entry.WeightIndex]
		}
	}

	for i, idx := range m.BiasIndices {
		u[i] += m.Parameter.Bias[idx]
	}
	return u, nil
}

func (m *Model) Predict(input Input) ([]float32, error) {
	u, err := m.Logits(input)
	if err != nil {
		return nil, err
	}
	return m.OutputLayer.Func(u), nil
}

func (m *Model) BackPropagate(input Input, t []float32, lossLayer PredictLossLayer) (GradBuffer, error) {
	u, err := m.Logits(input)
	if err != nil {
		return GradBuffer{}, err
	}
	y := m.OutputLayer.Func(u)

	dLdy := lossLayer.Derivative(y, t)
	dLdu := make([]float32, len(dLdy))

	if m.OutputLayer.Derivative == nil {
		/*
			出力層の導関数が定義されていない場合、連鎖律をそのまま通す。
			NewSoftmaxLayerForCrossEntropyはこの設計。
		*/
		for i := range dLdu {
			dLdu[i] = dLdy[i]
		}
	} else {
		dydu := m.OutputLayer.Derivative(y)
		for i := range dLdu {
			// dL/du = dL/dy * dy/du
			dLdu[i] = dLdy[i] * dydu[i]
		}
	}

	dLdw := make([]float32, len(m.Parameter.Weight))
	for row, entries := range input {
		for _, entry := range entries {
			// u = (x*w) + b
			// du/dw = x
			// dL/dw = dL/du * du/dw = dL/du * x
			dLdw[entry.WeightIndex] += dLdu[row] * entry.X
		}
	}

	dLdb := make([]float32, len(m.Parameter.Bias))
	for i, idx := range m.BiasIndices {
		// u = (x * w) + b
		// du / db = 1
		// dL/db = dL/du * du/db = dL/du * 1 = dL/db
		dLdb[idx] += dLdu[i]
	}

	return GradBuffer{
		Weight: dLdw,
		Bias:   dLdb,
	}, nil
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
	correctCounts := make([]int, p)

	err := parallel.For(n, p, func(workerId, idx int) error {
		input := inputs[idx]
		t := ts[idx]
		y, err := m.Predict(input)
		if err != nil {
			return err
		}

		yIdx := slicesx.Argsort(y)[len(y)-1]
		tIdx := slicesx.Argsort(t)[len(t)-1]
		if yIdx == tIdx {
			correctCounts[workerId] += 1
		}
		return nil
	})

	if err != nil {
		return 0.0, err
	}
	sum := omath.Sum(correctCounts...)
	return float32(sum) / float32(n), nil
}

func (m Model) ComputeGrad(inputs Inputs, ts [][]float32, lossLayer PredictLossLayer, p int) (GradBuffer, error) {
	n := len(inputs)
	if n != len(ts) {
		return GradBuffer{}, fmt.Errorf("バッチサイズが一致しません。")
	}

	grads := make(GradBuffers, p)
	for i := 0; i < p; i++ {
		grads[i] = m.Parameter.NewGradBufferZerosLike()
	}

	err := parallel.For(n, p, func(workerId, idx int) error {
		input := inputs[idx]
		t := ts[idx]
		grad, err := m.BackPropagate(input, t, lossLayer)
		if err != nil {
			return err
		}
		err = grads[workerId].Axpy(1.0, grad)
		return err
	})

	if err != nil {
		return GradBuffer{}, err
	}

	mean, err := grads.ReduceSum()
	if err != nil {
		return GradBuffer{}, err
	}
	mean.Scal(1.0 / float32(n))
	return mean, nil
}

func (m Model) EstimateGradBySPSA(c float32, lossFunc func(Model, int) (float32, error), rngs []*rand.Rand) (GradBuffer, error) {
	p := len(rngs)
	grads := make(GradBuffers, p)
	for i := 0; i < p; i++ {
		grads[i] = m.Parameter.NewGradBufferZerosLike()
	}
	n := p

	err := parallel.For(n, p, func(workerId, _ int) error {
		rng := rngs[workerId]
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
		err := plusModel.Parameter.Axpy(1.0, perturbation)
		if err != nil {
			return err
		}

		minusModel := m.Clone()
		err = minusModel.Parameter.Axpy(-1.0, perturbation)
		if err != nil {
			return err
		}

		plusLoss, err := lossFunc(plusModel, workerId)
		if err != nil {
			return err
		}

		minusLoss, err := lossFunc(minusModel, workerId)
		if err != nil {
			return err
		}

		grad := grads[workerId]
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

	mean, err := grads.ReduceSum()
	if err != nil {
		return GradBuffer{}, err
	}
	mean.Scal(1.0 / float32(n))
	return mean, nil
}

func (m Model) PartialDifferentiation(lossFunc func(Model, int) (float32, error), p int) (GradBuffer, error) {
	const h float32 = 1e-4
	grads := make(GradBuffers, p)
	for i := 0; i < p; i++ {
		grads[i] = m.Parameter.NewGradBufferZerosLike()
	}
	n := p

	err := parallel.For(n, p, func(workerId, _ int) error {
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
			grads[workerId].Weight[i] = cmath.CentralDifference(plusLoss, minusLoss, h)

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
			grads[workerId].Bias[i] = cmath.CentralDifference(plusLoss, minusLoss, h)

			// 元に戻す
			param.Bias[i] = tmp
		}
		return nil
	})

	if err != nil {
		return GradBuffer{}, err
	}

	mean, err := grads.ReduceSum()
	if err != nil {
		return GradBuffer{}, err
	}
	mean.Scal(1.0 / float32(n))
	return mean, nil
}
