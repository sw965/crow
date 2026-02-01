package mlp

// https://arxiv.org/abs/2512.04189

import (
	"fmt"
	"github.com/sw965/crow/model/binary/layer"
	"github.com/sw965/omw/mathx/bitsx"
	"math/rand/v2"
)

type H struct {
	Rows int
	Cols int
	Data []int8
}

func NewH(rows, cols int) H {
	return H{
		Rows: rows,
		Cols: cols,
		Data: make([]int8, rows*cols),
	}
}

func (h H) Index(r, c int) int {
	return (r * h.Cols) + c
}

type Model struct {
	Ws              bitsx.Matrices
	wTs             bitsx.Matrices
	H               H
	layersPerWorker layer.Sequences
	inputDim        int
}

func NewModel(numLayers, p int) Model {
	layersPerWorker := make(layer.Sequences, p)
	for workerI := range p {
		layersPerWorker[workerI] = make(layer.Sequence, 0, numLayers)
	}

	return Model{
		Ws:make(bitsx.Matrices, 0, numLayers),
		wTs:make(bitsx.Matrices, 0, numLayers),
		layersPerWorker:layersPerWorker,
	}
}

func (m *Model) AppendSignDotLayer(dim int, rng *rand.Rand) error {
	w, err := bitsx.NewRandMatrix(m.inputDim, dim, 0, rng)
	if err != nil {
		return err
	}
	m.Ws = append(m.Ws, w)

	wt, err := w.Transpose()
	if err != nil {
		return err
	}
	m.wTs = append(m.wTs, wt)

	for workerI := range m.layersPerWorker {
		signDotLayer, err := layer.NewSignDot(w, wt)
		if err != nil {
			return err
		}
		m.layersPerWorker[workerI] = append(m.layersPerWorker[workerI], signDotLayer)
	}
	m.inputDim = dim
	return nil
}

func (m *Model) Predict(x bitsx.Matrix) (bitsx.Matrix, error) {
	if len(m.layersPerWorker) == 0 {
		return bitsx.Matrix{}, fmt.Errorf("model has no workers")
	}
	return m.layersPerWorker[0].Predict(x)
}

func (m *Model) PredictLogits(x bitsx.Matrix, prototypes bitsx.Matrices) ([]int, error) {
	if len(m.layersPerWorker) == 0 {
		return nil, fmt.Errorf("model has no workers")
	}
	return m.layersPerWorker[0].PredictLogits(x, prototypes)
}

func (m *Model) Accuracy(xs []bitsx.Matrix, labels []int, prototypes bitsx.Matrices, p int) (float32, error) {
	if len(m.layersPerWorker) == 0 {
		return 0.0, fmt.Errorf("model has no workers")
	}
	return m.layersPerWorker[0].Accuracy(xs, labels, prototypes, p)
}

func (m *Model) ComputeSignDeltas(xs, ts bitsx.Matrices) ([]layer.Delta, error) {
	return m.layersPerWorker.ComputeSignDeltas(xs, ts)
}

// UpdateWeight は計算された勾配(deltas)に基づいて重みを更新します。
// rate: 学習率（ビットを反転させる確率）
func (m *Model) UpdateWeight(deltas []layer.Delta, rate float32, rng *rand.Rand) error {
	if len(deltas) != len(m.Ws) {
		return fmt.Errorf("layer count mismatch")
	}

	for i, delta := range deltas {
		// ポインタで取得して変更を反映させる
		w := &m.Ws[i]
		wt := &m.wTs[i]

		// layer.goの修正により、deltaは WT (Out x In) に対応する並びになっている
		rowsWT := w.Cols // Out
		colsWT := w.Rows // In

		if len(delta) != rowsWT*colsWT {
			return fmt.Errorf("delta size mismatch at layer %d", i)
		}

		for idx, d := range delta {
			if d == 0 {
				continue
			}
			// 確率的に更新をスキップ (Stochastic Update)
			if rng.Float32() > rate {
				continue
			}

			// deltaの並びは WT (Out x In) なので、座標もそれに合わせる
			rWT := idx / colsWT // Output Index
			cWT := idx % colsWT // Input Index

			// W における座標は (cWT, rWT)
			// 現在のビット値を取得 (0 or 1)
			val, err := w.Bit(cWT, rWT)
			if err != nil {
				return err
			}

			// 更新ロジック:
			// d > 0 (勾配が正): 重みを +1 (bit 1) にしたい -> 今 0 なら反転
			// d < 0 (勾配が負): 重みを -1 (bit 0) にしたい -> 今 1 なら反転
			if (d > 0 && val == 0) || (d < 0 && val == 1) {
				// W を更新
				if err := w.Toggle(cWT, rWT); err != nil {
					return err
				}
				// WT (転置) も整合性を保つため同時に更新
				if err := wt.Toggle(rWT, cWT); err != nil {
					return err
				}
			}
		}
	}
	return nil
}