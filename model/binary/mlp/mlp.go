package mlp

// https://arxiv.org/abs/2512.04189

import (
	"fmt"
	"github.com/sw965/crow/model/binary/layer"
	"github.com/sw965/omw/mathx/bitsx"
	"math/rand/v2"
	"math"
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
	Hs              []H
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
		Hs:make([]H, 0, numLayers),
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

	h := NewH(m.inputDim, dim)
	err = w.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
		wWord := w.Data[ctx.WordIndex]
		hWord := h.Data[ctx.GlobalStart:ctx.GlobalEnd]
		ctx.ScanBits(func(i, col, colT int) {
			wBit := wWord >> uint64(i) & 1
			if wBit == 1 {
				hWord[i] = 31 
			} else {
				hWord[i] = -31
			}
		})
		return nil
	})

	if err != nil {
		return err
	}

	m.Hs = append(m.Hs, h)

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

func (m *Model) UpdateWeight(deltas []layer.Delta, lr float32, rng *rand.Rand) error {
	if len(deltas) != len(m.Ws) {
		return fmt.Errorf("layer count mismatch")
	}

	if lr < 0.0 || lr > 1.0 {
		return fmt.Errorf("後でエラーメッセージを書く")
	}

	for layerI, w := range m.Ws {
		h := m.Hs[layerI]
		delta := deltas[layerI]
		err := w.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
			hWord := h.Data[ctx.GlobalStart:ctx.GlobalEnd]
			deltaWord := delta[ctx.GlobalStart:ctx.GlobalEnd]
			var flip uint64
			ctx.ScanBits(func(i, col, colT int) {
				if rng.Float32() > lr {
					return
				}

				old := hWord[i]
				// オーバーフロー対策に一旦intにする
				newVal := int(old) + int(deltaWord[i])
				clipped := int8(max(math.MinInt8, min(newVal, math.MaxInt8)))
				hWord[i] = clipped

				isOldPlus := old >= 0
				isNewPlus := clipped >= 0
				if isOldPlus != isNewPlus {
					flip |= (1 << uint64(i))
				}
			})
			w.Data[ctx.WordIndex] ^= flip
			return nil
		})

		if err != nil {
			return err
		}
	}
	return nil
}