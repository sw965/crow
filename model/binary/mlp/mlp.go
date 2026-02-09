package mlp

// https://arxiv.org/abs/2512.04189

import (
	"fmt"
	"github.com/sw965/crow/model/binary/layer"
	"github.com/sw965/omw/mathx/bitsx"
	"github.com/sw965/omw/mathx/randx"
	"math/rand/v2"
	"math"
	"github.com/sw965/omw/slicesx"
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
	Prototypes      bitsx.Matrices

	layersPerWorker layer.Sequences
	rngs            []*rand.Rand
	inputDim        int

	TrainingContext *layer.TrainingContext
}

func NewModel(inputDim, numLayers, p int) Model {
	layersPerWorker := make(layer.Sequences, p)
	for workerI := range p {
		layersPerWorker[workerI] = make(layer.Sequence, 0, numLayers)
	}

	rngs := make([]*rand.Rand, p)
	for i := range p {
		rngs[i] = randx.NewPCGFromGlobalSeed()
	}

	return Model{
		Ws:make(bitsx.Matrices, 0, numLayers),
		wTs:make(bitsx.Matrices, 0, numLayers),
		Hs:make([]H, 0, numLayers),
		layersPerWorker:layersPerWorker,
		rngs:rngs,
		inputDim:inputDim,
	}
}

func (m *Model) AppendDenseLayer(dim int) error {
	// rngs と layersPerWorkerの長さチェックを入れる？
	rng := m.rngs[0]
	w, err := bitsx.NewRandMatrix(dim, m.inputDim, 0, rng)
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
				hWord[i] = math.MaxInt8 / 4
			} else {
				hWord[i] = math.MinInt8 / 4
			}
		})
		return nil
	})

	if err != nil {
		return err
	}

	m.Hs = append(m.Hs, h)

	for workerI := range m.layersPerWorker {
		denseLayer, err := layer.NewDense(w, wt, m.inputDim, m.TrainingContext)
		if err != nil {
			return err
		}
		denseLayer.Rng = m.rngs[workerI]
		m.layersPerWorker[workerI] = append(m.layersPerWorker[workerI], denseLayer)
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

func (m *Model) PredictLogits(x bitsx.Matrix) ([]int, error) {
	if len(m.layersPerWorker) == 0 {
		return nil, fmt.Errorf("model has no workers")
	}
	return m.layersPerWorker[0].PredictLogits(x, m.Prototypes)
}

func (m *Model) PredictSoftmax(x bitsx.Matrix) ([]float32, error) {
	if len(m.layersPerWorker) == 0 {
		return nil, fmt.Errorf("model has no workers")
	}
	return m.layersPerWorker[0].PredictSoftmax(x, m.Prototypes)
}

func (m *Model) Accuracy(xs bitsx.Matrices, labels []int, p int) (float32, error) {
	if len(m.layersPerWorker) == 0 {
		return 0.0, fmt.Errorf("model has no workers")
	}
	return m.layersPerWorker[0].Accuracy(xs, labels, m.Prototypes, p)
}

// 後で並列化にする
func (m *Model) UpdateWeight(deltas []layer.Delta, lr float32, rng *rand.Rand) error {
	if lr < 0.0 || lr > 1.0 {
		return fmt.Errorf("後でエラーメッセージを書く")
	}

	for layerI, w := range m.Ws {
		delta := deltas[layerI]
		wt := m.wTs[layerI]
		h := m.Hs[layerI]
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
					err := wt.Toggle(col, ctx.Row)
					if err != nil {
						panic(err)
					}
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

// Trainメソッドにエラーを集約させる？
func (m *Model) Train(xs bitsx.Matrices, labels []int, config *TrainingConfig) error {
	if err := config.Validate(); err != nil {
		return err
	}

	n := len(xs)
	if config.MiniBatchSize <= 0 {
		return fmt.Errorf("後でエラーメッセージを書く")
	}

	miniBatchSize := config.MiniBatchSize
	if n < miniBatchSize {
		miniBatchSize = n
	}

	batchIdxs := config.Rng.Perm(n)
	for i := 0; i < n; i += miniBatchSize {
		end := i + miniBatchSize
		if end > n {
			end = n
		}

		miniIdxs := batchIdxs[i:end]
		miniXs, err := slicesx.ElementsByIndices(xs, miniIdxs...)
		if err != nil {
			return err
		}

		miniLabels, err := slicesx.ElementsByIndices(labels, miniIdxs...)
		if err != nil {
			return err
		}

		signDeltas, err := m.layersPerWorker.ComputeSignDeltas(miniXs, miniLabels, m.Prototypes, config.Margin)
		if err != nil {
			return err
		}

		err = m.UpdateWeight(signDeltas, config.LearningRate, config.Rng)
		if err != nil {
			return err
		}
	}
	return nil
}

type TrainingConfig struct {
	MiniBatchSize int
    LearningRate  float32
	Margin        float32
	Rng           *rand.Rand
}

func (tc *TrainingConfig) Validate() error {
	if tc.MiniBatchSize <= 0 {
		return fmt.Errorf("後でエラーメッセージを書く")
	}

	if tc.LearningRate <= 0.0 {
		return fmt.Errorf("後でエラーメッセージを書く")
	}

	if tc.Rng == nil {
		return fmt.Errorf("後でエラーメッセージを書く")
	}
	return nil
}