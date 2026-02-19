package binary

import (
	"fmt"
	"github.com/sw965/omw/encoding/gobx"
	"github.com/sw965/omw/mathx/bitsx"
	"github.com/sw965/omw/parallel"
	"github.com/sw965/omw/slicesx"
	"math"
	"math/rand/v2"
	"slices"
)

type Model struct {
	Backbone   Sequence
	Prototypes bitsx.Matrices
	// 昇順である事を何かしらの形で保証し、それに基づいた高速なアルゴリズムに変更すべし。今はまだ昇順が前提になっていない。
	Values []float32
	XRows  int
	XCols  int
}

func LoadModel(path string) (Model, error) {
	return gobx.Load[Model](path)
}

func (m *Model) AppendDenseLayer(wRows int, rng *rand.Rand) error {
	if m.XRows <= 0 || m.XCols <= 0 {
		return fmt.Errorf("model XRows and XCols must be set before appending layers")
	}

	var wCols int
	var err error
	if len(m.Backbone) == 0 {
		wCols = m.XCols
	} else {
		_, wCols, err = m.Backbone.OutputShape(m.XRows, m.XCols)
		if err != nil {
			return err
		}
	}

	denseLayer, err := NewDense(wRows, wCols, rng)
	if err != nil {
		return err
	}

	m.Backbone = append(m.Backbone, denseLayer)
	return nil
}

func (m *Model) SetClassPrototypes(numClasses int, rng *rand.Rand) error {
	// m.XRows, m.XCols の初期化を忘れててもエラーで弾ける
	yRows, yCols, err := m.Backbone.OutputShape(m.XRows, m.XCols)
	if err != nil {
		return err
	}

	totalBits := numClasses * yRows * yCols
	iters := 10 * int(float64(totalBits)*math.Log(float64(totalBits)))
	protos, err := bitsx.NewETFMatrices(numClasses, yRows, yCols, iters, rng)
	if err != nil {
		return err
	}
	m.Prototypes = protos
	return nil
}

func (m *Model) SetRegressionHighPrototypes(n int, sigma float32, rng *rand.Rand) error {
	yRows, yCols, err := m.Backbone.OutputShape(m.XRows, m.XCols)
	if err != nil {
		return err
	}
	protos, err := bitsx.NewRFFMatrices(n, yRows, yCols, sigma, rng)
	if err != nil {
		return err
	}
	m.Prototypes = protos
	return nil
}

func (m *Model) SetRegressionLowPrototypes(n int) error {
	yRows, yCols, err := m.Backbone.OutputShape(m.XRows, m.XCols)
	if err != nil {
		return err
	}
	protos, err := bitsx.NewThermometerMatrices(n, yRows, yCols)
	if err != nil {
		return err
	}
	m.Prototypes = protos
	return nil
}

func (m *Model) SetValues(minVal, maxVal float32) error {
	n := len(m.Prototypes)
	if n <= 1 {
		return fmt.Errorf("model prototypes must be set before defining target values")
	}

	if minVal >= maxVal {
		return fmt.Errorf("後でエラーメッセージを書く")
	}

	m.Values = make([]float32, n)
	step := (maxVal - minVal) / float32(n-1)
	for i := range n {
		m.Values[i] = minVal + float32(i)*step
	}
	return nil
}

func (m *Model) SetSigmoidValues() error {
	return m.SetValues(0.0, 1.0)
}

func (m *Model) SetTanhValues() error {
	return m.SetValues(-1.0, 1.0)
}

func (m *Model) ValueToLabel(val float32) int {
	n := len(m.Values)
	if n == 0 {
		return 0
	}

	// 線形探索で最も近い値のインデックスを特定
	bestIdx := 0
	minDiff := float32(math.Abs(float64(val - m.Values[0])))

	for i := 1; i < n; i++ {
		diff := float32(math.Abs(float64(val - m.Values[i])))
		if diff < minDiff {
			minDiff = diff
			bestIdx = i
		}
	}
	return bestIdx
}

func (m *Model) PredictLogits(x *bitsx.Matrix) ([]int, error) {
	y, err := m.Backbone.Predict(x)
	if err != nil {
		return nil, err
	}

	n := len(m.Prototypes)
	logits := make([]int, n)
	maxMatch := y.Rows * y.Cols

	for i, proto := range m.Prototypes {
		if err := y.ValidateSameShape(proto); err != nil {
			return nil, fmt.Errorf("後でエラーメッセージを書く")
		}
		mismatch, err := y.HammingDistance(proto)
		if err != nil {
			return nil, err
		}
		logits[i] = maxMatch - mismatch
	}
	return logits, nil
}

func (m *Model) PredictSoftmax(x *bitsx.Matrix) ([]float32, error) {
	logits, err := m.PredictLogits(x)
	if err != nil {
		return nil, err
	}

	maxLogit := slices.Max(logits)
	exps := make([]float64, len(logits))
	var sumExp float64
	for i, l := range logits {
		exps[i] = math.Exp(float64(l - maxLogit))
		sumExp += exps[i]
	}

	y := make([]float32, len(logits))
	for i, exp := range exps {
		y[i] = float32(exp / sumExp)
	}
	return y, nil
}

func (m *Model) PredictValue(x *bitsx.Matrix) (float32, error) {
	logits, err := m.PredictLogits(x)
	if err != nil {
		return 0.0, err
	}

	if len(logits) == 0 {
		return 0.0, fmt.Errorf("logits is empty")
	}

	if len(logits) != len(m.Values) {
		return 0.0, fmt.Errorf("後でエラーメッセージを書く")
	}

	max := slices.Max(logits)
	var sum float32
	var count int

	for i, l := range logits {
		if l == max {
			sum += m.Values[i]
			count++
		}
	}
	return sum / float32(count), nil
}

func (m *Model) Accuracy(xs bitsx.Matrices, labels []int, p int) (float32, error) {
	n := len(xs)
	if n != len(labels) {
		return 0.0, fmt.Errorf("length mismatch: xs %d != labels %d", n, len(labels))
	}
	correctCounts := make([]int, p)

	err := parallel.For(n, p, func(workerId, idx int) error {
		x := xs[idx]
		label := labels[idx]

		logits, err := m.PredictLogits(x)
		if err != nil {
			return err
		}

		if len(logits) == 0 {
			return fmt.Errorf("logits is empty")
		}

		yMaxIdx := slicesx.Argsort(logits)[len(logits)-1]
		if yMaxIdx == label {
			correctCounts[workerId]++
		}
		return nil
	})

	if err != nil {
		return 0.0, err
	}

	totalCorrect := 0
	for _, c := range correctCounts {
		totalCorrect += c
	}
	return float32(totalCorrect) / float32(n), nil
}

func (m *Model) Loss(xs bitsx.Matrices, labels []int, p int) (float32, error) {
	n := len(xs)
	if n != len(labels) {
		return 0.0, fmt.Errorf("length mismatch: xs %d != labels %d", n, len(labels))
	}

	if len(m.Values) == 0 {
		return 0.0, fmt.Errorf("model Values must be set to calculate regression loss")
	}

	losses := make([]float32, p)
	err := parallel.For(n, p, func(workerId, idx int) error {
		x := xs[idx]
		label := labels[idx]

		if label < 0 || label >= len(m.Values) {
			return fmt.Errorf("label index %d is out of range for model Values", label)
		}

		t := m.Values[label]
		y, err := m.PredictValue(x)
		if err != nil {
			return err
		}

		diff := y - t
		losses[workerId] += diff * diff
		return nil
	})

	if err != nil {
		return 0.0, err
	}

	var total float32
	for _, loss := range losses {
		total += loss
	}
	return total / float32(n), nil
}

func (m *Model) Save(path string) error {
	return gobx.Save(m, path)
}
