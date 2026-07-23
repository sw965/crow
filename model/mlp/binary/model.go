package binary

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"

	"github.com/sw965/omw/encoding/gobx"
	"github.com/sw965/omw/mathx/bitsx"
	"github.com/sw965/omw/parallel"
	"github.com/sw965/omw/slicesx"
)

type Model struct {
	Backbone   Sequence
	Prototypes bitsx.Matrices
	// Values は昇順である事を前提とする(ValueToLabelの二分探索や、PairwiseLabelsの値域計算が依存)。
	// SetValues系の関数は昇順で生成する。外部から直接設定する場合も昇順にする事。
	// 昇順でない場合、Trainer.Validate や PairwiseLabels がエラーで弾く。
	Values []float32
	XRows  int
	XCols  int
}

func LoadModel(path string) (Model, error) {
	return gobx.Load[Model](path)
}

func (m *Model) AppendDenseLayer(wRows int, rng *rand.Rand) error {
	if m.XRows <= 0 || m.XCols <= 0 {
		return fmt.Errorf("XRowsとXColsが未設定です: XRows = %d, XCols = %d: 層を追加する前に、どちらも正の値を設定するべき", m.XRows, m.XCols)
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
		return fmt.Errorf("Prototypesが不足: len(Prototypes) = %d: Valuesを設定する前に、2つ以上のPrototypesを設定するべき", n)
	}

	if minVal >= maxVal {
		return fmt.Errorf("範囲が不正(min >= max): min = %v, max = %v: min < max であるべき", minVal, maxVal)
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

// ValueToLabel は、valに最も近いValuesの値のインデックスを返す。
// Valuesが昇順である事を前提に、二分探索で特定する。
// 前後の値と等距離の場合は、小さい方のインデックスを返す。
func (m *Model) ValueToLabel(val float32) int {
	n := len(m.Values)
	if n == 0 {
		return 0
	}

	// idxは「val以上の最初の要素」の位置
	idx, found := slices.BinarySearch(m.Values, val)
	if found {
		return idx
	}
	if idx == 0 {
		return 0
	}
	if idx >= n {
		return n - 1
	}

	lowDiff := val - m.Values[idx-1]
	highDiff := m.Values[idx] - val
	if lowDiff <= highDiff {
		return idx - 1
	}
	return idx
}

// validateAscendingValues は、Valuesが昇順である事を確認する。
func (m *Model) validateAscendingValues() error {
	if !slices.IsSorted(m.Values) {
		return fmt.Errorf("Valuesが昇順ではありません: Values = %v: 昇順であるべき", m.Values)
	}
	return nil
}

func (m *Model) PredictLogits(x *bitsx.Matrix) ([]int, error) {
	if len(m.Prototypes) == 0 {
		return nil, fmt.Errorf("Prototypesが未設定です")
	}

	y, err := m.Backbone.Predict(x)
	if err != nil {
		return nil, err
	}

	n := len(m.Prototypes)
	logits := make([]int, n)
	maxMatch := y.Rows * y.Cols

	for i, proto := range m.Prototypes {
		if err := y.ValidateSameShape(proto); err != nil {
			return nil, fmt.Errorf("Prototypes[%d]と出力の形状が不一致: %w", i, err)
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
		return 0.0, fmt.Errorf("logitsが空です")
	}

	if len(logits) != len(m.Values) {
		return 0.0, fmt.Errorf("logitsとValuesの数が不一致: len(logits) = %d, len(Values) = %d", len(logits), len(m.Values))
	}

	maxLogit := slices.Max(logits)
	var sum float32
	var count int

	for i, logit := range logits {
		if logit == maxLogit {
			sum += m.Values[i]
			count++
		}
	}
	return sum / float32(count), nil
}

func (m *Model) Accuracy(xs bitsx.Matrices, labels []int, p int) (float32, error) {
	n := len(xs)
	if n != len(labels) {
		return 0.0, fmt.Errorf("長さが不一致: len(xs) = %d, len(labels) = %d", n, len(labels))
	}
	if n == 0 {
		return 0.0, fmt.Errorf("xsが空です")
	}
	if p <= 0 {
		return 0.0, fmt.Errorf("ワーカー数が不正: p = %d: p > 0 であるべき", p)
	}
	correctCounts := make([]int, p)

	err := parallel.For(n, p, func(workerID, idx int) error {
		x := xs[idx]
		label := labels[idx]

		logits, err := m.PredictLogits(x)
		if err != nil {
			return err
		}

		if len(logits) == 0 {
			return fmt.Errorf("logitsが空です")
		}

		predictedLabel := slicesx.Argsort(logits)[len(logits)-1]
		if predictedLabel == label {
			correctCounts[workerID]++
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
		return 0.0, fmt.Errorf("長さが不一致: len(xs) = %d, len(labels) = %d", n, len(labels))
	}
	if n == 0 {
		return 0.0, fmt.Errorf("xsが空です")
	}
	if p <= 0 {
		return 0.0, fmt.Errorf("ワーカー数が不正: p = %d: p > 0 であるべき", p)
	}

	if len(m.Values) == 0 {
		return 0.0, fmt.Errorf("Valuesが未設定です: Lossの計算にはValuesが必要です")
	}

	losses := make([]float32, p)
	err := parallel.For(n, p, func(workerID, idx int) error {
		x := xs[idx]
		label := labels[idx]

		if label < 0 || label >= len(m.Values) {
			return fmt.Errorf("labelが範囲外: label = %d: 0 <= label < %d であるべき", label, len(m.Values))
		}

		t := m.Values[label]
		y, err := m.PredictValue(x)
		if err != nil {
			return err
		}

		diff := y - t
		losses[workerID] += diff * diff
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

// TODO これだと「ローカルのファイルパス」に固定されてしまい、メモリ上のバッファに書きたい、ネットワーク越しに送りたい、テストでbytes.Bufferに対して検証したい、みたいな時に使えません。標準ライブラリの流儀に寄せるなら、io.Writer/io.Readerを受け取る形にして、「パスを開いてWriterを渡す」部分は呼び出し側(今回で言えばatomicfile寄りの薄い関数)に任せる方が、gobx自体の再利用性は上がります。
func (m *Model) Save(path string) error {
	return gobx.Save(m, path)
}
