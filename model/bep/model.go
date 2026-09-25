package bep

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"

	"github.com/sw965/omw/encoding/gobx"
	"github.com/sw965/omw/mathx/bitsx"
	"github.com/sw965/omw/parallel"
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
	wCols, err := m.nextLayerInputCols()
	if err != nil {
		return err
	}

	denseLayer, err := NewDense(wRows, wCols, rng)
	if err != nil {
		return err
	}

	m.Backbone = append(m.Backbone, denseLayer)
	return nil
}

func (m *Model) AppendProductDenseLayer(wRows, numBranches int, rng *rand.Rand) error {
	wCols, err := m.nextLayerInputCols()
	if err != nil {
		return err
	}

	productLayer, err := NewProductDense(wRows, wCols, numBranches, rng)
	if err != nil {
		return err
	}

	m.Backbone = append(m.Backbone, productLayer)
	return nil
}

func (m *Model) nextLayerInputCols() (int, error) {
	if m.XRows <= 0 || m.XCols <= 0 {
		return 0, fmt.Errorf("XRowsとXColsが未設定です: XRows = %d, XCols = %d: 層を追加する前に、どちらも正の値を設定するべき", m.XRows, m.XCols)
	}
	if len(m.Backbone) == 0 {
		return m.XCols, nil
	}
	_, wCols, err := m.Backbone.OutputShape(m.XRows, m.XCols)
	if err != nil {
		return 0, err
	}
	return wCols, nil
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

func (m *Model) SetRegressionPrototypes(n int) error {
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
		return fmt.Errorf("prototypesが不足: len(Prototypes) = %d: Valuesを設定する前に、2つ以上のPrototypesを設定するべき", n)
	}

	if minVal >= maxVal {
		return fmt.Errorf("範囲が不正(min >= max): min = %g, max = %g: min < max であるべき", minVal, maxVal)
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
		return fmt.Errorf("valuesが昇順ではありません: Values = %v: 昇順であるべき", m.Values)
	}
	return nil
}

func (m *Model) PredictLogits(x *bitsx.Matrix) ([]int, error) {
	if len(m.Prototypes) == 0 {
		return nil, errors.New("Prototypesが未設定です")
	}

	y, err := m.Backbone.Predict(x)
	if err != nil {
		return nil, err
	}

	n := len(m.Prototypes)
	logits := make([]int, n)
	maxMatch := y.Rows() * y.Cols()

	for i, proto := range m.Prototypes {
		if err := y.ValidateSameShape(proto); err != nil {
			return nil, fmt.Errorf("prototypes[%d]と出力の形状が不一致: %w", i, err)
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

func (m *Model) PredictLabel(x *bitsx.Matrix) (int, error) {
	logits, err := m.PredictLogits(x)
	if err != nil {
		return 0, err
	}
	if len(logits) == 0 {
		return 0, errors.New("logitsが空です")
	}
	return argmax(logits), nil
}

// 最も近いプロトタイプを1つ選ぶのではなく、出力の点灯数から値を読む。温度計のレベル i は先頭の
// ⌊i·L/(n−1)⌋ ビットが 1 なので、点灯数はそのままレベルの連続値になり、全ビットの票を使える。
// 連続量の回帰で MAE が 27〜44% 小さかった(PROTOTYPES.md B-13)。
// 温度計であることの確認は、推論のたびに行うと重いため Trainer.Validate で行う。
func (m *Model) PredictValue(x *bitsx.Matrix) (float32, error) {
	n := len(m.Prototypes)
	if n == 0 {
		return 0.0, errors.New("Prototypesが未設定です")
	}
	if n != len(m.Values) {
		return 0.0, fmt.Errorf("PrototypesとValuesの数が不一致: len(Prototypes) = %d, len(Values) = %d", n, len(m.Values))
	}

	y, err := m.Backbone.Predict(x)
	if err != nil {
		return 0.0, err
	}
	if err := y.ValidateSameShape(m.Prototypes[0]); err != nil {
		return 0.0, fmt.Errorf("prototypes[0]と出力の形状が不一致: %w", err)
	}
	return valueFromOnesCount(y.OnesCount(), y.Rows()*y.Cols(), m.Values), nil
}

func valueFromOnesCount(ones, totalBits int, values []float32) float32 {
	n := len(values)
	if n == 1 {
		return values[0]
	}
	level := float32(ones) * float32(n-1) / float32(totalBits)
	lower := int(level)
	if lower >= n-1 {
		return values[n-1]
	}
	frac := level - float32(lower)
	return values[lower] + frac*(values[lower+1]-values[lower])
}

func validateEvaluationSize(name string, n, p int) error {
	if n == 0 {
		return fmt.Errorf("%sが空です", name)
	}
	if p <= 0 {
		return fmt.Errorf("ワーカー数が不正: p = %d: p > 0 であるべき", p)
	}
	return nil
}

func sumParallel[T int | float32](n, p int, f func(idx int) (T, error)) (T, error) {
	sums := make([]T, p)
	err := parallel.For(n, p, func(workerID, idx int) error {
		v, err := f(idx)
		if err != nil {
			return err
		}
		sums[workerID] += v
		return nil
	})
	if err != nil {
		return 0, err
	}

	var total T
	for _, s := range sums {
		total += s
	}
	return total, nil
}

// 同点のときは後ろの添字を選ぶ。旧実装(安定ソートした Argsort の末尾)と同じ結果にするため。
func argmax(logits []int) int {
	best := 0
	for i, logit := range logits {
		if logit >= logits[best] {
			best = i
		}
	}
	return best
}

func (m *Model) Accuracy(xs bitsx.Matrices, labels []int, p int) (float32, error) {
	n := len(xs)
	if n != len(labels) {
		return 0.0, fmt.Errorf("長さが不一致: len(xs) = %d, len(labels) = %d", n, len(labels))
	}
	if err := validateEvaluationSize("xs", n, p); err != nil {
		return 0.0, err
	}

	totalCorrect, err := sumParallel(n, p, func(idx int) (int, error) {
		predictedLabel, err := m.PredictLabel(xs[idx])
		if err != nil {
			return 0, err
		}
		if predictedLabel == labels[idx] {
			return 1, nil
		}
		return 0, nil
	})
	if err != nil {
		return 0.0, err
	}
	return float32(totalCorrect) / float32(n), nil
}

func (m *Model) Loss(xs bitsx.Matrices, labels []int, p int) (float32, error) {
	n := len(xs)
	if n != len(labels) {
		return 0.0, fmt.Errorf("長さが不一致: len(xs) = %d, len(labels) = %d", n, len(labels))
	}
	if err := validateEvaluationSize("xs", n, p); err != nil {
		return 0.0, err
	}

	if len(m.Values) == 0 {
		return 0.0, errors.New("valuesが未設定です: Lossの計算にはValuesが必要です")
	}

	total, err := sumParallel(n, p, func(idx int) (float32, error) {
		label := labels[idx]
		if label < 0 || label >= len(m.Values) {
			return 0, fmt.Errorf("labelが範囲外: label = %d: 0 <= label < %d であるべき", label, len(m.Values))
		}

		t := m.Values[label]
		y, err := m.PredictValue(xs[idx])
		if err != nil {
			return 0, err
		}

		diff := y - t
		return diff * diff, nil
	})
	if err != nil {
		return 0.0, err
	}
	return total / float32(n), nil
}

// TODO これだと「ローカルのファイルパス」に固定されてしまい、メモリ上のバッファに書きたい、ネットワーク越しに送りたい、テストでbytes.Bufferに対して検証したい、みたいな時に使えません。標準ライブラリの流儀に寄せるなら、io.Writer/io.Readerを受け取る形にして、「パスを開いてWriterを渡す」部分は呼び出し側(今回で言えばatomicfile寄りの薄い関数)に任せる方が、gobx自体の再利用性は上がります。
func (m *Model) Save(path string) error {
	return gobx.Save(m, path)
}

// PredictValue は点灯数から値を読むため、回帰(Values あり)のプロトタイプは温度計である必要がある。
func validateThermometer(prototypes bitsx.Matrices) error {
	if len(prototypes) < 2 {
		return fmt.Errorf("回帰のPrototypesが不足: len(Prototypes) = %d: 2つ以上であるべき", len(prototypes))
	}
	want, err := bitsx.NewThermometerMatrices(len(prototypes), prototypes[0].Rows(), prototypes[0].Cols())
	if err != nil {
		return err
	}
	for i, p := range prototypes {
		if !p.Equal(want[i]) {
			return fmt.Errorf("prototypes[%d]が温度計ではありません: Valuesを使う回帰では SetRegressionPrototypes で設定するべき", i)
		}
	}
	return nil
}
