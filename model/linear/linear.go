package linear

import (
	"fmt"
	"math/rand"
	"runtime"
	crowmath "github.com/sw965/crow/math"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/crow/ml/1d"
	"github.com/sw965/omw/fn"
	omwjson "github.com/sw965/omw/json"
	omwrand "github.com/sw965/omw/math/rand"
	omwslices "github.com/sw965/omw/slices"
	"github.com/sw965/omw/parallel"
)

type GradBuffer struct {
	Weight tensor.D2
	Bias   tensor.D1
}

func (g *GradBuffer) NewZerosLike() GradBuffer {
	return GradBuffer{
		Weight:tensor.NewD2ZerosLike(g.Weight),
		Bias:tensor.NewD1ZerosLike(g.Bias),
	}
}

func (g *GradBuffer) Add(other *GradBuffer) error {
	err := g.Weight.Add(other.Weight)
	if err != nil {
		return err
	}
	err = g.Bias.Add(other.Bias)
	return err
}

type GradBuffers []GradBuffer

func (gs GradBuffers) Total() GradBuffer {
	total := gs[0].NewZerosLike()
	for _, g := range gs {
		total.Add(&g)
	}
	return total
}

type Parameter struct {
	Weight [][]*float64
	Bias   []*float64
}

func canonicalWeightCoordinate(coord WeightCoordinate, wf func(WeightCoordinate) WeightCoordinate) WeightCoordinate {
	orbit := []WeightCoordinate{coord}
	seen := map[WeightCoordinate]bool{coord: true}
	current := wf(coord)
	for !seen[current] {
		orbit = append(orbit, current)
		seen[current] = true
		current = wf(current)
	}
	canonical := orbit[0]
	for _, c := range orbit {
		if c.Row < canonical.Row || (c.Row == canonical.Row && c.Column < canonical.Column) {
			canonical = c
		}
	}
	return canonical
}

func canonicalBiasIndex(i int, bf func(int) int) int {
	orbit := []int{i}
	seen := map[int]bool{i: true}
	current := bf(i)
	for !seen[current] {
		orbit = append(orbit, current)
		seen[current] = true
		current = bf(current)
	}
	canonical := orbit[0]
	for _, idx := range orbit {
		if idx < canonical {
			canonical = idx
		}
	}
	return canonical
}

func NewParameter(sizes []int, wf func(WeightCoordinate) WeightCoordinate, bf func(int) int) Parameter {
	seenW := make(map[WeightCoordinate]*float64)
	w := make([][]*float64, len(sizes))
	for row, size := range sizes {
		w[row] = make([]*float64, size)
		for col := 0; col < size; col++ {
			coord := WeightCoordinate{Row: row, Column: col}
			shareCoord := canonicalWeightCoordinate(coord, wf)
			if ptr, ok := seenW[shareCoord]; ok {
				w[row][col] = ptr
			} else {
				newPtr := 1.0
				seenW[shareCoord] = &newPtr
				w[row][col] = &newPtr
			}
		}
	}

	seenB := make(map[int]*float64)
	b := make([]*float64, len(sizes))
	for row := range sizes {
		shareIdx := canonicalBiasIndex(row, bf)
		if ptr, ok := seenB[shareIdx]; ok {
			b[row] = ptr
		} else {
			newPtr := 0.0
			seenB[shareIdx] = &newPtr
			b[row] = &newPtr
		}
	}

	return Parameter{Weight:w, Bias:b}
}

func LoadParameterJSON(path string, wf func(WeightCoordinate) WeightCoordinate, bf func(int) int) (Parameter, error) {
	loadedParam, err := omwjson.Load[Parameter](path)
	if err != nil {
		return Parameter{}, err
	}

	sizes := make([]int, len(loadedParam.Bias))
	for i, row := range loadedParam.Weight {
		sizes[i] = len(row)
	}

	/*
		JSONで読み込んだパラメーターは、ポインターの関係が崩れているので、
		(共有パラメーターの関係が崩れている)
		適切なポインターの関係を保持したパラメーターを生成し、値を代入する。
	*/
	param := NewParameter(sizes, wf, bf)
	for i := range param.Weight {
		for j := range param.Weight[i] {
			*param.Weight[i][j] = *loadedParam.Weight[i][j]
		}
	}
	for i := range param.Bias {
		*param.Bias[i] = *loadedParam.Bias[i]
	}
	return param, nil
}

func (p *Parameter) WriteJSON(path string) error {
	err := omwjson.Write[Parameter](p, path)
	return err
}

func (p *Parameter) Clone() Parameter {
	seenW := make(map[*float64]*float64)
	newWeight := make([][]*float64, len(p.Weight))
	for i, wi := range p.Weight {
		newWi := make([]*float64, len(wi))
		for j, origPtr := range wi {
			//共有関係が崩れないように、origPtrに対応する新しいポインターを既に生成している場合、そのアドレスを割り当てる。
			if newPtr, ok := seenW[origPtr]; ok {
				newWi[j] = newPtr
			} else {
				newV := new(float64)
				*newV = *origPtr
				seenW[origPtr] = newV
				newWi[j] = newV
			}
		}
		newWeight[i] = newWi
	}

	seenB := make(map[*float64]*float64)
	newBias := make([]*float64, len(p.Bias))
	for i, origPtr := range p.Bias {
		//共有関係が崩れないように
		if newPtr, ok := seenB[origPtr]; ok {
			newBias[i] = newPtr
		} else {
			newV := new(float64)
			*newV = *origPtr
			seenB[origPtr] = newV
			newBias[i] = newV
		}
	}

	return Parameter{
		Weight: newWeight,
		Bias:   newBias,
	}
}

func (p *Parameter) AddGrad(grad *GradBuffer) error {
	w := p.Weight
	b := p.Bias
	gw := grad.Weight
	gb := grad.Bias

	/*
		パラメーターはポインターなので、加算をた場合、指定したインデックス以外の
		パラメーターにも影響するが、1つの変数に対して、
		複数の微分得られた場合は、微分結果を合計すればいいので、整合性に問題はない。
	*/

	for i := range w {
		for j := range w[i] {
			*w[i][j] += gw[i][j]
		}
	}

	for i := range b {
		*b[i] += gb[i]
	}
	return nil
}

type Optimizer func(*Model, *GradBuffer) error

type SGD struct {
	LearningRate float64
}

func (sgd *SGD) Optimizer(model *Model, grad *GradBuffer) error {
	lr := sgd.LearningRate
	grad.Weight.MulScalar(-lr)
	grad.Bias.MulScalar(-lr)
	err := model.Parameter.AddGrad(grad)
	return err
}

// https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/optimizer.py
type Momentum struct {
	LearningRate float64
	MomentumRate float64
	velocity     GradBuffer
}

func NewMomentum(model *Model) Momentum {
	w := model.Parameter.Weight
	zeroW := make(tensor.D2, len(w))
	for i := range w {
		zeroW[i] = make(tensor.D1, len(w[i]))
		for j := range w[i] {
			zeroW[i][j] = 0.0
		} 
	}

	b := model.Parameter.Bias
	zeroB := make(tensor.D1, len(b))
	for i := range b {
		zeroB[i] = 0.0
	}

	return Momentum{
		LearningRate: 0.01,
		MomentumRate: 0.9,
		velocity:     GradBuffer{Weight:zeroW, Bias:zeroB},
	}
}

func (m *Momentum) Optimizer(model *Model, grad *GradBuffer) error {
	lr := m.LearningRate

	m.velocity.Weight.MulScalar(m.MomentumRate)
	m.velocity.Bias.MulScalar(m.MomentumRate)

	grad.Weight.MulScalar(-lr)
	grad.Bias.MulScalar(-lr)

	err := m.velocity.Add(grad)
	if err != nil {
		return err
	}

	err = model.Parameter.AddGrad(&m.velocity)
	return err
}

type WeightCoordinate struct {
	Row int
	Column int
}

type Input map[WeightCoordinate]float64
type Inputs []Input

type Model struct {
	Parameter Parameter

	OutputFunc       func(tensor.D1) tensor.D1
	OutputDerivative func(tensor.D1) tensor.D1

	PredictionLossFunc       func(tensor.D1, tensor.D1) (float64, error)
	PredictionLossDerivative func(tensor.D1, tensor.D1) (tensor.D1, error)

	// SPSAの教師なし学習の為の損失関数。
	LossFunc func(*Model) (float64, error)
}

func (m Model) Clone() Model {
	m.Parameter = m.Parameter.Clone()
	return m
}

func (m *Model) SetSigmoid() {
	m.OutputFunc = ml1d.Sigmoid
	m.OutputDerivative = ml1d.SigmoidGrad
}

func (m *Model) SetSoftmaxForCrossEntropy() {
	m.OutputFunc = ml1d.Softmax
	m.OutputDerivative = fn.Identity[tensor.D1]
}

func (m *Model) SetSumSquaredError() {
	m.PredictionLossFunc = ml1d.SumSquaredError
	m.PredictionLossDerivative = ml1d.SumSquaredErrorDerivative
}

func (m *Model) SetCrossEntropyError() {
	m.PredictionLossFunc = ml1d.CrossEntropyError
	m.PredictionLossDerivative = ml1d.CrossEntropyErrorDerivative
}

func (m *Model) LinearSum(input Input) tensor.D1 {
	w := m.Parameter.Weight
	b := m.Parameter.Bias
	u := make(tensor.D1, len(w))
	for k, v := range input {
		r := k.Row
		u[r] += *w[r][k.Column] * v
	}
	for i, v := range b {
		u[i] += *v
	}
	return u
} 

func (m *Model) Predict(input Input) tensor.D1 {
	u := m.LinearSum(input)
	return m.OutputFunc(u)
}

func (m *Model) MeanLoss(inputs Inputs, ts tensor.D2) (float64, error) {
	n := len(inputs)
	if n != len(ts) {
		return 0.0, fmt.Errorf("バッチサイズが一致しません。")
	}

	sum := 0.0
	for i, input := range inputs {
		y := m.Predict(input)
		loss, err := m.PredictionLossFunc(y, ts[i])
		if err != nil {
			return 0.0, err
		}
		sum += loss
	}
	mean := sum / float64(n)
	return mean, nil
}

func (m *Model) Accuracy(inputs Inputs, ts tensor.D2) (float64, error) {
	n := len(inputs)
	if n != len(ts) {
		return 0.0, fmt.Errorf("バッチサイズが一致しません。")
	}

	correct := 0
	for i, input := range inputs {
		y := m.Predict(input)
		if omwslices.MaxIndex(y) == omwslices.MaxIndex(ts[i]) {
			correct += 1
		}
	}
	return float64(correct) / float64(n), nil
}

func (m *Model) BackPropagate(input Input, t tensor.D1) (GradBuffer, error) {
	u := m.LinearSum(input)
	y := m.OutputFunc(u)

	dLdy, err := m.PredictionLossDerivative(y, t)
	if err != nil {
		return GradBuffer{}, err
	}

	dydu := m.OutputDerivative(y)

	dLdu, err := tensor.D1Mul(dLdy, dydu)
	if err != nil {
		return GradBuffer{}, err
	}

	w := m.Parameter.Weight

	//∂L/∂w 
	dw := make(tensor.D2, len(w))
	for i := range dw {
		dw[i] = make(tensor.D1, len(w[i]))
	}

	for k, v := range input {
		r := k.Row
		//x*wのwについての微分はx。連鎖律に基づいて、dLdu[r]を掛ける。
		dw[r][k.Column] = v * dLdu[r]
	}

	//∂L/∂b
	db := dLdu
	return GradBuffer{
		Weight:dw,
		Bias:db,
	}, nil
}

func (m *Model) ComputeGrad(inputs Inputs, ts tensor.D2, p int) (GradBuffer, error) {
	n := len(inputs)
	if n != len(ts) {
		return GradBuffer{}, fmt.Errorf("バッチサイズが一致しません。")
	}

	firstGrad, err := m.BackPropagate(inputs[0], ts[0])
	if err != nil {
		return GradBuffer{}, err
	}
	gradBuffers := make(GradBuffers, p)
	for i := 0; i < p; i++ {
		gradBuffers[i] = firstGrad.NewZerosLike()
	}
	errCh := make(chan error, p)

	worker := func(idxs []int, goroutineI int) {
		for _, idx := range idxs {
			input := inputs[idx+1]
			t := ts[idx+1]
			grad, err := m.BackPropagate(input, t)
			if err != nil {
				errCh <- err
				return
			}
			gradBuffers[goroutineI].Add(&grad)
		}
		errCh <- nil
	}

	for gorutineI, idxs := range parallel.DistributeIndicesEvenly(n-1, p) {
		go worker(idxs, gorutineI)
	}

	for i := 0; i < p; i++ {
		if err := <-errCh; err != nil {
			return GradBuffer{}, err
		}
	}

	total := gradBuffers.Total()
	total.Add(&firstGrad)
	nf := float64(n)
	total.Weight.DivScalar(nf)
	total.Bias.DivScalar(nf)
	return total, nil
}

func (m *Model) EstimateGradBySPSA(c float64, r *rand.Rand) (GradBuffer, error) {
	deltaW := make(tensor.D2, len(m.Parameter.Weight))
	for i, wi := range m.Parameter.Weight {
		deltaW[i] = make(tensor.D1, len(wi))
		for j := range wi {
			if omwrand.Bool(r) {
				deltaW[i][j] = 1.0
			} else {
				deltaW[i][j] = -1.0
			}
		}
	}

	perturbationW := deltaW.Clone()
	perturbationW.MulScalar(c)

	deltaB := make(tensor.D1, len(m.Parameter.Bias))
	for i := range m.Parameter.Bias {
		if omwrand.Bool(r) {
			deltaB[i] = 1.0
		} else {
			deltaB[i] = -1.0
		}
	}

	perturbationB := deltaB.Clone()
	perturbationB.MulScalar(c)

	grad := GradBuffer{
		Weight: make(tensor.D2, len(m.Parameter.Weight)),
		Bias:   make(tensor.D1, len(m.Parameter.Bias)),
	}

	for i := range grad.Weight {
		grad.Weight[i] = make(tensor.D1, len(m.Parameter.Weight[i]))
	}

	plusModel := m.Clone()

	for i := range plusModel.Parameter.Weight {
		for j := range plusModel.Parameter.Weight[i] {
			*plusModel.Parameter.Weight[i][j] += perturbationW[i][j]
		}
	}

	for i := range plusModel.Parameter.Bias {
		*plusModel.Parameter.Bias[i] += perturbationB[i]
	}

	minusModel := m.Clone()

	for i := range minusModel.Parameter.Weight {
		for j := range minusModel.Parameter.Weight[i] {
			*minusModel.Parameter.Weight[i][j] -= perturbationW[i][j]
		}
	}

	for i := range minusModel.Parameter.Bias {
		*minusModel.Parameter.Bias[i] -= perturbationB[i]
	}

	plusLoss, err := m.LossFunc(&plusModel)
	if err != nil {
		return GradBuffer{}, err
	}

	minusLoss, err := m.LossFunc(&minusModel)
	if err != nil {
		return GradBuffer{}, err
	}

	for i := range grad.Weight {
		for j := range grad.Weight[i] {
			grad.Weight[i][j] = crowmath.CentralDifference(plusLoss, minusLoss, perturbationW[i][j])
		}
	}

	for i := range grad.Bias {
		grad.Bias[i] = crowmath.CentralDifference(plusLoss, minusLoss, perturbationB[i])
	}
	return grad, nil
}

type MiniBatchTeacher struct {
	Inputs        Inputs
	Labels        tensor.D2
	MiniBatchSize int
	Epoch         int
	Optimizer     Optimizer
	Parallel      int
}

func NewDefaultMiniBatchTeacher(model *Model, inputs Inputs, ts tensor.D2) MiniBatchTeacher {
	momentum := NewMomentum(model)
	return MiniBatchTeacher{
		Inputs:inputs,
		Labels:ts,
		MiniBatchSize:16,
		Epoch:1,
		Optimizer:momentum.Optimizer,
		Parallel:runtime.NumCPU(),
	}
}

func (mbt *MiniBatchTeacher) Teach(model *Model, r *rand.Rand) error {
	inputs := mbt.Inputs
	ts := mbt.Labels
	size := mbt.MiniBatchSize
	epoch := mbt.Epoch
	op := mbt.Optimizer
	p := mbt.Parallel
	n := len(inputs)

	if n < size {
		return fmt.Errorf("データ数 < バッチサイズである為、モデルの訓練を出来ません。")
	}
	if epoch <= 0 {
		return fmt.Errorf("エポック数が0以下である為、モデルの訓練を開始出来ません。")
	}

	iter := (n / size) * epoch
	for i := 0; i < iter; i++ {
		idxs := omwrand.Ints(size, 0, n, r)
		miniInputs := omwslices.ElementsByIndices(inputs, idxs...)
		miniTs := omwslices.ElementsByIndices(ts, idxs...)

		grad, err := model.ComputeGrad(miniInputs, miniTs, p)
		if err != nil {
			return err
		}

		err = op(model, &grad)
		if err != nil {
			return err
		}
	}
	return nil
}