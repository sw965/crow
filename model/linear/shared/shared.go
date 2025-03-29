package shared

import (
	"fmt"
	"math/rand"
	crowmath "github.com/sw965/crow/math"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/crow/ml/1d"
	"github.com/sw965/omw/fn"
	"golang.org/x/exp/maps"
	omwrand "github.com/sw965/omw/math/rand"
	omwslices "github.com/sw965/omw/slices"
	"github.com/sw965/omw/parallel"
)

type GradBuffer[Wk, Bk comparable] struct {
	Weight map[Wk]float64
	Bias   map[Bk]float64
}

func (g *GradBuffer[Wk, Bk]) NewZerosLike() GradBuffer[Wk, Bk] {
	zeroW := make(map[Wk]float64)
	for k := range g.Weight {
		zeroW[k] = 0.0
	}
	zeroB := make(map[Bk]float64)
	for k := range g.Bias {
		zeroB[k] = 0.0
	}
	return GradBuffer[Wk, Bk]{
		Weight: zeroW,
		Bias:   zeroB,
	}
}

func (g *GradBuffer[Wk, Bk]) MulScalar(s float64) {
	for k, v := range g.Weight {
		g.Weight[k] = v * s
	}
	for k, v := range g.Bias {
		g.Bias[k] = v * s
	}
}

func (g *GradBuffer[Wk, Bk]) DivScalar(s float64) {
	for k, v := range g.Weight {
		g.Weight[k] = v / s
	}
	for k, v := range g.Bias {
		g.Bias[k] = v / s
	}
}

func (g *GradBuffer[Wk, Bk]) Add(other *GradBuffer[Wk, Bk]) error {
	for k, v := range other.Weight {
		g.Weight[k] += v
	}
	for k, v := range other.Bias {
		g.Bias[k] += v
	}
	return nil
}

type GradBuffers[Wk, Bk comparable] []GradBuffer[Wk, Bk]

func (gs GradBuffers[Wk, Bk]) Total() GradBuffer[Wk, Bk] {
	total := gs[0].NewZerosLike()
	for _, g := range gs {
		total.Add(&g)
	}
	return total
}

type Parameter[Wk, Bk comparable] struct {
	Weight map[Wk]float64
	Bias   map[Bk]float64
}

func (p *Parameter[Wk, Bk]) Clone() Parameter[Wk, Bk] {
	return Parameter[Wk, Bk]{
		Weight: maps.Clone(p.Weight),
		Bias:   maps.Clone(p.Bias),
	}
}

func (p *Parameter[Wk, Bk]) AddGrad(grad *GradBuffer[Wk, Bk]) error {
	for k, v := range grad.Weight {
		p.Weight[k] += v
	}
	for k, v := range grad.Bias {
		p.Bias[k] += v
	}
	return nil
}

func NewInitParameter[Wk, Bk comparable](wks []Wk, bks []Bk) Parameter[Wk, Bk] {
	w := make(map[Wk]float64)
	for _, k := range wks {
		w[k] = 1.0
	}
	b := make(map[Bk]float64)
	for _, k := range bks {
		b[k] = 0.0
	}
	return Parameter[Wk, Bk]{Weight: w, Bias: b}
}

type Optimizer[Wk, Bk comparable] func(*Model[Wk, Bk], *GradBuffer[Wk, Bk]) error

type SGD[Wk, Bk comparable] struct {
	LearningRate float64
}

func (sgd *SGD[Wk, Bk]) Optimizer(model *Model[Wk, Bk], grad *GradBuffer[Wk, Bk]) error {
	lr := sgd.LearningRate
	grad.MulScalar(-lr)
	err := model.Parameter.AddGrad(grad)
	return err
}

type Momentum[Wk, Bk comparable] struct {
	LearningRate float64
	MomentumRate float64
	velocity     GradBuffer[Wk, Bk]
}

func NewMomentum[Wk, Bk comparable](model *Model[Wk, Bk]) Momentum[Wk, Bk] {
	v := GradBuffer[Wk, Bk]{
		Weight: make(map[Wk]float64),
		Bias:   make(map[Bk]float64),
	}
	// パラメーターと同じキーを初期化
	for k := range model.Parameter.Weight {
		v.Weight[k] = 0.0
	}
	for k := range model.Parameter.Bias {
		v.Bias[k] = 0.0
	}
	return Momentum[Wk, Bk]{
		LearningRate: 0.01,
		MomentumRate: 0.9,
		velocity:     v,
	}
}

func (m *Momentum[Wk, Bk]) Optimizer(model *Model[Wk, Bk], grad *GradBuffer[Wk, Bk]) error {
	lr := m.LearningRate
	m.velocity.MulScalar(m.MomentumRate)
	grad.MulScalar(-lr)
	m.velocity.Add(grad)
	err := model.Parameter.AddGrad(&m.velocity)
	return err
}

type Input[Wk comparable] []map[Wk]float64
type Inputs[Wk comparable] []Input[Wk]

type Model[Wk, Bk comparable] struct {
	Parameter Parameter[Wk, Bk]
	BiasKeys []Bk

	OutputFunc       func(tensor.D1) tensor.D1
	OutputDerivative func(tensor.D1) tensor.D1

	PredictionLossFunc       func(tensor.D1, tensor.D1) (float64, error)
	PredictionLossDerivative func(tensor.D1, tensor.D1) (tensor.D1, error)

	// SPSAの教師なし学習の為の損失関数。
	LossFunc func(*Model[Wk, Bk]) (float64, error)
}

func (m Model[Wk, Bk]) Clone() Model[Wk, Bk] {
	m.Parameter = m.Parameter.Clone()
	return m
}

func (m *Model[Wk, Bk]) SetSigmoid() {
	m.OutputFunc = ml1d.Sigmoid
	m.OutputDerivative = ml1d.SigmoidGrad
}

func (m *Model[Wk, Bk]) SetSoftmaxForCrossEntropy() {
	m.OutputFunc = ml1d.Softmax
	m.OutputDerivative = fn.Identity[tensor.D1]
}

func (m *Model[Wk, Bk]) SetSumSquaredError() {
	m.PredictionLossFunc = ml1d.SumSquaredError
	m.PredictionLossDerivative = ml1d.SumSquaredErrorDerivative
}

func (m *Model[Wk, Bk]) SetCrossEntropyError() {
	m.PredictionLossFunc = ml1d.CrossEntropyError
	m.PredictionLossDerivative = ml1d.CrossEntropyErrorDerivative
}

func (m *Model[Wk, Bk]) LinearSum(x Input[Wk]) tensor.D1 {
	w := m.Parameter.Weight
	b := m.Parameter.Bias
	u := make(tensor.D1, len(x))
	for i, sample := range x {
		sum := 0.0
		bk := m.BiasKeys[i]
		for k, v := range sample {
			sum += w[k] * v
		}
		u[i] = sum + b[bk]
	}
	return u
} 

func (m *Model[Wk, Bk]) Predict(x Input[Wk]) tensor.D1 {
	u := m.LinearSum(x)
	return m.OutputFunc(u)
}

func (m *Model[Wk, Bk]) MeanLoss(xs Inputs[Wk], ts tensor.D2) (float64, error) {
	n := len(xs)
	if n != len(ts) {
		return 0.0, fmt.Errorf("バッチサイズが一致しません。")
	}

	sum := 0.0
	for i, x := range xs {
		y := m.Predict(x)
		loss, err := m.PredictionLossFunc(y, ts[i])
		if err != nil {
			return 0.0, err
		}
		sum += loss
	}
	mean := sum / float64(n)
	return mean, nil
}


func (m *Model[Wk, Bk]) Accuracy(xs Inputs[Wk], ts tensor.D2) (float64, error) {
	n := len(xs)
	if n != len(ts) {
		return 0.0, fmt.Errorf("バッチサイズが一致しません。")
	}

	correct := 0
	for i, x := range xs {
		y := m.Predict(x)
		if omwslices.MaxIndex(y) == omwslices.MaxIndex(ts[i]) {
			correct += 1
		}
	}
	return float64(correct) / float64(n), nil
}

func (m *Model[Wk, Bk]) BackPropagate(x Input[Wk], t tensor.D1) (GradBuffer[Wk, Bk], error) {
	u := m.LinearSum(x)
	y := m.OutputFunc(u)

	dLdy, err := m.PredictionLossDerivative(y, t)
	if err != nil {
		return GradBuffer[Wk, Bk]{}, err
	}

	dydu := m.OutputDerivative(y)

	dLdu, err := tensor.D1Mul(dLdy, dydu)
	if err != nil {
		return GradBuffer[Wk, Bk]{}, err
	}

	//∂L/∂w 
	dw := map[Wk]float64{}
	for i, g := range dLdu {
		for k, v := range x[i] {
			/*
				wに対する微分は入力値(wxをwについて微分)
				連鎖律に基づき、gを掛ける。
				同じ変数に対して、複数の微分結果が得られた場合、合計すればいいので += で加算する。
			*/
			dw[k] += g * v
		}
	}

	//∂L/∂b
	db := map[Bk]float64{}
	for i, bk := range m.BiasKeys {
		/*
			バイアスに対する微分は1(... + b において、bについて微分すると、... の部分は定数と見なすので0となる)
			よって、連鎖律に基づき、1(bの微分) * dLdu[i]となり、そのままdLdu[i]が微分結果になる。
		*/
		db[bk] = dLdu[i]
	}

	return GradBuffer[Wk, Bk]{
		Weight:dw,
		Bias:db,
	}, nil
}

func (m *Model[Wk, Bk]) ComputeGrad(xs Inputs[Wk], ts tensor.D2, p int) (GradBuffer[Wk, Bk], error) {
	n := len(xs)
	if n != len(ts) {
		return GradBuffer[Wk, Bk]{}, fmt.Errorf("バッチサイズが一致しません。")
	}
	// 最初のサンプルの勾配を計算
	firstGrad, err := m.BackPropagate(xs[0], ts[0])
	if err != nil {
		return GradBuffer[Wk, Bk]{}, err
	}
	gradBuffers := make(GradBuffers[Wk, Bk], p)
	for i := 0; i < p; i++ {
		gradBuffers[i] = firstGrad.NewZerosLike()
	}
	errCh := make(chan error, p)

	worker := func(idxs []int, goroutineI int) {
		for _, idx := range idxs {
			x := xs[idx+1]
			t := ts[idx+1]
			grad, err := m.BackPropagate(x, t)
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
			return GradBuffer[Wk, Bk]{}, err
		}
	}

	total := gradBuffers.Total()
	total.Add(&firstGrad)
	total.DivScalar(float64(n))
	return total, nil
}

func (m *Model[Wk, Bk]) EstimateGradBySPSA(c float64, r *rand.Rand) (GradBuffer[Wk, Bk], error) {
	deltaW := make(map[Wk]float64)
	for k := range m.Parameter.Weight {
		var e float64
		if omwrand.Bool(r) {
			e = 1.0
		} else {
			e = -1.0
		}
		deltaW[k] = e
	}

	perturbationW := make(map[Wk]float64)
	for k, v := range deltaW {
		perturbationW[k] = c * v
	}

	deltaB := make(map[Bk]float64)
	for k := range m.Parameter.Bias {
		var e float64
		if omwrand.Bool(r) {
			e = 1.0
		} else {
			e = -1.0
		}
		deltaB[k] = e
	}

	perturbationB := make(map[Bk]float64)
	for k, v := range deltaB {
		perturbationB[k] = c * v
	}

	grad := GradBuffer[Wk, Bk]{
		Weight: make(map[Wk]float64),
		Bias:   make(map[Bk]float64),
	}

	plusModel := m.Clone()
	for k, v := range perturbationW {
		plusModel.Parameter.Weight[k] += v
	}
	for k, v := range perturbationB {
		plusModel.Parameter.Bias[k] += v
	}

	minusModel := m.Clone()
	for k, v := range perturbationW {
		minusModel.Parameter.Weight[k] -= v
	}
	for k, v := range perturbationB {
		minusModel.Parameter.Bias[k] -= v
	}

	plusLoss, err := m.LossFunc(&plusModel)
	if err != nil {
		return GradBuffer[Wk, Bk]{}, err
	}

	minusLoss, err := m.LossFunc(&minusModel)
	if err != nil {
		return GradBuffer[Wk, Bk]{}, err
	}

	for k := range deltaW {
		grad.Weight[k] = crowmath.CentralDifference(plusLoss, minusLoss, perturbationW[k])
	}

	for k := range deltaB {
		grad.Bias[k] = crowmath.CentralDifference(plusLoss, minusLoss, perturbationB[k])
	}
	return grad, nil
}

type MiniBatchTeacher[Wk, Bk comparable] struct {
	Inputs        Inputs[Wk]
	Labels        tensor.D2
	MiniBatchSize int
	Epoch         int
	Optimizer     Optimizer[Wk, Bk]
	Parallel      int
}

func (mbt *MiniBatchTeacher[Wk, Bk]) Teach(model *Model[Wk, Bk], r *rand.Rand) error {
	xs := mbt.Inputs
	ts := mbt.Labels
	size := mbt.MiniBatchSize
	epoch := mbt.Epoch
	op := mbt.Optimizer
	p := mbt.Parallel
	n := len(xs)

	if n < size {
		return fmt.Errorf("データ数 < バッチサイズである為、モデルの訓練を出来ません。")
	}
	if epoch <= 0 {
		return fmt.Errorf("エポック数が0以下である為、モデルの訓練を開始出来ません。")
	}

	iter := (n / size) * epoch
	for i := 0; i < iter; i++ {
		idxs := omwrand.Ints(size, 0, n, r)
		miniXs := omwslices.ElementsByIndices(xs, idxs...)
		miniTs := omwslices.ElementsByIndices(ts, idxs...)

		grad, err := model.ComputeGrad(miniXs, miniTs, p)
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