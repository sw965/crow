package linear

import (
	"fmt"
	"math/rand"
	"github.com/sw965/omw/fn"
	omwrand "github.com/sw965/omw/math/rand"
	omwslices "github.com/sw965/omw/slices"
	omwjson "github.com/sw965/omw/json"
	"github.com/sw965/omw/parallel"
	crowmath "github.com/sw965/crow/math"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/crow/ml/1d"
	"github.com/sw965/crow/ml/2d"
)

type GradBuffer struct {
	Weight tensor.D2
	Bias tensor.D1
}

func (g *GradBuffer) NewZerosLike() GradBuffer {
	return GradBuffer{
		Bias:tensor.NewD1ZerosLike(g.Bias),
		Weight:tensor.NewD2ZerosLike(g.Weight),
	}
}

func (g *GradBuffer) Add(other *GradBuffer) error {
	err := g.Bias.Add(other.Bias)
	if err != nil {
		return err
	}
	err = g.Weight.Add(other.Weight)
	if err != nil {
		return err
	}
	return nil
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
	Weight tensor.D2
	Bias tensor.D1
}

func LoadParameterJSON(path string) (Parameter, error) {
	param, err := omwjson.Load[Parameter](path)
	return param, err
}

func (p *Parameter) WriteJSON(path string) error {
	err := omwjson.Write[Parameter](p, path)
	return err
}

func (p *Parameter) Clone() Parameter {
	return Parameter{
		Weight:p.Weight.Clone(),
		Bias:p.Bias.Clone(),
	}
}

func (p *Parameter) AddGrad(grad *GradBuffer) error {
	err := p.Weight.Add(grad.Weight)
	if err != nil {
		return err
	}

	err = p.Bias.Add(grad.Bias)
	return err
}

func (p *Parameter) SubGrad(grad *GradBuffer) error {
	err := p.Weight.Sub(grad.Weight)
	if err != nil {
		return err
	}

	err = p.Bias.Sub(grad.Bias)
	return err
}

func NewInitParameter(wns []int) Parameter {
	xn := len(wns)
	w := make(tensor.D2, xn)
	b := make(tensor.D1, xn)
	for i, n := range wns {
		w[i] = tensor.NewD1Ones(n)
		b[i] = 0.0
	}
	return Parameter{Weight:w, Bias:b}
}

type Optimizer func(*Model, *GradBuffer) error 

type SGD struct {
	LearningRate float64
}

func (sgd *SGD) Optimizer(model *Model, grad *GradBuffer) error {
	lr := sgd.LearningRate
	grad.Bias.MulScalar(-lr)
	grad.Weight.MulScalar(-lr)
	err := model.Parameter.AddGrad(grad)
	return err
}

type Momentum struct {
	LearningRate float64
	MomentumRate float64
	velocity GradBuffer
}

func NewMomentum(model *Model) Momentum {
	v := GradBuffer{
		Weight:tensor.NewD2ZerosLike(model.Parameter.Weight),
		Bias:tensor.NewD1ZerosLike(model.Parameter.Bias),
	}

	return Momentum{
		LearningRate:0.01,
		MomentumRate:0.9,
		velocity:v,
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

type Model struct {
	Parameter Parameter

	YCalculator func(tensor.D1) tensor.D1
	YDifferentiator func(tensor.D1) tensor.D1

	YLossCalculator func(tensor.D1, tensor.D1) (float64, error)
	YLossDifferentiator func(tensor.D1, tensor.D1) (tensor.D1, error)

	//SPSAによる教師なし学習の為の損失関数。
	ModelLossCaluclator func(*Model) (float64, error)
}

func (m Model) Clone() Model {
	m.Parameter = m.Parameter.Clone()
	return m
}

func (m *Model) SetSigmoid() {
	m.YCalculator = ml1d.Sigmoid
	m.YDifferentiator = ml1d.SigmoidGrad
}

func (m *Model) SetSoftmaxForCrossEntropy() {
	m.YCalculator = ml1d.Softmax
	m.YDifferentiator = fn.Identity[tensor.D1]
}

func (m *Model) SetSumSquaredError() {
	m.YLossCalculator = ml1d.SumSquaredError
	m.YLossDifferentiator = ml1d.SumSquaredErrorDerivative
}

func (m *Model) SetCrossEntropyError() {
	m.YLossCalculator = ml1d.CrossEntropyError
	m.YLossDifferentiator = ml1d.CrossEntropyErrorDerivative
}

func (m *Model) Predict(x tensor.D2) (tensor.D1, error) {
	u, err := ml2d.LinearSum(x, m.Parameter.Weight, m.Parameter.Bias)
	if err != nil {
		return nil, nil
	}
	y := m.YCalculator(u)
	return y, err
}

func (m *Model) MeanLoss(xs tensor.D3, ts tensor.D2) (float64, error) {
	n := len(xs)
	if n != len(ts) {
		return 0.0, fmt.Errorf("バッチサイズが一致しません。")
	}

	sum := 0.0
	for i := range xs {
		y, err := m.Predict(xs[i])
		if err != nil {
			return 0.0, err
		}
		yLoss, err := m.YLossCalculator(y, ts[i])
		if err != nil {
			return 0.0, err
		}
		sum += yLoss
	}
	mean := sum / float64(n)
	return mean, nil
}

func (m *Model) Accuracy(xs tensor.D3, ts tensor.D2) (float64, error) {
	n := len(xs)
	if n != len(ts) {
		return 0.0, fmt.Errorf("バッチサイズが一致しません。")
	}

	correct := 0
	for i := range xs {
		y, err := m.Predict(xs[i])
		if err != nil {
			return 0.0, err
		}
		if omwslices.MaxIndex(y) == omwslices.MaxIndex(ts[i]) {
			correct += 1
		}
	}
	return float64(correct) / float64(n), nil
}

func (m *Model) BackPropagate(x tensor.D2, t tensor.D1) (GradBuffer, error) {
	gb := GradBuffer{}
	w := m.Parameter.Weight
	b := m.Parameter.Bias

	u, err := ml2d.LinearSum(x, w, b)
	if err != nil {
		return GradBuffer{}, err
	}

	y := m.YCalculator(u)

	dLdy, err := m.YLossDifferentiator(y, t)
	if err != nil {
		return GradBuffer{}, err
	}

	dydu := m.YDifferentiator(y)

	dLdu, err := tensor.D1Mul(dLdy, dydu)
	if err != nil {
		return GradBuffer{}, err
	}

	//∂L/∂w 
	dw, err := tensor.D2MulD1Col(x, dLdu)
	if err != nil {
		return GradBuffer{}, err
	}
	gb.Weight = dw

	//∂L/∂b
	gb.Bias = dLdu

	return gb, nil
}

func (m *Model) ComputeGrad(xs tensor.D3, ts tensor.D2, p int) (GradBuffer, error) {
	firstGradBuffer, err := m.BackPropagate(xs[0], ts[0])
	if err != nil {
		return GradBuffer{}, err
	}

	gradBuffers := make(GradBuffers, p)
	for i := 0; i < p; i++ {
		gradBuffers[i] = firstGradBuffer.NewZerosLike()
	}

	n := len(xs)
	errCh := make(chan error, p)
	defer close(errCh)

	write := func(idxs []int, gorutineI int) {
		for _, idx := range idxs {
			//firstGradBufferで、0番目のデータの勾配は計算済みなので0にアクセスしないように、+1とする。
			x := xs[idx+1]
			t := ts[idx+1]
			gradBuffer, err := m.BackPropagate(x, t)
			if err != nil {
				errCh <- err
				return
			}
			gradBuffers[gorutineI].Add(&gradBuffer)
		}
		errCh <- nil
	}

	for gorutineI, idxs := range parallel.DistributeIndicesEvenly(n-1, p) {
		go write(idxs, gorutineI)
	}
	
	for i := 0; i < p; i++ {
		err := <- errCh
		if err != nil {
			return GradBuffer{}, err
		}
	}
	
	total := gradBuffers.Total()
	total.Add(&firstGradBuffer)
	
	nf := float64(n)
	total.Bias.DivScalar(nf)
	total.Weight.DivScalar(nf)
	return total, nil
}

func (m *Model) EstimateGradBySPSA(c float64, r *rand.Rand) (GradBuffer, error) {
	deltaW := tensor.NewD2ZerosLike(m.Parameter.Weight)
	for i := range deltaW {
		for j := range deltaW[i] {
			var e float64
			if omwrand.Bool(r) {
				e = 1
			} else {
				e = -1
			}
			deltaW[i][j] = e
		}
	}

	perturbationW := deltaW.Clone()
	perturbationW.MulScalar(c)

	deltaB := tensor.NewD1ZerosLike(m.Parameter.Bias)
	for i := range deltaB {
		var e float64
		if omwrand.Bool(r) {
			e = 1
		} else {
			e = -1
		}
		deltaB[i] = e
	}

	perturbationB := deltaB.Clone()
	perturbationB.MulScalar(c)

	grad := GradBuffer{
		Weight:tensor.NewD2ZerosLike(m.Parameter.Weight),
		Bias:tensor.NewD1ZerosLike(m.Parameter.Bias),
	}

	plusModel := m.Clone()

	err := plusModel.Parameter.Weight.Add(perturbationW)
	if err != nil {
		return GradBuffer{}, err
	}

	err = plusModel.Parameter.Bias.Add(perturbationB)
	if err != nil {
		return GradBuffer{}, err
	}

	minusModel := m.Clone()

	err = minusModel.Parameter.Weight.Sub(perturbationW)
	if err != nil {
		return GradBuffer{}, err
	}

	err = minusModel.Parameter.Bias.Sub(perturbationB)
	if err != nil {
		return GradBuffer{}, err
	}

	plusModelLoss, err := m.ModelLossCaluclator(&plusModel)
	if err != nil {
		return GradBuffer{}, err
	}

	minusModelLoss, err := m.ModelLossCaluclator(&minusModel)
	if err != nil {
		return GradBuffer{}, err
	}

	for i := range deltaW {
		for j := range deltaW[i] {
			grad.Weight[i][j] = crowmath.CentralDifference(plusModelLoss, minusModelLoss, perturbationW[i][j])
		}
	}

	for i := range deltaB {
		grad.Bias[i] = crowmath.CentralDifference(plusModelLoss, minusModelLoss, perturbationB[i])
	}
	return grad, nil
}

type MiniBatchTeacher struct {
	Inputs tensor.D3
	Labels tensor.D2
	MiniBatchSize int
	Epoch int
	Optimizer Optimizer
	Parallel int
}

func (mbt *MiniBatchTeacher) Teach(model *Model, r *rand.Rand) error {
	xs := mbt.Inputs
	ts := mbt.Labels
	size := mbt.MiniBatchSize
	epoch := mbt.Epoch
	opt := mbt.Optimizer
	p := mbt.Parallel
	n := len(xs)

	if n < size {
		return fmt.Errorf("データ数 < バッチサイズである為、モデルの訓練を出来ません、")
	}

	if epoch <= 0 {
		return fmt.Errorf("エポック数が0以下である為、モデルの訓練を開始出来ません。")
	}

	iter := n / size * epoch
	for i := 0; i < iter; i++ {
		idxs := omwrand.Ints(size, 0, n, r)
		miniXs := omwslices.ElementsByIndices(xs, idxs...)
		miniTs := omwslices.ElementsByIndices(ts, idxs...)

		grad, err := model.ComputeGrad(miniXs, miniTs, p)
		if err != nil {
			return err
		}

		err = opt(model, &grad)
		if err != nil {
			return err
		}
	}
	return nil
}