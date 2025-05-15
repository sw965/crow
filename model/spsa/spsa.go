package spsa

import (
	"fmt"
	"github.com/sw965/crow/model/consts"
	"github.com/sw965/crow/tensor"
	tmath "github.com/sw965/crow/tensor/math"
	"github.com/sw965/crow/tensor/nn"
	cmath "github.com/sw965/crow/math"
	oslices "github.com/sw965/omw/slices"
	"math/rand"
	"slices"
)

type GradBuffer struct {
	Filter tensor.D4
	Weight tensor.D2
	Gamma  tensor.D1
	Bias   tensor.D1
}

func (g GradBuffer) NewZerosLike() GradBuffer {
	return GradBuffer{
		Filter:g.Filter.NewZerosLike(),
		Weight:g.Weight.NewZerosLike(),
		Gamma:g.Gamma.NewZerosLike(),
		Bias:g.Bias.NewZerosLike(),
	}
}

func (g GradBuffer) Clone() GradBuffer {
	return GradBuffer{
		Filter:g.Filter.Clone(),
		Weight:g.Weight.Clone(),
		Gamma:g.Gamma.Clone(),
		Bias:g.Bias.Clone(),
	}
}

func (g *GradBuffer) AxpyInPlace(alpha float32, x *GradBuffer) {
	if g.Filter.Batches != 0 {
		g.Filter.AxpyInPlace(alpha, x.Filter)
	}

	if g.Weight.Rows != 0 {
		g.Weight.AxpyInPlace(alpha, x.Weight)
	}

	if g.Gamma.N != 0 {
		g.Gamma.AxpyInPlace(alpha, x.Gamma)
	}

	if g.Bias.N != 0 {
		g.Bias.AxpyInPlace(alpha, x.Bias)
	}
}

func (g *GradBuffer) ScalInPlace(alpha float32) {
	if g.Filter.Batches != 0 {
		g.Filter.ScalInPlace(alpha)
	}

	if g.Weight.Rows != 0 {
		g.Weight.ScalInPlace(alpha)
	}

	if g.Gamma.N != 0 {
		g.Gamma.ScalInPlace(alpha)
	}

	if g.Bias.N != 0 {
		g.Bias.ScalInPlace(alpha)
	}
}

type GradBuffers []GradBuffer

func (gs GradBuffers) NewZerosLike() GradBuffers {
	zeros := make(GradBuffers, len(gs))
	for i, g := range gs {
		zeros[i] = g.NewZerosLike()
	}
	return zeros
}

func (gs GradBuffers) Clone() GradBuffers {
	clone := make(GradBuffers, len(gs))
	for i, g := range gs {
		clone[i] = g.Clone()
	}
	return clone
}

func (gs GradBuffers) AxpyInPlace(alpha float32, xs GradBuffers) {
	for i := range gs {
		gs[i].AxpyInPlace(alpha, &xs[i])
	}
}

func (gs GradBuffers) ScalInPlace(alpha float32) {
	for i := range gs {
		gs[i].ScalInPlace(alpha)
	}
}

type Parameter struct {
	Filter tensor.D4
	Weight tensor.D2
	Gamma  tensor.D1
	Bias   tensor.D1
}

func (p Parameter) NewGradRademacherLike(rng *rand.Rand) GradBuffer {
	return GradBuffer{
		Filter: p.Filter.NewRademacherLike(rng),
		Weight: p.Weight.NewRademacherLike(rng),
		Gamma : p.Gamma.NewRademacherLike(rng),
		Bias:   p.Bias.NewRademacherLike(rng),
	}
}

func (p Parameter) NewGradZerosLike() GradBuffer {
	return GradBuffer{
		Filter: p.Filter.NewZerosLike(),
		Weight: p.Weight.NewZerosLike(),
		Gamma : p.Gamma.NewZerosLike(),
		Bias:   p.Bias.NewZerosLike(),
	}
}

func (p Parameter) Clone() Parameter {
	return Parameter{
		Filter: p.Filter.Clone(),
		Weight: p.Weight.Clone(),
		Gamma : p.Gamma.Clone(),
		Bias:   p.Bias.Clone(),
	}
}

func (p *Parameter) AxpyInPlaceGrad(alpha float32, grad GradBuffer) {
	if p.Filter.Batches != 0 {
		p.Filter.AxpyInPlace(alpha, grad.Filter)
	}

	if p.Weight.Rows != 0 {
		p.Weight.AxpyInPlace(alpha, grad.Weight)
	}

	if p.Gamma.N != 0 {
		p.Gamma.AxpyInPlace(alpha, grad.Gamma)
	}

	if p.Bias.N != 0 {
		p.Bias.AxpyInPlace(alpha, grad.Bias)
	}
}

type Parameters []Parameter

func (ps Parameters) NewGradsRademacherLike(rng *rand.Rand) GradBuffers {
	grads := make(GradBuffers, len(ps))
	for i, p := range ps {
		grads[i] = p.NewGradRademacherLike(rng)
	}
	return grads
}

func (ps Parameters) NewGradsZerosLike() GradBuffers {
	grads := make(GradBuffers, len(ps))
	for i, p := range ps {
		grads[i] = p.NewGradZerosLike()
	}
	return grads
}

func (ps Parameters) Clone() Parameters {
	clone := make(Parameters, len(ps))
	for i, p := range ps {
		clone[i] = p.Clone()
	}
	return clone
}

func (ps Parameters) AxpyInPlaceGrads(alpha float32, grads GradBuffers) {
	for i := range ps {
		ps[i].AxpyInPlaceGrad(alpha, grads[i])
	}
}

type Forward func(tensor.D1, Parameter) (tensor.D1, error)
type Forwards []Forward

func (fs Forwards) Propagate(x tensor.D1, params Parameters) (tensor.D1, error) {
	if len(fs) != len(params) {
		return tensor.D1{}, fmt.Errorf("フォワードとパラメーターの長さが一致しない")
	}

	var err error
	for i, f := range fs {
		x, err = f(x, params[i])
		if err != nil {
			return tensor.D1{}, err
		}
	}
	return x, nil
}

type LossFunc func(Model, int) (float32, error)

type Model struct {
	Parameters  Parameters
	Forwards    Forwards
	LayerTypes  []consts.LayerType
}

func (m *Model) AppendDot(xn, yn int, rng *rand.Rand) {
	param := Parameter{
		Weight: tensor.NewD2He(xn, yn, rng),
	}
	m.Parameters = append(m.Parameters, param)

	forward := func(x tensor.D1, p Parameter) (tensor.D1, error) {
		return x.DotNoTrans2D(p.Weight), nil
	}
	m.Forwards = append(m.Forwards, forward)
	m.LayerTypes = append(m.LayerTypes, consts.DotLayer)
}

func (m *Model) AppendLeakyReLU(alpha float32) {
	param := Parameter{}
	m.Parameters = append(m.Parameters, param)

	forward := func(x tensor.D1, _ Parameter) (tensor.D1, error) {
		return nn.ReLUD1WithAlpha(x, alpha), nil
	}
	m.Forwards = append(m.Forwards, forward)
	m.LayerTypes = append(m.LayerTypes, consts.LeakyReLULayer)
}

func (m *Model) AppendInstanceNormalization(n int) {
	gamma := tensor.NewD1Ones(n)
	beta := tensor.NewD1Zeros(n)
	param := Parameter{
		Gamma:  gamma,
		Bias:   beta,
	}

	m.Parameters = append(m.Parameters, param)
	forward := func(x tensor.D1, p Parameter) (tensor.D1, error) {
		z := tmath.Standardize(x)
		y := z.Hadamard(p.Gamma).Axpy(1.0, p.Bias)
		return y, nil
	}
	m.Forwards = append(m.Forwards, forward)
	m.LayerTypes = append(m.LayerTypes, consts.InstanceNormalizationLayer)
}

func (m *Model) AppendSoftmax() {
	param := Parameter{}
	m.Parameters = append(m.Parameters, param)

	forward := func(x tensor.D1, _ Parameter) (tensor.D1, error) {
		return tmath.Softmax(x), nil
	}
	m.Forwards = append(m.Forwards, forward)
	m.LayerTypes = append(m.LayerTypes, consts.SoftmaxLayer)
}

func (m Model) Clone() Model {
	return Model{
		Parameters: m.Parameters.Clone(),
		Forwards:   m.Forwards,
		LayerTypes: slices.Clone(m.LayerTypes),
	}
}

func (m Model) Predict(x tensor.D1) (tensor.D1, error) {
	y, err := m.Forwards.Propagate(x, m.Parameters)
	return y, err
}

func (m Model) Accuracy(xs, ts []tensor.D1) (float32, error) {
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
		if oslices.MaxIndices(y.Data)[0] == oslices.MaxIndices(ts[i].Data)[0] {
			correct += 1
		}
	}
	return float32(correct) / float32(n), nil
}

func (m Model) EstimateGrads(lossFunc LossFunc, c float32, rngs []*rand.Rand) (GradBuffers, error) {
	p := len(rngs)
	gradsByParallel := make([]GradBuffers, p)
	errCh := make(chan error, p)

	worker := func(workerIdx int) {
		rng := rngs[workerIdx]
		deltas := m.Parameters.NewGradsRademacherLike(rng)

		plusModel := m.Clone()
		plusModel.Parameters.AxpyInPlaceGrads(c, deltas)

		minusModel := m.Clone()
		minusModel.Parameters.AxpyInPlaceGrads(-c, deltas)

		plusLoss, err := lossFunc(plusModel, workerIdx)
		if err != nil {
			errCh <- err
			return
		}

		minusLoss, err := lossFunc(minusModel, workerIdx)
		if err != nil {
			errCh <- err
			return
		}

		grads := m.Parameters.NewGradsZerosLike()
		for i, delta := range deltas {
			for j, d := range delta.Filter.Data {
				grads[i].Filter.Data[j] = cmath.CentralDifference(plusLoss, minusLoss, c*d)
			}

			for j, d := range delta.Weight.Data {
				grads[i].Weight.Data[j] = cmath.CentralDifference(plusLoss, minusLoss, c*d)
			}

			for j, d := range delta.Gamma.Data {
				grads[i].Gamma.Data[j] = cmath.CentralDifference(plusLoss, minusLoss, c*d)
			}

			for j, d := range delta.Bias.Data {
				grads[i].Bias.Data[j] = cmath.CentralDifference(plusLoss, minusLoss, c*d)
			}
		}

		gradsByParallel[workerIdx] = grads
		errCh <- err
	}

	for i := 0; i < p; i++ {
		go worker(i)
	}

	for i := 0; i < p; i++ {
		if err := <-errCh; err != nil {
			return nil, err
		}
	}

	firstGrads := gradsByParallel[0]
	firstGrads.ScalInPlace(1.0 / float32(p))
	for _, grads := range gradsByParallel[1:] {
		firstGrads.AxpyInPlace(1.0/float32(p), grads)
	}
	return firstGrads, nil
}

func (m Model) NumericalGrads(lossFunc LossFunc) (GradBuffers, error) {
	const h float32 = 1e-4
	grads := m.Parameters.NewGradsZerosLike()

	for i := range m.Parameters {
		p := &m.Parameters[i]

		for j := range p.Filter.Data {
			tmp := p.Filter.Data[j]

			//プラス方向への微小変化
			p.Filter.Data[j] = tmp + h
			plusLoss, err := lossFunc(m, 0)
			if err != nil {
				return nil, err
			}

			//マイナス方向への微小変化
			p.Filter.Data[j] = tmp - h
			minusLoss, err := lossFunc(m, 0)
			if err != nil {
				return nil, err
			}

			grads[i].Filter.Data[j] = cmath.CentralDifference(plusLoss, minusLoss, h)
			//微小変化前に戻す
			p.Filter.Data[j] = tmp
		}

		for j := range p.Weight.Data {
			tmp := p.Weight.Data[j]

			//プラス方向への微小変化
			p.Weight.Data[j] = tmp + h
			plusLoss, err := lossFunc(m, 0)
			if err != nil {
				return nil, err
			}

			//マイナス方向への微小変化
			p.Weight.Data[j] = tmp - h
			minusLoss, err := lossFunc(m, 0)
			if err != nil {
				return nil, err
			}

			grads[i].Weight.Data[j] = cmath.CentralDifference(plusLoss, minusLoss, h)
			//微小変化前に戻す
			p.Weight.Data[j] = tmp
		}

		for j := range p.Gamma.Data {
			tmp := p.Gamma.Data[j]

			//プラス方向への微小変化
			p.Gamma.Data[j] = tmp + h
			plusLoss, err := lossFunc(m, 0)
			if err != nil {
				return nil, err
			}

			//マイナス方向への微小変化
			p.Gamma.Data[j] = tmp - h
			minusLoss, err := lossFunc(m, 0)
			if err != nil {
				return nil, err
			}

			grads[i].Gamma.Data[j] = cmath.CentralDifference(plusLoss, minusLoss, h)
			//微小変化前に戻す
			p.Gamma.Data[j] = tmp
		}

		for j := range p.Bias.Data {
			tmp := p.Bias.Data[j]

			//プラス方向への微小変化
			p.Bias.Data[j] = tmp + h
			plusLoss, err := lossFunc(m, 0)
			if err != nil {
				return nil, err
			}

			//マイナス方向への微小変化
			p.Bias.Data[j] = tmp - h
			minusLoss, err := lossFunc(m, 0)
			if err != nil {
				return nil, err
			}

			grads[i].Bias.Data[j] = cmath.CentralDifference(plusLoss, minusLoss, h)
			//微小変化前に戻す
			p.Bias.Data[j] = tmp
		}
	}

	return grads, nil
}