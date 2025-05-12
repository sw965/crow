package spsa

import (
	"fmt"
	"github.com/sw965/crow/model/consts"
	"github.com/sw965/crow/blas32/tensor/2d"
	"github.com/sw965/crow/blas32/vector"
	cmath "github.com/sw965/crow/math"
	oslices "github.com/sw965/omw/slices"
	"gonum.org/v1/gonum/blas/blas32"
	"math/rand"
)

type GradBuffer struct {
	Weight blas32.General
	Gamma  blas32.Vector
	Bias   blas32.Vector
}

func (g *GradBuffer) NewZerosLike() GradBuffer {
	return GradBuffer{
		Weight:tensor2d.NewZerosLike(g.Weight),
		Gamma:vector.NewZerosLike(g.Gamma),
		Bias:vector.NewZerosLike(g.Bias),
	}
}

func (g GradBuffer) Clone() GradBuffer {
	return GradBuffer{
		Weight:tensor2d.Clone(g.Weight),
		Gamma:vector.Clone(g.Gamma),
		Bias:vector.Clone(g.Bias),
	}
}

func (g *GradBuffer) Axpy(alpha float32, x *GradBuffer) {
	if x.Weight.Rows != 0 {
		tensor2d.Axpy(alpha, x.Weight, g.Weight)
	}

	if x.Gamma.N != 0 {
		blas32.Axpy(alpha, x.Gamma, g.Gamma)
	}

	if x.Bias.N != 0 {
		blas32.Axpy(alpha, x.Bias, g.Bias)
	}
}

func (g *GradBuffer) Scal(alpha float32) {
	if g.Weight.Rows != 0 {
		tensor2d.Scal(alpha, g.Weight)
	}

	if g.Gamma.N != 0 {
		blas32.Scal(alpha, g.Gamma)
	}

	if g.Bias.N != 0 {
		blas32.Scal(alpha, g.Bias)
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

func (gs GradBuffers) Axpy(alpha float32, xs GradBuffers) {
	for i := range gs {
		gs[i].Axpy(alpha, &xs[i])
	}
}

func (gs GradBuffers) Scal(alpha float32) {
	for i := range gs {
		gs[i].Scal(alpha)
	}
}

type Parameter struct {
	Weight blas32.General
	Gamma  blas32.Vector
	Bias   blas32.Vector
}

func (p *Parameter) NewGradRademacherLike(rng *rand.Rand) GradBuffer {
	return GradBuffer{
		Weight: tensor2d.NewRademacherLike(p.Weight, rng),
		Gamma : vector.NewRademacherLike(p.Gamma, rng),
		Bias:   vector.NewRademacherLike(p.Bias, rng),
	}
}

func (p *Parameter) NewGradZerosLike() GradBuffer {
	return GradBuffer{
		Weight: tensor2d.NewZerosLike(p.Weight),
		Gamma : vector.NewZerosLike(p.Gamma),
		Bias:   vector.NewZerosLike(p.Bias),
	}
}

func (p *Parameter) Clone() Parameter {
	return Parameter{
		Weight: tensor2d.Clone(p.Weight),
		Gamma :vector.Clone(p.Gamma),
		Bias:   vector.Clone(p.Bias),
	}
}

func (p *Parameter) AxpyGrad(alpha float32, grad *GradBuffer) {
	if p.Weight.Rows != 0 {
		tensor2d.Axpy(alpha, grad.Weight, p.Weight)
	}

	if p.Gamma.N != 0 {
		blas32.Axpy(alpha, grad.Gamma, p.Gamma)
	}

	if p.Bias.N != 0 {
		blas32.Axpy(alpha, grad.Bias, p.Bias)
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

func (ps Parameters) AxpyGrads(alpha float32, grads GradBuffers) {
	for i := range ps {
		ps[i].AxpyGrad(alpha, &grads[i])
	}
}

type Forward func(blas32.Vector, *Parameter) (blas32.Vector, error)
type Forwards []Forward

func (fs Forwards) Propagate(x blas32.Vector, params Parameters) (blas32.Vector, error) {
	if len(fs) != len(params) {
		return blas32.Vector{}, fmt.Errorf("フォワードとパラメーターの長さが一致しない")
	}

	var err error
	for i, f := range fs {
		x, err = f(x, &params[i])
		if err != nil {
			return blas32.Vector{}, err
		}
	}
	return x, nil
}

type LossFunc func(*Model, int) (float32, error)

type Model struct {
	Parameters  Parameters
	Forwards    Forwards
	LayerTypes  []consts.LayerType
}

func (m *Model) AppendDot(xn, yn int, rng *rand.Rand) {
	param := Parameter{
		Weight: tensor2d.NewHe(xn, yn, rng),
	}
	m.Parameters = append(m.Parameters, param)

	forward := func(x blas32.Vector, param *Parameter) (blas32.Vector, error) {
		return vector.DotNoTrans2D(x, param.Weight), nil
	}
	m.Forwards = append(m.Forwards, forward)
	m.LayerTypes = append(m.LayerTypes, consts.DotLayer)
}

func (m *Model) AppendLeakyReLU(alpha float32) {
	param := Parameter{}
	m.Parameters = append(m.Parameters, param)

	forward := func(x blas32.Vector, _ *Parameter) (blas32.Vector, error) {
		return vector.LeakyReLU(x, alpha), nil
	}
	m.Forwards = append(m.Forwards, forward)
	m.LayerTypes = append(m.LayerTypes, consts.LeakyReLULayer)
}

func (m *Model) AppendInstanceNormalization(n int) {
	gamma := vector.NewOnes(n)
	beta := vector.NewZeros(n)
	param := Parameter{
		Gamma:  gamma,
		Bias:   beta,
	}

	m.Parameters = append(m.Parameters, param)
	forward := func(x blas32.Vector, param *Parameter) (blas32.Vector, error) {
		u, err := vector.Standardize(x)
		if err != nil {
			return blas32.Vector{}, err
		}

		y, err := vector.Hadamard(u, param.Gamma)
		if err != nil {
			return blas32.Vector{}, err
		}
		blas32.Axpy(1.0, param.Bias, y)
		return y, nil
	}
	m.Forwards = append(m.Forwards, forward)
	m.LayerTypes = append(m.LayerTypes, consts.InstanceNormalizationLayer)
}

func (m *Model) AppendSoftmax() {
	param := Parameter{}
	m.Parameters = append(m.Parameters, param)

	forward := func(x blas32.Vector, _ *Parameter) (blas32.Vector, error) {
		return vector.Softmax(x), nil
	}
	m.Forwards = append(m.Forwards, forward)
	m.LayerTypes = append(m.LayerTypes, consts.SoftmaxLayer)
}

func (m Model) Clone() Model {
	return Model{
		Parameters: m.Parameters.Clone(),
		Forwards:   m.Forwards,
	}
}

func (m *Model) Predict(x blas32.Vector) (blas32.Vector, error) {
	y, err := m.Forwards.Propagate(x, m.Parameters)
	return y, err
}

func (m *Model) Accuracy(xs, ts []blas32.Vector) (float32, error) {
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

func (m *Model) EstimateGrads(lossFunc LossFunc, c float32, rngs []*rand.Rand) (GradBuffers, error) {
	p := len(rngs)
	gradsByParallel := make([]GradBuffers, p)
	errCh := make(chan error, p)

	worker := func(workerIdx int) {
		rng := rngs[workerIdx]
		deltas := m.Parameters.NewGradsRademacherLike(rng)

		plusModel := m.Clone()
		plusModel.Parameters.AxpyGrads(c, deltas)

		minusModel := m.Clone()
		minusModel.Parameters.AxpyGrads(-c, deltas)

		plusLoss, err := lossFunc(&plusModel, workerIdx)
		if err != nil {
			errCh <- err
			return
		}

		minusLoss, err := lossFunc(&minusModel, workerIdx)
		if err != nil {
			errCh <- err
			return
		}

		grads := m.Parameters.NewGradsZerosLike()
		for i, delta := range deltas {
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
	firstGrads.Scal(1.0 / float32(p))
	for _, grads := range gradsByParallel[1:] {
		firstGrads.Axpy(1.0/float32(p), grads)
	}
	return firstGrads, nil
}

func (m *Model) NumericalGrads(lossFunc LossFunc) (GradBuffers, error) {
	const h float32 = 1e-4
	grads := m.Parameters.NewGradsZerosLike()

	for i := range m.Parameters {
		p := &m.Parameters[i]

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