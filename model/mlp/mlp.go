package mlp

import (
	"fmt"
	"github.com/chewxy/math32"
	"github.com/sw965/crow/blas32/tensor/2d"
	"github.com/sw965/crow/blas32/vector"
	cmath "github.com/sw965/crow/math"
	omath "github.com/sw965/omw/math"
	oslices "github.com/sw965/omw/slices"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
	"math"
	"math/rand"
	"slices"
)

type GradBuffer struct {
	Weight blas32.General
	Bias   blas32.Vector
}

func (g *GradBuffer) Axpy(alpha float32, x *GradBuffer) {
	tensor2d.Axpy(alpha, x.Weight, g.Weight)
	blas32.Axpy(alpha, x.Bias, g.Bias)
}

func (g *GradBuffer) Scal(alpha float32) {
	tensor2d.Scal(alpha, g.Weight)
	blas32.Scal(alpha, g.Bias)
}

type GradBuffers []GradBuffer

func (gs GradBuffers) Axpy(alpha float32, xs GradBuffers) {
	for i, g := range gs {
		g.Axpy(alpha, &xs[i])
	}
}

func (gs GradBuffers) Scal(alpha float32) {
	for _, g := range gs {
		g.Scal(alpha)
	}
}

type Parameter struct {
	Weight blas32.General
	Bias   blas32.Vector
}

func (p *Parameter) NewGradRademacherLike(rng *rand.Rand) GradBuffer {
	return GradBuffer{
		Weight: tensor2d.NewRademacherLike(p.Weight, rng),
		Bias:   vector.NewRademacherLike(p.Bias, rng),
	}
}

func (p *Parameter) NewGradZerosLike() GradBuffer {
	return GradBuffer{
		Weight: tensor2d.NewZerosLike(p.Weight),
		Bias:   vector.NewZerosLike(p.Bias),
	}
}

func (p *Parameter) Clone() Parameter {
	return Parameter{
		Weight: tensor2d.Clone(p.Weight),
		Bias:   vector.Clone(p.Bias),
	}
}

func (p *Parameter) AxpyGrad(alpha float32, grad *GradBuffer) {
	if p.Weight.Rows != 0 {
		tensor2d.Axpy(alpha, grad.Weight, p.Weight)
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
	for i, p := range ps {
		p.AxpyGrad(alpha, &grads[i])
	}
}

type Forward func(blas32.Vector, *Parameter) (blas32.Vector, Backward, error)
type Forwards []Forward

func (fs Forwards) Propagate(x blas32.Vector, params Parameters) (blas32.Vector, Backwards, error) {
	var err error
	var backward Backward
	backwards := make(Backwards, len(fs))
	for i, f := range fs {
		x, backward, err = f(x, &params[i])
		if err != nil {
			return blas32.Vector{}, nil, err
		}
		backwards[i] = backward
	}
	y := x
	slices.Reverse(backwards)
	return y, backwards, nil
}

type Backward func(blas32.Vector) (blas32.Vector, GradBuffer, error)
type Backwards []Backward

func AffineForward(x blas32.Vector, param *Parameter) (blas32.Vector, Backward, error) {
	yn := param.Weight.Cols
	y := blas32.Vector{N: yn, Inc: 1, Data: make([]float32, yn)}
	blas32.Copy(param.Bias, y)
	blas32.Gemv(blas.Trans, 1.0, param.Weight, x, 1.0, y)

	var backward Backward
	backward = func(chain blas32.Vector) (blas32.Vector, GradBuffer, error) {
		wRows := param.Weight.Rows
		wCols := param.Weight.Cols

		dx := blas32.Vector{
			N:    wRows,
			Inc:  1,
			Data: make([]float32, wRows),
		}
		blas32.Gemv(blas.NoTrans, 1.0, param.Weight, chain, 1.0, dx)
		fmt.Println("dx =", dx)

		dw := blas32.General{
			Rows:   wRows,
			Cols:   wCols,
			Stride: wCols,
			Data:   make([]float32, wRows*wCols),
		}
		blas32.Ger(1.0, x, chain, dw)
		fmt.Println("dw =", dw)

		db := blas32.Vector{
			N:    chain.N,
			Inc:  1,
			Data: make([]float32, chain.N),
		}
		blas32.Copy(chain, db)
		fmt.Println("db =", db)

		grad := GradBuffer{
			Weight: dw,
			Bias:   db,
		}

		return dx, grad, nil
	}
	return y, backward, nil
}

func NewLeakyReLU1DForward(alpha float32) Forward {
	return func(x blas32.Vector, _ *Parameter) (blas32.Vector, Backward, error) {
		xData := x.Data
		yData := make([]float32, x.N)
		for i := range yData {
			e := xData[i]
			if e > 0 {
				yData[i] = e
			} else {
				yData[i] = alpha * e
			}
		}

		y := blas32.Vector{
			N:    x.N,
			Inc:  x.Inc,
			Data: yData,
		}

		var backward Backward
		backward = func(chain blas32.Vector) (blas32.Vector, GradBuffer, error) {
			chainData := chain.Data
			dxData := make([]float32, chain.N)
			for i, e := range xData {
				if e > 0 {
					dxData[i] = chainData[i]
				} else {
					dxData[i] = alpha * chainData[i]
				}
			}
			dx := blas32.Vector{
				N:    chain.N,
				Inc:  chain.Inc,
				Data: dxData,
			}
			return dx, GradBuffer{}, nil
		}

		return y, backward, nil
	}
}

func SoftmaxForOutputForward(x blas32.Vector, _ *Parameter) (blas32.Vector, Backward, error) {
	xData := x.Data
	maxX := omath.Max(xData...) // オーバーフロー対策
	expX := make([]float32, x.N)
	sumExpX := float32(0.0)
	for i, e := range xData {
		expX[i] = math32.Exp(e - maxX)
		sumExpX += expX[i]
	}

	yData := make([]float32, x.N)
	for i := range expX {
		yData[i] = expX[i] / sumExpX
	}

	y := blas32.Vector{
		N:    x.N,
		Inc:  x.Inc,
		Data: yData,
	}

	var backward Backward
	backward = func(chain blas32.Vector) (blas32.Vector, GradBuffer, error) {
		//クロスエントロピーが損失関数である事を前提
		dx := chain
		return dx, GradBuffer{}, nil
	}
	return y, backward, nil
}

type PredictLoss struct {
	Func       func(blas32.Vector, blas32.Vector) (float32, error)
	Derivative func(blas32.Vector, blas32.Vector) (blas32.Vector, error)
}

func NewCrossEntropyLossForSoftmax() PredictLoss {
	f := func(y, t blas32.Vector) (float32, error) {
		loss := float32(0.0)
		e := float32(0.0001)
		for i := range y.Data {
			ye := float64(omath.Max(y.Data[i], e))
			te := t.Data[i]
			loss += -te * float32(math.Log(ye))
		}
		return loss, nil
	}

	d := func(y, t blas32.Vector) (blas32.Vector, error) {
		dx := blas32.Vector{
			N:    y.N,
			Inc:  y.Inc,
			Data: make([]float32, y.N),
		}
		blas32.Copy(y, dx)
		blas32.Axpy(-1.0, t, dx)
		return dx, nil
	}

	return PredictLoss{
		Func:       f,
		Derivative: d,
	}
}

type Model struct {
	Parameters  Parameters
	Forwards    Forwards
	PredictLoss PredictLoss
	LossFunc    func(*Model, int) (float32, error)
}

func (m *Model) AppendAffine(xn, yn int, rng *rand.Rand) {
	param := Parameter{
		Weight: tensor2d.NewHe(xn, yn, rng),
		Bias:   vector.NewZeros(yn),
	}
	m.Parameters = append(m.Parameters, param)
	m.Forwards = append(m.Forwards, AffineForward)
}

func (m *Model) AppendLeakyReLU(alpha float32) {
	param := Parameter{
		Weight: blas32.General{Rows: 0, Cols: 0, Stride: 0, Data: []float32{}},
		Bias:   blas32.Vector{N: 0, Inc: 0, Data: []float32{}},
	}
	m.Parameters = append(m.Parameters, param)
	m.Forwards = append(m.Forwards, NewLeakyReLU1DForward(alpha))
}

func (m *Model) AppendOutputSoftmaxAndSetCrossEntropyLoss() {
	param := Parameter{
		Weight: blas32.General{Rows: 0, Cols: 0, Stride: 0, Data: []float32{}},
		Bias:   blas32.Vector{N: 0, Inc: 0, Data: []float32{}},
	}
	m.Parameters = append(m.Parameters, param)
	m.Forwards = append(m.Forwards, SoftmaxForOutputForward)
	m.PredictLoss = NewCrossEntropyLossForSoftmax()
}

func (m Model) Clone() Model {
	return Model{
		Parameters: m.Parameters.Clone(),
		Forwards:   m.Forwards,
		LossFunc:   m.LossFunc,
	}
}

func (m *Model) Predict(x blas32.Vector) (blas32.Vector, error) {
	y, _, err := m.Forwards.Propagate(x, m.Parameters)
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

func (m *Model) ComputeGradByTeacher(xs, ts blas32.Vector) (GradBuffers, error) {
	
}

func (m *Model) EstimateGradsBySPSA(c float32, rngs []*rand.Rand) (GradBuffers, error) {
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

		plusLoss, err := m.LossFunc(&plusModel, workerIdx)
		if err != nil {
			errCh <- err
			return
		}

		minusLoss, err := m.LossFunc(&minusModel, workerIdx)
		if err != nil {
			errCh <- err
			return
		}

		grads := m.Parameters.NewGradsZerosLike()
		for i, delta := range deltas {
			for j, d := range delta.Weight.Data {
				grads[i].Weight.Data[j] = cmath.CentralDifference(plusLoss, minusLoss, c*d)
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
