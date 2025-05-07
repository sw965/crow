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
	"github.com/sw965/omw/parallel"
)

type GradBuffer struct {
	Weight blas32.General
	Bias   blas32.Vector
}

func (g *GradBuffer) NewZerosLike() GradBuffer {
	return GradBuffer{
		Weight:tensor2d.NewZerosLike(g.Weight),
		Bias:vector.NewZerosLike(g.Bias),
	}
}

func (g GradBuffer) Clone() GradBuffer {
	return GradBuffer{
		Weight:tensor2d.Clone(g.Weight),
		Bias:vector.Clone(g.Bias),
	}
}

func (g *GradBuffer) Axpy(alpha float32, x *GradBuffer) {
	if x.Weight.Rows != 0 {
		tensor2d.Axpy(alpha, x.Weight, g.Weight)
	}

	if x.Bias.N != 0 {
		blas32.Axpy(alpha, x.Bias, g.Bias)
	}
}

func (g *GradBuffer) Scal(alpha float32) {
	if g.Weight.Rows != 0 {
		tensor2d.Scal(alpha, g.Weight)
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

type Forward3D func(tensor3d.General, ) (tensor3d.General, Backwards3D, error)


type Backward func(blas32.Vector) (blas32.Vector, GradBuffer, error)
type Backwards []Backward

func (bs Backwards) Propagate(chain blas32.Vector) (blas32.Vector, GradBuffers, error) {
	grads := make(GradBuffers, len(bs))
	var grad GradBuffer
	var err error
	for i, b := range bs {
		chain, grad, err = b(chain)
		if err != nil {
			return blas32.Vector{}, nil, err
		}
		grads[i] = grad
	}
	dx := chain
	slices.Reverse(grads)
	return dx, grads, nil
}

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

		dw := blas32.General{
			Rows:   wRows,
			Cols:   wCols,
			Stride: wCols,
			Data:   make([]float32, wRows*wCols),
		}
		blas32.Ger(1.0, x, chain, dw)

		db := blas32.Vector{
			N:    chain.N,
			Inc:  1,
			Data: make([]float32, chain.N),
		}
		blas32.Copy(chain, db)

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

func (m *Model) BackPropagateByTeacher(x, t blas32.Vector) (blas32.Vector, GradBuffers, error) {
	y, backwards, err := m.Forwards.Propagate(x, m.Parameters)
	if err != nil {
		return blas32.Vector{}, nil, err
	}
	firstChain, err := m.PredictLoss.Derivative(y, t)
	if err != nil {
		return blas32.Vector{}, nil, err
	}
	return backwards.Propagate(firstChain)
}

func (m *Model) ComputeGradByTeacher(xs, ts []blas32.Vector, rng *rand.Rand, p int) (GradBuffers, error) {	
	n := len(xs)
	if n != len(ts) {
		return nil, fmt.Errorf("バッチサイズが一致しません。")
	}

	_, firstGrads, err := m.BackPropagateByTeacher(xs[0], ts[0])
	if err != nil {
		return nil, err
	}

	gradBuffersByParallel := make([]GradBuffers, p)
	for i := range gradBuffersByParallel {
		gradBuffersByParallel[i] = firstGrads.NewZerosLike()
	}

	errCh := make(chan error, p)
	worker := func(workerIdx int, idxs []int) {
		for _, idx := range idxs {
			x := xs[idx+1]
			t := ts[idx+1]
			_, grads, err := m.BackPropagateByTeacher(x, t)
			if err != nil {
				errCh <- err
				return
			}
			gradBuffersByParallel[workerIdx].Axpy(1.0, grads)
		}
		errCh <- err
	}

	for workerIdx, idxs := range parallel.DistributeIndicesEvenly(n -1, p) {
		go worker(workerIdx, idxs)
	}

	for i := 0; i < p; i++ {
		if err := <-errCh; err != nil {
			return nil, err
		}
	}

	total := firstGrads.Clone()
	for _, g := range gradBuffersByParallel {
		total.Axpy(1.0, g)
	}
	total.Scal(1.0/float32(n))
	//メモリ解法
	firstGrads = make(GradBuffers, 0)
	gradBuffersByParallel = make([]GradBuffers, 0)
	return total, nil
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

type Adam struct {
	LearningRate float32
	Beta1        float32
	Beta2        float32
	Epsilon      float32

	iter int
	m    GradBuffers
	v    GradBuffers
}

// NewAdam creates a new Adam optimizer whose内部状態
// (1st/2nd momentバッファ)は引数の Parameters と同じ形状で 0 初期化されます。
func NewAdam(params Parameters) *Adam {
	return &Adam{
		LearningRate: 0.001,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-7,
		iter:         0,
		m:            params.NewGradsZerosLike(),
		v:            params.NewGradsZerosLike(),
	}
}

// Optimizer updates model.Parameters in‑place using Adam rule.
//  * model  – 更新対象モデル
//  * grads  – SPSA 等で推定した勾配（model と同じ長さ）
func (a *Adam) Optimizer(model *Model, grads GradBuffers) error {
	if len(model.Parameters) != len(grads) {
		return fmt.Errorf("Adam: parameters/grads size mismatch")
	}

	// 念のため lazy‑init（NewAdam を通らず生成された場合に備え）
	if a.m == nil || len(a.m) == 0 {
		a.m = model.Parameters.NewGradsZerosLike()
		a.v = model.Parameters.NewGradsZerosLike()
	}

	a.iter++
	beta1, beta2 := a.Beta1, a.Beta2
	lrt := a.LearningRate *
		float32(math32.Sqrt(1-math32.Pow(beta2, float32(a.iter)))) /
		(1 - math32.Pow(beta1, float32(a.iter)))

	for i := range grads {
		// --- Weight 部分 ---
		for j, g := range grads[i].Weight.Data {
			a.m[i].Weight.Data[j] += (1 - beta1) * (g - a.m[i].Weight.Data[j])
			a.v[i].Weight.Data[j] += (1 - beta2) * (g*g - a.v[i].Weight.Data[j])

			update := lrt * a.m[i].Weight.Data[j] /
				(float32(math32.Sqrt(a.v[i].Weight.Data[j])) + a.Epsilon)
			model.Parameters[i].Weight.Data[j] -= update
		}
		// --- Bias 部分 ---
		for j, g := range grads[i].Bias.Data {
			a.m[i].Bias.Data[j] += (1 - beta1) * (g - a.m[i].Bias.Data[j])
			a.v[i].Bias.Data[j] += (1 - beta2) * (g*g - a.v[i].Bias.Data[j])

			update := lrt * a.m[i].Bias.Data[j] /
				(float32(math32.Sqrt(a.v[i].Bias.Data[j])) + a.Epsilon)
			model.Parameters[i].Bias.Data[j] -= update
		}
	}
	return nil
}