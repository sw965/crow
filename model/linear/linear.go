package linear

// import (
// 	"fmt"
// 	"math"
// 	"math/rand"
// 	"runtime"
// 	crowmath "github.com/sw965/crow/math"
// 	"github.com/sw965/crow/tensor"
// 	"github.com/sw965/crow/ml/1d"
// 	"github.com/sw965/omw/fn"
// 	omwjson "github.com/sw965/omw/json"
// 	omwrand "github.com/sw965/omw/math/rand"
// 	omwslices "github.com/sw965/omw/slices"
// 	"github.com/sw965/omw/parallel"
// )

// type GradBuffer struct {
// 	Weight tensor.D2
// 	Bias   tensor.D1
// }

// func (g *GradBuffer) NewZerosLike() GradBuffer {
// 	return GradBuffer{
// 		Weight:tensor.NewD2ZerosLike(g.Weight),
// 		Bias:tensor.NewD1ZerosLike(g.Bias),
// 	}
// }

// func (g *GradBuffer) Add(other *GradBuffer) error {
// 	err := g.Weight.Add(other.Weight)
// 	if err != nil {
// 		return err
// 	}
// 	err = g.Bias.Add(other.Bias)
// 	return err
// }

// type GradBuffers []GradBuffer

// func (gs GradBuffers) Total() GradBuffer {
// 	total := gs[0].NewZerosLike()
// 	for _, g := range gs {
// 		total.Add(&g)
// 	}
// 	return total
// }

// type Parameter struct {
// 	Weight [][]*float32
// 	Bias   []*float32
// }

// func LoadParameterJSON(path string, param *Parameter) error {
// 	loadedParam, err := omwjson.Load[Parameter](path)
// 	if err != nil {
// 		return err
// 	}

// 	/*
// 		読み込んだパラメーターは、ポインター(共有関係)が崩れている為、
// 		引数に正しいポインターを保持したパラメーターを渡し、そのパラメーターに値を書き込む形にする事で、
// 		正しい共有関係を保つようにする。
// 	*/
// 	w := loadedParam.Weight
// 	for i, wi := range w {
// 		for j, wij := range wi {
// 			*param.Weight[i][j] = *wij
// 		}
// 	}

// 	b := loadedParam.Bias
// 	for i, bi := range b {
// 		*param.Bias[i] = *bi
// 	}
// 	return nil
// }

// func (p *Parameter) WriteJSON(path string) error {
// 	err := omwjson.Write[Parameter](p, path)
// 	return err
// }

// func (p *Parameter) Clone() Parameter {
// 	seenW := make(map[*float32]*float32)
// 	newWeight := make([][]*float32, len(p.Weight))
// 	for i, wi := range p.Weight {
// 		newWi := make([]*float32, len(wi))
// 		for j, origPtr := range wi {
// 			//共有関係が崩れないように、origPtrに対応する新しいポインターを既に生成している場合、そのアドレスを割り当てる。
// 			if newPtr, ok := seenW[origPtr]; ok {
// 				newWi[j] = newPtr
// 			} else {
// 				newV := new(float32)
// 				*newV = *origPtr
// 				seenW[origPtr] = newV
// 				newWi[j] = newV
// 			}
// 		}
// 		newWeight[i] = newWi
// 	}

// 	seenB := make(map[*float32]*float32)
// 	newBias := make([]*float32, len(p.Bias))
// 	for i, origPtr := range p.Bias {
// 		//共有関係が崩れないように
// 		if newPtr, ok := seenB[origPtr]; ok {
// 			newBias[i] = newPtr
// 		} else {
// 			newV := new(float32)
// 			*newV = *origPtr
// 			seenB[origPtr] = newV
// 			newBias[i] = newV
// 		}
// 	}

// 	return Parameter{
// 		Weight: newWeight,
// 		Bias:   newBias,
// 	}
// }

// func (p *Parameter) AddGrad(grad *GradBuffer) error {
// 	w := p.Weight
// 	b := p.Bias
// 	gw := grad.Weight
// 	gb := grad.Bias

// 	/*
// 		パラメーターはポインターなので、加算した場合、指定したインデックス以外の
// 		パラメーターにも影響するが、1つの変数に対して、
// 		複数の微分得られた場合は、微分結果を合計すればいいので、整合性に問題はない。
// 	*/

// 	for i := range w {
// 		for j := range w[i] {
// 			*w[i][j] += gw[i][j]
// 		}
// 	}

// 	for i := range b {
// 		*b[i] += gb[i]
// 	}
// 	return nil
// }

// type Optimizer func(*Model, *GradBuffer) error

// type SGD struct {
// 	LearningRate float32
// }

// func (sgd *SGD) Optimizer(model *Model, grad *GradBuffer) error {
// 	lr := sgd.LearningRate
// 	grad.Weight.MulScalar(-lr)
// 	grad.Bias.MulScalar(-lr)
// 	err := model.Parameter.AddGrad(grad)
// 	return err
// }

// // https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/optimizer.py
// type Momentum struct {
// 	LearningRate float32
// 	MomentumRate float32
// 	velocity     GradBuffer
// }

// func NewMomentum(model *Model) Momentum {
// 	w := model.Parameter.Weight
// 	zeroW := make(tensor.D2, len(w))
// 	for i := range w {
// 		zeroW[i] = make(tensor.D1, len(w[i]))
// 		for j := range w[i] {
// 			zeroW[i][j] = 0.0
// 		} 
// 	}

// 	b := model.Parameter.Bias
// 	zeroB := make(tensor.D1, len(b))
// 	for i := range b {
// 		zeroB[i] = 0.0
// 	}

// 	return Momentum{
// 		LearningRate: 0.01,
// 		MomentumRate: 0.9,
// 		velocity:     GradBuffer{Weight:zeroW, Bias:zeroB},
// 	}
// }

// func (m *Momentum) Optimizer(model *Model, grad *GradBuffer) error {
// 	lr := m.LearningRate

// 	m.velocity.Weight.MulScalar(m.MomentumRate)
// 	m.velocity.Bias.MulScalar(m.MomentumRate)

// 	grad.Weight.MulScalar(-lr)
// 	grad.Bias.MulScalar(-lr)

// 	err := m.velocity.Add(grad)
// 	if err != nil {
// 		return err
// 	}

// 	err = model.Parameter.AddGrad(&m.velocity)
// 	return err
// }

// type Test struct {
// 	Row int
// 	Column int
// 	Value float32
// }

// type Input []Test
// type Inputs []Input

// type Model struct {
// 	Parameter Parameter
// 	OutputFunc       func(tensor.D1) tensor.D1
// 	// SPSAの教師なし学習の為の損失関数。
// 	LossFunc func(*Model) (float32, error)
// }

// func (m Model) Clone() Model {
// 	m.Parameter = m.Parameter.Clone()
// 	return m
// }

// func (m *Model) Predict(input Input) tensor.D1 {
// 	w := m.Parameter.Weight
// 	b := m.Parameter.Bias
// 	u := make(tensor.D1, len(w))
// 	for k, v := range input {
// 		r := k.Row
// 		u[r] += *w[r][k.Column] * v
// 	}
// 	for i, v := range b {
// 		u[i] += *v
// 	}
// 	return m.OutputFunc(u)
// }

// func (m *Model) Accuracy(inputs Inputs, ts tensor.D2) (float32, error) {
// 	n := len(inputs)
// 	if n != len(ts) {
// 		return 0.0, fmt.Errorf("バッチサイズが一致しません。")
// 	}

// 	correct := 0
// 	for i, input := range inputs {
// 		y := m.Predict(input)
// 		if omwslices.MaxIndex(y) == omwslices.MaxIndex(ts[i]) {
// 			correct += 1
// 		}
// 	}
// 	return float32(correct) / float32(n), nil
// }

// func (m *Model) ComputeGrad(inputs Inputs, ts tensor.D2, p int) (GradBuffer, error) {
// 	n := len(inputs)
// 	if n != len(ts) {
// 		return GradBuffer{}, fmt.Errorf("バッチサイズが一致しません。")
// 	}

// 	firstGrad, err := m.BackPropagate(inputs[0], ts[0])
// 	if err != nil {
// 		return GradBuffer{}, err
// 	}
// 	gradBuffers := make(GradBuffers, p)
// 	for i := 0; i < p; i++ {
// 		gradBuffers[i] = firstGrad.NewZerosLike()
// 	}
// 	errCh := make(chan error, p)

// 	worker := func(idxs []int, goroutineI int) {
// 		for _, idx := range idxs {
// 			input := inputs[idx+1]
// 			t := ts[idx+1]
// 			grad, err := m.BackPropagate(input, t)
// 			if err != nil {
// 				errCh <- err
// 				return
// 			}
// 			gradBuffers[goroutineI].Add(&grad)
// 		}
// 		errCh <- nil
// 	}

// 	for gorutineI, idxs := range parallel.DistributeIndicesEvenly(n-1, p) {
// 		go worker(idxs, gorutineI)
// 	}

// 	for i := 0; i < p; i++ {
// 		if err := <-errCh; err != nil {
// 			return GradBuffer{}, err
// 		}
// 	}

// 	total := gradBuffers.Total()
// 	total.Add(&firstGrad)
// 	nf := float32(n)
// 	total.Weight.DivScalar(nf)
// 	total.Bias.DivScalar(nf)
// 	return total, nil
// }

func (m *Model) EstimateGradBySPSA(c float32, rng *rand.Rand, p int) (GradBuffer, error) {
	grads := make(GradBuffers, p)
	for i := 0; i < p; i++ {
		grads[i] = GradBuffer{Weight:general.NewZeroLike(m.Weight), Bias:vector.NewZeroLike(m.Bias)}
	}
	errCh := make(chan error, p)

	worker := func(workerId int) {
		deltaW := general.NewRademacher(m.Parameter.Weight.Rows, m.Parameter.Weight.Cols)	
		perturbationW := general.Clone()
		general.MulScalar(perturbationW, c)

		deltaB := general.NewRademacher(m.Parameter.Bias.N)
		perturbationB := deltaB.Clone()
		vector.Scal(c, perturbationB)

		plusModel := m.Clone()
		general.Add(plusModel.Parameter.Weight, perturbationW)
		vector.Add(plusModel.Parameter.Bias, perturbationB)

		minusModel := m.Clone()
		general.Sub(plusModel.Parameter.Weight, perturbationW)
		vector.Sub(plusModel.Parameter.Bias, perturbationB)
	
		plusLoss, err := m.LossFunc(&plusModel)
		if err != nil {
			return GradBuffer{}, err
		}
	
		minusLoss, err := m.LossFunc(&minusModel)
		if err != nil {
			return GradBuffer{}, err
		}

		for i := range grad.Weight.Data {
			grads[routine].Weight.Data[i] += crowmath.CentralDifference(plusLoss, minusLoss, perturbationW[i])
		}
	
		for i := range grad.Bias.Data {
			grads[routine].Bias.Data[i] += crowmath.CentralDifference(plusLoss, minusLoss, perturbationB[i])
		}
		errCh <- nil
	}

	for workerI := 0; i < p; i++ {
		go worker(workerI)
	}

	for i := 0; i < p; i++ {
		if err := <-errCh; err != nil {
			return GradBuffer{}, err
		}
	}
}

// func (m *Model) SoftmaxActionSelection(input Input, temperature float32, exclude func(int) bool, r *rand.Rand, epsilon float32) int {
// 	y := m.Predict(input)
// 	for i := range y {
// 		if exclude(i) {
// 			y[i] = 0.0
// 		} else {
// 			y[i] += epsilon
// 		}
// 	}

// 	if temperature == 0.0 {
// 		idxs := omwslices.MaxIndices(y)
// 		return omwrand.Choice(idxs, r)
// 	}

// 	ws := make([]float32, len(y))
// 	for i, yi := range y {
// 		ws[i] = float32(math.Pow(float64(yi), float64(1.0/temperature)))
// 	}
// 	return omwrand.IntByWeight(ws, r)
// }