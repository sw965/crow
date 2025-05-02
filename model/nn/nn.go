package nn

// import (
// 	"fmt"
// 	"github.com/sw965/crow/layer/1d"
// 	"github.com/sw965/crow/tensor"
// 	"github.com/sw965/crow/ml/1d"
// 	omwjson "github.com/sw965/omw/json"
// 	omwslices "github.com/sw965/omw/slices"
// 	omwrand "github.com/sw965/omw/math/rand"
// 	"math/rand"
// 	"github.com/sw965/omw/parallel"
// )

// type Parameter struct {
// 	Weights tensor.D3
// 	Biases tensor.D2
// }

// func LoadParameterJSON(path string) (Parameter, error) {
// 	param, err := omwjson.Load[Parameter](path)
// 	return param, err
// }

// func (p *Parameter) WriteJSON(path string) error {
// 	err := omwjson.Write[Parameter](p, path)
// 	return err
// }

// type FullyConnected struct {
// 	Parameter Parameter
// 	Forwards layer1d.Forwards

// 	YLossCalculator     func(tensor.D1, tensor.D1) (float32, error)
// 	YLossDifferentiator func(tensor.D1, tensor.D1) (tensor.D1, error)
// }

// func (nn *FullyConnected) SetParameter(param *Parameter) {
// 	nn.Parameter.Weights.Copy(param.Weights)
// 	nn.Parameter.Biases.Copy(param.Biases)
// }

// func (nn *FullyConnected) SetSumSquaredError() {
// 	nn.YLossCalculator = ml1d.SumSquaredError
// 	nn.YLossDifferentiator = ml1d.SumSquaredErrorDerivative
// }

// func (nn *FullyConnected) SetCrossEntropyError() {
// 	nn.YLossCalculator = ml1d.CrossEntropyError
// 	nn.YLossDifferentiator = ml1d.CrossEntropyErrorDerivative
// }

// func (nn *FullyConnected) AppendFullyConnectedLayer(r, c int, rn *rand.Rand) {
// 	he := tensor.NewD2He(r, c, rn)
// 	b := tensor.NewD1Zeros(c)
// 	nn.Parameter.Weights = append(nn.Parameter.Weights, he)
// 	nn.Parameter.Biases = append(nn.Parameter.Biases, b)
// 	nn.Forwards = append(nn.Forwards, layer1d.NewFullyConnectedForward(he, b))
// }

// func (nn *FullyConnected) AppendReLULayer() {
// 	nn.Forwards = append(nn.Forwards, layer1d.ReLUForward)
// }

// func (nn *FullyConnected) AppendLeakyReLULayer(alpha float32) {
// 	nn.Forwards = append(nn.Forwards, layer1d.NewLeakyReLUForward(alpha))
// }

// func (nn *FullyConnected) AppendSigmoidLayer() {
// 	nn.Forwards = append(nn.Forwards, layer1d.SigmoidForward)
// }

// func (nn *FullyConnected) AppendSoftmaxForCrossEntropyLayer() {
// 	nn.Forwards = append(nn.Forwards, layer1d.SoftmaxForwardForCrossEntropy)
// }

// func (nn *FullyConnected) Predict(x tensor.D1) (tensor.D1, error) {
// 	y, _, err := nn.Forwards.Propagate(x)
// 	return y, err
// }

// func (nn *FullyConnected) MeanLoss(xs, ts tensor.D2) (float32, error) {
// 	n := len(xs)
// 	if n != len(ts) {
// 		return 0.0, fmt.Errorf("バッチサイズが一致しません。")
// 	}

// 	var sum float32 = 0.0
// 	for i := range xs {
// 		y, err := nn.Predict(xs[i])
// 		if err != nil {
// 			return 0.0, err
// 		}
// 		yLoss, err := nn.YLossCalculator(y, ts[i])
// 		if err != nil {
// 			return 0.0, err
// 		}
// 		sum += yLoss
// 	}
// 	mean := sum / float32(n)
// 	return mean, nil
// }

// func (nn *FullyConnected) Accuracy(xs, ts tensor.D2) (float32, error) {
// 	n := len(xs)
// 	if n != len(ts) {
// 		return 0.0, fmt.Errorf("バッチサイズが一致しません。")
// 	}

// 	correct := 0
// 	for i := range xs {
// 		y, err := nn.Predict(xs[i])
// 		if err != nil {
// 			return 0.0, err
// 		}
// 		if omwslices.MaxIndex(y) == omwslices.MaxIndex(ts[i]) {
// 			correct += 1
// 		}
// 	}
// 	return float32(correct) / float32(n), nil
// }

// func (nn *FullyConnected) BackPropagate(x, t tensor.D1) (layer1d.GradBuffer, error) {
// 	y, backwards, err := nn.Forwards.Propagate(x)
// 	if err != nil {
// 		return layer1d.GradBuffer{}, err
// 	}

// 	dLdy, err := nn.YLossDifferentiator(y, t)
// 	if err != nil {
// 		return layer1d.GradBuffer{}, err
// 	}

// 	_, gradBuffer, err := backwards.Propagate(dLdy)
// 	return gradBuffer, err
// }

// func (nn *FullyConnected) ComputeGrad(xs, ts tensor.D2, p int) (layer1d.GradBuffer, error) {
// 	firstGradBuffer, err := nn.BackPropagate(xs[0], ts[0])
// 	if err != nil {
// 		return layer1d.GradBuffer{}, err
// 	}
	
// 	gradBuffers := make(layer1d.GradBuffers, p)
// 	for i := 0; i < p; i++ {
// 		gradBuffers[i] = firstGradBuffer.NewZerosLike()
// 	}
	
// 	n := len(xs)
// 	errCh := make(chan error, p)
// 	defer close(errCh)
	
// 	write := func(idxs []int, gorutineI int) {
// 		for _, idx := range idxs {
// 			//firstGradBufferで、0番目のデータの勾配は計算済みなので0にアクセスしないように、+1とする。
// 			x := xs[idx+1]
// 			t := ts[idx+1]
// 			gradBuffer, err := nn.BackPropagate(x, t)
// 			if err != nil {
// 				errCh <- err
// 				return
// 			}
// 			gradBuffers[gorutineI].Add(&gradBuffer)
// 		}
// 		errCh <- nil
// 	}
	
// 	for gorutineI, idxs := range parallel.DistributeIndicesEvenly(n-1, p) {
// 		go write(idxs, gorutineI)
// 	}
	
// 	for i := 0; i < p; i++ {
// 		err := <- errCh
// 		if err != nil {
// 			return layer1d.GradBuffer{}, err
// 		}
// 	}
	
// 	total := gradBuffers.Total()
// 	total.Add(&firstGradBuffer)
	
// 	nf := float32(n)
// 	total.Biases.DivScalar(nf)
// 	total.Weights.DivScalar(nf)
// 	return total, nil
// }

// func (nn *FullyConnected) Train(xs, ts tensor.D2, c *MiniBatchConfig, r *rand.Rand) error {
// 	lr := c.LearningRate
// 	batchSize := c.BatchSize
// 	p := c.Parallel

// 	n := len(xs)

// 	if n < batchSize {
// 		return fmt.Errorf("データ数 < バッチサイズである為、モデルの訓練を出来ません、")
// 	}

// 	if c.Epoch <= 0 {
// 		return fmt.Errorf("エポック数が0以下である為、モデルの訓練を開始出来ません。")
// 	}

// 	iter := n / batchSize * c.Epoch
// 	for i := 0; i < iter; i++ {
// 		idxs := omwrand.Ints(batchSize, 0, n, r)
// 		miniXs := omwslices.ElementsByIndices(xs, idxs...)
// 		miniTs := omwslices.ElementsByIndices(ts, idxs...)
// 		grad, err := nn.ComputeGrad(miniXs, miniTs, p)
// 		if err != nil {
// 			return err
// 		}

// 		grad.Biases.MulScalar(lr)
// 		grad.Weights.MulScalar(lr)
		
// 		nn.Parameter.Biases.Sub(grad.Biases)
// 		nn.Parameter.Weights.Sub(grad.Weights)
// 	}
// 	return nil
// }

// type MiniBatchConfig struct {
// 	LearningRate float32
// 	BatchSize int
// 	Epoch int
// 	Parallel int
// }