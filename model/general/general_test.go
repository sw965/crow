package general_test

import (
	"fmt"
	//cmath "github.com/sw965/crow/math"
	"github.com/sw965/crow/model/general"
	"testing"
	//oslices "github.com/sw965/omw/slices"
	"github.com/sw965/crow/dataset/mnist"
	orand "github.com/sw965/omw/math/rand"
	"gonum.org/v1/gonum/blas/blas32"
	//"math/rand"
	"runtime"
	"runtime/debug"
)

func TestModel(t *testing.T) {
	rng := orand.NewMt19937()
	model := general.Model{}
	model.AppendAffine(784, 784, rng)
	model.AppendLeakyReLU(0.1)

	model.AppendAffine(784, 32, rng)
	model.AppendLeakyReLU(0.1)

	model.AppendAffine(32, 10, rng)
	model.AppendLeakyReLU(0.1)

	model.AppendOutputSoftmaxAndSetCrossEntropyLoss()

	trainXs, err := mnist.LoadTrainFlatImages()
	if err != nil {
		panic(err)
	}

	trainYs, err := mnist.LoadTrainLabels()
	if err != nil {
		panic(err)
	}

	testXs, err := mnist.LoadTestFlatImages()
	if err != nil {
		panic(err)
	}

	testYs, err := mnist.LoadTestLabels()
	if err != nil {
		panic(err)
	}

	p := 4
	miniBatchSize := 16*p

	// rngs := make([]*rand.Rand, 4)
	// for i := range rngs {
	// 	rngs[i] = orand.NewMt19937()
	// }

	// model.LossFunc = func(model *mlp.Model, workerIdx int) (float32, error) {
	// 	rng := rngs[workerIdx]
	// 	sum := float32(0.0)
	// 	for i := 0; i < miniBatchSize; i++ {
	// 		idx := rng.Intn(60000)
	// 		x := trainXs[idx]
	// 		y, err := model.Predict(x)
	// 		if err != nil {
	// 			return 0.0, err
	// 		}
	// 		t := trainYs[idx]
	// 		loss, err := cmath.CrossEntropyError(y, t)
	// 		if err != nil {
	// 			return 0.0, err
	// 		}
	// 		sum += loss
	// 	}
	// 	ceeLoss := sum / float32(miniBatchSize)
	// 	return ceeLoss, nil
	// }

	trainNum := 128000
	//opt := mlp.NewAdam(model.Parameters)
	lr := float32(0.01)
	//c := float32(0.2)

	for i := 0; i < trainNum; i++ {
		// grads, err := model.EstimateGradsBySPSA(c, rngs)
		// if err != nil {
		// 	panic(err)
		// }

		//model.Parameters.AxpyGrads(-lr, grads)

		miniXs := make([]blas32.Vector, miniBatchSize)
		miniTs := make([]blas32.Vector, miniBatchSize)
		for j := 0; j < miniBatchSize; j++ {
			idx := rng.Intn(60000)
			miniXs[j] = trainXs[idx]
			miniTs[j] = trainYs[idx]
		}

		grad, err := model.ComputeGrad(miniXs, miniTs, p)
		if err != nil {
			panic(err)
		}
		model.Parameter.AxpyGrad(-lr, &grad)
		//opt.Optimizer(&model, grads)

		if i%512 == 0 {
			acc, err := model.Accuracy(testXs[:1000], testYs[:1000])
			if err != nil {
				panic(err)
			}
			fmt.Println(i, acc)
			runtime.GC()
			debug.FreeOSMemory()
		}
	}
}