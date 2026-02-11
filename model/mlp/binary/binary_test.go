package binary_test

import (
	"github.com/sw965/crow/model/mlp/binary"
	"github.com/sw965/crow/dataset"
	"github.com/sw965/omw/mathx/bitsx"
	"testing"
	"github.com/sw965/omw/mathx/randx"
	// "fmt"
	"time"
)

func Test(t *testing.T) {
	rng := randx.NewPCGFromGlobalSeed()
	mnist, err := dataset.LoadFashionMNIST()
	if err != nil {
		panic(err)
	}

	p := 4
	classNum := 10
	outputSize := 1024
	prototypes, err := bitsx.NewBEFMatrices(classNum, 1, outputSize, 10000, rng)
	if err != nil {
		panic(err)
	}

	model, err := binary.NewDenseLayers([]int{784, 512, outputSize}, rng)
	if err != nil {
		panic(err)
	}

	sharedContext := binary.SharedContext{
		GateThresholdScale:1.0,
		NoiseStdScale:0.5,
		GroupSize:4,
	}
	model.SetSharedContext(&sharedContext)

	trainer := binary.NewTrainer(model, 0.25, p)
	trainer.Prototypes = prototypes
	trainer.LR = 0.1

	epochs := 100
	for i := range epochs {
		err := trainer.Fit(mnist.TrainImages, mnist.TrainLabels, 128)
		if err != nil {
			panic(err)
		}

		acc, err := model.Accuracy(mnist.TestImages, mnist.TestLabels, prototypes, p)
		if err != nil {
			panic(err)
		}
		t.Logf("[%s] Epoch %d: Acc %.4f", time.Now().Format("15:04:05"), i, acc)
	}
}