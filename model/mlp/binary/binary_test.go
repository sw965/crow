package binary_test

import (
	"fmt"
	"github.com/sw965/crow/dataset"
	"github.com/sw965/crow/model/mlp/binary"
	"github.com/sw965/omw/mathx/randx"
	"testing"
	"time"
)

func Test(t *testing.T) {
	// err := dataset.Clean()
	// if err != nil {
	// 	panic(err)
	// }

	// return
	mnist, err := dataset.LoadFashionMNIST()
	if err != nil {
		panic(err)
	}

	p := 4
	numClasses := 10
	outputSize := 1024
	rng := randx.NewPCG()

	model := binary.Model{XRows: 1, XCols: 784}
	err = model.AppendDenseLayer(512, rng)
	if err != nil {
		panic(err)
	}

	err = model.AppendDenseLayer(outputSize, rng)
	if err != nil {
		panic(err)
	}

	err = model.SetClassPrototypes(numClasses, rng)
	if err != nil {
		panic(err)
	}

	sharedContext := binary.SharedContext{
		GateDropThresholdScale: 1.0,
		NoiseStdScale:          0.5,
		GroupSize:              4,
	}
	model.Backbone.SetSharedContext(&sharedContext)

	trainer := binary.NewTrainer(model, p)
	trainer.Margin = 0.25
	// 確認用
	fmt.Printf("Prototype Checksum: %d\n", model.Prototypes[0].Data[0])

	acc1, _ := model.Accuracy(mnist.TestImages, mnist.TestLabels, p)
	t.Logf("Initial Accuracy (Loaded): %.4f", acc1)

	epochs := 25
	for i := range epochs {
		err := trainer.Train(mnist.TrainImages, mnist.TrainLabels)
		if err != nil {
			panic(err)
		}

		acc, err := model.Accuracy(mnist.TestImages, mnist.TestLabels, p)
		if err != nil {
			panic(err)
		}
		t.Logf("[%s] Epoch %d: Acc %.4f", time.Now().Format("15:04:05"), i, acc)
	}

	err = model.Save("test.gob")
	if err != nil {
		panic(err)
	}
}
