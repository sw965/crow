// Fashion-MNISTをバイナリMLPで学習するデモプログラム。
// 元は model/mlp/binary のテストとして置かれていた実験コードを、実行可能な例として移設したもの。
//
// 実行方法:
//
//	go run github.com/sw965/crow/examples/mnist
package main

import (
	"fmt"
	"log"
	"runtime"
	"time"

	"github.com/sw965/crow/dataset"
	"github.com/sw965/crow/model/mlp/binary"
	"github.com/sw965/omw/mathx/randx"
)

func main() {
	logf := func(format string, a ...any) {
		fmt.Printf(format, a...)
	}
	mnist, err := dataset.LoadFashionMNIST(logf)
	if err != nil {
		log.Fatal(err)
	}

	p := runtime.NumCPU()
	fmt.Println("p =", p)
	numClasses := 10
	outputSize := 1024
	rng := randx.NewPCG()

	model := binary.Model{XRows: 1, XCols: 784}
	if err := model.AppendDenseLayer(512, rng); err != nil {
		log.Fatal(err)
	}

	if err := model.AppendDenseLayer(outputSize, rng); err != nil {
		log.Fatal(err)
	}

	if err := model.SetClassPrototypes(numClasses, rng); err != nil {
		log.Fatal(err)
	}

	sharedHyperparameters := binary.NewSharedHyperparameters()
	if err := model.Backbone.SetSharedHyperparameters(&sharedHyperparameters); err != nil {
		log.Fatal(err)
	}

	trainer := binary.NewTrainer(model, p)
	trainer.MiniBatchSize = 1024
	fmt.Println("ミニバッチサイズ", trainer.MiniBatchSize)
	// 論文スケール(r)。旧実装スケールの 0.25 と同じ実効マージン
	trainer.Margin = 0.5

	acc, err := model.Accuracy(mnist.TestImages, mnist.TestLabels, p)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Initial Accuracy: %.4f\n", acc)

	epochs := 50
	for i := range epochs {
		if err := trainer.Train(mnist.TrainImages, mnist.TrainLabels); err != nil {
			log.Fatal(err)
		}

		acc, err := model.Accuracy(mnist.TestImages, mnist.TestLabels, p)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("[%s] Epoch %d: Acc %.4f\n", time.Now().Format("15:04:05"), i, acc)
	}

	if err := model.Save("fashion_mnist_model.gob"); err != nil {
		log.Fatal(err)
	}
	fmt.Println("モデルを fashion_mnist_model.gob に保存しました")
}
