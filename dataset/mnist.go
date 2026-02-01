package dataset

import (
	"embed"
	"encoding/gob"
	"fmt"
)

//go:embed mnist.gob
var mnistFile embed.FS

type Mnist struct {
	TrainImages [][]float32 // 60,000 x 784
	TrainLabels []float32   // 60,000
	TestImages  [][]float32 // 10,000 x 784
	TestLabels  []float32   // 10,000
}

func LoadMnist() (Mnist, error) {
	f, err := mnistFile.Open("mnist.gob")
	if err != nil {
		// 後でエラーメッセージを日本語にする
		return Mnist{}, fmt.Errorf("failed to open embedded mnist.gob: %w", err)
	}
	defer f.Close()

	var data Mnist
	if err := gob.NewDecoder(f).Decode(&data); err != nil {
		// 後でエラーメッセージを日本語にする
		return Mnist{}, fmt.Errorf("failed to decode mnist data: %w", err)
	}
	return data, nil
}

//go:embed fashion_mnist.gob
var fashionMnistFile embed.FS

// FashionMnist は Fashion-MNISTデータセットの構造体です。
// フォーマットはMNISTと同一です。
type FashionMnist struct {
	TrainImages [][]float32 // 60,000 x 784
	TrainLabels []float32   // 60,000
	TestImages  [][]float32 // 10,000 x 784
	TestLabels  []float32   // 10,000
}

// LoadFashionMnist は埋め込まれたgobファイルからデータを読み込みます。
func LoadFashionMnist() (FashionMnist, error) {
	f, err := fashionMnistFile.Open("fashion_mnist.gob")
	if err != nil {
		return FashionMnist{}, fmt.Errorf("failed to open embedded fashion_mnist.gob: %w", err)
	}
	defer f.Close()

	var data FashionMnist
	if err := gob.NewDecoder(f).Decode(&data); err != nil {
		return FashionMnist{}, fmt.Errorf("failed to decode fashion_mnist data: %w", err)
	}
	return data, nil
}