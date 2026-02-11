package dataset

import (
	"fmt"
	"github.com/sw965/omw/encoding/gobx"
	"github.com/sw965/omw/mathx/bitsx"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

// このpackage全体的に命名にbinaryを付ける？あるいはdataset/binaryにする？

const (
	baseURL = "https://github.com/sw965/crow/releases/download/v0.1.0-test/"

	// MNIST
	mnistTrainImg   = "mnist_flat_binary_imgs.gob"
	mnistTrainLabel = "mnist_int_labels.gob"
	mnistTestImg    = "mnist_test_flat_binary_imgs.gob"
	mnistTestLabel  = "mnist_test_int_labels.gob"

	// Fashion-MNIST
	fashionTrainImg   = "fashion_mnist_train_flat_binary_imgs.gob"
	fashionTrainLabel = "fashion_mnist_train_int_labels.gob"
	fashionTestImg    = "fashion_mnist_test_flat_binary_imgs.gob"
	fashionTestLabel  = "fashion_mnist_test_int_labels.gob"
)

// BinaryDataset は訓練用とテスト用の画像・ラベルを保持します
type BinaryDataset struct {
	TrainImages bitsx.Matrices
	TrainLabels []int
	TestImages  bitsx.Matrices
	TestLabels  []int
}

// LoadMNIST は通常のMNISTデータを読み込みます
func LoadMNIST() (BinaryDataset, error) {
	return loadDataset(mnistTrainImg, mnistTrainLabel, mnistTestImg, mnistTestLabel)
}

// LoadFashionMNIST はFashion-MNISTデータを読み込みます
func LoadFashionMNIST() (BinaryDataset, error) {
	return loadDataset(fashionTrainImg, fashionTrainLabel, fashionTestImg, fashionTestLabel)
}

// 内部共有用の読み込み関数
func loadDataset(trImg, trLbl, teImg, teLbl string) (BinaryDataset, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return BinaryDataset{}, fmt.Errorf("ホームディレクトリの取得に失敗: %w", err)
	}

	dataDir := filepath.Join(home, ".crow_dataset")
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return BinaryDataset{}, err
	}

	// 4つのファイルすべてに対してダウンロードとロードを実行
	targetFiles := []string{trImg, trLbl, teImg, teLbl}
	for _, name := range targetFiles {
		path := filepath.Join(dataDir, name)
		if err := ensureFile(path, baseURL+name); err != nil {
			return BinaryDataset{}, err
		}
	}

	// 各ファイルをGOBとしてデコード
	trImgs, err := gobx.Load[bitsx.Matrices](filepath.Join(dataDir, trImg))
	if err != nil {
		return BinaryDataset{}, err
	}
	trLbls, err := gobx.Load[[]int](filepath.Join(dataDir, trLbl))
	if err != nil {
		return BinaryDataset{}, err
	}
	teImgs, err := gobx.Load[bitsx.Matrices](filepath.Join(dataDir, teImg))
	if err != nil {
		return BinaryDataset{}, err
	}
	teLbls, err := gobx.Load[[]int](filepath.Join(dataDir, teLbl))
	if err != nil {
		return BinaryDataset{}, err
	}

	return BinaryDataset{
		TrainImages: trImgs,
		TrainLabels: trLbls,
		TestImages:  teImgs,
		TestLabels:  teLbls,
	}, nil
}

func ensureFile(path, url string) error {
	if _, err := os.Stat(path); err == nil {
		return nil
	}

	fmt.Printf("Downloading %s...\n", url)
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	out, err := os.Create(path)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	return err
}