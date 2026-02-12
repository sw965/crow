package dataset

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"github.com/sw965/omw/encoding/gobx"
	"github.com/sw965/omw/mathx/bitsx"
)

//dataset/binaryパッケージに変更する。
//でバックの為の、cui上でmnistの数字とラベルが一致しているかを見る

const (
	baseURL = "https://github.com/sw965/crow/releases/download/v0.1.0-test/"

	// MNIST
	mnistTrainImg   = "mnist_train_flat_binary_imgs.gob"
	mnistTrainLabel = "mnist_train_int_labels.gob"
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
// 初回実行時はGitHubからデータをダウンロードし、ホームディレクトリにキャッシュします。
func LoadMNIST() (BinaryDataset, error) {
	return loadDataset(mnistTrainImg, mnistTrainLabel, mnistTestImg, mnistTestLabel)
}

// LoadFashionMNIST はFashion-MNISTデータを読み込みます
// 初回実行時はGitHubからデータをダウンロードし、ホームディレクトリにキャッシュします。
func LoadFashionMNIST() (BinaryDataset, error) {
	return loadDataset(fashionTrainImg, fashionTrainLabel, fashionTestImg, fashionTestLabel)
}

// Clean はキャッシュされたデータセットディレクトリ(.crow_dataset)を完全に削除します。
// ディスク容量を空けたい場合や、データセットを再ダウンロードしたい場合に使用します。
func Clean() error {
	dir, err := getCacheDir()
	if err != nil {
		return err
	}

	// ディレクトリが存在しない場合は何もしない
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		return nil
	}

	fmt.Printf("Removing dataset cache: %s\n", dir)
	return os.RemoveAll(dir)
}

// ----------------------------------------------------------------------------
// Internal Helpers
// ----------------------------------------------------------------------------

// 内部共有用の読み込み関数
func loadDataset(trImg, trLbl, teImg, teLbl string) (BinaryDataset, error) {
	dataDir, err := getCacheDir()
	if err != nil {
		return BinaryDataset{}, err
	}

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
		return BinaryDataset{}, fmt.Errorf("failed to load %s: %w", trImg, err)
	}
	trLbls, err := gobx.Load[[]int](filepath.Join(dataDir, trLbl))
	if err != nil {
		return BinaryDataset{}, fmt.Errorf("failed to load %s: %w", trLbl, err)
	}
	teImgs, err := gobx.Load[bitsx.Matrices](filepath.Join(dataDir, teImg))
	if err != nil {
		return BinaryDataset{}, fmt.Errorf("failed to load %s: %w", teImg, err)
	}
	teLbls, err := gobx.Load[[]int](filepath.Join(dataDir, teLbl))
	if err != nil {
		return BinaryDataset{}, fmt.Errorf("failed to load %s: %w", teLbl, err)
	}

	return BinaryDataset{
		TrainImages: trImgs,
		TrainLabels: trLbls,
		TestImages:  teImgs,
		TestLabels:  teLbls,
	}, nil
}

// getCacheDir はOSごとのホームディレクトリ配下のキャッシュパスを返します
func getCacheDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("failed to get user home directory: %w", err)
	}
	return filepath.Join(home, ".crow_dataset"), nil
}

// ensureFile はファイルが存在しない場合のみURLからダウンロードします
func ensureFile(path, url string) error {
	if _, err := os.Stat(path); err == nil {
		return nil // 既に存在するのでスキップ
	}

	fmt.Printf("Downloading %s...\n", url)
	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("download failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	out, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	if err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}
	return nil
}