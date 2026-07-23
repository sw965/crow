package dataset

import (
	"fmt"
	"net/http"
	"os"
	"path/filepath"

	"github.com/sw965/omw/encoding/atomicfile"
	"github.com/sw965/omw/encoding/gobx"
	"github.com/sw965/omw/mathx/bitsx"
)

const (
	defaultBaseURL = "https://github.com/sw965/crow/releases/download/v0.1.0-test/"

	// MNIST
	mnistTrainImagesFile = "mnist_train_flat_binary_imgs.gob"
	mnistTrainLabelsFile = "mnist_train_int_labels.gob"
	mnistTestImagesFile  = "mnist_test_flat_binary_imgs.gob"
	mnistTestLabelsFile  = "mnist_test_int_labels.gob"

	// Fashion-MNIST
	fashionTrainImagesFile = "fashion_mnist_train_flat_binary_imgs.gob"
	fashionTrainLabelsFile = "fashion_mnist_train_int_labels.gob"
	fashionTestImagesFile  = "fashion_mnist_test_flat_binary_imgs.gob"
	fashionTestLabelsFile  = "fashion_mnist_test_int_labels.gob"
)

// LogFunc は、進捗の出力先。fmt.Printf や t.Logf 相当を渡す。nilの場合は無音。
type LogFunc func(format string, a ...any)

// PrintLog は、進捗を標準出力に表示する LogFunc の実装。
// dataset.LoadMNIST(dataset.PrintLog) のように渡して使う。
var PrintLog LogFunc = func(format string, a ...any) {
	fmt.Printf(format, a...)
}

// BinaryDataset は訓練用とテスト用の画像・ラベルを保持します
type BinaryDataset struct {
	TrainImages bitsx.Matrices
	TrainLabels []int
	TestImages  bitsx.Matrices
	TestLabels  []int
}

// LoadMNIST は通常のMNISTデータを読み込みます
// 初回実行時はGitHubからデータをダウンロードし、ホームディレクトリにキャッシュします。
func LoadMNIST(logf LogFunc) (BinaryDataset, error) {
	dataDir, err := getCacheDir()
	if err != nil {
		return BinaryDataset{}, err
	}
	return loadDataset(defaultBaseURL, dataDir, mnistTrainImagesFile, mnistTrainLabelsFile, mnistTestImagesFile, mnistTestLabelsFile, logf)
}

// LoadFashionMNIST はFashion-MNISTデータを読み込みます
// 初回実行時はGitHubからデータをダウンロードし、ホームディレクトリにキャッシュします。
func LoadFashionMNIST(logf LogFunc) (BinaryDataset, error) {
	dataDir, err := getCacheDir()
	if err != nil {
		return BinaryDataset{}, err
	}
	return loadDataset(defaultBaseURL, dataDir, fashionTrainImagesFile, fashionTrainLabelsFile, fashionTestImagesFile, fashionTestLabelsFile, logf)
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
	return os.RemoveAll(dir)
}

// ----------------------------------------------------------------------------
// Internal Helpers
// ----------------------------------------------------------------------------

// 内部共有用の読み込み関数
func loadDataset(baseURL, dataDir, trainImagesFile, trainLabelsFile, testImagesFile, testLabelsFile string, logf LogFunc) (BinaryDataset, error) {
	if logf == nil {
		logf = func(string, ...any) {}
	}

	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return BinaryDataset{}, err
	}

	trainImages, err := loadWithRecovery[bitsx.Matrices](filepath.Join(dataDir, trainImagesFile), baseURL+trainImagesFile, logf)
	if err != nil {
		return BinaryDataset{}, fmt.Errorf("%s の読み込みに失敗: %w", trainImagesFile, err)
	}
	trainLabels, err := loadWithRecovery[[]int](filepath.Join(dataDir, trainLabelsFile), baseURL+trainLabelsFile, logf)
	if err != nil {
		return BinaryDataset{}, fmt.Errorf("%s の読み込みに失敗: %w", trainLabelsFile, err)
	}
	testImages, err := loadWithRecovery[bitsx.Matrices](filepath.Join(dataDir, testImagesFile), baseURL+testImagesFile, logf)
	if err != nil {
		return BinaryDataset{}, fmt.Errorf("%s の読み込みに失敗: %w", testImagesFile, err)
	}
	testLabels, err := loadWithRecovery[[]int](filepath.Join(dataDir, testLabelsFile), baseURL+testLabelsFile, logf)
	if err != nil {
		return BinaryDataset{}, fmt.Errorf("%s の読み込みに失敗: %w", testLabelsFile, err)
	}

	return BinaryDataset{
		TrainImages: trainImages,
		TrainLabels: trainLabels,
		TestImages:  testImages,
		TestLabels:  testLabels,
	}, nil
}

// loadWithRecovery は、ファイルを確保してgobとして読み込む。
// デコードに失敗した場合、破損したキャッシュとみなして削除し、再ダウンロードした上で
// 1回だけ再試行する（無限に再試行はしない）。
func loadWithRecovery[T any](path, url string, logf LogFunc) (T, error) {
	var zero T

	if err := ensureFile(path, url, logf); err != nil {
		return zero, err
	}

	data, err := gobx.Load[T](path)
	if err == nil {
		return data, nil
	}

	if removeErr := os.Remove(path); removeErr != nil {
		return zero, fmt.Errorf("キャッシュの読み込みに失敗し(%w)、破損したキャッシュの削除にも失敗しました: %w", err, removeErr)
	}

	if err := ensureFile(path, url, logf); err != nil {
		return zero, err
	}

	data, err = gobx.Load[T](path)
	if err != nil {
		return zero, fmt.Errorf("キャッシュが破損していた為、再取得しましたが、それでも読み込みに失敗しました: %w", err)
	}
	return data, nil
}

// getCacheDir はOSごとのホームディレクトリ配下のキャッシュパスを返します
func getCacheDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("ホームディレクトリの取得に失敗: %w", err)
	}
	return filepath.Join(home, ".crow_dataset"), nil
}

// ensureFile はファイルが存在しない場合のみURLからダウンロードします。
// ダウンロードが途中で失敗した場合に壊れたファイルが残らないよう、
// atomicfile.WriteFrom で一時ファイル経由の安全な保存を行います。
// レスポンスの内容は全体をメモリに保持せず、ストリーミングで一時ファイルへ書き込みます。
func ensureFile(path, url string, logf LogFunc) error {
	if logf == nil {
		logf = func(string, ...any) {}
	}

	if _, err := os.Stat(path); err == nil {
		return nil // 既に存在するのでスキップ
	}

	logf("Downloading %s...\n", url)
	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("ダウンロードに失敗: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTPステータスが不正: %s", resp.Status)
	}

	if err := atomicfile.WriteFrom(path, resp.Body, 0644); err != nil {
		return fmt.Errorf("ファイルの保存に失敗: %w", err)
	}
	return nil
}
