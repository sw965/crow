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

const (
	mnistBaseURL           = "https://github.com/sw965/crow/releases/download/v0.1.0-test/"
	mnistBinaryImgFileName = "mnist_flat_binary_imgs.gob"
	mnistIntLabelFileName  = "mnist_int_labels.gob"
)

type BinaryMNIST struct {
	FlatImgs bitsx.Matrices
	Labels   []int
}

func LoadBinaryMNIST() (BinaryMNIST, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return BinaryMNIST{}, fmt.Errorf("ホームディレクトリの取得に失敗: %w", err)
	}

	dataDir := filepath.Join(home, ".crow_dataset")
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return BinaryMNIST{}, err
	}

	imgPath := filepath.Join(dataDir, mnistBinaryImgFileName)
	labelPath := filepath.Join(dataDir, mnistIntLabelFileName)

	if err := ensureFile(imgPath, mnistBaseURL+mnistBinaryImgFileName); err != nil {
		return BinaryMNIST{}, err
	}
	if err := ensureFile(labelPath, mnistBaseURL+mnistIntLabelFileName); err != nil {
		return BinaryMNIST{}, err
	}

	imgs, err := gobx.Load[bitsx.Matrices](imgPath)
	if err != nil {
		return BinaryMNIST{}, err
	}

	labels, err := gobx.Load[[]int](labelPath)
	if err != nil {
		return BinaryMNIST{}, err
	}

	return BinaryMNIST{FlatImgs: imgs, Labels: labels}, nil
}

func ensureFile(path, url string) error {
	if _, err := os.Stat(path); err == nil {
		return nil // 既に存在するので何もしない
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
