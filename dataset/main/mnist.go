package main

import (
	"compress/gzip"
	"encoding/binary"
	// "fmt" // 削除: 使用されていないため
	"github.com/sw965/crow/dataset"
	"github.com/sw965/omw/encoding/gobx"
	"io"
	"log"
	"os"
)

const (
	trainImagesFile = "train-images-idx3-ubyte.gz"
	trainLabelsFile = "train-labels-idx1-ubyte.gz"
	testImagesFile  = "t10k-images-idx3-ubyte.gz"
	testLabelsFile  = "t10k-labels-idx1-ubyte.gz"
	outputFile      = "mnist.gob"
)

func main() {
	log.Println("MNISTデータの読み込みを開始します (float32モード)...")

	data := dataset.Mnist{}

	var err error
	log.Println("学習用画像を読み込んでいます...")
	data.TrainImages, err = readImages(trainImagesFile)
	if err != nil {
		log.Fatalf("学習用画像の読み込み失敗: %v", err)
	}

	log.Println("学習用ラベルを読み込んでいます...")
	// 型変換済みの関数を呼び出すのでエラーは解消されます
	data.TrainLabels, err = readLabels(trainLabelsFile)
	if err != nil {
		log.Fatalf("学習用ラベルの読み込み失敗: %v", err)
	}

	log.Println("テスト用画像を読み込んでいます...")
	data.TestImages, err = readImages(testImagesFile)
	if err != nil {
		log.Fatalf("テスト用画像の読み込み失敗: %v", err)
	}

	log.Println("テスト用ラベルを読み込んでいます...")
	data.TestLabels, err = readLabels(testLabelsFile)
	if err != nil {
		log.Fatalf("テスト用ラベルの読み込み失敗: %v", err)
	}

	log.Printf("読み込み完了: Train[%d], Test[%d]", len(data.TrainImages), len(data.TestImages))

	log.Println("gobファイルに保存しています...")
	if err := gobx.Save(data, outputFile); err != nil {
		log.Fatalf("保存失敗: %v", err)
	}

	log.Printf("完了しました！ '%s' に保存されました。", outputFile)
}

func readImages(filename string) ([][]float32, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	gr, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer gr.Close()

	// ヘッダー読み飛ばし (16バイト)
	header := make([]byte, 16)
	if _, err := io.ReadFull(gr, header); err != nil {
		return nil, err
	}

	count := binary.BigEndian.Uint32(header[4:8])
	rows := binary.BigEndian.Uint32(header[8:12])
	cols := binary.BigEndian.Uint32(header[12:16])
	imageSize := int(rows * cols)

	log.Printf("File: %s, Count: %d, Size: %dx%d", filename, count, rows, cols)

	// float32のスライスとして確保
	images := make([][]float32, count)

	// 一時読み込み用のバッファ
	buf := make([]byte, imageSize)

	for i := range images {
		if _, err := io.ReadFull(gr, buf); err != nil {
			return nil, err
		}

		// byte(0-255) -> float32(0.0-1.0) に変換
		floatRow := make([]float32, imageSize)
		for j, b := range buf {
			floatRow[j] = float32(b) / 255.0
		}
		images[i] = floatRow
	}
	return images, nil
}

// 戻り値を []float32 に変更
func readLabels(filename string) ([]float32, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	gr, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer gr.Close()

	header := make([]byte, 8)
	if _, err := io.ReadFull(gr, header); err != nil {
		return nil, err
	}

	count := binary.BigEndian.Uint32(header[4:8])

	// 一旦バイト列として読み込む
	byteLabels := make([]byte, count)
	if _, err := io.ReadFull(gr, byteLabels); err != nil {
		return nil, err
	}

	// MnistData構造体に合わせて float32 に変換
	labels := make([]float32, count)
	for i, b := range byteLabels {
		labels[i] = float32(b)
	}
	return labels, nil
}