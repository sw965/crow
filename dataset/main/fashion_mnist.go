package main

import (
	"compress/gzip"
	"encoding/binary"
	"github.com/sw965/crow/dataset" // プロジェクトの構造に合わせて調整してください
	"github.com/sw965/omw/encoding/gobx"
	"io"
	"log"
	"os"
)

// Fashion-MNISTのファイル名（MNISTと区別するためにプレフィックスを付けて保存することを想定）
const (
	trainImagesFile = "train-images-idx3-ubyte.gz"
	trainLabelsFile = "train-labels-idx1-ubyte.gz"
	testImagesFile  = "t10k-images-idx3-ubyte.gz"
	testLabelsFile  = "t10k-labels-idx1-ubyte.gz"
	outputFile      = "fashion_mnist.gob"
)

func main() {
	log.Println("Fashion-MNISTデータの読み込みを開始します...")

	data := dataset.FashionMnist{}

	var err error
	log.Println("学習用画像を読み込んでいます...")
	data.TrainImages, err = readImages(trainImagesFile)
	if err != nil {
		log.Fatalf("学習用画像の読み込み失敗: %v", err)
	}

	log.Println("学習用ラベルを読み込んでいます...")
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

	log.Printf("完了！ '%s' に保存されました。", outputFile)
}

// 以下の関数は既存のコードと同様
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

	header := make([]byte, 16)
	if _, err := io.ReadFull(gr, header); err != nil {
		return nil, err
	}

	count := binary.BigEndian.Uint32(header[4:8])
	rows := binary.BigEndian.Uint32(header[8:12])
	cols := binary.BigEndian.Uint32(header[12:16])
	imageSize := int(rows * cols)

	images := make([][]float32, count)
	buf := make([]byte, imageSize)

	for i := range images {
		if _, err := io.ReadFull(gr, buf); err != nil {
			return nil, err
		}
		floatRow := make([]float32, imageSize)
		for j, b := range buf {
			floatRow[j] = float32(b) / 255.0
		}
		images[i] = floatRow
	}
	return images, nil
}

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
	byteLabels := make([]byte, count)
	if _, err := io.ReadFull(gr, byteLabels); err != nil {
		return nil, err
	}

	labels := make([]float32, count)
	for i, b := range byteLabels {
		labels[i] = float32(b)
	}
	return labels, nil
}