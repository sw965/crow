package mlp_test

import (
	"fmt"
	"testing"

	"github.com/sw965/crow/dataset"
	"github.com/sw965/crow/model/binary/mlp"
	"github.com/sw965/omw/mathx/bitsx"
	"github.com/sw965/omw/mathx/randx"
	"github.com/sw965/omw/slicesx"
	"github.com/sw965/crow/model/binary/layer"
)

// addBias は各行列に対し、末尾に n ビット(値はすべて1)を追加します。
func addBias(xs []bitsx.Matrix, n int) ([]bitsx.Matrix, error) {
	if n < 0 {
		return nil, fmt.Errorf("bias count must be non-negative")
	}
	if n == 0 {
		return xs, nil
	}

	newXs := make([]bitsx.Matrix, len(xs))
	for i, x := range xs {
		// 元の列数 + n
		newCols := x.Cols + n
		newMat, err := bitsx.NewZerosMatrix(x.Rows, newCols)
		if err != nil {
			return nil, err
		}

		srcStride := x.Stride
		dstStride := newMat.Stride
		srcData := x.Data
		dstData := newMat.Data

		for r := 0; r < x.Rows; r++ {
			srcOffset := r * srcStride
			dstOffset := r * dstStride
			copy(dstData[dstOffset:dstOffset+srcStride], srcData[srcOffset:srcOffset+srcStride])

			// バイアスビットの設定 (1にする)
			for k := 0; k < n; k++ {
				if err := newMat.Set(r, x.Cols+k); err != nil {
					return nil, err
				}
			}
		}

		newMat.ApplyMask()
		newXs[i] = newMat
	}
	return newXs, nil
}

// ensembleAccuracy は複数のモデルの予測ロジットを合算(Soft Voting)して精度を計算します
func ensembleAccuracy(models []*mlp.Model, xs []bitsx.Matrix, labels []int) (float32, error) {
	n := len(xs)
	if n != len(labels) {
		return 0.0, fmt.Errorf("length mismatch")
	}

	correctCount := 0
	numClasses := 10 // FashionMNISTは10クラス

	// ※ 高速化のために parallel.For を使うことも可能ですが、ここではシンプルに実装します
	for i, x := range xs {
		label := labels[i]
		
		// 全モデルのロジットを合計するための配列
		sumLogits := make([]int, numClasses)

		for _, model := range models {
			// 各モデルの予測スコア（プロトタイプとの類似度）を取得
			logits, err := model.PredictLogits(x)
			if err != nil {
				return 0.0, err
			}

			// ロジットを加算
			for classIdx, score := range logits {
				sumLogits[classIdx] += score
			}
		}

		// 合計スコアが最大のクラスを選択
		predMaxIdx := slicesx.Argsort(sumLogits)[len(sumLogits)-1]
		if predMaxIdx == label {
			correctCount++
		}
	}

	return float32(correctCount) / float32(n), nil
}

func TestEnsemble(t *testing.T) {
	rng := randx.NewPCGFromGlobalSeed()

	mnist, err := dataset.LoadFashionMNIST()
	if err != nil {
		panic(err)
	}

	// === 設定: バイアスの数 ===
	const biasN = 128
	// === 設定: アンサンブルするモデルの数 ===
	const numModels = 1
	// ========================

	// バイアス項を追加
	mnist.TrainImages, err = addBias(mnist.TrainImages, biasN)
	if err != nil {
		panic(err)
	}
	mnist.TestImages, err = addBias(mnist.TestImages, biasN)
	if err != nil {
		panic(err)
	}

	inputDim := 784 + biasN
	fmt.Printf("Input Dimension: %d (784 + %d bias)\n", inputDim, biasN)
	fmt.Printf("Ensemble Size: %d models\n", numModels)

	// すべてのモデルで共通のターゲット（プロトタイプ）を使用することで、
	// ロジットの空間を揃えてアンサンブルしやすくします。
	prototypes, err := bitsx.NewBEFPrototypeMatrices(10, 1, 1024, 10000, rng)
	if err != nil {
		panic(err)
	}

	// モデルのスライスを作成
	models := make([]*mlp.Model, numModels)

	for i := 0; i < numModels; i++ {
		// モデル初期化 (それぞれ異なる乱数シードで初期化される)
		m := mlp.NewModel(inputDim, 3, 6)
		m.TrainingContext = &layer.TrainingContext{
			NoiseStdScale:0.5,
			GateThresholdScale:1.0,
			GroupSize:4,
		}

		err = m.AppendDenseLayer(1024)
		if err != nil {
			panic(err)
		}

		err = m.AppendDenseLayer(1024)
		if err != nil {
			panic(err)
		}

		err = m.AppendDenseLayer(1024)
		if err != nil {
			panic(err)
		}

		// 共通のプロトタイプをセット
		m.Prototypes = prototypes
		models[i] = &m
	}

	// 学習ループ
	for epoch := 0; epoch < 200; epoch++ {
		// 各モデルを個別に学習
		for mIdx, model := range models {
			// rngを共有して渡していますが、ループが進むごとにrngの状態も進むため
			// 各モデルでバッチのシャッフル順序は異なります（多様性に寄与）
			err := model.TrainForClassification(mnist.TrainImages, mnist.TrainLabels, 600, 0.1, rng)
			if err != nil {
				panic(fmt.Errorf("model %d train error: %v", mIdx, err))
			}
		}

		// アンサンブル精度を評価
		acc, err := ensembleAccuracy(models, mnist.TestImages, mnist.TestLabels)
		if err != nil {
			panic(err)
		}
		
		fmt.Printf("Epoch %d: Ensemble Accuracy = %.4f\n", epoch, acc)
	}
}