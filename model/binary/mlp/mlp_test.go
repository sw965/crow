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
	"math/rand/v2"
	"math"
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
	return
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
	prototypes, err := bitsx.NewBEFMatrices(10, 1, 1024, 10000, rng)
	if err != nil {
		panic(err)
	}

	// モデルのスライスを作成
	models := make([]*mlp.Model, numModels)
	p := 6
	for i := 0; i < numModels; i++ {
		// モデル初期化 (それぞれ異なる乱数シードで初期化される)
		m := mlp.NewModel(inputDim, 3, p)
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
			err := model.Train(mnist.TrainImages, mnist.TrainLabels, &mlp.TrainingConfig{MiniBatchSize:600, LearningRate:0.1, Rng:rng, Margin:0.25})
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

// splitMatrixToRows は、1つの大きな行列を行ごとの Matrix スライスに分割します。
// bitsx.NewRandMatrix の結果を mlp.Train で使える形式に変換するために使用します。
func splitMatrixToRows(m bitsx.Matrix) bitsx.Matrices {
	res := make(bitsx.Matrices, m.Rows)
	stride := m.Stride
	for r := 0; r < m.Rows; r++ {
		// 行ベクトルとして新しいMatrixを作成
		rowMat := bitsx.Matrix{
			Rows:     1,
			Cols:     m.Cols,
			Stride:   stride,
			WordMask: m.WordMask,
			Data:     make([]uint64, stride),
		}
		// 該当行のデータをコピー
		start := r * stride
		copy(rowMat.Data, m.Data[start:start+stride])
		res[r] = rowMat
	}
	return res
}

// generateRegressionData は、回帰タスク用のデータを生成します。
// 入力: inputBits ビットのランダム行列
// ラベル: 入力のPopCount * 2
func generateRegressionData(n, inputBits int, rng *rand.Rand) (bitsx.Matrices, []int, error) {
	// 1. ランダムな入力行列を生成 (k=0: 一様ランダム)
	// バリエーションを持たせたい場合は、kを変えて複数の行列を生成して結合すると良いですが、
	// ここではシンプルに一様ランダムとします。
	rawX, err := bitsx.NewRandMatrix(n, inputBits, 0, rng)
	if err != nil {
		return nil, nil, err
	}

	xs := splitMatrixToRows(rawX)
	labels := make([]int, n)

	for i, x := range xs {
		// ルール: 入力の1の数 × 2 を正解ラベルとする
		// 例: 50bit中 25bitが1なら、ラベルは 50
		count := x.PopCount()
		labels[i] = count * 2
	}

	return xs, labels, nil
}

// evaluateMAE は、モデルの予測値（温度計プロトタイプのインデックス）と正解ラベルの
// 平均絶対誤差 (Mean Absolute Error) を計算します。
func evaluateMAE(model *mlp.Model, xs bitsx.Matrices, labels []int) (float64, error) {
	var totalError float64
	n := len(xs)

	for i, x := range xs {
		// ロジット（各プロトタイプとの類似度）を取得
		logits, err := model.PredictLogits(x)
		if err != nil {
			return 0.0, err
		}

		// ロジットが最大のインデックスを予測値とする
		// 温度計プロトタイプの場合、インデックスがそのまま「数値」を表す
		predIdx := 0
		maxLogit := -1
		for idx, score := range logits {
			if score > maxLogit {
				maxLogit = score
				predIdx = idx
			}
		}

		// 誤差を加算
		diff := float64(predIdx - labels[i])
		totalError += math.Abs(diff)
	}

	return totalError / float64(n), nil
}

func TestThermometerRegression(t *testing.T) {
	// return
	rng := randx.NewPCGFromGlobalSeed()

	// === 設定 ===
	const (
		inputBits  = 50
		outputBits = 100 // 出力層のビット数
		trainSize  = 60000
		testSize   = 10000
		epochs     = 1500
	)

	// 1. データセット生成
	trainXs, trainLabels, err := generateRegressionData(trainSize, inputBits, rng)
	if err != nil {
		t.Fatal(err)
	}
	testXs, testLabels, err := generateRegressionData(testSize, inputBits, rng)
	if err != nil {
		t.Fatal(err)
	}

	// 2. プロトタイプの準備
	// 0〜100個の「1」に対応する 101個のプロトタイプを生成
	// インデックス k のプロトタイプは、k個のビットが立っている
	// 正解ラベル (PopCount * 2) がそのままインデックスとして使える
	prototypes, err := bitsx.NewThermometerMatrices(101, 1, outputBits)
	if err != nil {
		t.Fatal(err)
	}

	// 3. モデル構築
	// 入力50 -> 隠れ128 -> 隠れ128 -> 出力100
	p := 4
	model := mlp.NewModel(inputBits, 3, p) // 4並列
	model.TrainingContext = &layer.TrainingContext{
		NoiseStdScale:      0.0,
		GateThresholdScale: 1.0,
		GroupSize:          1,
	}

	// 層の追加
	if err := model.AppendDenseLayer(512); err != nil {
		t.Fatal(err)
	}
	if err := model.AppendDenseLayer(512); err != nil {
		t.Fatal(err)
	}
	// 最終層はプロトタイプと同じ100次元
	if err := model.AppendDenseLayer(outputBits); err != nil {
		t.Fatal(err)
	}

	// プロトタイプをセット
	model.Prototypes = prototypes

	fmt.Println("Start Training Regression Model...")
	fmt.Printf("Input: %d bits -> Output: %d bits (Thermometer)\n", inputBits, outputBits)

	// 4. 学習ループ
	for epoch := 0; epoch < epochs; epoch++ {
		// 温度計プロトタイプは隣接クラスが似ているため、Marginの設定が重要
		// ここでは 0.2 (100ビット中20ビット差) を設定
		config := &mlp.TrainingConfig{
			MiniBatchSize: 100,
			LearningRate:  0.1,
			Rng:           rng,
			Margin:        0.01,
		}

		err := model.Train(trainXs, trainLabels, config)
		if err != nil {
			t.Fatalf("Train error at epoch %d: %v", epoch, err)
		}

		if (epoch+1)%10 == 0 {
			mae, err := evaluateMAE(&model, testXs, testLabels)
			if err != nil {
				t.Fatal(err)
			}
			// 精度(正解率)も一応計算
			acc, err := model.Accuracy(testXs, testLabels, 1)
			if err != nil {
				t.Fatal(err)
			}
			fmt.Printf("Epoch %d: MAE = %.4f, Accuracy(完全一致) = %.4f\n", epoch+1, mae, acc)

			mae, err = evaluateMAE(&model, trainXs, trainLabels)
			if err != nil {
				t.Fatal(err)
			}

			acc, err = model.Accuracy(trainXs, trainLabels, 1)
			if err != nil {
				t.Fatal(err)
			}
			fmt.Printf("Epoch %d: MAE = %.4f, Accuracy(完全一致) = %.4f\n", epoch+1, mae, acc)
			fmt.Println("")			
		}
	}
}

func TestHDRPERegression(t *testing.T) {
	// return
	// 乱数シードの固定（再現性のため）
	rng := randx.NewPCGFromGlobalSeed()

	// === 設定 ===
	const (
		inputBits  = 50           // 入力ビット数
		maxLabel   = inputBits * 2 // ラベルの最大値 (PopCount * 2) -> 100
		outputBits = 2048         // HDRPEの次元数 (ホログラフィック性を高めるため少し大きめに確保)
		sigma      = 30.0          // HDRPEの帯域幅パラメータ (滑らかさと識別性のバランス)
		
		trainSize  = 10000
		testSize   = 2000
		epochs     = 2500
	)

	// 1. データセット生成
	// 入力の1の数 x 2 を正解ラベルとするデータ
	trainXs, trainLabels, err := generateRegressionData(trainSize, inputBits, rng)
	if err != nil {
		t.Fatal(err)
	}
	testXs, testLabels, err := generateRegressionData(testSize, inputBits, rng)
	if err != nil {
		t.Fatal(err)
	}

	// 2. プロトタイプの準備 (HDRPE)
	// 0 から maxLabel までの値を表現するため、maxLabel + 1 個のプロトタイプを作成
	// HDRPEは、値が近いプロトタイプ同士のハミング距離が近く、遠いと遠くなるように符号化される
	numPrototypes := maxLabel + 1
	prototypes, err := bitsx.NewHDRPEMatrices(numPrototypes, 1, outputBits, sigma, rng)
	if err != nil {
		t.Fatal(err)
	}

	// 3. モデル構築
	// 入力50 -> 隠れ512 -> 隠れ512 -> 出力2048 (HDRPE次元)
	// 並列数 p は適当に設定
	p := 4
	model := mlp.NewModel(inputBits, 3, p)
	model.TrainingContext = &layer.TrainingContext{
		NoiseStdScale:      0.0,
		GateThresholdScale: 1.0,
		GroupSize:          1, // 学習の安定性を制御
	}

	// 隠れ層
	if err := model.AppendDenseLayer(512); err != nil {
		t.Fatal(err)
	}
	if err := model.AppendDenseLayer(512); err != nil {
		t.Fatal(err)
	}
	// 出力層 (HDRPEの次元数に合わせる)
	if err := model.AppendDenseLayer(outputBits); err != nil {
		t.Fatal(err)
	}

	// 生成したHDRPEプロトタイプをモデルにセット
	model.Prototypes = prototypes

	fmt.Println("Start Training HDRPE Regression Model...")
	fmt.Printf("Input: %d bits -> Output: %d bits (HDRPE, Sigma=%.1f)\n", inputBits, outputBits, sigma)

	// 4. 学習ループ
	for epoch := 0; epoch < epochs; epoch++ {
		config := &mlp.TrainingConfig{
			MiniBatchSize: 100,
			LearningRate:  0.1,
			Rng:           rng,
			// HDRPEはThermometerと同様に近傍の類似度が高いため、適切なマージンが必要
			// ここでは全体のビット数の20%程度の差を要求してみる
			Margin:        0.05, 
		}

		err := model.Train(trainXs, trainLabels, config)
		if err != nil {
			t.Fatalf("Train error at epoch %d: %v", epoch, err)
		}

		// 10エポックごとに評価
		if (epoch+1)%10 == 0 {
			// テストデータでのMAE (平均絶対誤差)
			mae, err := evaluateMAE(&model, testXs, testLabels)
			if err != nil {
				t.Fatal(err)
			}
			
			// テストデータでの完全一致率 (Accuracy)
			// 回帰タスクだが、離散的なプロトタイプへの分類として見た場合の正解率
			acc, err := model.Accuracy(testXs, testLabels, 1)
			if err != nil {
				t.Fatal(err)
			}

			fmt.Printf("Epoch %3d: MAE = %.4f, Accuracy(Exact) = %.4f\n", epoch+1, mae, acc)
		}
	}
}

func TestHDRPE_GroupSizeAnalysis(t *testing.T) {
	return
	// 固定パラメータ（前回の良設定）
	sigma := 30.0
	margin := float32(0.05)
	
	// 検証パラメータ: GroupSize を小さくして更新密度を上げる
	// GroupSize=1 は「間違ったニューロンは全て更新する」設定（最も可塑性が高い）
	groupSizes := []int{1, 2, 4}

	const (
		inputBits  = 50
		maxLabel   = 100
		outputBits = 2048
		trainSize  = 10000
		testSize   = 2000
		epochs     = 50
	)

	rngData := rand.New(rand.NewPCG(777, 777))
	trainXs, trainLabels, err := generateRegressionData(trainSize, inputBits, rngData)
	if err != nil {
		t.Fatal(err)
	}
	testXs, testLabels, err := generateRegressionData(testSize, inputBits, rngData)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Printf("=== HDRPE GroupSize Analysis (Sigma=%.1f, Margin=%.2f) ===\n", sigma, margin)
	fmt.Printf("| GroupSize | Final MAE | Final Acc |\n")
	fmt.Printf("|-----------|-----------|-----------|\n")

	for _, gs := range groupSizes {
		rngModel := rand.New(rand.NewPCG(1, 1))

		prototypes, err := bitsx.NewHDRPEMatrices(maxLabel+1, 1, outputBits, sigma, rngModel)
		if err != nil {
			t.Fatal(err)
		}

		p := 4
		model := mlp.NewModel(inputBits, 3, p)
		model.TrainingContext = &layer.TrainingContext{
			NoiseStdScale:      0.5,
			GateThresholdScale: 1.0,
			GroupSize:          gs, // ここを変える
		}
		model.AppendDenseLayer(512)
		model.AppendDenseLayer(512)
		model.AppendDenseLayer(outputBits)
		model.Prototypes = prototypes

		var finalMAE float64
		var finalAcc float32

		for e := 0; e < epochs; e++ {
			config := &mlp.TrainingConfig{
				MiniBatchSize: 100,
				LearningRate:  0.1, // 学習率は0.1に戻す
				Rng:           rngModel,
				Margin:        margin,
			}
			model.Train(trainXs, trainLabels, config)

			if e == epochs-1 {
				finalMAE, _ = evaluateMAE(&model, testXs, testLabels)
				finalAcc, _ = model.Accuracy(testXs, testLabels, 1)
			}
		}

		fmt.Printf("| %9d | %9.4f | %9.4f |\n", gs, finalMAE, finalAcc)
	}
}