package mlp_test

import (
	"fmt"
	"github.com/sw965/crow/dataset"
	"github.com/sw965/crow/model/binary/mlp"
	"github.com/sw965/omw/mathx/bitsx"
	"github.com/sw965/omw/mathx/randx"
	"log"
	"math/bits"
	"math/rand/v2"
	"testing"
	"github.com/sw965/omw/slicesx"
)

func Test(t *testing.T) {
	// --- ハイパーパラメータ設定 ---
	a := float32(0.001)      // 入力画像の二値化閾値
	epochs := 100          // エポック数
	batchSize := 1000      // ミニバッチサイズ
	parallelism := 6       // 並列ワーカー数
	lr := float32(0.1)     // 更新確率(prob)
	numBiasBits := 160     // 追加するバイアス項(定数1)の数 (n個)
	
	// BEF (Binary Equiangular Frame) プロトタイプ生成用パラメータ
	outputDim := 1024       // 出力層の次元数 K_L
	befIterations := 20000  // 局所探索の反復回数 (クラス数 * 次元数 * 10程度が目安)

	// 1. MNISTデータの読み込み
	mnist, err := dataset.LoadFashionMnist()
	if err != nil {
		log.Fatal(err)
	}

	// 2. 学習画像の二値化
	log.Println("Binarizing train images...")
	xTrainRaw, err := dataset.BinarizeImages(mnist.TrainImages, a)
	if err != nil {
		log.Fatal(err)
	}

	// バイアス項の追加 (Train)
	log.Printf("Adding %d bias bits to train images...", numBiasBits)
	xTrain, err := addBias(xTrainRaw, numBiasBits)
	if err != nil {
		log.Fatal(err)
	}

	rngs := make([]*rand.Rand, parallelism)
	for i := range rngs {
		rngs[i] = randx.NewPCGFromGlobalSeed()
	}
	rng := rand.New(rand.NewPCG(1, 2))

	// 3. モデルの初期化
	// Note: SetPrototypesを使うため、ターゲット生成の前にモデルを作成する必要があります
	inputDim := 784 + numBiasBits
	model, err := mlp.NewModel(
		[]int{
			inputDim,
			512,
			512,
			512,
			outputDim,
		},
		parallelism,
		randx.NewPCGFromGlobalSeed(),
	)
	if err != nil {
		panic(err)
	}

	// 4. プロトタイプ（正解の型）の作成と設定 [cite: 571]
	// ハミング距離を最大化・均一化したBEFプロトタイプを生成し、モデルにセットします
	log.Println("Generating BEF prototypes...")
	prototypes, err := bitsx.NewBEFPrototypeMatrices(10, outputDim, befIterations, rng)
	if err != nil {
		log.Fatal(err)
	}
	model.Prototypes = prototypes

	// 5. ラベルをプロトタイプ(ビット行列)に変換
	// model.Prototypes に生成された最適なビットパターンが入っているため、それを使用します
	log.Println("Converting labels to targets...")
	tTrain, err := dataset.LabelsToTargets(mnist.TrainLabels, model.Prototypes)
	if err != nil {
		log.Fatal(err)
	}

	// 6. テストデータの準備
	log.Println("Binarizing test images...")
	xTestRaw, err := dataset.BinarizeImages(mnist.TestImages, a)
	if err != nil {
		log.Fatal(err)
	}

	// バイアス項の追加 (Test)
	log.Printf("Adding %d bias bits to test images...", numBiasBits)
	xTest, err := addBias(xTestRaw, numBiasBits)
	if err != nil {
		log.Fatal(err)
	}

	tTestLabels := mnist.TestLabels

	log.Printf("Ready for training! Samples: %d, InputDim: %d, OutputDim: %d", len(xTrain), inputDim, outputDim)
	log.Println("Start training loop...")

	numTrain := len(xTrain)
	numBatches := numTrain / batchSize

	for epoch := 0; epoch < epochs; epoch++ {
		// データのシャッフル
		perm := rng.Perm(numTrain)

		model.SetIsTraining(true)
		// 1エポック分のミニバッチ学習
		for b := 0; b < numBatches; b++ {
			start := b * batchSize

			batchXs := make([]bitsx.Matrix, batchSize)
			batchTs := make([]bitsx.Matrix, batchSize)
			batchLabels := make([]int, batchSize)

			for i := 0; i < batchSize; i++ {
				idx := perm[start+i]
				batchXs[i] = xTrain[idx]
				batchTs[i] = tTrain[idx]
				batchLabels[i] = int(mnist.TrainLabels[idx])
			}

			// 1. 勾配（DeltaH）の計算
			// margin は元のコードにあった 0.5 を使用 [cite: 127]
			deltaHs, err := model.ComputeSignDeltas(batchXs, batchTs, 0.5, rngs)
			if err != nil {
				t.Fatalf("ComputeDeltaHs failed: %v", err)
			}

			// 2. 重みの更新
			if err := model.UpdateWeight(deltaHs, lr, rng); err != nil {
				t.Fatalf("UpdateWeight failed: %v", err)
			}

			// 3. 【追加】強化ステップ (Reinforcement Step)
            // 更新の後、確率的に重みを強化する [cite: 197]
            // if err := model.ReinforceWeight(pr, rng); err != nil {
            //     t.Fatalf("ReinforceWeight failed: %v", err)
            // }
		}

		model.SetIsTraining(false)
		// --- エポックごとの精度検証 ---
		correctCount := 0
		for i, x := range xTest {
			// PredictLogits は内部で予測ベクトルとプロトタイプの内積(スコア)を計算します [cite: 123]
			logits, err := model.PredictLogits(x, rng)
			if err != nil {
				t.Fatalf("PredictLogits failed: %v", err)
			}

			// スコアが最大のクラスを予測ラベルとする
			yMaxIdx := slicesx.Argsort(logits)[len(logits)-1]

			if float32(yMaxIdx) == tTestLabels[i] {
				correctCount++
			}
		}

		accuracy := float32(correctCount) / float32(len(xTest)) * 100
		log.Printf("Epoch %d/%d. Test Accuracy: %.2f%%", epoch+1, epochs, accuracy)
	}
}

// addBias はビット行列のリストを受け取り、各行列の末尾に n 個の「1」ビットを追加して返します。
// これにより、重み行列の最後の n 列がバイアス項として機能します。
func addBias(data []bitsx.Matrix, n int) ([]bitsx.Matrix, error) {
	if n <= 0 {
		return data, nil
	}

	newData := make([]bitsx.Matrix, len(data))
	for i, mat := range data {
		rows, cols := mat.Rows, mat.Cols
		newCols := cols + n

		// 新しいサイズの行列を作成 (初期値は全て0)
		newMat, err := bitsx.NewZerosMatrix(rows, newCols)
		if err != nil {
			return nil, fmt.Errorf("failed to create matrix with bias: %w", err)
		}

		// 元のデータをコピー
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				bit, err := mat.Bit(r, c)
				if err != nil {
					return nil, err
				}
				if bit == 1 {
					// Set は 1 をセットする (NewMatrix初期値は0)
					if err := newMat.Set(r, c); err != nil {
						return nil, err
					}
				}
			}
			// バイアスビット(定数1)を末尾にセット
			for b := 0; b < n; b++ {
				if err := newMat.Set(r, cols+b); err != nil {
					return nil, err
				}
			}
		}
		newData[i] = newMat
	}
	return newData, nil
}

// computeSimilarity は2つのビット行列の一致ビット数(PopCount of XNOR)を返します
// ※テストコード内では PredictLogits に置き換わりましたが、ロジック確認用に残す場合は以下
func computeSimilarity(a, b bitsx.Matrix) (int, error) {
	if a.Cols != b.Cols {
		return 0, fmt.Errorf("dimension mismatch: %d != %d", a.Cols, b.Cols)
	}

	count := 0
	for i := range a.Data {
		// XNOR: ビットが一致している場所が1になる
		match := ^(a.Data[i] ^ b.Data[i])
		count += bits.OnesCount64(match)
	}
	return count, nil
}