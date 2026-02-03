package layer

import (
	// "cmp"
	// "fmt"
	"log"
	"math"
	// "math/bits"
	"math/rand/v2"
	"testing"

	"github.com/sw965/crow/dataset"
	"github.com/sw965/omw/mathx/bitsx"
	"github.com/sw965/omw/mathx/randx"
	"github.com/sw965/omw/parallel"
	// "github.com/sw965/omw/slicesx"
)

// --- 補助構造体: 隠れ重み H (mlp.go から移植) ---
type H struct {
	Rows int
	Cols int
	Data []int8
}

func NewH(rows, cols int) H {
	return H{
		Rows: rows,
		Cols: cols,
		Data: make([]int8, rows*cols),
	}
}

func (h H) Index(r, c int) int {
	return (r * h.Cols) + c
}

// --- 学習用ラッパー: 1層分の状態管理 ---
type LayerState struct {
	SD *SignDot // layer.go の SignDot (Forward/Backwardロジック)
	H  H        // 隠れ重み (学習の本体)
}

// --- モデル全体 ---
type LayerModel struct {
	Layers     []*LayerState
	Sequence   Sequence // Forward/Backward用のインターフェーススライス
	Prototypes []bitsx.Matrix
}

func TestLayerMNIST(t *testing.T) {
	// return
	// --- ハイパーパラメータ (mlp_test.go と完全に一致) ---
	a := float32(0.001)
	epochs := 50
	batchSize := 1000
	parallelism := 6
	lr := float32(0.1)
	numBiasBits := 160
	outputDim := 1024
	befIterations := 20000

	// 1. データ読み込み & 前処理
	mnist, err := dataset.LoadFashionMnist()
	if err != nil {
		log.Fatal(err)
	}

	log.Println("Binarizing train images...")
	xTrainRaw, err := dataset.BinarizeImages(mnist.TrainImages, a)
	if err != nil {
		log.Fatal(err)
	}

	log.Printf("Adding %d bias bits...", numBiasBits)
	xTrain, err := addBias(xTrainRaw, numBiasBits)
	if err != nil {
		log.Fatal(err)
	}

	// 2. モデル初期化
	inputDim := 784 + numBiasBits
	layerDims := []int{inputDim, 512, 512, 512, outputDim}

	// 並列ワーカーごとの乱数生成器
	rngs := make([]*rand.Rand, parallelism)
	for i := range rngs {
		rngs[i] = randx.NewPCGFromGlobalSeed()
	}
	masterRng := rand.New(rand.NewPCG(1, 2))

	// プロトタイプ生成
	log.Println("Generating BEF prototypes...")
	prototypes, err := bitsx.NewBEFPrototypeMatrices(10, 1, outputDim, befIterations, masterRng)
	if err != nil {
		log.Fatal(err)
	}

	// ラベル変換
	tTrain, err := dataset.LabelsToTargets(mnist.TrainLabels, prototypes)
	if err != nil {
		log.Fatal(err)
	}

	// テストデータ準備
	xTestRaw, err := dataset.BinarizeImages(mnist.TestImages, a)
	if err != nil {
		log.Fatal(err)
	}
	xTest, err := addBias(xTestRaw, numBiasBits)
	if err != nil {
		log.Fatal(err)
	}
	tTestLabels := mnist.TestLabels

	// --- Layerモデルの構築 (並列ワーカー分作成) ---
	// 各ワーカーは重み(W)は共有するが、計算状態(SignDot内のdeltaやzs)は独立して持つ必要がある
	// ここでは、WのDataスライスを共有しつつ、SignDot構造体自体は別にする

	// まずマスターの重み(HとW)を作成
	masterLayers := make([]*LayerState, len(layerDims)-1)
	for i := 0; i < len(layerDims)-1; i++ {
		cols := layerDims[i]
		rows := layerDims[i+1]

		// W初期化
		w, _ := bitsx.NewRandMatrix(rows, cols, 0, masterRng)
		wt, _ := w.Transpose()

		// H初期化 (±31)
		h := NewH(rows, cols)
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				bit, _ := w.Bit(r, c)
				idx := h.Index(r, c)
				if bit == 1 {
					h.Data[idx] = 31
				} else {
					h.Data[idx] = -31
				}
			}
		}

		// マスター用のSignDot (これは学習ループでは使わず、更新の同期用)
		sd, _ := NewSignDot(w, wt)
		masterLayers[i] = &LayerState{
			SD: sd,
			H:  h,
		}
	}

	// ワーカーごとのモデルを作成
	workerModels := make([]LayerModel, parallelism)
	for p := 0; p < parallelism; p++ {
		workerLayers := make([]*LayerState, len(masterLayers))
		workerSeq := make(Sequence, len(masterLayers))

		for i, master := range masterLayers {
			// W, WTはマスターとデータを共有 (参照渡し)
			// 注意: Forward/Backward中にWは書き換わらないので安全
			// Dataスライスは共有されている
			w := master.SD.W
			wt := master.SD.WT

			sd, _ := NewSignDot(w, wt)
			sd.IsNoisy = true
			sd.Rng = rngs[p] // ワーカー固有のRNG

			state := &LayerState{
				SD: sd,
				H:  master.H, // Hも参照共有 (更新は同期時に行うのでOK)
			}
			workerLayers[i] = state
			workerSeq[i] = sd // Interfaceへのポインタ
		}

		workerModels[p] = LayerModel{
			Layers:     workerLayers,
			Sequence:   workerSeq,
			Prototypes: prototypes,
		}
	}

	log.Printf("Start training loop... (Use Layer Package)")
	numTrain := len(xTrain)
	numBatches := numTrain / batchSize

	for epoch := 0; epoch < epochs; epoch++ {
		perm := masterRng.Perm(numTrain)

		// 1エポック
		for b := 0; b < numBatches; b++ {
			start := b * batchSize

			// --- 1. Forward & Backward (Parallel) ---
			err := parallel.For(batchSize, parallelism, func(workerId, i int) error {
				idx := perm[start+i]
				x := xTrain[idx] // 入力 (1xInputDim)
				t := tTrain[idx] // 正解プロトタイプ (1xOutputDim)

				model := workerModels[workerId]

				// Forward
				y, err := model.Sequence.Forwards(x)
				if err != nil {
					return err
				}

				// --- マージン判定ロジック (mlp.go相当) ---
				// 正解クラスのスコア
				correctCounts, _ := y.Dot(t)
				correctScore := correctCounts[0]

				// 不正解クラスの最大スコア探索
				maxWrongScore := -1
				for _, proto := range model.Prototypes {
					counts, _ := y.Dot(proto)
					score := counts[0]

					if score == correctScore {
						// 正解クラス自身かどうかチェック
						checkCounts, _ := t.Dot(proto)
						if checkCounts[0] == t.Cols {
							continue // 完全一致＝正解クラス
						}
					}
					if score > maxWrongScore {
						maxWrongScore = score
					}
				}

				// マージンチェック (r=0.5)
				limit := int(float32(y.Cols) * 0.5)
				if (correctScore - maxWrongScore) >= limit {
					return nil // 十分自信があるので学習しない
				}

				// Backward (targetを渡す)
				_, err = model.Sequence.Backwards(t)
				return err
			})
			if err != nil {
				t.Fatalf("Parallel loop failed: %v", err)
			}

			// --- 2. 重みの更新 (Aggregation & Update) ---
			// 各ワーカーのSignDot.deltaを集約して、マスターのHを更新する

			// レイヤーごとに処理
			for lIdx, masterL := range masterLayers {
				h := masterL.H
				w := masterL.SD.W
				wt := masterL.SD.WT

				// 全ワーカーのDeltaを合計するためのバッファ (int16で加算すると溢れる可能性があるのでintで)
				// サイズは W.Rows * W.Cols
				totalDelta := make([]int, len(h.Data))

				for p := 0; p < parallelism; p++ {
					workerDelta := workerModels[p].Layers[lIdx].SD.delta
					for k, d := range workerDelta {
						if d != 0 {
							totalDelta[k] += int(d)
						}
					}
					// 重要: 次のバッチのためにワーカーのDeltaをゼロクリアする
					// layer.goにはメソッドがないので直接クリア
					clear(workerDelta)
				}

				// マスターHの更新
				// update prob = lr
				updateCount := 0
				for r := 0; r < h.Rows; r++ {
					for c := 0; c < h.Cols; c++ {
						idx := h.Index(r, c)
						dVal := totalDelta[idx]

						if dVal == 0 {
							continue
						}

						// 符号化 (Sign Delta)
						signD := int(0)
						if dVal > 0 {
							signD = 1
						} else if dVal < 0 {
							signD = -1
						}

						// 確率的更新
						if masterRng.Float32() > lr {
							continue
						}

						old := h.Data[idx]
						newVal := int(old) + int(2*signD)
						clipped := int8(max(math.MinInt8, min(math.MaxInt8, newVal)))
						h.Data[idx] = clipped

						// 符号反転チェック -> Wの更新
						isOldPlus := old >= 0
						isNewPlus := clipped >= 0

						if isOldPlus != isNewPlus {
							w.Toggle(r, c)
							wt.Toggle(c, r) // WTも同期
							updateCount++
						}
					}
				}
				// ワーカーはW, WTのDataスライスを共有しているので、ここでマスターがWを書き換えれば
				// 自動的に全ワーカーに反映される (ただし競合しないようこのフェーズはシングルスレッド)
			}
		}

		// --- 検証 (Accuracy) ---
		correctCount := 0
		// テストはシングルスレッドで簡易実行 (マスターモデルを使用)
		// テスト時はノイズOFF
		for _, l := range masterLayers {
			l.SD.IsNoisy = false
		}
		// Sequenceインターフェースを作る
		testSeq := make(Sequence, len(masterLayers))
		for i, l := range masterLayers {
			testSeq[i] = l.SD
		}

		for i, x := range xTest {
			y, _ := testSeq.Forwards(x)

			// PredictLogits相当
			maxScore := -999999
			predLabel := -1

			for labelIdx, proto := range prototypes {
				counts, _ := y.Dot(proto)
				if counts[0] > maxScore {
					maxScore = counts[0]
					predLabel = labelIdx
				}
			}

			if float32(predLabel) == tTestLabels[i] {
				correctCount++
			}
		}

		// 学習モードに戻す
		for _, l := range masterLayers {
			l.SD.IsNoisy = true
		}

		acc := float32(correctCount) / float32(len(xTest)) * 100
		log.Printf("Epoch %d/%d. Test Accuracy: %.2f%%", epoch+1, epochs, acc)
	}
}

// --- Helper Functions (copied from mlp_test.go) ---

func addBias(data []bitsx.Matrix, n int) ([]bitsx.Matrix, error) {
	if n <= 0 {
		return data, nil
	}
	newData := make([]bitsx.Matrix, len(data))
	for i, mat := range data {
		rows, cols := mat.Rows, mat.Cols
		newCols := cols + n
		newMat, err := bitsx.NewZerosMatrix(rows, newCols)
		if err != nil {
			return nil, err
		}
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				bit, _ := mat.Bit(r, c)
				if bit == 1 {
					newMat.Set(r, c)
				}
			}
			for b := 0; b < n; b++ {
				newMat.Set(r, cols+b)
			}
		}
		newData[i] = newMat
	}
	return newData, nil
}