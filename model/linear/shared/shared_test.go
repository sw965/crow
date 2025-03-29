package shared_test

import (
	"testing"
	"fmt"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/crow/model/linear/shared"
	omwrand "github.com/sw965/omw/math/rand"
)

func TestSupervisedLearning(t *testing.T) {
	// --- テスト用データ ---
	// 各サンプルは、入力ニューロンを表すマップ（キー "w1"）で表現
	inputs := shared.Inputs[string]{
		shared.Input[string]{{"w1": 0.0}},
		shared.Input[string]{{"w1": 1.0}},
	}
	// 目標値：x=0 のとき 0.1、x=1 のとき 0.9（各出力は長さ1のスライス）
	labels := tensor.D2{
		{0.1},
		{0.9},
	}

	// --- モデルの初期化 ---
	// 重み "w1" は初期値 1.0、バイアス "b1" は初期値 0.0
	param := shared.NewInitParameter[string, string]([]string{"w1"}, []string{"b1"})
	model := shared.Model[string, string]{
		Parameter: param,
		BiasKeys:[]string{"b1"},
	}
	// 活性化関数に Sigmoid、損失関数に二乗和誤差を設定
	model.SetSigmoid()
	model.SetSumSquaredError()

	// 訓練前の損失を計算
	initialLoss, err := model.MeanLoss(inputs, labels)
	if err != nil {
		t.Fatalf("初期損失の計算エラー: %v", err)
	}

	// --- ミニバッチ学習の設定 ---
	// ミニバッチサイズ1、エポック数1000（各サンプルにつき1000回更新）
	teacher := shared.MiniBatchTeacher[string, string]{
		Inputs:        inputs,
		Labels:        labels,
		MiniBatchSize: 1,
		Epoch:         10000,
		Optimizer:     (&shared.SGD[string, string]{LearningRate: 0.1}).Optimizer,
		Parallel:      1,
	}

	r := omwrand.NewMt19937()
	if err := teacher.Teach(&model, r); err != nil {
		t.Fatalf("学習中のエラー: %v", err)
	}

	// 訓練後の損失を計算
	finalLoss, err := model.MeanLoss(inputs, labels)
	if err != nil {
		t.Fatalf("学習後の損失計算エラー: %v", err)
	}

	t.Logf("初期損失: %f, 学習後損失: %f", initialLoss, finalLoss)
	if finalLoss >= initialLoss {
		t.Errorf("学習後の損失 (%f) が初期損失 (%f) 以上です。", finalLoss, initialLoss)
	}

	// 各サンプルに対する予測をログ出力
	for i, inp := range inputs {
		pred := model.Predict(inp)
		t.Logf("サンプル %d: 予測: %v, 目標: %v", i, pred, labels[i])
	}
}

// TestSPSA は、SPSA による勾配推定の動作をテストします。
func TestSPSA(t *testing.T) {
	// --- テスト用データ ---
	inputs := shared.Inputs[string]{
		shared.Input[string]{{"w1": 0.0}},
		shared.Input[string]{{"w1": 1.0}},
	}
	labels := tensor.D2{
		tensor.D1{0.1},
		tensor.D1{0.9},
	}
	biasKeys := []string{"b1"}

	// --- モデルの初期化 ---
	param := shared.NewInitParameter[string, string]([]string{"w1"}, biasKeys)
	model := shared.Model[string, string]{
		Parameter: param,
		BiasKeys:biasKeys,
	}
	model.SetSigmoid()
	model.SetSumSquaredError()
	// SPSA では、データ全体に対する損失関数を利用
	model.LossFunc = func(m *shared.Model[string, string]) (float64, error) {
		return m.MeanLoss(inputs, labels)
	}

	r := omwrand.NewMt19937()
	c := 0.01 // 摂動の大きさ

	grad, err := model.EstimateGradBySPSA(c, r)
	if err != nil {
		t.Fatalf("SPSA による勾配推定エラー: %v", err)
	}

	t.Logf("SPSA 推定勾配 - 重み: %v, バイアス: %v", grad.Weight, grad.Bias)

	// 勾配が正しく計算されているか、キーの存在をチェック
	if _, ok := grad.Weight["w1"]; !ok {
		t.Error("w1 の勾配が計算されていません。")
	}
	if _, ok := grad.Bias["b1"]; !ok {
		t.Error("b1 の勾配が計算されていません。")
	}
}

func TestLearning(t *testing.T) {
	r := omwrand.NewMt19937()

	// --- モデル初期化 ---
	// 入力2次元、2クラス分類タスクを想定。
	// 重みキーは "w1" と "w2"、バイアスキーは "b1" と "b2" を使用。
	param := shared.NewInitParameter[string, string]([]string{"w1", "w2"}, []string{"b1", "b2"})
	model := shared.Model[string, string]{
		Parameter: param,
		// LinearSum の計算では、各入力サンプルの i 番目のマップに対応して BiasKeys[i] を使用するため、
		// サンプル毎に 2 つのマップが必要です。
		BiasKeys: []string{"b1", "b2"},
	}
	// 分類タスク用に Softmax とクロスエントロピー誤差を設定
	model.SetSoftmaxForCrossEntropy()
	model.SetCrossEntropyError()

	// --- 学習用データセット生成 ---
	// 各サンプルは、2つの入力マップから構成される
	// 1つ目のマップはキー "w1" に対応する入力値 a、
	// 2つ目のマップはキー "w2" に対応する入力値 b。
	// ターゲットは、 a > b なら [1, 0]、それ以外なら [0, 1] とする。
	sampleN := 10000
	inputs := make(shared.Inputs[string], sampleN)
	labels := make(tensor.D2, sampleN)

	for i := 0; i < sampleN; i++ {
		a := r.Float64()
		b := r.Float64()
		sample := shared.Input[string]{
			{"w1": a},
			{"w2": b},
		}
		inputs[i] = sample
		if a > b {
			labels[i] = tensor.D1{1.0, 0.0}
		} else {
			labels[i] = tensor.D1{0.0, 1.0}
		}
	}

	// --- ミニバッチ学習 ---
	// Momentum オプティマイザを利用（学習率は小さめに設定）
	momentum := shared.NewMomentum[string, string](&model)
	momentum.LearningRate = 0.00001

	teacher := &shared.MiniBatchTeacher[string, string]{
		Inputs:        inputs,
		Labels:        labels,
		MiniBatchSize: 32,
		Epoch:         1000,
		Optimizer:     momentum.Optimizer,
		Parallel:      4,
	}

	if err := teacher.Teach(&model, r); err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// --- 正解率評価 ---
	acc, err := model.Accuracy(inputs, labels)
	if err != nil {
		t.Fatalf("Accuracy evaluation failed: %v", err)
	}
	t.Logf("Training accuracy: %.2f%%", acc*100)

	if acc < 0.9 {
		t.Fatalf("Training accuracy too low: %.2f%%", acc*100)
	}

	fmt.Println(model.Parameter)
}

// TestSPSA2 は、SPSA による勾配推定が解析的勾配と十分近いかを確認するテストです。
func TestSPSA2(t *testing.T) {
	r := omwrand.NewMt19937()

	// --- モデル初期化 ---
	// 2ニューロンモデルとして、重みとバイアスはマップで管理する
	param := shared.NewInitParameter[string, string]([]string{"w1", "w2"}, []string{"b1", "b2"})
	// 特定の値に設定
	param.Weight["w1"] = 1.0
	param.Weight["w2"] = 2.0
	param.Bias["b1"] = 3.0
	param.Bias["b2"] = 4.0

	model := shared.Model[string, string]{
		Parameter: param,
		// SPSA では入力は使用しないため、BiasKeys は順番に "b1", "b2" としておく
		BiasKeys: []string{"b1", "b2"},
	}

	// 損失関数として、二乗誤差を用いる
	// ターゲットは、各パラメータが以下の値になるように設定
	targetWeight := map[string]float64{
		"w1": 0.5,
		"w2": 1.5,
	}
	targetBias := map[string]float64{
		"b1": 2.5,
		"b2": 3.5,
	}
	model.LossFunc = func(m *shared.Model[string, string]) (float64, error) {
		loss := 0.0
		for k, w := range m.Parameter.Weight {
			diff := w - targetWeight[k]
			loss += 0.5 * diff * diff
		}
		for k, b := range m.Parameter.Bias {
			diff := b - targetBias[k]
			loss += 0.5 * diff * diff
		}
		return loss, nil
	}

	// 解析的な勾配は、パラメータとターゲットの差
	trueGradWeight := map[string]float64{
		"w1": 1.0 - 0.5, // 0.5
		"w2": 2.0 - 1.5, // 0.5
	}
	trueGradBias := map[string]float64{
		"b1": 3.0 - 2.5, // 0.5
		"b2": 4.0 - 3.5, // 0.5
	}

	// SPSA による勾配推定を多数回行い平均値を算出
	iterations := 10000
	sumGradW := map[string]float64{
		"w1": 0.0,
		"w2": 0.0,
	}
	sumGradB := map[string]float64{
		"b1": 0.0,
		"b2": 0.0,
	}

	for i := 0; i < iterations; i++ {
		grad, err := model.EstimateGradBySPSA(1e-3, r)
		if err != nil {
			t.Fatalf("EstimateGradBySPSA failed: %v", err)
		}
		for k, v := range grad.Weight {
			sumGradW[k] += v
		}
		for k, v := range grad.Bias {
			sumGradB[k] += v
		}
	}

	avgGradW := make(map[string]float64)
	avgGradB := make(map[string]float64)
	for k, v := range sumGradW {
		avgGradW[k] = v / float64(iterations)
	}
	for k, v := range sumGradB {
		avgGradB[k] = v / float64(iterations)
	}

	t.Logf("Analytical Grad Weight: %v", trueGradWeight)
	t.Logf("SPSA Estimated Grad Weight (avg): %v", avgGradW)
	t.Logf("Analytical Grad Bias: %v", trueGradBias)
	t.Logf("SPSA Estimated Grad Bias (avg): %v", avgGradB)

	// 許容誤差 tol 内に収まっているかチェック
	tol := 1e-2
	for k, v := range avgGradW {
		if diff := v - trueGradWeight[k]; diff < -tol || diff > tol {
			t.Fatalf("Weight gradient mismatch for %s: estimated %v, analytical %v", k, v, trueGradWeight[k])
		}
	}
	for k, v := range avgGradB {
		if diff := v - trueGradBias[k]; diff < -tol || diff > tol {
			t.Fatalf("Bias gradient mismatch for %s: estimated %v, analytical %v", k, v, trueGradBias[k])
		}
	}
}