package main

import (
	"errors"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"runtime"
	"slices"
	"time"

	"github.com/sw965/crow/dataset"
	"github.com/sw965/omw/mathx/bitsx"
	"github.com/sw965/omw/parallel"
)

// ---------------------------------------------------------------------------
// モデル(固定プロトタイプ + BEPバックボーン)
// ---------------------------------------------------------------------------

type model struct {
	layers []*dense
	protos bitsx.Matrices
	xRows  int
	xCols  int
}

func (m *model) outputShape() (int, int, error) {
	rows, cols := m.xRows, m.xCols
	var err error
	for i, l := range m.layers {
		rows, cols, err = l.outputShape(rows, cols)
		if err != nil {
			return 0, 0, fmt.Errorf("layer %d: %w", i, err)
		}
	}
	return rows, cols, nil
}

func (m *model) appendLayer(wRows int, rng *rand.Rand) error {
	wCols := m.xCols
	if len(m.layers) > 0 {
		_, c, err := m.outputShape()
		if err != nil {
			return err
		}
		wCols = c
	}
	l, err := newDense(wRows, wCols, rng)
	if err != nil {
		return err
	}
	m.layers = append(m.layers, l)
	return nil
}

func (m *model) forward(x *bitsx.Matrix, noiseScale float32, rng *rand.Rand) (*bitsx.Matrix, []backward, error) {
	bws := make([]backward, len(m.layers))
	var err error
	var bw backward
	for i, l := range m.layers {
		x, bw, err = l.forward(x, noiseScale, rng)
		if err != nil {
			return nil, nil, err
		}
		bws[i] = bw
	}
	return x, bws, nil
}

func (m *model) predict(x *bitsx.Matrix) (*bitsx.Matrix, error) {
	var err error
	for _, l := range m.layers {
		x, err = l.predict(x)
		if err != nil {
			return nil, err
		}
	}
	return x, nil
}

func (m *model) totalBits() int { return m.protos[0].Rows() * m.protos[0].Cols() }

func (m *model) logits(y *bitsx.Matrix) ([]int, error) {
	total := m.totalBits()
	out := make([]int, len(m.protos))
	for i, p := range m.protos {
		hd, err := y.HammingDistance(p)
		if err != nil {
			return nil, err
		}
		out[i] = total - hd
	}
	return out, nil
}

func (m *model) accuracy(xs bitsx.Matrices, labels []int, p int) (float64, error) {
	counts := make([]int, p)
	err := parallel.For(len(xs), p, func(workerID, i int) error {
		y, err := m.predict(xs[i])
		if err != nil {
			return err
		}
		lg, err := m.logits(y)
		if err != nil {
			return err
		}
		best, bestVal := 0, lg[0]
		for c, v := range lg {
			if v > bestVal {
				bestVal, best = v, c
			}
		}
		if best == labels[i] {
			counts[workerID]++
		}
		return nil
	})
	if err != nil {
		return 0, err
	}
	total := 0
	for _, c := range counts {
		total += c
	}
	return float64(total) / float64(len(xs)), nil
}

// ---------------------------------------------------------------------------
// SWA(隠れ重み H の走算平均)
// ---------------------------------------------------------------------------

// swa は各層の隠れ重み H の総和と、加えたモデル数を保持する。
// 平均は sum/count の整数除算で求め、可視重みは W = sign(平均H) で作り直す。
// 全て整数演算で、浮動小数点は使わない。
type swa struct {
	sums  [][]int32
	count int
	// vote が真なら、H の平均ではなく可視重み W の多数決で集約する。
	// 二値重みに対する素直な「平均」の別解釈。
	vote  bool
	votes [][]int32
}

func newSWA(m *model) *swa {
	s := &swa{sums: make([][]int32, len(m.layers)), votes: make([][]int32, len(m.layers))}
	for i, l := range m.layers {
		s.sums[i] = make([]int32, len(l.h))
		s.votes[i] = make([]int32, len(l.h))
	}
	return s
}

func (s *swa) add(m *model) error {
	if len(m.layers) != len(s.sums) {
		return errors.New("層数が一致しません")
	}
	for i, l := range m.layers {
		if len(l.h) != len(s.sums[i]) {
			return fmt.Errorf("layer %d: 重み数が一致しません", i)
		}
		for j, v := range l.h {
			s.sums[i][j] += int32(v)
			// 可視重みの ±1 表現を票として数える
			if v >= 0 {
				s.votes[i][j]++
			} else {
				s.votes[i][j]--
			}
		}
	}
	s.count++
	return nil
}

// writeAverage は平均した H を dst へ書き込み、W と WT を作り直す。
func (s *swa) writeAverage(dst *model) error {
	if s.count == 0 {
		return errors.New("平均する対象がありません")
	}
	for i, l := range dst.layers {
		for j := range l.h {
			sum := s.sums[i][j]
			var v int32
			if s.vote {
				// 多数決。同点(票が0、モデル数が偶数のとき起きる)は
				// H の総和の符号で決める。単純に +1 へ倒すと、
				// 偶数個のときだけ大量の重みが誤って +1 になってしまう。
				tally := s.votes[i][j]
				if tally == 0 {
					tally = sum
				}
				if tally >= 0 {
					v = hInitAbs
				} else {
					v = -hInitAbs
				}
			} else {
				// 平均。整数除算は0方向へ切り捨てるため、平均の絶対値が1未満だと
				// 0 になり、sign(0)=+1 の規約で負の重みが +1 に化ける。
				// 符号は総和から取り、大きさだけ平均を使う。
				v = sum / int32(s.count)
				if v == 0 && sum != 0 {
					if sum > 0 {
						v = 1
					} else {
						v = -1
					}
				}
			}
			l.h[j] = int8(max(math.MinInt8, min(int(v), math.MaxInt8)))
		}
		if err := rebuildVisible(l); err != nil {
			return err
		}
	}
	return nil
}

// rebuildVisible は H の符号から可視重み W と転置 WT を作り直す。
func rebuildVisible(l *dense) error {
	if err := l.w.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
		hWord := l.h[ctx.GlobalStart:ctx.GlobalEnd]
		var word uint64
		if err := ctx.ScanBits(func(i, col, colT int) error {
			if hWord[i] >= 0 {
				word |= 1 << uint64(i)
			}
			return nil
		}); err != nil {
			return err
		}
		return l.w.SetWord(ctx.WordIndex, word)
	}); err != nil {
		return err
	}
	wt, err := l.w.Transpose()
	if err != nil {
		return err
	}
	l.wt = wt
	return nil
}

func cloneModel(src *model) *model {
	dst := &model{xRows: src.xRows, xCols: src.xCols, protos: src.protos}
	for _, l := range src.layers {
		nl := &dense{
			w: l.w.Clone(), wt: l.wt.Clone(),
			h:        make([]int8, len(l.h)),
			gateBase: l.gateBase, noiseStd: l.noiseStd,
			gateScale: l.gateScale, groupSize: l.groupSize,
		}
		copy(nl.h, l.h)
		dst.layers = append(dst.layers, nl)
	}
	return dst
}

// weightAgreement は2つのモデルの可視重みの一致率を返す。
// 独立に学習したモデル同士がどれだけ対応していないかの診断に使う。
// 無関係なら 0.5 前後になる。
func weightAgreement(a, b *model) (float64, error) {
	var same, total int
	for i := range a.layers {
		hd, err := a.layers[i].w.HammingDistance(b.layers[i].w)
		if err != nil {
			return 0, err
		}
		n := a.layers[i].w.Rows() * a.layers[i].w.Cols()
		same += n - hd
		total += n
	}
	return float64(same) / float64(total), nil
}

// ---------------------------------------------------------------------------
// 学習
// ---------------------------------------------------------------------------

type trainer struct {
	model         *model
	miniBatchSize int
	lr            float32
	margin        float32
	noiseScale    float32

	workerRNGs []*rand.Rand
	shuffleRNG *rand.Rand
	updateRNG  *rand.Rand

	workerDeltas [][]*delta
	aggregated   []*delta

	// targets が非nilなら、プロトタイプの代わりにサンプルごとの目標符号を使う。
	// 蒸留で「教師の最終活性」を目標にするための入口。
	targets bitsx.Matrices
}

func newTrainer(m *model, p int, seed uint64) (*trainer, error) {
	if p <= 0 {
		return nil, errors.New("ワーカー数は1以上であるべき")
	}
	rngs := make([]*rand.Rand, p)
	for i := range rngs {
		rngs[i] = rand.New(rand.NewPCG(seed, 0x9E3779B97F4A7C15+uint64(i)))
	}
	wd := make([][]*delta, p)
	for i := range p {
		ds := make([]*delta, len(m.layers))
		for l, layer := range m.layers {
			ds[l] = layer.newDelta()
		}
		wd[i] = ds
	}
	agg := make([]*delta, len(m.layers))
	for l, layer := range m.layers {
		agg[l] = layer.newDelta()
	}
	return &trainer{
		model: m, miniBatchSize: 1024, lr: 0.1, margin: 0.5, noiseScale: 0.5,
		workerRNGs:   rngs,
		shuffleRNG:   rand.New(rand.NewPCG(seed, 0xD1B54A32D192ED03)),
		updateRNG:    rand.New(rand.NewPCG(seed, 0xA24BAED4963EE407)),
		workerDeltas: wd, aggregated: agg,
	}, nil
}

// satisfiesUpdateCriterion は ../train.go と同じ判定を、既に求めたロジットに対して行う。
func satisfiesUpdateCriterion(logits []int, label int, totalBits int, margin float32) bool {
	marginBits := int(float32(totalBits) * margin / 2)
	for c, v := range logits {
		if c == label {
			continue
		}
		if logits[label]-v < marginBits {
			return true
		}
	}
	return false
}

func (t *trainer) trainEpoch(xs bitsx.Matrices, labels []int) error {
	n := len(xs)
	batch := min(t.miniBatchSize, n)
	perm := t.shuffleRNG.Perm(n)
	totalBits := t.model.totalBits()

	for start := 0; start < n; start += batch {
		end := min(start+batch, n)
		idxs := perm[start:end]

		for _, ds := range t.workerDeltas {
			for _, d := range ds {
				d.clear()
			}
		}

		p := len(t.workerRNGs)
		if err := parallel.For(len(idxs), p, func(workerID, i int) error {
			rng := t.workerRNGs[workerID]
			x := xs[idxs[i]]
			label := labels[idxs[i]]

			y, bws, err := t.model.forward(x, t.noiseScale, rng)
			if err != nil {
				return err
			}
			lg, err := t.model.logits(y)
			if err != nil {
				return err
			}
			if !satisfiesUpdateCriterion(lg, label, totalBits, t.margin) {
				return nil
			}
			target := t.model.protos[label]
			if t.targets != nil {
				target = t.targets[idxs[i]]
			}
			for li := range slices.Backward(bws) {
				target, err = bws[li](target, nil, t.workerDeltas[workerID][li])
				if err != nil {
					return err
				}
			}
			return nil
		}); err != nil {
			return err
		}

		for _, d := range t.aggregated {
			d.clear()
		}
		for _, ds := range t.workerDeltas {
			for li, d := range ds {
				t.aggregated[li].add(d)
			}
		}
		for _, d := range t.aggregated {
			d.sign()
		}
		for li, layer := range t.model.layers {
			if err := layer.update(t.aggregated[li], t.lr, t.updateRNG); err != nil {
				return err
			}
		}
	}
	return nil
}

// ---------------------------------------------------------------------------

func splitTrainValidation(xs bitsx.Matrices, labels []int, valRatio float64, rng *rand.Rand) (
	bitsx.Matrices, []int, bitsx.Matrices, []int, error) {
	n := len(xs)
	valN := int(float64(n) * valRatio)
	if valN <= 0 || valN >= n {
		return nil, nil, nil, nil, fmt.Errorf("valRatio %g では分割できません (n = %d)", valRatio, n)
	}
	perm := rng.Perm(n)
	valXs := make(bitsx.Matrices, 0, valN)
	valLabels := make([]int, 0, valN)
	trXs := make(bitsx.Matrices, 0, n-valN)
	trLabels := make([]int, 0, n-valN)
	for i, idx := range perm {
		if i < valN {
			valXs = append(valXs, xs[idx])
			valLabels = append(valLabels, labels[idx])
		} else {
			trXs = append(trXs, xs[idx])
			trLabels = append(trLabels, labels[idx])
		}
	}
	return trXs, trLabels, valXs, valLabels, nil
}

func main() {
	var (
		dsName     = flag.String("dataset", "mnist", "mnist | fashion")
		mode       = flag.String("mode", "independent", "independent | continued | trajectory | ensemble | distill")
		distillTgt = flag.String("distill", "hard", "distill時の目標: hard(アンサンブルの予測クラス) | vote(教師の最終活性の多数決)")
		numModels  = flag.Int("models", 5, "平均するモデル数")
		epochs     = flag.Int("epochs", 20, "モデル1つあたり(スナップショット間隔)のエポック数")
		warmup     = flag.Int("warmup", 0, "trajectory: 平均を取り始める前に学習するエポック数")
		aggMode    = flag.String("agg", "mean", "mean(Hの平均) | vote(Wの多数決)")
		h1         = flag.Int("h1", 512, "隠れ層1の幅(0で省略)")
		h2         = flag.Int("h2", 1024, "隠れ層2(最終)の幅")
		lr         = flag.Float64("lr", 0.1, "確率的学習率")
		margin     = flag.Float64("margin", 0.5, "更新判定のマージン")
		groupSize  = flag.Int("gsize", 4, "GroupSize")
		gateScale  = flag.Float64("gate", 1.0, "GateDropThresholdScale")
		noiseScale = flag.Float64("noise", 0.5, "NoiseStdScale")
		batch      = flag.Int("batch", 1024, "ミニバッチサイズ")
		valRatio   = flag.Float64("valratio", 0.1, "検証に分ける割合")
		seed       = flag.Uint64("seed", 1, "乱数シード")
		threads    = flag.Int("threads", 0, "ワーカー数 (0 = NumCPU)")
	)
	flag.Parse()

	workers := *threads
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	switch *mode {
	case "independent", "continued", "trajectory", "ensemble", "distill":
	default:
		log.Fatalf("modeが不正: %s", *mode)
	}
	if *distillTgt != "hard" && *distillTgt != "vote" {
		log.Fatalf("distillが不正: %s", *distillTgt)
	}

	var ds dataset.Binary[int]
	var err error
	if *dsName == "fashion" {
		ds, err = dataset.LoadFashionMNIST(nil)
	} else {
		ds, err = dataset.LoadMNIST(nil)
	}
	if err != nil {
		log.Fatal(err)
	}

	splitRNG := rand.New(rand.NewPCG(*seed, 0xC2B2AE3D27D4EB4F))
	trainXs, trainLabels, valXs, valLabels, err := splitTrainValidation(
		ds.TrainInputs, ds.TrainLabels, *valRatio, splitRNG)
	if err != nil {
		log.Fatal(err)
	}

	newModel := func(initSeed uint64) (*model, error) {
		rng := rand.New(rand.NewPCG(initSeed, initSeed+1))
		m := &model{xRows: 1, xCols: 784}
		if *h1 > 0 {
			if err := m.appendLayer(*h1, rng); err != nil {
				return nil, err
			}
		}
		if err := m.appendLayer(*h2, rng); err != nil {
			return nil, err
		}
		for _, l := range m.layers {
			l.groupSize = *groupSize
			l.gateScale = float32(*gateScale)
		}
		return m, nil
	}

	base, err := newModel(*seed)
	if err != nil {
		log.Fatal(err)
	}
	yRows, yCols, err := base.outputShape()
	if err != nil {
		log.Fatal(err)
	}

	// プロトタイプは全モデルで共有する(違うと目標符号が変わり、平均する意味が無くなる)
	protoRNG := rand.New(rand.NewPCG(*seed, 0xBF58476D1CE4E5B9))
	totalBits := numClasses * yRows * yCols
	iters := 10 * int(float64(totalBits)*math.Log(float64(totalBits)))
	protos, err := bitsx.NewETFMatrices(numClasses, yRows, yCols, iters, protoRNG)
	if err != nil {
		log.Fatal(err)
	}
	base.protos = protos

	fmt.Printf("dataset=%s mode=%s agg=%s distill=%s warmup=%d models=%d epochs=%d 構成=784->%d->%d lr=%g margin=%g gsize=%d gate=%g noise=%g seed=%d train=%d val=%d test=%d\n",
		*dsName, *mode, *aggMode, *distillTgt, *warmup, *numModels, *epochs, *h1, *h2, *lr, *margin, *groupSize,
		*gateScale, *noiseScale, *seed, len(trainXs), len(valXs), len(ds.TestInputs))

	if *aggMode != "mean" && *aggMode != "vote" {
		log.Fatalf("aggが不正: %s", *aggMode)
	}
	trainOne := func(m *model, trainSeed uint64) error {
		tr, err := newTrainer(m, workers, trainSeed)
		if err != nil {
			return err
		}
		tr.miniBatchSize = *batch
		tr.lr = float32(*lr)
		tr.margin = float32(*margin)
		tr.noiseScale = float32(*noiseScale)
		for range *epochs {
			if err := tr.trainEpoch(trainXs, trainLabels); err != nil {
				return err
			}
		}
		return nil
	}

	if *mode == "ensemble" || *mode == "distill" {
		ens := &ensemble{}
		for k := 1; k <= *numModels; k++ {
			t0 := time.Now()
			m, err := newModel(*seed + uint64(k)*7919)
			if err != nil {
				log.Fatal(err)
			}
			m.protos = protos
			if err := trainOne(m, *seed+uint64(k)*104729); err != nil {
				log.Fatal(err)
			}
			soloVal, err := m.accuracy(valXs, valLabels, workers)
			if err != nil {
				log.Fatal(err)
			}
			soloTest, err := m.accuracy(ds.TestInputs, ds.TestLabels, workers)
			if err != nil {
				log.Fatal(err)
			}
			ens.add(m)
			ensVal, err := ens.accuracy(valXs, valLabels, workers)
			if err != nil {
				log.Fatal(err)
			}
			ensTest, err := ens.accuracy(ds.TestInputs, ds.TestLabels, workers)
			if err != nil {
				log.Fatal(err)
			}
			fmt.Printf("教師%d: 単体 val %.4f / test %.4f | アンサンブル(%d個) val %.4f / test %.4f | %.1fs\n",
				k, soloVal, soloTest, k, ensVal, ensTest, time.Since(t0).Seconds())
		}

		if *mode == "ensemble" {
			return
		}

		// --- 蒸留 ---
		t0 := time.Now()
		pseudo, err := ens.predictLabels(trainXs, workers)
		if err != nil {
			log.Fatal(err)
		}
		agreeLabel := 0
		for i, v := range pseudo {
			if v == trainLabels[i] {
				agreeLabel++
			}
		}
		student, err := newModel(*seed + 999331)
		if err != nil {
			log.Fatal(err)
		}
		student.protos = protos
		st, err := newTrainer(student, workers, *seed+555557)
		if err != nil {
			log.Fatal(err)
		}
		st.miniBatchSize = *batch
		st.lr = float32(*lr)
		st.margin = float32(*margin)
		st.noiseScale = float32(*noiseScale)

		if *distillTgt == "vote" {
			codes, err := ens.voteCodes(trainXs, workers)
			if err != nil {
				log.Fatal(err)
			}
			agree, err := codeAgreement(codes, trainLabels, protos)
			if err != nil {
				log.Fatal(err)
			}
			st.targets = codes
			fmt.Printf("蒸留の目標: 教師の最終活性の多数決符号。正解プロトタイプとの一致率 %.3f\n", agree)
		} else {
			fmt.Print("蒸留の目標: アンサンブルの予測クラスのプロトタイプ\n")
		}
		fmt.Printf("疑似ラベルと正解ラベルの一致率(学習集合) %.4f | 準備 %.1fs\n",
			float64(agreeLabel)/float64(len(pseudo)), time.Since(t0).Seconds())

		bestVal, testAtBest, bestEpoch := -1.0, 0.0, 0
		for e := 1; e <= *epochs; e++ {
			te := time.Now()
			if err := st.trainEpoch(trainXs, pseudo); err != nil {
				log.Fatal(err)
			}
			v, err := student.accuracy(valXs, valLabels, workers)
			if err != nil {
				log.Fatal(err)
			}
			tst, err := student.accuracy(ds.TestInputs, ds.TestLabels, workers)
			if err != nil {
				log.Fatal(err)
			}
			if v > bestVal {
				bestVal, testAtBest, bestEpoch = v, tst, e
			}
			fmt.Printf("蒸留 epoch %d: val %.4f / test %.4f (best val %.4f @%d) %.1fs\n",
				e, v, tst, bestVal, bestEpoch, time.Since(te).Seconds())
		}
		fmt.Printf("DISTILLED val %.4f @epoch %d / TEST %.4f\n", bestVal, bestEpoch, testAtBest)
		return
	}

	acc := newSWA(base)
	acc.vote = *aggMode == "vote"
	var first *model

	// trajectory は1本の学習を続け、Eエポックごとにスナップショットを取る
	var runningTrainer *trainer
	if *mode == "trajectory" {
		runningTrainer, err = newTrainer(base, workers, *seed)
		if err != nil {
			log.Fatal(err)
		}
		runningTrainer.miniBatchSize = *batch
		runningTrainer.lr = float32(*lr)
		runningTrainer.margin = float32(*margin)
		runningTrainer.noiseScale = float32(*noiseScale)
		// 平均を取り始める前に、良い領域まで学習を進めておく(標準的なSWAの手順)
		for range *warmup {
			if err := runningTrainer.trainEpoch(trainXs, trainLabels); err != nil {
				log.Fatal(err)
			}
		}
	}

	bestAvgVal, bestAvgTest, bestAt := -1.0, 0.0, 0

	for k := 1; k <= *numModels; k++ {
		t0 := time.Now()
		var m *model

		switch *mode {
		case "independent":
			m, err = newModel(*seed + uint64(k)*7919)
			if err != nil {
				log.Fatal(err)
			}
			m.protos = protos
			if err := trainOne(m, *seed+uint64(k)*104729); err != nil {
				log.Fatal(err)
			}
		case "continued":
			if k == 1 {
				m = base
			} else {
				// 現在の平均モデルから学習を再開する
				m = cloneModel(base)
				if err := acc.writeAverage(m); err != nil {
					log.Fatal(err)
				}
			}
			if err := trainOne(m, *seed+uint64(k)*104729); err != nil {
				log.Fatal(err)
			}
		case "trajectory":
			for range *epochs {
				if err := runningTrainer.trainEpoch(trainXs, trainLabels); err != nil {
					log.Fatal(err)
				}
			}
			m = cloneModel(base)
		}

		soloVal, err := m.accuracy(valXs, valLabels, workers)
		if err != nil {
			log.Fatal(err)
		}
		soloTest, err := m.accuracy(ds.TestInputs, ds.TestLabels, workers)
		if err != nil {
			log.Fatal(err)
		}

		agree := math.NaN()
		if first == nil {
			first = cloneModel(m)
		} else {
			agree, err = weightAgreement(first, m)
			if err != nil {
				log.Fatal(err)
			}
		}

		if err := acc.add(m); err != nil {
			log.Fatal(err)
		}
		avg := cloneModel(base)
		if err := acc.writeAverage(avg); err != nil {
			log.Fatal(err)
		}
		avgVal, err := avg.accuracy(valXs, valLabels, workers)
		if err != nil {
			log.Fatal(err)
		}
		avgTest, err := avg.accuracy(ds.TestInputs, ds.TestLabels, workers)
		if err != nil {
			log.Fatal(err)
		}
		if avgVal > bestAvgVal {
			bestAvgVal, bestAvgTest, bestAt = avgVal, avgTest, k
		}

		fmt.Printf("モデル%d: 単体 val %.4f / test %.4f | SWA(%d個) val %.4f / test %.4f | 1個目との重み一致率 %.3f | %.1fs\n",
			k, soloVal, soloTest, k, avgVal, avgTest, agree, time.Since(t0).Seconds())
	}

	fmt.Printf("BEST_SWA val %.4f @%d個 / TEST %.4f\n", bestAvgVal, bestAt, bestAvgTest)
}
