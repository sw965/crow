package binary

import (
	"fmt"

	"github.com/sw965/omw/mathx/bitsx"
	"github.com/sw965/omw/parallel"
)

type RankPairX struct {
	High *bitsx.Matrix
	Low  *bitsx.Matrix
}

type RankPairXs []RankPairX

type RankPairLabel struct {
	High int
	Low  int
}

func (m *Model) PairwiseLabels(pairX RankPairX, margin float32) (RankPairLabel, bool, error) {
	if margin > 1.0 || margin < 0.0 {
		return RankPairLabel{}, false, fmt.Errorf("marginが不正: margin = %v: 0.0 <= margin <= 1.0 であるべき", margin)
	}

	if err := m.validateAscendingValues(); err != nil {
		return RankPairLabel{}, false, err
	}

	highY, err := m.PredictValue(pairX.High)
	if err != nil {
		return RankPairLabel{}, false, err
	}

	lowY, err := m.PredictValue(pairX.Low)
	if err != nil {
		return RankPairLabel{}, false, err
	}

	// Valuesは昇順である事が保証されている為、両端の差が値域になる
	valueRange := m.Values[len(m.Values)-1] - m.Values[0]
	absMargin := margin * valueRange

	if highY-lowY >= absMargin {
		return RankPairLabel{}, false, nil
	}

	n := len(m.Prototypes)
	currentHighLabel := m.ValueToLabel(highY)
	nextHighLabel := currentHighLabel + 1
	// 上限を超えたらクリップする
	if nextHighLabel >= n {
		nextHighLabel = n - 1
	}

	currentLowLabel := m.ValueToLabel(lowY)
	nextLowLabel := currentLowLabel - 1
	// 下限を下回ったらクリップする
	if nextLowLabel < 0 {
		nextLowLabel = 0
	}

	label := RankPairLabel{High: nextHighLabel, Low: nextLowLabel}
	return label, true, nil
}

func (m *Model) PairwiseAccuracy(pairXs RankPairXs, p int) (float32, error) {
	n := len(pairXs)
	if n == 0 {
		return 0.0, fmt.Errorf("pairXsが空です")
	}

	correctCounts := make([]int, p)

	err := parallel.For(n, p, func(workerID, idx int) error {
		pairX := pairXs[idx]

		yHigh, err := m.PredictValue(pairX.High)
		if err != nil {
			return err
		}

		yLow, err := m.PredictValue(pairX.Low)
		if err != nil {
			return err
		}

		if yHigh > yLow {
			correctCounts[workerID]++
		}
		return nil
	})

	if err != nil {
		return 0.0, err
	}

	var totalCorrect int
	for _, c := range correctCounts {
		totalCorrect += c
	}
	return float32(totalCorrect) / float32(n), nil
}

func (t *Trainer) TrainPairwise(pairXs RankPairXs) error {
	var xs bitsx.Matrices
	var labels []int

	// 並列化対象
	for _, pairX := range pairXs {
		pairLabel, shouldUpdate, err := t.model.PairwiseLabels(pairX, t.Margin)
		if err != nil {
			return err
		}

		if shouldUpdate {
			xs = append(xs, pairX.High, pairX.Low)
			labels = append(labels, pairLabel.High, pairLabel.Low)
		}
	}

	if len(xs) > 0 {
		err := t.Train(xs, labels)
		if err != nil {
			return err
		}
	}
	return nil
}
