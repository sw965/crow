package bep

import (
	"fmt"

	"github.com/sw965/omw/mathx/bitsx"
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
		return RankPairLabel{}, false, fmt.Errorf("marginが不正: margin = %g: 0.0 <= margin <= 1.0 であるべき", margin)
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
	nextHighLabel := min(m.ValueToLabel(highY)+1, n-1)
	nextLowLabel := max(m.ValueToLabel(lowY)-1, 0)

	label := RankPairLabel{High: nextHighLabel, Low: nextLowLabel}
	return label, true, nil
}

func (m *Model) PairwiseAccuracy(pairXs RankPairXs, p int) (float32, error) {
	n := len(pairXs)
	if err := validateEvaluationSize("pairXs", n, p); err != nil {
		return 0.0, err
	}

	totalCorrect, err := sumParallel(n, p, func(idx int) (int, error) {
		pairX := pairXs[idx]

		yHigh, err := m.PredictValue(pairX.High)
		if err != nil {
			return 0, err
		}

		yLow, err := m.PredictValue(pairX.Low)
		if err != nil {
			return 0, err
		}

		if yHigh > yLow {
			return 1, nil
		}
		return 0, nil
	})
	if err != nil {
		return 0.0, err
	}
	return float32(totalCorrect) / float32(n), nil
}

func (t *Trainer) TrainPairwise(pairXs RankPairXs) error {
	var xs bitsx.Matrices
	var labels []int

	// TODO 並列化対象
	for _, pairX := range pairXs {
		pairLabel, shouldUpdate, err := t.model.PairwiseLabels(pairX, t.PairwiseMargin)
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
