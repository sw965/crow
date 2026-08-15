package dataset

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/sw965/omw/mathx/bitsx"
)

type Label interface {
	int | float32
}

type Binary[L Label] struct {
	TrainInputs bitsx.Matrices
	TrainLabels []L
	TestInputs  bitsx.Matrices
	TestLabels  []L
}

func loadBinary[L Label](baseURL, dataDir, trainInputsFileName, trainLabelsFileName, testInputsFileName, testLabelsFileName string, logf LogFunc) (Binary[L], error) {
	if logf == nil {
		logf = func(string, ...any) {}
	}

	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return Binary[L]{}, err
	}

	trainInputs, err := loadWithRecovery[bitsx.Matrices](filepath.Join(dataDir, trainInputsFileName), baseURL+trainInputsFileName, logf)
	if err != nil {
		return Binary[L]{}, fmt.Errorf("%s の読み込みに失敗: %w", trainInputsFileName, err)
	}
	trainLabels, err := loadWithRecovery[[]L](filepath.Join(dataDir, trainLabelsFileName), baseURL+trainLabelsFileName, logf)
	if err != nil {
		return Binary[L]{}, fmt.Errorf("%s の読み込みに失敗: %w", trainLabelsFileName, err)
	}
	testInputs, err := loadWithRecovery[bitsx.Matrices](filepath.Join(dataDir, testInputsFileName), baseURL+testInputsFileName, logf)
	if err != nil {
		return Binary[L]{}, fmt.Errorf("%s の読み込みに失敗: %w", testInputsFileName, err)
	}
	testLabels, err := loadWithRecovery[[]L](filepath.Join(dataDir, testLabelsFileName), baseURL+testLabelsFileName, logf)
	if err != nil {
		return Binary[L]{}, fmt.Errorf("%s の読み込みに失敗: %w", testLabelsFileName, err)
	}

	if len(trainInputs) != len(trainLabels) {
		return Binary[L]{}, fmt.Errorf("訓練入力とラベルの件数が不一致: inputs=%d, labels=%d", len(trainInputs), len(trainLabels))
	}
	if len(testInputs) != len(testLabels) {
		return Binary[L]{}, fmt.Errorf("テスト入力とラベルの件数が不一致: inputs=%d, labels=%d", len(testInputs), len(testLabels))
	}

	return Binary[L]{
		TrainInputs: trainInputs,
		TrainLabels: trainLabels,
		TestInputs:  testInputs,
		TestLabels:  testLabels,
	}, nil
}
