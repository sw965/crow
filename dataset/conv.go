package dataset

import (
	"fmt"
	"github.com/sw965/omw/mathx/bitsx"
	"math/rand/v2"
)

// BinarizeImages は float32の画像データを bitsx.Matrix に変換します。
func BinarizeImages(images [][]float32, threshold float32) ([]bitsx.Matrix, error) {
	result := make([]bitsx.Matrix, len(images))
	if len(images) == 0 {
		return result, nil
	}

	// 画像サイズ (784)
	size := len(images[0])

	for i, img := range images {
		if len(img) != size {
			return nil, fmt.Errorf("image size mismatch at index %d", i)
		}

		// 1行 x size列 の行列を作成
		mat, err := bitsx.NewZerosMatrix(1, size)
		if err != nil {
			return nil, err
		}

		for j, val := range img {
			// 閾値以上ならビットを立てる
			if val >= threshold {
				if err := mat.Set(0, j); err != nil {
					return nil, err
				}
			}
		}
		result[i] = mat
	}
	return result, nil
}

// GeneratePrototypes は、クラス数分のランダムなプロトタイプ（正解ビット列）を生成します。
// numClasses: クラス数 (MNISTなら10)
// dim: 最終層の次元数 (例: 1024)
func GeneratePrototypes(numClasses, dim int, rng *rand.Rand) ([]bitsx.Matrix, error) {
	prototypes := make([]bitsx.Matrix, numClasses)
	for i := 0; i < numClasses; i++ {
		// ランダムなビット行列を生成
		p, err := bitsx.NewRandMatrix(1, dim, 0, rng)
		if err != nil {
			return nil, err
		}
		prototypes[i] = p
	}
	return prototypes, nil
}

// LabelsToTargets は、ラベル(0-9)を対応するプロトタイプ(bitsx.Matrix)に変換します。
func LabelsToTargets(labels []float32, prototypes []bitsx.Matrix) ([]bitsx.Matrix, error) {
	targets := make([]bitsx.Matrix, len(labels))
	numClasses := len(prototypes)

	for i, labelFloat := range labels {
		label := int(labelFloat)
		if label < 0 || label >= numClasses {
			return nil, fmt.Errorf("label out of range at index %d: %d", i, label)
		}
		// 対応するプロトタイプのコピーを割り当てる
		// (参照渡しでも良いが、学習中に書き換わらないよう念のため)
		targets[i] = prototypes[label]
	}
	return targets, nil
}