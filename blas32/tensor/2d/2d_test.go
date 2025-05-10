package tensor2d_test

import (
	"testing"
	"slices"
	"gonum.org/v1/gonum/blas/blas32"
	"github.com/sw965/crow/blas32/tensor/2d"
)

func TestTranspose(t *testing.T) {
	x := blas32.General{
		Rows:3,
		Cols:5,
		Stride:5,
		Data:[]float32{
			1, 2, 3, 4, 5,
			2, 5, 4, 1, 3,
			3, 1, 5, 2, 4,
		},
	}

	result := tensor2d.Transpose(x)
	expected := blas32.General{
		Rows:5,
		Cols:3,
		Stride:3,
		Data:[]float32{
			1, 2, 3,
			2, 5, 1,
			3, 4, 5,
			4, 1, 2,
			5, 3, 4,
		},
	}

	if result.Rows != expected.Rows {
		t.Errorf("テスト失敗")
	}

	if result.Cols != expected.Cols {
		t.Errorf("テスト失敗")
	}

	if result.Stride != expected.Stride {
		t.Errorf("テスト失敗")
	}

	if !slices.Equal(result.Data, expected.Data) {
		t.Errorf("テスト失敗")
	}
}