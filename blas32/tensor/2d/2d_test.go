package tensor2d_test

import (
	"testing"
	"fmt"
	"slices"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
	"github.com/sw965/crow/blas32/tensor/2d"
)

func TestT(t *testing.T) {
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

func TestNoTransDotNoTrans(t *testing.T) {
	x := blas32.General{
		Rows:4,
		Cols:3,
		Stride:3,
		Data:[]float32{
			0.2, 0.3, 0.1,
			0.1, 0.3, 0.2,
			0.7, 0.9, 0.4,
			0.9, 0.8, 0.6,
		},
	}

	w := blas32.General{
		Rows:3,
		Cols:4,
		Stride:4,
		Data:[]float32{
			0.1, 0.2, 0.3, 0.4,
			0.3, 0.2, 0.4, 0.1,
			0.5, 0.6, 0.7, 0.8,
		},
	}

	result := tensor2d.Dot(blas.NoTrans, blas.NoTrans, x, w)
	fmt.Println(result)
}

func TestSum0(t *testing.T) {
	x := blas32.General{
		Rows:4,
		Cols:3,
		Stride:3,
		Data:[]float32{
			0.2, 0.3, 0.1,
			0.1, 0.3, 0.2,
			0.7, 0.9, 0.4,
			0.9, 0.8, 0.6,
		},
	}

	sum0 := tensor2d.Sum0(x)
	fmt.Println(sum0)
}

func TestSum1(t *testing.T) {
	x := blas32.General{
		Rows:4,
		Cols:3,
		Stride:3,
		Data:[]float32{
			0.2, 0.3, 0.1,
			0.1, 0.3, 0.2,
			0.7, 0.9, 0.4,
			0.9, 0.8, 0.6,
		},
	}

	sum1 := tensor2d.Sum1(x)
	fmt.Println(sum1)
}