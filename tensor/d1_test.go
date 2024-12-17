package tensor_test

import (
	"testing"
	"github.com/sw965/crow/tensor"
)

func Test(t *testing.T) {
	x := tensor.D1{
		1, 2, 3, 4, 5, 6, 7, 8, 9,
		10, 11, 12, 13, 14, 15, 16, 17, 18,
		19, 20, 21, 22, 23, 24, 25, 26, 27,
	}

	result, err := x.Reshape3D(3, 3, 3)
	if err != nil {
		panic(err)
	}
	expected := tensor.D3{
		tensor.D2{
			tensor.D1{1, 2, 3},
			tensor.D1{4, 5, 6},
			tensor.D1{7, 8, 9},
		},

		tensor.D2{
			tensor.D1{10, 11, 12},
			tensor.D1{13, 14, 15},
			tensor.D1{16, 17, 18},
		},

		tensor.D2{
			tensor.D1{19, 20, 21},
			tensor.D1{22, 23, 24},
			tensor.D1{25, 26, 27},
		},
	}

	if !result.Equal(expected) {
		t.Errorf("テスト失敗")
	}
}

