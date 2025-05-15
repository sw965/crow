package tensor_test

import (
	"testing"
	"github.com/sw965/crow/tensor"
	"fmt"
	"math"
)

func TestOuter(t *testing.T) {
	x := tensor.D1{
		N:4,
		Inc:1,
		Data:[]float32{1.0, 2.0, 3.0, 5.0},
	}

	w := tensor.D1{
		N:4,
		Inc:1,
		Data:[]float32{0.3, 0.2, 1.0, 1.5},
	}

	result := x.Outer(w)
	fmt.Println(result)
}

func TestDotNoTrans2D(t *testing.T) {
	x := tensor.D1{
		N:5,
		Inc:1,
		Data:[]float32{
			1.0, 2.0, 3.0, 4.0, 5.0,
		},
	}

	w := tensor.D2{
		Rows:5,
		Cols:3,
		Stride:3,
		Data:[]float32{
			0.1, 0.2, 0.3,
			0.2, 1.0, 3.0,
			3.0, 5.0, 2.5,
			6.0, 2.0, 1.5,
			0.3, 2.0, 3.0,
		},
	}

	result := x.DotNoTrans2D(w)
	fmt.Println(result)
}

func TestDotTrans2D_Correct(t *testing.T) {
    // もとの行列 W は 5×3
    w := tensor.D2{
        Rows:   5,
        Cols:   3,
        Stride: 3,
        Data: []float32{
            0.1, 0.2, 0.3,
            0.2, 1.0, 3.0,
            3.0, 5.0, 2.5,
            6.0, 2.0, 1.5,
            0.3, 2.0, 3.0,
        },
    }
    // 逆伝播で受け取る勾配 chain は、出力サイズに合わせて長さ 3
    chain := tensor.D1{
        N:    3,
        Inc:  1,
        Data: []float32{1.0, 2.0, 3.0},
    }

    // 期待値を手計算すると:
    // dx[i] = sum_{j=0..2} W[i,j] * chain[j]
    //    = [0.1*1 + 0.2*2 + 0.3*3,
    //       0.2*1 + 1.0*2 + 3.0*3,
    //       3.0*1 + 5.0*2 + 2.5*3,
    //       6.0*1 + 2.0*2 + 1.5*3,
    //       0.3*1 + 2.0*2 + 3.0*3]
    expected := []float32{1.4, 11.2, 20.5, 14.5, 13.3}

    got := chain.DotTrans2D(w)
    for i, v := range got.Data {
        if math.Abs(float64(v-expected[i])) > 1e-6 {
            t.Errorf("DotTrans2D: at %d, got %v, want %v", i, v, expected[i])
        }
    }
}