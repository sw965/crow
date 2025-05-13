package tensor_test

import (
	"testing"
	"github.com/sw965/crow/tensor"
	"fmt"
)

func TestTranspose(t *testing.T) {
	x := tensor.D2{
		Rows:3,
		Cols:4,
		Stride:4,
		Data:[]float32{
			1.0, 2.0, 3.0, 4.0,
			5.0, 6.0, 7.0, 8.0,
			9.0, 10.0, 11.0, 12.0,
		},
	}

	fmt.Println(x.Transpose())
}

