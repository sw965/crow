package tensor3d_test

import (
	"testing"
	"fmt"
	"github.com/sw965/crow/blas32/tensor/3d"
)

func TestTranspose(t *testing.T) {
	x := tensor3d.General{
		Channels:2,
		Rows:3,
		Cols:5,
		ChannelStride:3*5,
		RowStride:5,
		Data:[]float32{
			1, 2, 3, 4, 5,
			2, 1, 4, 3, 5,
			3, 2, 5, 1, 4,

			3, 2, 1, 0, 5,
			2, 4, 3, 1, 0,
			1, 0, 2, 3, 4,
		},
	}

	transposed021 := x.Transpose021()
	fmt.Println(transposed021)

	transposed102 := x.Transpose102()
	fmt.Println(transposed102)

	transposed120 := x.Transpose120()
	fmt.Println(transposed120)

	transposed201 := x.Transpose201()
	fmt.Println(transposed201)

	transposed210 := x.Transpose210()
	fmt.Println(transposed210)
}