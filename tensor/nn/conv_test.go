package nn_test

import (
	"testing"
	"fmt"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/crow/tensor/nn"
)

func TestIm2Col(t *testing.T) {
	img := tensor.D3{
		Channels:     3,
		Rows:         5,
		Cols:         6,
		ChannelStride: 30,
		RowStride:     6,
		Data: []float32{
			// Channel 0
			2.3, -8.7, 0.0, 9.9, -1.2, 4.5,
			-3.4, 7.6, -9.9, 1.1, 0.8, -2.2,
			5.5, -6.6, 3.3, -0.4, 8.8, -7.7,
			9.0, -1.0, 2.2, -3.3, 4.4, -5.5,
			6.6, -8.8, 1.9, -0.1, 7.7, -9.8,

			// Channel 1
			-4.4, 3.3, -2.2, 1.1, 0.0, 9.9,
			8.8, -7.7, 6.6, -5.5, 4.4, -3.3,
			2.2, -1.1, 0.0, 9.9, -8.8, 7.7,
			-6.6, 5.5, -4.4, 3.3, -2.2, 1.1,
			0.0, -9.9, 8.8, -7.7, 6.6, -5.5,

			// Channel 2
			5.5, -4.4, 3.3, -2.2, 1.1, 0.0,
			-9.9, 8.8, -7.7, 6.6, -5.5, 4.4,
			-3.3, 2.2, -1.1, 0.0, 9.9, -8.8,
			7.7, -6.6, 5.5, -4.4, 3.3, -2.2,
			1.1, 0.0, -9.9, 8.8, -7.7, 6.6,
		},
	}

	col1 := nn.Im2Col(img, 3, 3, 1)
	fmt.Println(col1)
	fmt.Println("")

	col2 := nn.Im2Col(img, 3, 3, 2)
	fmt.Println(col2)
	fmt.Println("")

	col3 := nn.Im2Col(img, 2, 2, 3)
	fmt.Println(col3)
	fmt.Println("")
}