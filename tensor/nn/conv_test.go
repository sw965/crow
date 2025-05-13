package nn_test

import (
	"testing"
	"fmt"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/crow/tensor/nn"
)

func TestIm2Col(t *testing.T) {
	img1 := tensor.D3{}
	col1 := nn.Im2Col(img1, 3, 3, 1)
	fmt.Println(col1)

	img2 := tensor.D3{}
	col2 := nn.Im2Col(img2, 3, 3, 2)
	fmt.Println(col2)

	img3 := tensor.D3{}
	col3 := nn.Im2Col(img3, 2, 2, 3)
	fmt.Println(col3)
}