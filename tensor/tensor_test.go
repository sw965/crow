package tensor_test

import (
	"testing"
	"github.com/sw965/crow/tensor"
	"fmt"
)

func TestDotProduct(t *testing.T) {
	tensor1 := tensor.D2{
		tensor.D1{1.0, 2.0, 3.0},
		tensor.D1{4.0, 5.0, 6.0},
	}

	tensor2 := tensor.D2{
		tensor.D1{1.0, 2.0},
		tensor.D1{3.0, 4.0},
		tensor.D1{5.0, 6.0},
	}

	fmt.Println(tensor1.DotProduct(tensor2))
}