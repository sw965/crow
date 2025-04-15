package linear_test

import (
	"fmt"
	"testing"
	omwrand "github.com/sw965/omw/math/rand"
	"github.com/sw965/crow/model/linear"
)

func TestSoftmaxActionSelection(t *testing.T) {
	wn := 5
	w := make([][]*float64, wn)
	win := 1
	for i := range w {
		w[i] = make([]*float64, win)
		for j := range w[i] {
			w[i][j] = new(float64)
			*w[i][j] = 1.0
		}
	}

	b := make([]*float64, wn)
	for i := range b {
		b[i] = new(float64)
		*b[i] = 0.0
	}

	model := linear.Model{Parameter:linear.Parameter{Weight:w, Bias:b}}
	model.SetSoftmaxForCrossEntropy()
	model.SetCrossEntropyError()
	input := linear.Input{
		linear.WeightCoordinate{Row:0, Column:0}:1.0,
		linear.WeightCoordinate{Row:1, Column:0}:0.5,
		linear.WeightCoordinate{Row:2, Column:0}:0.3,
		linear.WeightCoordinate{Row:3, Column:0}:0.1,
		linear.WeightCoordinate{Row:4, Column:0}:0.0,
	}
	y := model.Predict(input)
	fmt.Println("y =", y)
	r := omwrand.NewMt19937()
	counter := map[int]int{}
	for i := 0; i < 10000; i++ {
		actionIdx := model.SoftmaxActionSelection(input, 1.0, func(idx int) bool { return true }, r)
		counter[actionIdx] += 1
	}

	fmt.Println("counter =", counter)
}