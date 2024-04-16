package layer_test

import (
	"testing"
	"github.com/sw965/omw"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/crow/layer"
)

func TestPReLUForward(t *testing.T) {
	r := omw.NewMt19937()
	h := 0.001
	for i := 0; i < 128; i++ {
		x := tensor.NewD1He(10, r)
		alpha := r.Float64()
		dAlpha := 0.0
		forwards := layer.NewD1PReLUForward(*alpha, *dAlpha)
		backwards := layer.D1Backwards{}
		_, backwards, err := forwards(x, backwards)
		_, err := backwards[0](tensor.D1{1.0})

		tmp := alpha

		alpha += h
		y1, _, err := forwards(x, backwards)
		if err != nil {
			panic(err)
		}

		alpha -= h
		y2, _, err := forwards(x, backwards)
		if err != nil {
			panic(err)
		}
		alpha = tmp

		numericalGrad := (y1 - y2) / (2*h)
		fmt.Println(numericalGrad, )
	}
}