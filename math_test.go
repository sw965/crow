package crow

import (
	"fmt"
	"testing"
	"github.com/sw965/omw"
)

func Test(t *testing.T) {
	f := func(t float64) func([]float64) float64 {
		return func(x []float64) float64 {
			y := omw.Sum(x...)
			return 0.5 * (y - t) * (y - t)
		}
	}
	x := []float64{10, 20, 30, 40, 50, 60, 70, 80}
	grad := NumericalGradient(x, f(500))
	fmt.Println(grad)
}