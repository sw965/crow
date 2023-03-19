package crow

import (
	"math"
	"golang.org/x/exp/constraints"
)

type UCBFunc func(float64, int, int) float64

func UpperConfidenceBound1(c float64) UCBFunc {
	return func(v float64, n, a int) float64 {
		fn := float64(n)
		return v + (c * math.Sqrt(fn) / float64(a))
	}
}

func NumericalGradient[X constraints.Float](xs []X, f func([]X) X) []X {
	h := X(0.0001)
	n := len(xs)
	grad := make([]X, n)
	for i := 0; i < n; i++ {
		tmp := xs[i]
		xs[i] = tmp + h
		y1 := f(xs)

		xs[i] = tmp - h
		y2 := f(xs)

		grad[i] = (y1 - y2) / (h * 2)
		xs[i] = tmp
	}
	return grad
}