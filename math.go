package crow

import (
	"golang.org/x/exp/constraints"
)

func NumericalGradient[T constraints.Float](x []T, f func([]T) T) []T {
	h := T(0.0001)
	n := len(x)
	grad := make([]T, n)
	for i := 0; i < n; i++ {
		tmp := x[i]
		x[i] = tmp + h
		y1 := f(x)

		x[i] = tmp - h
		y2 := f(x)

		grad[i] = (y1 - y2) / (h * 2)
		x[i] = tmp
	}
	return grad
}