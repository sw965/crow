package math

import (
	"math"
)

func PolicyUpperConfidenceBound(c, v, p float64, n, a int) float64 {
	m := float64(n)
	return v + (c * p * math.Sqrt(m) / float64(a))
}