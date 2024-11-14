package ml

import (
	"github.com/sw965/crow/math"
)

func ComputeL2NormClipFactor(maxNorm float64, es ...float64) float64 {
	if maxNorm <= 0.0 {
		return 1.0
	}
	norm := math.ComputeL2Norm(es...)
	c := maxNorm / norm
	if c < 1.0 {
		return c
	}
	return 1.0
}