package mlfuncs

import (
	"math"
	"github.com/sw965/crow/tensor"
)

func PolicyUpperConfidenceBound(v, p, c float64, totalN, actionN int) float64 {
	total := float64(totalN)
	n := float64(actionN+1)
	return v + (p * c * math.Sqrt(total) / n)
}

func L2Norm(d1 tensor.D1, d2 tensor.D2, d3 tensor.D3) float64 {
	sqSum := 0.0
	for i := range d1 {
		d1i := d1[i]
		sqSum += (d1i * d1i)
	}

	for i := range d2 {
		d2i := d2[i]
		for j := range d2i {
			d2ij := d2i[j]
			sqSum += (d2ij * d2ij)
		}
	}

	for i := range d3 {
		d3i := d3[i]
		for j := range d3i {
			d3ij := d3i[j]
			for k := range d3ij {
				d3ijk := d3ij[k]
				sqSum += (d3ijk * d3ijk)
			}
		}
	}
	return math.Sqrt(sqSum)
}

func ClipL2Norm(d1 tensor.D1, d2 tensor.D2, d3 tensor.D3, threshold float64) {
	norm := L2Norm(d1, d2, d3)
	scale := threshold / norm
	if scale < 1.0 {
		d1.MulScalar(scale)
		d2.MulScalar(scale)
		d3.MulScalar(scale)
	}
}