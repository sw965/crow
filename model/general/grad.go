package general

import (
	"github.com/sw965/crow/tensor"
	omath "github.com/sw965/omw/math"
	"github.com/chewxy/math32"
)

type GradBuffer struct {
	Filters tensor.D4Slice
	Weights tensor.D2Slice
	Gammas  tensor.D1Slice
	Biases  tensor.D1Slice
}

func (g GradBuffer) NewZerosLike() GradBuffer {
	return GradBuffer{
		Filters:g.Filters.NewZerosLike(),
		Weights:g.Weights.NewZerosLike(),
		Gammas:g.Gammas.NewZerosLike(),
		Biases:g.Biases.NewZerosLike(),
	}
}

func (g GradBuffer) Clone() GradBuffer {
	return GradBuffer{
		Filters:g.Filters.Clone(),
		Weights:g.Weights.Clone(),
		Gammas:g.Gammas.Clone(),
		Biases:g.Biases.Clone(),
	}
}

func (g *GradBuffer) AxpyInPlace(alpha float32, x GradBuffer) {
	g.Filters.AxpyInPlace(alpha, x.Filters)
	g.Weights.AxpyInPlace(alpha, x.Weights)
	g.Gammas.AxpyInPlace(alpha, x.Gammas)
	g.Biases.AxpyInPlace(alpha, x.Biases)
}

func (g *GradBuffer) ScalInPlace(alpha float32) {
	g.Filters.ScalInPlace(alpha)
	g.Weights.ScalInPlace(alpha)
	g.Gammas.ScalInPlace(alpha)
	g.Biases.ScalInPlace(alpha)
}

func (g GradBuffer) CompareMaxDiff(other GradBuffer) ([]float32, []float32, []float32, []float32) {
	fMaxDiffs := make([]float32, len(g.Filters))
	wMaxDiffs := make([]float32, len(g.Weights))
	gMaxDiffs := make([]float32, len(g.Gammas))
	bMaxDiffs := make([]float32, len(g.Biases))

	for i, gf := range g.Filters {
		of := other.Filters[i]
		of = of.Clone()
		of.AxpyInPlace(-1.0, gf)
		for i, e := range of.Data {
			of.Data[i] = math32.Abs(e)
		}
		maxDiff := omath.Max(of.Data...)
		fMaxDiffs[i] = maxDiff
	}

	for i, gw := range g.Weights {
		ow := other.Weights[i]
		ow = ow.Clone()
		ow.AxpyInPlace(-1.0, gw)
		for i, e := range ow.Data {
			ow.Data[i] = math32.Abs(e)
		}
		maxDiff := omath.Max(ow.Data...)
		wMaxDiffs[i] = maxDiff
	}

	compareVecs := func(vs1, vs2 tensor.D1Slice, result []float32) {
		for i, v1 := range vs1 {
			v2 := vs2[i]
			v2 = v2.Clone()
			v2.AxpyInPlace(-1.0, v1)
			for i, e := range v2.Data {
				v2.Data[i] = math32.Abs(e)
			}
			maxDiff := omath.Max(v2.Data...)
			result[i] = maxDiff
		}
	}

	compareVecs(g.Gammas, other.Gammas, gMaxDiffs)
	compareVecs(g.Biases, other.Biases, bMaxDiffs)
	return fMaxDiffs, wMaxDiffs, gMaxDiffs, bMaxDiffs
}

type GradBuffers []GradBuffer

func (gs GradBuffers) Total() GradBuffer {
	total := gs[0].Clone()
	for _, g := range gs[1:] {
		total.AxpyInPlace(1.0, g)
	}
	return total
}