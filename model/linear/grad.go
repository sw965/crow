package linear

import (
	"github.com/chewxy/math32"
)

type GradBuffer struct {
	Weight []float32
	Bias   []float32
}

func (g *GradBuffer) NewZerosLike() GradBuffer {
	return GradBuffer{
		Weight: make([]float32, len(g.Weight)),
		Bias:   make([]float32, len(g.Bias)),
	}
}

func (g *GradBuffer) Axpy(alpha float32, x GradBuffer) {
	for i := range g.Weight {
		g.Weight[i] += (alpha * x.Weight[i])
	}

	for i := range g.Bias {
		g.Bias[i] += (alpha * x.Bias[i])
	}
}

func (g *GradBuffer) Scal(alpha float32) {
	for i := range g.Weight {
		g.Weight[i] *= alpha
	}

	for i := range g.Bias {
		g.Bias[i] *= alpha
	}
}

func (g *GradBuffer) MaxAbs() (float32, float32) {
	wMax := float32(0.0)
	for i := range g.Weight {
		wi := math32.Abs(g.Weight[i])
		if wi > wMax {
			wMax = wi
		}
	}

	bMax := float32(0.0)
	for i := range g.Bias {
		bi := math32.Abs(g.Bias[i])
		if bi > bMax {
			bMax = bi
		}
	}
	return wMax, bMax
}

type GradBuffers []GradBuffer

func (gs GradBuffers) Total() GradBuffer {
	total := gs[0].NewZerosLike()
	for _, g := range gs {
		total.Axpy(1.0, g)
	}
	return total
}

func (gs GradBuffers) Average() GradBuffer {
	avg := gs[0].NewZerosLike()
	for _, g := range gs {
		avg.Axpy(1.0, g)
	}
	avg.Scal(1.0 / float32(len(gs)))
	return avg
}
