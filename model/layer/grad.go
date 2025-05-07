package layer

import (
	"gonum.org/v1/gonum/blas/blas32"
	"github.com/sw965/crow/blas32/vectors"
	"github.com/sw965/crow/blas32/tensors/2d"
)

type GradBuffer struct {
	Weights []blas32.General
	Biases  []blas32.Vector
}

func (g *GradBuffer) NewZerosLike() GradBuffer {
	return GradBuffer{
		Weights:tensors2d.NewZerosLike(g.Weights),
		Biases:vectors.NewZerosLike(g.Biases),
	}
}

func (g *GradBuffer) Clone() GradBuffer {
	return GradBuffer{
		Weights:tensors2d.Clone(g.Weights),
		Biases:vectors.Clone(g.Biases),
	}
}

func (g *GradBuffer) Axpy(alpha float32, x *GradBuffer) {
	tensors2d.Axpy(alpha, x.Weights, g.Weights)
	vectors.Axpy(alpha, x.Biases, g.Biases)
}

func (g *GradBuffer) Scal(alpha float32) {
	tensors2d.Scal(alpha, g.Weights)
	vectors.Scal(alpha, g.Biases)
}

type GradBuffers []GradBuffer

func (gs GradBuffers) Total() GradBuffer {
	total := gs[0].Clone()
	for _, g := range gs[1:] {
		total.Axpy(1.0, &g)
	}
	return total
}

func (gs GradBuffers) Average() GradBuffer {
	total := gs.Total()
	total.Scal(1.0 / float32(len(gs)))
	return total
}