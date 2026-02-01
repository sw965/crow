package linear

import (
	"fmt"
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

func (g *GradBuffer) Axpy(alpha float32, x GradBuffer) error {
	if len(g.Weight) != len(x.Weight) || len(g.Bias) != len(x.Bias) {
		return fmt.Errorf("GradBuffer sizes do not match in Axpy")
	}

	for i := range g.Weight {
		g.Weight[i] += (alpha * x.Weight[i])
	}

	for i := range g.Bias {
		g.Bias[i] += (alpha * x.Bias[i])
	}
	return nil
}

func (g *GradBuffer) Scal(alpha float32) {
	for i := range g.Weight {
		g.Weight[i] *= alpha
	}

	for i := range g.Bias {
		g.Bias[i] *= alpha
	}
}

type GradBuffers []GradBuffer

func (gs GradBuffers) ReduceSum() (GradBuffer, error) {
	if len(gs) == 0 {
		return GradBuffer{}, nil
	}

	sum := gs[0].NewZerosLike()
	for _, g := range gs {
		err := sum.Axpy(1.0, g)
		if err != nil {
			return GradBuffer{}, err
		}
	}
	return sum, nil
}
