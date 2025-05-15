package general

import (
	"github.com/sw965/crow/tensor"
)

type Parameter struct {
	Filters tensor.D4Slice
	Weights tensor.D2Slice
	Gammas  tensor.D1Slice
	Biases  tensor.D1Slice 
}

func (p Parameter) NewGradZerosLike() GradBuffer {
	return GradBuffer{
		Filters: p.Filters.NewZerosLike(),
		Weights: p.Weights.NewZerosLike(),
		Gammas:  p.Gammas.NewZerosLike(),
		Biases:  p.Biases.NewZerosLike(),
	}
}

func (p Parameter) Clone() Parameter {
	return Parameter{
		Filters: p.Filters.Clone(),
		Weights: p.Weights.Clone(),
		Gammas:  p.Gammas.Clone(),
		Biases:  p.Biases.Clone(),
	}
}

func (p *Parameter) AxpyInPlaceGrad(alpha float32, grad *GradBuffer) {
	p.Filters.AxpyInPlace(alpha, grad.Filters)
	p.Weights.AxpyInPlace(alpha, grad.Weights)
	p.Gammas.AxpyInPlace(alpha, grad.Gammas)
	p.Biases.AxpyInPlace(alpha, grad.Biases)
}