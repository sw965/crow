package linear

import (
	"slices"
	ojson "github.com/sw965/omw/encoding/json"
)

type Parameter struct {
	Weight []float32
	Bias   []float32
}

func LoadParameterJSON(path string) (Parameter, error) {
	return ojson.Load[Parameter](path)
}

func (p Parameter) SaveJSON(path string) error {
	err := ojson.Save[Parameter](&p, path)
	return err
}

func (p Parameter) Clone() Parameter {
	return Parameter{
		Weight: slices.Clone(p.Weight),
		Bias:   slices.Clone(p.Bias),
	}
}

func (p *Parameter) NewGradBufferZerosLike() GradBuffer {
	return GradBuffer{
		Weight: make([]float32, len(p.Weight)),
		Bias:   make([]float32, len(p.Bias)),
	}
}

func (p *Parameter) Axpy(alpha float32, x Parameter) {
	for i := range p.Weight {
		p.Weight[i] += (alpha * x.Weight[i])
	}

	for i := range p.Bias {
		p.Bias[i] += (alpha * x.Bias[i])
	}
}

func (p *Parameter) Scal(alpha float32) {
	for i := range p.Weight {
		p.Weight[i] *= alpha
	}

	for i := range p.Bias {
		p.Bias[i] *= alpha
	}
}

func (p *Parameter) AxpyGrad(alpha float32, grad GradBuffer) {
	for i := range p.Weight {
		p.Weight[i] += (alpha * grad.Weight[i])
	}

	for i := range p.Bias {
		p.Bias[i] += (alpha * grad.Bias[i])
	}
}
