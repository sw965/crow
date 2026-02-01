package linear

import (
	"github.com/sw965/omw/encoding/jsonx"
	"slices"
	"fmt"
)

type Parameter struct {
	Weight []float32
	Bias   []float32
}

func LoadParameterJSON(path string) (Parameter, error) {
	return jsonx.Load[Parameter](path)
}

func (p Parameter) SaveJSON(path string) error {
	err := jsonx.Save[Parameter](p, path)
	return err
}

func (p Parameter) Clone() Parameter {
	return Parameter{
		Weight: slices.Clone(p.Weight),
		Bias:   slices.Clone(p.Bias),
	}
}

func (p Parameter) NewGradBufferZerosLike() GradBuffer {
	return GradBuffer{
		Weight: make([]float32, len(p.Weight)),
		Bias:   make([]float32, len(p.Bias)),
	}
}

func (p *Parameter) Axpy(alpha float32, x Parameter) error {
	if len(p.Weight) != len(x.Weight) || len(p.Bias) != len(x.Bias) {
		return fmt.Errorf("Parameter sizes do not match in Axpy")
	}

	for i := range p.Weight {
		p.Weight[i] += (alpha * x.Weight[i])
	}

	for i := range p.Bias {
		p.Bias[i] += (alpha * x.Bias[i])
	}
	return nil
}

func (p *Parameter) Scal(alpha float32) {
	for i := range p.Weight {
		p.Weight[i] *= alpha
	}

	for i := range p.Bias {
		p.Bias[i] *= alpha
	}
}

func (p *Parameter) AxpyGrad(alpha float32, grad GradBuffer) error {
	err := p.Axpy(alpha, Parameter(grad))
	return err
}