package linear

import (
	"fmt"
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

func (p Parameter) NewGradBufferZerosLike() GradBuffer {
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
	p.Axpy(alpha, Parameter(grad))
}

type Parameters []Parameter

func (ps Parameters) Sum() (Parameter, error) {
    if len(ps) == 0 {
        return Parameter{}, fmt.Errorf("Parameters is empty")
    }
    wn, bn := len(ps[0].Weight), len(ps[0].Bias)
    sumW := make([]float32, wn)
    sumB := make([]float32, bn)

    for _, p := range ps {
        if len(p.Weight) != wn || len(p.Bias) != bn {
            return Parameter{}, fmt.Errorf("shape mismatch")
        }
        for i := 0; i < wn; i++ {
            sumW[i] += p.Weight[i]
        }
        for i := 0; i < bn; i++ {
            sumB[i] += p.Bias[i]
        }
    }
    return Parameter{Weight: sumW, Bias: sumB}, nil
}