package convert

import (
	"github.com/sw965/crow/model/consts"
	"github.com/sw965/crow/model/general"
	"github.com/sw965/crow/model/spsa"
	"gonum.org/v1/gonum/blas/blas32"
	"github.com/sw965/crow/blas32/vector"
	"github.com/sw965/crow/blas32/tensor/2d"
)

func GeneralGradToSPSAGrads(genGrad general.GradBuffer, layerTypes []consts.LayerType) spsa.GradBuffers {
	wIdx := 0
	gIdx := 0
	bIdx := 0

	spsaGrads := make(spsa.GradBuffers, len(layerTypes))
	for i, layerType := range layerTypes {
		spsaGrad := spsa.GradBuffer{}

		if layerType == consts.DotLayer {
			spsaGrad.Weight = tensor2d.Clone(genGrad.Weights[wIdx])
			wIdx++
		}

		if layerType == consts.InstanceNormalizationLayer {
			spsaGrad.Gamma = vector.Clone(genGrad.Gammas[gIdx])
			gIdx++

			spsaGrad.Bias = vector.Clone(genGrad.Biases[bIdx])
			bIdx++
		}
		spsaGrads[i] = spsaGrad
	}
	return spsaGrads
}

func SPSAGradsToGeneralGrad(spsaGrads spsa.GradBuffers) general.GradBuffer {
	n := len(spsaGrads)
	genGrad := general.GradBuffer{
		Weights:make([]blas32.General, 0, n),
		Gammas:make([]blas32.Vector, 0, n),
		Biases:make([]blas32.Vector, 0, n),
	}
	for _, spsaGrad := range spsaGrads {
		if spsaGrad.Weight.Rows != 0 {
			genGrad.Weights = append(genGrad.Weights, tensor2d.Clone(spsaGrad.Weight))
		}

		if spsaGrad.Gamma.N != 0 {
			genGrad.Gammas = append(genGrad.Gammas, vector.Clone(spsaGrad.Gamma))
		}

		if spsaGrad.Bias.N != 0 {
			genGrad.Biases = append(genGrad.Biases, vector.Clone(spsaGrad.Bias))
		}
	}
	return genGrad
}


func GeneralParameterToSPSAParameters(genParam *general.Parameter, layerTypes []consts.LayerType) spsa.Parameters {
	wIdx := 0
	gIdx := 0
	bIdx := 0

	spsaParams := make(spsa.Parameters, len(layerTypes))
	for i, layerType := range layerTypes {
		spsaParam := spsa.Parameter{}

		if layerType == consts.DotLayer {
			spsaParam.Weight = tensor2d.Clone(genParam.Weights[wIdx])
			wIdx++
		}

		if layerType == consts.InstanceNormalizationLayer {
			spsaParam.Gamma = vector.Clone(genParam.Gammas[gIdx])
			gIdx++

			spsaParam.Bias = vector.Clone(genParam.Biases[bIdx])
			bIdx++
		}
		spsaParams[i] = spsaParam 
	}
	return spsaParams
}

func SPSAParametersToGeneralParameter(spsaParams spsa.Parameters) general.Parameter {
	n := len(spsaParams)
	genParam := general.Parameter{
		Weights:make([]blas32.General, 0, n),
		Gammas:make([]blas32.Vector, 0, n),
		Biases:make([]blas32.Vector, 0, n),
	}
	for _, spsaParam := range spsaParams {
		if spsaParam.Weight.Rows != 0 {
			genParam.Weights = append(genParam.Weights, tensor2d.Clone(spsaParam.Weight))
		}

		if spsaParam.Gamma.N != 0 {
			genParam.Gammas = append(genParam.Gammas, vector.Clone(spsaParam.Gamma))
		}

		if spsaParam.Bias.N != 0 {
			genParam.Biases = append(genParam.Biases, vector.Clone(spsaParam.Bias))
		}
	}
	return genParam
}