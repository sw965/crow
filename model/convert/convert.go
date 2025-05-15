package convert

import (
	"github.com/sw965/crow/model/consts"
	"github.com/sw965/crow/model/general"
	"github.com/sw965/crow/model/spsa"
	"github.com/sw965/crow/tensor"
)

func GeneralGradToSPSAGrads(genGrad general.GradBuffer, layerTypes []consts.LayerType) spsa.GradBuffers {
	fIdx := 0
	wIdx := 0
	gIdx := 0
	bIdx := 0

	spsaGrads := make(spsa.GradBuffers, len(layerTypes))
	for i, layerType := range layerTypes {
		spsaGrad := spsa.GradBuffer{}
		if layerType == consts.ConvLayer {
			spsaGrad.Filter = genGrad.Filters[fIdx].Clone()
			fIdx++
		}

		if layerType == consts.DotLayer {
			spsaGrad.Weight = genGrad.Weights[wIdx].Clone()
			wIdx++
		}

		if layerType == consts.InstanceNormalizationLayer {
			spsaGrad.Gamma = genGrad.Gammas[gIdx].Clone()
			gIdx++

			spsaGrad.Bias = genGrad.Biases[bIdx].Clone()
			bIdx++
		}
		spsaGrads[i] = spsaGrad
	}
	return spsaGrads
}

func SPSAGradsToGeneralGrad(spsaGrads spsa.GradBuffers) general.GradBuffer {
	n := len(spsaGrads)
	genGrad := general.GradBuffer{
		Filters:make(tensor.D4Slice, 0, n),
		Weights:make(tensor.D2Slice, 0, n),
		Gammas:make(tensor.D1Slice, 0, n),
		Biases:make(tensor.D1Slice, 0, n),
	}
	for _, spsaGrad := range spsaGrads {
		if spsaGrad.Filter.Batches != 0 {
			genGrad.Filters = append(genGrad.Filters, spsaGrad.Filter.Clone())
		}

		if spsaGrad.Weight.Rows != 0 {
			genGrad.Weights = append(genGrad.Weights, spsaGrad.Weight.Clone())
		}

		if spsaGrad.Gamma.N != 0 {
			genGrad.Gammas = append(genGrad.Gammas, spsaGrad.Gamma.Clone())
		}

		if spsaGrad.Bias.N != 0 {
			genGrad.Biases = append(genGrad.Biases, spsaGrad.Bias.Clone())
		}
	}
	return genGrad
}


func GeneralParameterToSPSAParameters(genParam *general.Parameter, layerTypes []consts.LayerType) spsa.Parameters {
	fIdx := 0
	wIdx := 0
	gIdx := 0
	bIdx := 0

	spsaParams := make(spsa.Parameters, len(layerTypes))
	for i, layerType := range layerTypes {
		spsaParam := spsa.Parameter{}

		if layerType == consts.ConvLayer {
			spsaParam.Filter = genParam.Filters[fIdx].Clone()
			fIdx++
		}

		if layerType == consts.DotLayer {
			spsaParam.Weight = genParam.Weights[wIdx].Clone()
			wIdx++
		}

		if layerType == consts.InstanceNormalizationLayer {
			spsaParam.Gamma = genParam.Gammas[gIdx].Clone()
			gIdx++

			spsaParam.Bias = genParam.Biases[bIdx].Clone()
			bIdx++
		}
		spsaParams[i] = spsaParam 
	}
	return spsaParams
}

func SPSAParametersToGeneralParameter(spsaParams spsa.Parameters) general.Parameter {
	n := len(spsaParams)
	genParam := general.Parameter{
		Filters:make(tensor.D4Slice, 0, n),
		Weights:make(tensor.D2Slice, 0, n),
		Gammas:make(tensor.D1Slice, 0, n),
		Biases:make(tensor.D1Slice, 0, n),
	}
	for _, spsaParam := range spsaParams {
		if spsaParam.Filter.Batches != 0 {
			genParam.Filters = append(genParam.Filters, spsaParam.Filter.Clone())
		}

		if spsaParam.Weight.Rows != 0 {
			genParam.Weights = append(genParam.Weights, spsaParam.Weight.Clone())
		}

		if spsaParam.Gamma.N != 0 {
			genParam.Gammas = append(genParam.Gammas, spsaParam.Gamma.Clone())
		}

		if spsaParam.Bias.N != 0 {
			genParam.Biases = append(genParam.Biases, spsaParam.Bias.Clone())
		}
	}
	return genParam
}