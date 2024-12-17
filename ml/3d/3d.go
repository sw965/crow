package ml3d

import (
	"github.com/sw965/crow/ml/2d"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/omw/fn"
	omwmath "github.com/sw965/omw/math"
)

func Sigmoid(x tensor.D3) tensor.D3 {
	return fn.Map[tensor.D3](x, ml2d.Sigmoid)
}

func SigmoidGrad(y tensor.D3) tensor.D3 {
	return fn.Map[tensor.D3](y, ml2d.SigmoidGrad)
}

func ReLU(x tensor.D3) tensor.D3 {
	return fn.Map[tensor.D3](x, ml2d.ReLU)
}

func ReLUDerivative(x tensor.D3) tensor.D3 {
	return fn.Map[tensor.D3](x, ml2d.ReLUDerivative)
}

func LeakyReLU(alpha float64) func(tensor.D3) tensor.D3 {
	return func(x tensor.D3) tensor.D3 {
		return fn.Map[tensor.D3](x, ml2d.LeakyReLU(alpha))
	}
}

func LeakyReLUDerivative(alpha float64) func(tensor.D3) tensor.D3 {
	return func(x tensor.D3) tensor.D3 {
		return fn.Map[tensor.D3](x, ml2d.LeakyReLUDerivative(alpha))
	}
}

func Convolution(x, filter tensor.D3, stride int) tensor.D3 {
	xDepth := len(x)
	xHeight := len(x[0])
	xWidth := len(x[0][0])

	filterDepth := len(filter)
	filterHeight := len(filter[0])
	filterWidth := len(filter[0][0])

	yDepth := xDepth - filterDepth + 1
	yHeight := (xHeight-filterHeight)/stride + 1
	yWidth := (xWidth-filterWidth)/stride + 1

	y := make(tensor.D3, yDepth)
	for d := range y {
		y[d] = make(tensor.D2, yHeight)
		for h := range y[d] {
			y[d][h] = make(tensor.D1, yWidth)
		}
	}

	for d := 0; d <= xDepth-filterDepth; d++ {
		for h := 0; h <= xHeight-filterHeight; h += stride {
			for w := 0; w <= xWidth-filterWidth; w += stride {
				sum := 0.0
				for fd := 0; fd < filterDepth; fd++ {
					for fh := 0; fh < filterHeight; fh++ {
						for fw := 0; fw < filterWidth; fw++ {
							sum += x[d+fd][h+fh][w+fw] * filter[fd][fh][fw]
						}
					}
				}
				y[d][h/stride][w/stride] = sum
			}
		}
	}
	return y
}

func ConvolutionDerivative(x, filter, chain tensor.D3, stride int) (tensor.D3, tensor.D3) {
	xDepth := len(x)
	xHeight := len(x[0])
	xWidth := len(x[0][0])

	filterDepth := len(filter)
	filterHeight := len(filter[0])
	filterWidth := len(filter[0][0])

	gradX := make(tensor.D3, xDepth)
	for d := range gradX {
		gradX[d] = make(tensor.D2, xHeight)
		for h := range gradX[d] {
			gradX[d][h] = make(tensor.D1, xWidth)
		}
	}

	gradFilter := make(tensor.D3, filterDepth)
	for d := range gradFilter {
		gradFilter[d] = make(tensor.D2, filterHeight)
		for h := range gradFilter[d] {
			gradFilter[d][h] = make(tensor.D1, filterWidth)
		}
	}

	for d := 0; d <= xDepth-filterDepth; d++ {
		for h := 0; h <= xHeight-filterHeight; h += stride {
			for w := 0; w <= xWidth-filterWidth; w += stride {
				cv := chain[d][h/stride][w/stride]
				for fd := 0; fd < filterDepth; fd++ {
					for fh := 0; fh < filterHeight; fh++ {
						for fw := 0; fw < filterWidth; fw++ {
							gradFilter[fd][fh][fw] += cv * x[d+fd][h+fh][w+fw]
							gradX[d+fd][h+fh][w+fw] += cv * filter[fd][fh][fw]
						}
					}
				}
			}
		}
	}

	return gradX, gradFilter
}

func L2Regularization(c float64) func(tensor.D3) float64 {
	return func(w tensor.D3) float64 {
		return omwmath.Sum(fn.Map[tensor.D1](w, ml2d.L2Regularization(c))...)
	}
}

func L2RegularizationDerivative(c float64) func(tensor.D3) tensor.D3 {
	return func(w tensor.D3) tensor.D3 {
		return fn.Map[tensor.D3](w, ml2d.L2RegularizationDerivative(c))
	}
}

func NumericalDifferentiation(x tensor.D3, f func(tensor.D3) float64) tensor.D3 {
	h := 0.001
	grad := tensor.NewD3ZerosLike(x)
	for i := range x {
		gradi := grad[i]
		xi := x[i]
		for j := range xi {
			gradij := gradi[j]
			xij := xi[j]
			for k := range xij {
				tmp := xij[k]

				xij[k] = tmp + h
				y1 := f(x)

				xij[k] = tmp - h
				y2 := f(x)

				gradij[k] = (y1 - y2) / (2 * h)
				xij[k] = tmp
			}
		}
	}
	return grad
}
