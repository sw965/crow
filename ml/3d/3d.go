package ml3d

import (
	"github.com/sw965/crow/ml/2d"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/omw/fn"
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

func Conv(x tensor.D3, filter tensor.D4) tensor.D3 {
    xD := len(x)
    xH := len(x[0])
    xW := len(x[0][0])

    yD := len(filter)          // 出力チャネル数(=フィルタ数)
    fH := len(filter[0][0])    // フィルタの高さ
    fW := len(filter[0][0][0]) // フィルタの幅

    // 出力 (y) の次元計算（ストライド1、パディングなし想定）
    yH := xH - fH + 1
    yW := xW - fW + 1

    y := make(tensor.D3, yD)
    for od := 0; od < yD; od++ {
        y[od] = make(tensor.D2, yH)
        for h := 0; h < yH; h++ {
            y[od][h] = make(tensor.D1, yW)
        }
    }

    for od := 0; od < yD; od++ {
        for h := 0; h < yH; h++ {
            for wi := 0; wi < yW; wi++ {
                sum := 0.0
                for id := 0; id < xD; id++ {
                    for kh := 0; kh < fH; kh++ {
                        for kw := 0; kw < fW; kw++ {
                            sum += x[id][h+kh][wi+kw] * filter[od][id][kh][kw]
                        }
                    }
                }
                y[od][h][wi] = sum
            }
        }
    }

    return y
}

func ConvDerivative(x tensor.D3, filter tensor.D4, chain tensor.D3) (tensor.D3, tensor.D4) {
    xD := len(x)
    xH := len(x[0])
    xW := len(x[0][0])

    yD := len(filter)
    fD := len(filter[0])
    fH := len(filter[0][0])
    fW := len(filter[0][0][0])

    yH := xH - fH + 1
    yW := xW - fW + 1

    // dx, dfilter の初期化
    dx := make(tensor.D3, xD)
    for id := 0; id < xD; id++ {
        dx[id] = make(tensor.D2, xH)
        for h := 0; h < xH; h++ {
            dx[id][h] = make(tensor.D1, xW)
        }
    }

    dfilter := make(tensor.D4, yD)
    for od := 0; od < yD; od++ {
        dfilter[od] = make(tensor.D3, fD)
        for id := 0; id < fD; id++ {
            dfilter[od][id] = make(tensor.D2, fH)
            for kh := 0; kh < fH; kh++ {
                dfilter[od][id][kh] = make(tensor.D1, fW)
            }
        }
    }

    for od := 0; od < yD; od++ {
        for id := 0; id < fD; id++ {
            for kh := 0; kh < fH; kh++ {
                for kw := 0; kw < fW; kw++ {
                    sum := 0.0
                    for h := 0; h < yH; h++ {
                        for w := 0; w < yW; w++ {
                            sum += chain[od][h][w] * x[id][h+kh][w+kw]
                        }
                    }
                    dfilter[od][id][kh][kw] = sum
                }
            }
        }
    }

    for id := 0; id < xD; id++ {
        for iH := 0; iH < xH; iH++ {
            for iW := 0; iW < xW; iW++ {
                sum := 0.0
                for od := 0; od < yD; od++ {
                    for kh := 0; kh < fH; kh++ {
                        for kw := 0; kw < fW; kw++ {
                            h := iH - kh
                            w := iW - kw
                            if h >= 0 && h < yH && w >= 0 && w < yW {
                                sum += chain[od][h][w] * filter[od][id][kh][kw]
                            }
                        }
                    }
                }
                dx[id][iH][iW] = sum
            }
        }
    }

    return dx, dfilter
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
