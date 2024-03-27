package optimizer

import (
	"github.com/sw965/crow/tensor"
)

type D2Momentum struct {
	Momentum float64
	V tensor.D2
}

func NewD2Momentum(momentum float64, row, col int) D2Momentum {
	return D2Momentum{Momentum:momentum, V:tensor.NewD2Zeros(row, col)}
}

func(opt *D2Momentum) Train(params, grads tensor.D2, lr float64) {
	for i := range params {
		for j := range params[i] {
			opt.V[i][j] = (opt.Momentum * opt.V[i][j]) - (lr * grads[i][j])
			params[i][j] += opt.V[i][j]
		}
	}
}