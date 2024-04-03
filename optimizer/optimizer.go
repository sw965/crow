package optimizer

import (
	"github.com/sw965/crow/tensor"
)

type D1Momentum struct {
	Momentum float64
	Velocity tensor.D1
}

func NewD2Momentum(momentum float64, n int) D2Momentum {
	return D2Momentum{Momentum:momentum, Velocity:make(tensor.D1, n)}
}

func(opt *D2Momentum) Train(w, grad tensor.D1, lr float64) {
	for i := range w {
		opt.Velocity[i] = (opt.Momentum * opt.Velocity[i]) - (lr * grad[i])
		w[i] += opt.Velocity[i]
	}
}

type D2Momentum struct {
	Momentum float64
	Velocity tensor.D2
}

func NewD2Momentum(momentum float64, row, col int) D2Momentum {
	return D2Momentum{Momentum:momentum, Velocity:tensor.NewD2Zeros(row, col)}
}

func(opt *D2Momentum) Train(w, grad tensor.D2, lr float64) {
	for i := range w {
		for j := range w[i] {
			opt.Velocity[i][j] = (opt.Momentum * opt.Velocity[i][j]) - (lr * grad[i][j])
			w[i][j] += opt.Velocity[i][j]
		}
	}
}