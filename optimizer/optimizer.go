package optimizer

import (
	"github.com/sw965/crow/tensor"
)

type D1Momentum struct {
	Momentum float64
	Velocity tensor.D1
}

func NewD1Momentum(n int) D1Momentum {
	momentum := 0.9
	return D1Momentum{Momentum:momentum, Velocity:make(tensor.D1, n)}
}

func(opt *D1Momentum) Update(w, grad tensor.D1, lr float64) {
	for i := range w {
		opt.Velocity[i] = (opt.Momentum * opt.Velocity[i]) - (grad[i] * lr)
		w[i] += opt.Velocity[i]
	}
}

type D2Momentum struct {
	Momentum float64
	Velocity tensor.D2
}

func NewD2Momentum(row, col int) D2Momentum {
	momentum := 0.9
	return D2Momentum{Momentum:momentum, Velocity:tensor.NewD2Zeros(row, col)}
}

func(opt *D2Momentum) Update(w, grad tensor.D2, lr float64) {
	for i := range w {
		for j := range w[i] {
			opt.Velocity[i][j] = (opt.Momentum * opt.Velocity[i][j]) - (grad[i][j] * lr)
			w[i][j] += opt.Velocity[i][j]
		}
	}
}

func (opt *D2Momentum) Reset() {
	r := len(opt.Velocity)
	c := len(opt.Velocity[0])
	opt.Velocity = tensor.NewD2Zeros(r, c)
}