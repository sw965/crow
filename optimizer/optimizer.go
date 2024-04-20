package optimizer

import (
	"github.com/sw965/crow/tensor"
)

type D1Momentum struct {
	momentum float64
	velocity tensor.D1
}

func NewD1Momentum(momentum float64, velocity tensor.D1) D1Momentum {
	return D1Momentum{momentum:momentum, velocity:velocity}
}

func (opt *D1Momentum) Train(w, grad tensor.D1, lr float64) {
	for i := range w {
		opt.velocity[i] =  (opt.momentum * opt.velocity[i]) - (lr * grad[i])
		w[i] += opt.velocity[i]
	}
}

type D2Momentum struct {
	momentum float64
	velocity tensor.D2
}

func NewD2Momentum(momentum float64, velocity tensor.D2) D2Momentum {
	return D2Momentum{momentum:momentum, velocity:velocity}
}

func(opt *D2Momentum) Train(w, grad tensor.D2, lr float64) {
	for i := range w {
		vi := opt.velocity[i]
		wi := w[i]
		gradi := grad[i]
		for j := range wi {
			vi[j] = (opt.momentum * vi[j]) - (lr * gradi[j])
			wi[j] += vi[j]
		}
	}
}

type D3Momentum struct {
    momentum float64
    velocity tensor.D3
}

func NewD3Momentum(momentum float64, velocity tensor.D3) D3Momentum {
	return D3Momentum{momentum:momentum, velocity:velocity}
}

func (opt *D3Momentum) Train(w, grad tensor.D3, lr float64) {
    for i := range w {
        vi := opt.velocity[i]
        wi := w[i]
        gradi := grad[i]
        for j := range wi {
            vij := vi[j]
            wij := wi[j]
            gradij := gradi[j]
            for k := range wij {
                vij[k] = (opt.momentum * vij[k]) - (lr * gradij[k])
                wij[k] += vij[k]
            }
        }
    }
}