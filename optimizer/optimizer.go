package optimizer

import (
	"github.com/sw965/crow/tensor"
)

type D1Momentum struct {
	Momentum float64
	Velocity tensor.D1
}

func (opt *D1Momentum) Train(w, grad tensor.D1, lr float64) {
	for i := range w {
		opt.Velocity[i] =  (opt.Momentum * opt.Velocity[i]) - (lr * grad[i])
		w[i] += opt.Velocity[i]
	}
}

type D2Momentum struct {
	Momentum float64
	Velocity tensor.D2
}

func(opt *D2Momentum) Train(w, grad tensor.D2, lr float64) {
	for i := range w {
		vi := opt.Velocity[i]
		wi := w[i]
		gradi := grad[i]
		for j := range wi {
			vi[j] = (opt.Momentum * vi[j]) - (lr * gradi[j])
			wi[j] += vi[j]
		}
	}
}

type D3Momentum struct {
    Momentum float64
    Velocity tensor.D3
}

func (opt *D3Momentum) Train(w, grad tensor.D3, lr float64) {
    for i := range w {
        vi := opt.Velocity[i]
        wi := w[i]
        gradi := grad[i]
        for j := range wi {
            vij := vi[j]
            wij := wi[j]
            gradij := gradi[j]
            for k := range wij {
                vij[k] = (opt.Momentum * vij[k]) - (lr * gradij[k])
                wij[k] += vij[k]
            }
        }
    }
}