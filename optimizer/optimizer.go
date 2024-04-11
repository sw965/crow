package optimizer

import (
	"github.com/sw965/crow/tensor"
	"github.com/sw965/crow/layer"
)

type Momentum struct {
	Momentum float64
	Velocity tensor.D2
}

func NewMomentum(row, col int) Momentum {
	momentum := 0.9
	return Momentum{Momentum:momentum, Velocity:tensor.NewD2Zeros(row, col)}
}

func(opt *Momentum) Update(pg *layer.ParameterGradientPairManager, lr float64) {
	w := pg.Pa
	for key := range pg.Pa {
		w := pg.Pa[key]
		grad := pg.Pa[key]
		for i := range w {
			for j := range w[i] {
				opt.Velocity[i][j] = (grad[i][j] * lr) - (opt.Momentum * opt.Velocity[i][j])
				w[i][j] += opt.Velocity[i][j]
			}
		}
	}
}

type Momentums []Momentum

for (opts Momentums) Update(pgs layer.ParameterGradientPairManagers, lr float64) {
	for i := range opts {
		opts[i].Update(pgs[i], lr)
	}
}
