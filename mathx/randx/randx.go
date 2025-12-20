package randx

import (
	"math/rand/v2"
	"github.com/sw965/omw/mathx/randx"
)

func Rademacher(rng *rand.Rand) float32 {
	if randx.Bool(rng) {
		return 1.0
	}
	return -1.0
}