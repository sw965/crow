package rand

import (
	"math/rand"
	omwrand "github.com/sw965/omw/math/rand"
)

func Rademacher(rng *rand.Rand) float32 {
	if omwrand.Bool(rng) {
		return 1.0
	}
	return -1.0
}