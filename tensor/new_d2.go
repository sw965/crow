package tensor

import (
	"github.com/sw965/omw/fn"
	"math/rand"
)

func NewD2Zeros(r, c int) D2 {
	ret := make(D2, r)
	for i := range ret {
		ret[i] = NewD1Zeros(c)
	}
	return ret
}

func NewD2ZerosLike(d2 D2) D2 {
	return fn.Map[D2](d2, NewD1ZerosLike)
}

func NewD2Ones(r, c int) D2 {
	ret := make(D2, r)
	for i := range ret {
		ret[i] = NewD1Ones(c)
	}
	return ret
}

func NewD2OnesLike(x D2) D2 {
	return fn.Map[D2](x, NewD1OnesLike)
}

func NewD2RandUniform(r, c int, min, max float64, rng *rand.Rand) D2 {
	ret := make(D2, r)
	for i := range ret {
		ret[i] = NewD1RandUniform(c, min, max, rng)
	}
	return ret
}

func NewD2He(r, c int, rng *rand.Rand) D2 {
	he := make(D2, r)
	for i := range he {
		he[i] = NewD1He(c, rng)
	}
	return he
}
