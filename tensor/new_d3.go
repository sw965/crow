package tensor

import (
	"math/rand"
	"github.com/sw965/omw"
)

func NewD3Zeros(r, c, d int) D3 {
	ret := make(D3, r)
	for i := range ret {
		ret[i] = NewD2Zeros(c, d)
	}
	return ret
}

func NewD3ZerosLike(d3 D3) D3 {
	return omw.MapFunc[D3](d3, NewD2ZerosLike)
}

func NewD3Ones(r, c, d int) D3 {
	ret := make(D3, r)
	for i := range ret {
		ret[i] = NewD2Ones(c, d)
	}
	return ret
}

func NewD3OnesLike(d3 D3) D3 {
	return omw.MapFunc[D3](d3, NewD2OnesLike)
}

func NewD3RandUniform(r, c, d int, min, max float64, rng *rand.Rand) D3 {
	ret := make(D3, r)
	for i := range ret {
		ret[i] = NewD2RandUniform(c, d, min, max, rng)
	}
	return ret
}