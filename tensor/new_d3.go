package tensor

import (
	"github.com/sw965/omw/fn"
	"math/rand"
)

func NewD3Zeros(r, c, d int) D3 {
	ret := make(D3, r)
	for i := range ret {
		ret[i] = NewD2Zeros(c, d)
	}
	return ret
}

func NewD3ZerosLike(d3 D3) D3 {
	return fn.Map[D3](d3, NewD2ZerosLike)
}

func NewD3Ones(r, c, d int) D3 {
	ret := make(D3, r)
	for i := range ret {
		ret[i] = NewD2Ones(c, d)
	}
	return ret
}

func NewD3OnesLike(d3 D3) D3 {
	return fn.Map[D3](d3, NewD2OnesLike)
}

func NewD3RandUniform(d, r, c int, min, max float64, rng *rand.Rand) D3 {
	ret := make(D3, d)
	for i := range ret {
		ret[i] = NewD2RandUniform(r, c, min, max, rng)
	}
	return ret
}

func NewD3He(d, r, c int, rng *rand.Rand) D3 {
	he := make(D3, d)
	for i := range he {
		he[i] = NewD2He(r, c, rng)
	}
	return he
}