package tensor

import (
	"math"
	"math/rand"
	"github.com/sw965/omw"
)

func NewD1Zeros(n int) D1 {
    return make(D1, n)
} 

func NewD1ZerosLike(x D1) D1 {
	n := len(x)
    return NewD1Zeros(n)
}

func NewD1Ones(n int) D1 {
	ret := make(D1, n)
	for i := range ret {
		ret[i] = 1.0 
	}
	return ret
}

func NewD1OnesLike(x D1) D1 {
	n := len(x)
    return NewD1Ones(n)
}

func NewD1RandomUniform(n int, min, max float64, r *rand.Rand) D1 {
	ret := make(D1, n)
	for i := range ret {
		ret[i] = omw.RandFloat64(min, max, r)
	}
	return ret
}

func NewD1He(n int, r *rand.Rand) D1 {
    std := math.Sqrt(2.0 / float64(n))
    he := make(D1, n)
    for i := range he {
        he[i] = r.NormFloat64() * std
    }
    return he
}