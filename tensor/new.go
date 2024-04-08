package tensor

import (
	"math"
	"math/rand"

	"github.com/sw965/omw"
)

func NewD1Ones(n int) D1 {
	y := make(D1, n)
	for i := range y {
		y[i] = 1.0 
	}
	return y
}

func NewD1Random(n int, min, max float64, r *rand.Rand) D1 {
	y := make(D1, n)
	for i := range y {
		y[i] = omw.RandFloat64(min, max, r)
	}
	return y

}

func NewD1He(n int, r *rand.Rand) D1 {
    std := math.Sqrt(2.0 / float64(n))
    y := make(D1, n)
    for i := range y {
        y[i] = r.NormFloat64() * std
    }
    return y
}

func NewD2Zeros(r, c int) D2 {
	y := make(D2, r)
	for i := range y {
		y[i] = make(D1, c)
	}
	return y
}

func NewD2Ones(r, c int) D2 {
	y := make(D2, r)
	for i := range y {
		y[i] = NewD1Ones(c)
	}
	return y
}

func NewD2Random(r, c int, min, max float64, random *rand.Rand) D2 {
    y := make(D2, r)
    for i := range y {
        y[i] = NewD1Random(c, min, max, random)
    }
    return y
}

func NewD2He(r, c int, random *rand.Rand) D2 {
    y := make(D2, r)
    for i := range y {
        y[i] = NewD1He(c, random)
    }
    return y
}