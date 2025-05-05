package vectors_test

import (
	"testing"
	"fmt"
	"gonum.org/v1/gonum/blas/blas32"
	"github.com/sw965/crow/blas32/vectors"
	orand "github.com/sw965/omw/math/rand"
)

func TestNewZerosLike(t *testing.T) {
	vecs := []blas32.Vector{
		blas32.Vector{
			N:7,
			Inc:1,
			Data:[]float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7},
		},

		blas32.Vector{
			N:5,
			Inc:1,
			Data:[]float32{0.1, 0.2, 0.3, 0.4, 0.5},
		},
		
		blas32.Vector{
			N:3,
			Inc:1,
			Data:[]float32{0.1, 0.2, 0.3},
		},
	}

	result := vectors.NewZerosLike(vecs)
	for _, vec := range result {
		fmt.Println(vec)
	}
}

func TestNewRademacherLike(t *testing.T) {
	rng := orand.NewMt19937()

	vecs := []blas32.Vector{
		blas32.Vector{
			N:7,
			Inc:1,
			Data:[]float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7},
		},

		blas32.Vector{
			N:5,
			Inc:1,
			Data:[]float32{0.1, 0.2, 0.3, 0.4, 0.5},
		},
		
		blas32.Vector{
			N:3,
			Inc:1,
			Data:[]float32{0.1, 0.2, 0.3},
		},
	}

	result := vectors.NewRademacherLike(vecs, rng)
	for _, vec := range result {
		fmt.Println(vec)
	}
}