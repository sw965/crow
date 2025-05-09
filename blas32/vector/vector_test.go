package vector_test

import (
	"testing"
	"fmt"
	"github.com/sw965/crow/blas32/vector"
	"gonum.org/v1/gonum/blas/blas32"
	orand "github.com/sw965/omw/math/rand"
)

func TestNewZeros(t *testing.T) {
	result := vector.NewZeros(7)
	fmt.Println(result)
}

func TestNewZerosLike(t *testing.T) {
	vec := blas32.Vector{
		N:3,
		Inc:1,
		Data:[]float32{100.0, -200.0, 300.0},
	}
	result := vector.NewZerosLike(vec)
	fmt.Println(result)
}

func TestNewRademacher(t *testing.T) {
	rng := orand.NewMt19937()
	result := vector.NewRademacher(10, rng)
	fmt.Println(result)
}

func TestNewRademacherLike(t *testing.T) {
	rng := orand.NewMt19937()
	vec := blas32.Vector{
		N:5,
		Inc:1,
		Data:[]float32{0.0, 0.1, 0.2, 0.3, 0.4},
	}
	result := vector.NewRademacherLike(vec, rng)
	fmt.Println(result)
}

func TestClone(t *testing.T) {
	vec := blas32.Vector{
		N:8,
		Inc:1,
		Data:[]float32{-1.0, -2.0, -3.0, -4.0, 1.0, 2.0, 3.0, 4.0},
	}

	result := vector.Clone(vec)
	result.Data[0] = 1000.0

	fmt.Println(vec)
	fmt.Println(result)
}

func TestAffine(t *testing.T) {
	xn := 5
	x := blas32.Vector{
		N:xn,
		Inc:1,
		Data:[]float32{1.0, 2.0, 3.0, 4.0, 5.0},
	}

	yn := 3
	w := blas32.General{
		Rows:xn,
		Cols:yn,
		Stride:yn,
		Data:[]float32{
			-1.0, 2.0, -3.0,
			-3.0, -2.0, -0.1,
			4.0,  -0.5, 5.0,
			0.0,  0.0, -2.0,
			5.0, -5.0, 0.0,
		},
	}

	b := blas32.Vector{
		N:yn,
		Inc:1,
		Data:[]float32{-1.0, 0.0, 1.0},
	}

	result := vector.Affine(x, w, b)
	expected := blas32.Vector{
		N:3,
		Inc:1,
		Data:[]float32{29.0, -28.5, 4.8},
	}

	if result.N != expected.N {
		t.Errorf("テスト失敗")
	}

	if result.Inc != expected.Inc {
		t.Errorf("テスト失敗")
	}

	fmt.Println(result.Data, expected.Data)
}