package math

import (
    "gonum.org/v1/gonum/blas/blas32"
    "fmt"
    "math"
    omath "github.com/sw965/omw/math"
)

func CentralDifference(plusY, minusY, h float32) float32 {
	return (plusY - minusY) / (2.0 * h)
}

func Softmax(x blas32.Vector) blas32.Vector {
    data := x.Data
    maxX := omath.Max(data...) // オーバーフロー対策
    expX := make([]float32, x.N)
    var sumExpX float32 = 0.0
    for i, e := range data {
        expX[i] = float32(math.Exp(float64(e - maxX)))
        sumExpX += expX[i]
    }

    newData := make([]float32, x.N)
    for i := range expX {
        newData[i] = expX[i] / sumExpX
    }
    x.Data = newData
    return x
}

func SumSquaredError(y, t blas32.Vector) (float32, error) {
	if y.N != t.N {
		return 0.0, fmt.Errorf("len(y) != len(t) であるため、SumSquaredErrorを計算できません。")
	}
	var sqSum float32 = 0.0
	for i := range y.Data {
		diff := y.Data[i] - t.Data[i]
		sqSum += (diff * diff)
	}
	return 0.5 * sqSum, nil
}

func CrossEntropyError(y, t blas32.Vector) (float32, error) {
    if y.N != t.N {
        return 0.0, fmt.Errorf("len(y) != len(t) であるため、CrossEntropyErrorを計算できません。")
    }
    var loss float32 = 0.0
	var e float32 = 0.0001
	for i := range y.Data {
		yi := float64(omath.Max(y.Data[i], e))
		ti := t.Data[i]
		loss += -ti * float32(math.Log(yi))
	}
    return loss, nil
}