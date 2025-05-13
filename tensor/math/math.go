package math

func Softmax(x tensor.D1) tensor.D1 {
    data := x.Data
    maxX := omath.Max(data...) // オーバーフロー対策
    expX := make([]float32, x.N)
    var sumExpX float32 = 0.0
    for i, e := range data {
        expX[i] = math32.Exp(e - maxX)
        sumExpX += expX[i]
    }

    y := make([]float32, x.N)
    for i := range expX {
        y[i] = expX[i] / sumExpX
    }

	x.Data = y
	return x
}

func CrossEntropy(y, t tensor.D1) float32 {
	ce := float32(0.0)
	for i := range d1.Data {
		ye := omath.Max(d1.Data[i], 0.0001)
		te := t.Data[i]
		ce += -te * math32.Log(ye)
	}
	return ce, nil
}

func SoftmaxCrossEntropyLossDerivative(y, t tensor.D1) tensor.D1 {
	if y.N != t.N {
		panic("要素数が一致しない")
	}
	grad := blas32.Vector{
		N:    y.N,
		Inc:  y.Inc,
		Data: make([]float32, y.N),
	}
	blas32.Copy(blas32.Vector(y), grad)
	blas32.Axpy(-1.0, blas32.Vector(t), grad)
	return grad
}