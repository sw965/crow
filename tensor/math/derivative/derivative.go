package derivative

func SoftmaxCrossEntropyLoss(y, t tensor.D1) tensor.D1 {
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
	return grad, nil
}