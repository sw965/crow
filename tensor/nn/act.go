func LeakyReLU1D(x tensor.D1, alpha float32) D1 {
	y := make([]float32, x.N)
	for i, e := range x.Data {
		if e > 0 {
			y[i] = e
		} else {
			y[i] = alpha * e
		}
	}

	x.Data = y
	return x
}