package nn



func LeakyReLU1DDerivative(x tensor.D1, alpha float32) tensor.D1 {
	grad := make([]float32, x.N)
	for i, e := range x.Data {
		if e > 0 {
			grad[i] = 1.0
		} else {
			grad[i] = alpha
		}
	}
	x.Data = grad
	return x
}