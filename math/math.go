package math

func CentralDifference(plusY, minusY, h float64) float64 {
	return (plusY - minusY) / (2.0 * h)
}