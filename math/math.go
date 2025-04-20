package math

func CentralDifference(plusY, minusY, h float32) float32 {
	return (plusY - minusY) / (2.0 * h)
}