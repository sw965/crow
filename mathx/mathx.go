package mathx

func ConvertScale(x, xMin, xMax, yMin, yMax float32) float32 {
    return yMin + (yMax - yMin) * (x - xMin) / (xMax - xMin)
}

func CentralDifference(plusY, minusY, h float32) float32 {
	return (plusY - minusY) / (2.0 * h)
}