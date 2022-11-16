package crow

import (
	"math"
)

func UpperConfidenceBound1(v float64, totalTrial, eachTrial int, X float64) float64 {
	floatTotalTrial := float64(totalTrial)
	floatEachTrial := float64(eachTrial)
	return v + (X * math.Sqrt(2 * math.Log(floatTotalTrial) / floatEachTrial))
}

func PolynomialUpperConfidenceBound(v, p float64, totalTrial, eachTrial int, X float64) float64 {
	floatTotalTrial := float64(totalTrial)
	floatEachTrial := float64(eachTrial + 1)
	return v + (X * p * math.Sqrt(floatTotalTrial) / floatEachTrial)
}