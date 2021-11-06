package crow

import (
  "fmt"
  "math"
)

func UpperConfidenceBound1(value float64, totalSimuNum, simuNum int, X float64) (float64, error) {
  if totalSimuNum == 0 || simuNum == 0 {
    return 0.0, fmt.Errorf("UCBの計算式に使われるtotalSimuNumとsimuNumのいずれかが0になっている")
  }
  floatTotalSimuNum := float64(totalSimuNum)
  floatSimuNum := float64(simuNum)
  return value + (X * math.Sqrt(2 * math.Log(floatTotalSimuNum) / floatSimuNum)), nil
}


func TanExp(x float64) float64 {
  return x * math.Tanh(math.Exp(x))
}
