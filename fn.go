package crow

import (
  "math"
)

func TanExp(x float64) float64 {
  return x * math.Tanh(math.Exp(x))
}
