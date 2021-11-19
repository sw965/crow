package crow

import (
  "fmt"
  "math"
)

type UpperConfidenceBound1 struct {
  AccumReward float64
  Trial int
}

func (ucb1 *UpperConfidenceBound1) AverageReward() (float64, error) {
  if ucb1.Trial == 0 {
    return 0.0, fmt.Errorf("zero division error")
  }
  return float64(ucb1.AccumReward) / float64(ucb1.Trial), nil
}

func (ucb1 *UpperConfidenceBound1) Get(totalTrial int, X float64) (float64, error) {
  if totalTrial == 0 {
    return 0.0, fmt.Errorf("totalTrial == 0")
  }

  if ucb1.Trial == 0 {
    return 0.0, fmt.Errorf("ucb1.Trial == 0")
  }

  floatTotalTrial := float64(totalTrial)
  floatTrial := float64(ucb1.Trial)
  averageReward, err := ucb1.AverageReward()
  return averageReward + (X * math.Sqrt(2 * math.Log(floatTotalTrial) / floatTrial)), err
}
