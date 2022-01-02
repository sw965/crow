package crow

import (
  "testing"
  "fmt"
  "math"
  "math/rand"
  "github.com/seehuhn/mt19937"
  "time"
)

type Slot func(*rand.Rand) int

func Slot0(random *rand.Rand) int {
  //期待値4.5
  coins := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  index := random.Intn(len(coins))
  return coins[index]
}

func Slot1(random *rand.Rand) int {
  //期待値1
  coins := make([]int, 10)
  coins[0] = 10
  index := random.Intn(len(coins))
  return coins[index]
}

func Slot2(random *rand.Rand) int {
  return 5
}

func Slot3(random *rand.Rand) int {
  //期待値5.5
  coins := []int{1, 10}
  index := random.Intn(len(coins))
  return coins[index]
}

func Slot4(random *rand.Rand) int {
  //期待値10
  coins := make([]int, 100)
  coins[0] = 1000
  index := random.Intn(len(coins))
  return coins[index]
}

type Slots []Slot
type SlotUCB1s []UpperConfidenceBound1

func (slotUCB1s SlotUCB1s) TotalTrial() int {
  result := 0
  for _, ucb1 := range slotUCB1s {
    result += ucb1.Trial
  }
  return result
}

func (slotUCB1s SlotUCB1s) Max(X float64) (float64, error) {
  totalTrial := slotUCB1s.TotalTrial()
  result, err := slotUCB1s[0].Get(totalTrial, X)

  if err != nil {
    return 0.0, err
  }

  for _, ucb1 := range slotUCB1s[1:] {
    ucb1v, err := ucb1.Get(totalTrial, X)
    if err != nil {
      return 0.0, err
    }

    if ucb1v > result {
      result = ucb1v
    }
  }
  return result, nil
}

func (slotUCB1s SlotUCB1s) MaxIndices(X float64) ([]int, error) {
  max, err := slotUCB1s.Max(X)
  if err != nil {
    return []int{}, err
  }
  result := []int{}

  totalTrial := slotUCB1s.TotalTrial()
  for i, ucb1 := range slotUCB1s {
    ucb1v, err := ucb1.Get(totalTrial, X)
    if err != nil {
      return []int{}, err
    }

    if ucb1v == max {
      result = append(result, i)
    }
  }
  return result, nil
}

func (slotUCB1s SlotUCB1s) MaxIndexRandomChoice(X float64, random *rand.Rand) (int, error) {
  maxIndices, err := slotUCB1s.MaxIndices(X)
  if err != nil {
    return 0, err
  }
  index := random.Intn(len(maxIndices))
  return maxIndices[index], nil
}

var SLOTS = Slots{Slot0, Slot1, Slot2, Slot3, Slot4}

func TestUCB1(t *testing.T) {
  mtRandom := rand.New(mt19937.New())
  mtRandom.Seed(time.Now().UnixNano())

  X := math.Sqrt(100)
  TRIAL_NUM := 5120000
  slotUCB1s := make(SlotUCB1s, len(SLOTS))

  //初期化
  for i := 0; i < len(slotUCB1s); i++ {
    slot := SLOTS[i]
    coin := slot(mtRandom)
    slotUCB1s[i].AccumReward += float64(coin)
    slotUCB1s[i].Trial += 1
  }

  for i := 0; i <TRIAL_NUM; i++ {
    slotIndex, err := slotUCB1s.MaxIndexRandomChoice(X, mtRandom)
    if err != nil {
      panic(err)
    }
    slot := SLOTS[slotIndex]
    coin := slot(mtRandom)
    slotUCB1s[slotIndex].AccumReward += float64(coin)
    slotUCB1s[slotIndex].Trial += 1
  }

  for i, ucb1 := range slotUCB1s {
    averageReward, err := ucb1.AverageReward()
    if err != nil {
      panic(err)
    }

    testMsg := fmt.Sprintf("slotNum = %v averageReward = %v trial = %v", i, averageReward, ucb1.Trial)
    fmt.Println(testMsg)
  }
}
