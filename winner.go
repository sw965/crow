package crow

type Winner struct {
  IsP1 bool
  IsP2 bool
}

var (
  WINNER_P1 = Winner{IsP1:true, IsP2:false}
  WINNER_P2 = Winner{IsP1:false, IsP2:true}
  DRAW = Winner{IsP1:false, IsP2:false}
)

type WinnerData_ struct {
  SigmoidReward float64
  TanhReward float64
}

type WinnerData map[Winner]*WinnerData_

var WINNER_DATA = func() WinnerData {
  result := WinnerData{}
  result[WINNER_P1] = &WinnerData_{SigmoidReward:1.0, TanhReward:1.0}
  result[WINNER_P2] = &WinnerData_{SigmoidReward:0.0, TanhReward:-1.0}
  result[DRAW] = &WinnerData_{SigmoidReward:0.5, TanhReward:0.0}
  return result
}()
