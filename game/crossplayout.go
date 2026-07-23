package game

import (
	"fmt"
	"maps"
)

// PlayPermutationFunc は、ActorCriticの並び(permIdx番目)1組分の全対局を実行し、
// 全試合の記録と、その並びでの「エージェント→ActorCritic名」の対応を返す。
type PlayPermutationFunc[R any, Ag comparable] func(permIdx, initStepsCap int) ([]R, map[Ag]ActorCriticName, error)

// ResultScoreFromRecordFunc は、1試合分の記録から、結果スコアを取り出す。
type ResultScoreFromRecordFunc[R any, Ag comparable] func(R) ResultScoreByAgent[Ag]

// CrossPlayoutRecorder は、複数のActorCriticを総当たり(全ての並び)で対戦させ、
// 記録とスコアを集計する。
// 対戦の実行方法(逐次手番・同時手番)には依存せず、並び1組分の対戦を実行する関数を外部から受け取る。
// Rは1試合分の記録の型。
type CrossPlayoutRecorder[R any, Ag comparable] struct {
	numPerms                  int
	numInits                  int
	playPermutationFunc       PlayPermutationFunc[R, Ag]
	resultScoreFromRecordFunc ResultScoreFromRecordFunc[R, Ag]
	initStepsCap              int

	currentIdx           int
	numGames             int
	totalScoreByAccrName map[ActorCriticName]float32
	numGamesByAccrName   map[ActorCriticName]int
}

func NewCrossPlayoutRecorder[R any, Ag comparable](
	accrNames []ActorCriticName,
	numPerms, numInits int,
	playPermutationFunc PlayPermutationFunc[R, Ag],
	resultScoreFromRecordFunc ResultScoreFromRecordFunc[R, Ag],
) *CrossPlayoutRecorder[R, Ag] {
	totalScoreByAccrName := make(map[ActorCriticName]float32, len(accrNames))
	numGamesByAccrName := make(map[ActorCriticName]int, len(accrNames))
	for _, name := range accrNames {
		totalScoreByAccrName[name] = 0
		numGamesByAccrName[name] = 0
	}

	return &CrossPlayoutRecorder[R, Ag]{
		numPerms:                  numPerms,
		numInits:                  numInits,
		playPermutationFunc:       playPermutationFunc,
		resultScoreFromRecordFunc: resultScoreFromRecordFunc,
		initStepsCap:              256,
		totalScoreByAccrName:      totalScoreByAccrName,
		numGamesByAccrName:        numGamesByAccrName,
	}
}

func (cp *CrossPlayoutRecorder[R, Ag]) NumGames() int {
	return cp.numGames
}

func (cp *CrossPlayoutRecorder[R, Ag]) SetInitStepsCap(c int) {
	cp.initStepsCap = c
}

func (cp *CrossPlayoutRecorder[R, Ag]) TotalScoreByActorCriticName() map[ActorCriticName]float32 {
	return maps.Clone(cp.totalScoreByAccrName)
}

func (cp *CrossPlayoutRecorder[R, Ag]) AverageScoreByActorCriticName() (map[ActorCriticName]float32, error) {
	if cp.numGames <= 0 {
		return nil, fmt.Errorf("ゲームがまだ行われていないので、平均スコアを計算出来ません。")
	}
	avg := make(map[ActorCriticName]float32, len(cp.totalScoreByAccrName))
	for k, v := range cp.totalScoreByAccrName {
		numGames := cp.numGamesByAccrName[k]
		if numGames > 0 {
			avg[k] = v / float32(numGames)
		} else {
			avg[k] = 0
		}
	}
	return avg, nil
}

func (cp *CrossPlayoutRecorder[R, Ag]) NumGamesByActorCriticName() map[ActorCriticName]int {
	return maps.Clone(cp.numGamesByAccrName)
}

func (cp *CrossPlayoutRecorder[R, Ag]) Next() ([]R, bool, error) {
	if cp.currentIdx >= cp.numPerms {
		return nil, false, nil
	}

	records, accrNameByAgent, err := cp.playPermutationFunc(cp.currentIdx, cp.initStepsCap)
	if err != nil {
		return nil, false, err
	}

	// スコアの集計
	for _, record := range records {
		for agent, score := range cp.resultScoreFromRecordFunc(record) {
			accrName := accrNameByAgent[agent]
			cp.totalScoreByAccrName[accrName] += score
			cp.numGamesByAccrName[accrName]++
		}
	}

	cp.currentIdx++
	cp.numGames += len(records)
	return records, true, nil
}

func (cp *CrossPlayoutRecorder[R, Ag]) Collect() ([]R, error) {
	remainingPerms := cp.numPerms - cp.currentIdx
	if remainingPerms <= 0 {
		return nil, nil
	}

	c := remainingPerms * cp.numInits
	collected := make([]R, 0, c)
	for {
		records, hasNext, err := cp.Next()
		if err != nil {
			return nil, err
		}
		if !hasNext {
			break
		}
		collected = append(collected, records...)
	}
	return collected, nil
}
