package sequential

import (
	"fmt"
	"sort"
	"math/rand"
	omwrand "github.com/sw965/omw/math/rand"
)

type Player[S any, As ~[]A, A comparable] func(*S, As) (A, error)
type PlayerPerAgent[S any, As ~[]A, A, G comparable] map[G]Player[S, As, A]

type LegalActionsProvider[S any, As ~[]A, A comparable] func(*S) As
type Transitioner[S any, A comparable] func(S, *A) (S, error)
type Comparator[S any] func(*S, *S) bool
type CurrentTurnAgentGetter[S any, G comparable] func(*S) G

// ゲームが終了していない場合は、空である事を想定。
type PlacementPerAgent[G comparable] map[G]int

func NewPlacementPerAgent[Gss ~[]Gs, Gs ~[]G, G comparable](agentTable Gss) (PlacementPerAgent[G], error) {
	placements := PlacementPerAgent[G]{}
	rank := 1
	for _, agents := range agentTable {
		if len(agents) == 0 {
			return PlacementPerAgent[G]{}, fmt.Errorf("順位 %d に対応するエージェントが存在しません", rank+1)
		}

		for _, agent := range agents {
			if _, ok := placements[agent]; ok {
				return PlacementPerAgent[G]{}, fmt.Errorf("エージェント %v が複数回出現しています", agent)
			}
			placements[agent] = rank
		}
		rank += len(agents)
	}
	return placements, nil
}

func (p PlacementPerAgent[G]) Validate() error {
	n := len(p)
	if n == 0 {
		return nil
	}

	ranks := make([]int, 0, n)
	for _, rank := range p {
		if rank < 1 {
			return fmt.Errorf("順位は1以上の正の整数でなければなりません")
		}
		ranks = append(ranks, rank)
	}
	sort.Ints(ranks)

	current := ranks[0]
	if current != 1 {
		return fmt.Errorf("最小順位が1ではありません: 最小順位 = %d", current)
	}
	expected := current + 1

	for _, rank := range ranks[1:] {
		// 同順の場合
		if rank == current {
			expected += 1
		// 順位が切り替わった場合
		} else if rank == expected {
			current = rank
			expected = rank + 1
		} else {
			return fmt.Errorf("順位が連続していません")
		}
	}
	return nil
}

type PlacementJudger[S any, G comparable] func(*S) (PlacementPerAgent[G], error)
type ResultScorePerAgent[G comparable] map[G]float64
type ResultScoreEvaluator[G comparable] func(PlacementPerAgent[G]) (ResultScorePerAgent[G], error)

type TotalResultScorePerAgent[G comparable] map[G]float64

func(t TotalResultScorePerAgent[G]) Add(scores ResultScorePerAgent[G]) {
	for k, v := range scores {
		if _, ok := t[k]; !ok {
			t[k] = 0
		}
		t[k] += v
	}
}

func (t TotalResultScorePerAgent[G]) ToAverage(n int) AverageResultScorePerAgent[G] {
	avg := AverageResultScorePerAgent[G]{}
	for k, v := range t {
		avg[k] = v / float64(n)
	}
	return avg
}

type AverageResultScorePerAgent[G comparable] map[G]float64

type Logic[S any, As ~[]A, A, G comparable] struct {
	LegalActionsProvider    LegalActionsProvider[S, As, A]
	Transitioner            Transitioner[S, A]
	Comparator              Comparator[S]
	CurrentTurnAgentGetter  CurrentTurnAgentGetter[S, G]
	PlacementJudger         PlacementJudger[S, G]
	ResultScoreEvaluator    ResultScoreEvaluator[G]
}

func (l *Logic[S, As, A, G]) IsEnd(state *S) bool {
	placements, _ := l.PlacementJudger(state)
	return len(placements) != 0
}

func (l *Logic[S, As, A, G]) SetStandardResultScoreEvaluator() {
	l.ResultScoreEvaluator = func(placements PlacementPerAgent[G]) (ResultScorePerAgent[G], error) {
		if err := placements.Validate(); err != nil {
			return ResultScorePerAgent[G]{} , err
		}

		n := len(placements)
		counts := map[int]int{}
		for _, rank := range placements {
			if _, ok := counts[rank]; !ok {
				counts[rank] = 1
			} else {
				counts[rank] += 1
			}
		}

		scores := ResultScorePerAgent[G]{}
		for agent, rank := range placements {
			score := 1.0 - ((float64(rank) - 1.0) / (float64(n) - 1.0))
			scores[agent] = score / float64(counts[rank])
		}
		return scores, nil
	}
}

func (l *Logic[S, As, A, G]) EvaluateResultScorePerAgent(state *S) (ResultScorePerAgent[G], error) {
	placements, err := l.PlacementJudger(state)
	if err != nil {
		return ResultScorePerAgent[G]{}, err
	}
	return l.ResultScoreEvaluator(placements)
}

func (l *Logic[S, As, A, G]) NewRandActionPlayer(r *rand.Rand) Player[S, As, A] {
	return func(state *S, legalActions As) (A, error) {
		return omwrand.Choice(legalActions, r), nil
	}
}

func (l *Logic[S, As, A, G]) Play(players PlayerPerAgent[S, As, A, G], state S, f func(*S) bool) (S, error) {
	for {
		isEnd := l.IsEnd(&state)
		if isEnd || f(&state) {
			break
		}

		agent := l.CurrentTurnAgentGetter(&state)
		player := players[agent]
		legalActions := l.LegalActionsProvider(&state)

		action, err := player(&state, legalActions)
		if err != nil {
			var zero S
			return zero, err
		}

		state, err = l.Transitioner(state, &action)
		if err != nil {
			var zero S
			return zero, err
		}
	}
	return state, nil
}

func (l *Logic[S, As, A, G]) Playout(players PlayerPerAgent[S, As, A, G], state S) (S, error) {
	return l.Play(players, state, func(_ *S) bool { return false })
}

func (l Logic[S, As, A, G]) ComparePlayerStrength(players PlayerPerAgent[S, As, A, G], gameNum int, state S) (AverageResultScorePerAgent[G], error) {
	total := TotalResultScorePerAgent[G]{}
	for i := 0; i < gameNum; i++ {
		final, err := l.Playout(players, state)
		if err != nil {
			return AverageResultScorePerAgent[G]{}, err
		}

		scores, err := l.EvaluateResultScorePerAgent(&final)
		if err != nil {
			return AverageResultScorePerAgent[G]{}, err
		}
		total.Add(scores)
	}
	return total.ToAverage(gameNum), nil
}