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

type PlacementsJudger[S any, G comparable] func(*S) (PlacementPerAgent[G], error)
type ResultScorePerAgent[G comparable] map[G]float64
type ResultScoresEvaluator[G comparable] func(PlacementPerAgent[G]) (ResultScorePerAgent[G], error)

type TotalResultScorePerAgent[G comparable] map[G]float64

func(ts TotalResultScorePerAgent[G]) Add(scores ResultScorePerAgent[G]) {
	for k, v := range scores {
		if _, ok := ts[k]; !ok {
			ts[k] = 0
		}
		ts[k] += v
	}
}

func (ts TotalResultScorePerAgent[G]) ToAverage(n int) AverageResultScorePerAgent[G] {
	avgs := AverageResultScorePerAgent[G]{}
	for k, v := range ts {
		avgs[k] = v / float64(n)
	}
	return avgs
}

type AverageResultScorePerAgent[G comparable] map[G]float64

type Logic[S any, As ~[]A, A, G comparable] struct {
	LegalActionsProvider     LegalActionsProvider[S, As, A]
	Transitioner             Transitioner[S, A]
	Comparator               Comparator[S]
	CurrentTurnAgentGetter   CurrentTurnAgentGetter[S, G]
	PlacementsJudger         PlacementsJudger[S, G]
	ResultScoresEvaluator    ResultScoresEvaluator[G]
}

func (l *Logic[S, As, A, G]) IsEnd(state *S) bool {
	placements, _ := l.PlacementsJudger(state)
	return len(placements) != 0
}

func (l *Logic[S, As, A, G]) SetStandardResultScoresEvaluator() {
	l.ResultScoresEvaluator = func(placements PlacementPerAgent[G]) (ResultScorePerAgent[G], error) {
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
			// 同順の人数で割る
			scores[agent] = score / float64(counts[rank])
		}
		return scores, nil
	}
}

func (l *Logic[S, As, A, G]) EvaluateResultScorePerAgent(state *S) (ResultScorePerAgent[G], error) {
	placements, err := l.PlacementsJudger(state)
	if err != nil {
		return ResultScorePerAgent[G]{}, err
	}
	return l.ResultScoresEvaluator(placements)
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
	totals := TotalResultScorePerAgent[G]{}
	for i := 0; i < gameNum; i++ {
		final, err := l.Playout(players, state)
		if err != nil {
			return nil, err
		}

		scores, err := l.EvaluateResultScorePerAgent(&final)
		if err != nil {
			return nil, err
		}
		totals.Add(scores)
	}
	return totals.ToAverage(gameNum), nil
}