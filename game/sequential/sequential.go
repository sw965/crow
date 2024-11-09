package sequential

import (
	"fmt"
	"sort"
	"math/rand"
	omwrand "github.com/sw965/omw/math/rand"
	"github.com/sw965/crow/game/solver"
)

type Player[S any, As ~[]A, A comparable] func(*S, As) (A, error)

func NewRandActionPlayer[S any, As ~[]A, A comparable](r *rand.Rand) Player[S, As, A] {
	return func(_ *S, legalActions As) (A, error) {
		return omwrand.Choice(legalActions, r), nil
	}
}

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
			return PlacementPerAgent[G]{}, fmt.Errorf("順位 %d に対応するエージェントが存在しません", rank)
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

func (ss ResultScorePerAgent[G]) ToEvalPerAgent() solver.EvalPerAgent[G] {
	es := solver.EvalPerAgent[G]{}
	for k, v := range ss {
		es[k] = solver.Eval(v)
	}
	return es
}

func (ss ResultScorePerAgent[G]) Add(other ResultScorePerAgent[G]) {
	for k := range ss {
		ss[k] += other[k]
	}
}

func (ss ResultScorePerAgent[G]) DivScalar(a float64) {
	for k := range ss {
		ss[k] /= a
	}
}

type ResultScoresEvaluator[G comparable] func(PlacementPerAgent[G]) (ResultScorePerAgent[G], error)

type AverageResultScorePerAgent[G comparable] map[G]float64

type Logic[S any, As ~[]A, A, G comparable] struct {
	LegalActionsProvider     LegalActionsProvider[S, As, A]
	Transitioner             Transitioner[S, A]
	Comparator               Comparator[S]
	CurrentTurnAgentGetter   CurrentTurnAgentGetter[S, G]
	PlacementsJudger         PlacementsJudger[S, G]
	ResultScoresEvaluator    ResultScoresEvaluator[G]
}

func (l *Logic[S, As, A, G]) IsEnd(state *S) (bool, error) {
	placements, err := l.PlacementsJudger(state)
	return len(placements) != 0, err
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

		if n == 1 {
            for agent := range placements {
                scores[agent] = 1.0
            }
            return scores, nil
        }

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

type Engine[S any, As ~[]A, A, G comparable] struct {
	Logic          Logic[S, As, A, G]
	PlayerPerAgent PlayerPerAgent[S, As, A, G]
}

func (e *Engine[S, As, A, G]) GetCurrentPlayer(state *S) Player[S, As, A] {
	agent := e.Logic.CurrentTurnAgentGetter(state)
	return e.PlayerPerAgent[agent]
}

func (e *Engine[S, As, A, G]) Play(state S, f func(*S) bool) (S, error) {
	for {
		isEnd, err := e.Logic.IsEnd(&state)
		if err != nil {
			var zero S
			return zero, err
		}

		if isEnd || f(&state) {
			break
		}

		player := e.GetCurrentPlayer(&state)
		legalActions := e.Logic.LegalActionsProvider(&state)

		action, err := player(&state, legalActions)
		if err != nil {
			var zero S
			return zero, err
		}

		state, err = e.Logic.Transitioner(state, &action)
		if err != nil {
			var zero S
			return zero, err
		}
	}
	return state, nil
}

func (e *Engine[S, As, A, G]) Playout(state S) (S, error) {
	return e.Play(state, func(_ *S) bool { return false })
}

func (e *Engine[S, As, A, G]) ComparePlayerStrength(state S, n int) (ResultScorePerAgent[G], error) {
	avgs := ResultScorePerAgent[G]{}
	for i := 0; i < n; i++ {
		final, err := e.Playout(state)
		if err != nil {
			return nil, err
		}

		scores, err := e.Logic.EvaluateResultScorePerAgent(&final)
		if err != nil {
			return nil, err
		}
		avgs.Add(scores)
	}
	avgs.DivScalar(float64(n))
	return avgs, nil
}