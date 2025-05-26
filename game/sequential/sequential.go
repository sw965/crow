package sequential

import (
	"fmt"
	"sort"
	"math/rand"
	omwrand "github.com/sw965/omw/math/rand"
	"github.com/sw965/omw/parallel"
)

type LegalActionsProvider[S any, As ~[]A, A comparable] func(S) As
type Transitioner[S any, A comparable] func(S, A) (S, error)
type Comparator[S any] func(S, S) bool
type CurrentTurnAgentGetter[S any, G comparable] func(S) G

// ゲームが終了していない場合は、戻り値は空あるいはnilである事を想定。
type PlacementByAgent[G comparable] map[G]int

func NewPlacementByAgent[Gss ~[]Gs, Gs ~[]G, G comparable](agentTable Gss) (PlacementByAgent[G], error) {
	placements := PlacementByAgent[G]{}
	rank := 1
	for _, agents := range agentTable {
		if len(agents) == 0 {
			return PlacementByAgent[G]{}, fmt.Errorf("順位 %d に対応するエージェントが存在しません", rank)
		}

		for _, agent := range agents {
			if _, ok := placements[agent]; ok {
				return PlacementByAgent[G]{}, fmt.Errorf("エージェント %v が複数回出現しています", agent)
			}
			placements[agent] = rank
		}
		rank += len(agents)
	}
	return placements, nil
}

func (ps PlacementByAgent[G]) Validate() error {
	n := len(ps)
	if n == 0 {
		return nil
	}

	ranks := make([]int, 0, n)
	for _, rank := range ps {
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

type PlacementsJudger[S any, G comparable] func(S) (PlacementByAgent[G], error)
type ResultScoreByAgent[G comparable] map[G]float32

type ResultScoresEvaluator[G comparable] func(PlacementByAgent[G]) (ResultScoreByAgent[G], error)

type Logic[S any, As ~[]A, A, G comparable] struct {
	LegalActionsProvider     LegalActionsProvider[S, As, A]
	Transitioner             Transitioner[S, A]
	Comparator               Comparator[S]
	CurrentTurnAgentGetter   CurrentTurnAgentGetter[S, G]
	PlacementsJudger         PlacementsJudger[S, G]
	ResultScoresEvaluator    ResultScoresEvaluator[G]
}

func (l Logic[S, As, A, G]) IsEnd(state S) (bool, error) {
	placements, err := l.PlacementsJudger(state)
	return len(placements) != 0, err
}

func (l *Logic[S, As, A, G]) SetStandardResultScoresEvaluator() {
	l.ResultScoresEvaluator = func(placements PlacementByAgent[G]) (ResultScoreByAgent[G], error) {
		if err := placements.Validate(); err != nil {
			return ResultScoreByAgent[G]{} , err
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

		scores := ResultScoreByAgent[G]{}

		if n == 1 {
            for agent := range placements {
                scores[agent] = 1.0
            }
            return scores, nil
        }

		for agent, rank := range placements {
			score := 1.0 - ((float32(rank) - 1.0) / (float32(n) - 1.0))
			// 同順の人数で割る
			scores[agent] = score / float32(counts[rank])
		}
		return scores, nil
	}
}

func (l Logic[S, As, A, G]) EvaluateResultScoreByAgent(state S) (ResultScoreByAgent[G], error) {
	placements, err := l.PlacementsJudger(state)
	if err != nil {
		return ResultScoreByAgent[G]{}, err
	}
	return l.ResultScoresEvaluator(placements)
}

func (l Logic[S, As, A, G]) Playout(state S, players PlayerByAgent[S, As, A, G]) (S, error) {
	for {
		isEnd, err := l.IsEnd(state)
		if err != nil {
			var s S
			return s, err
		}

		if isEnd {
			break
		}

		agent := l.CurrentTurnAgentGetter(state)
		player := players[agent]
		legalActions := l.LegalActionsProvider(state)

		action, err := player(&state, legalActions)
		if err != nil {
			var s S
			return s, err
		}

		state, err = l.Transitioner(state, action)
		if err != nil {
			var s S
			return s, err
		}
	}
	return state, nil
}

func (l Logic[S, As, A, G]) Playouts(states []S, playersByWorker []PlayerByAgent[S, As, A, G]) ([]S, error) {
	n := len(states)
	p := len(playersByWorker)
	finals := make([]S, n)
	errCh := make(chan error, p)

	worker := func(workerIdx int, statesIdxs []int) {
		players := playersByWorker[workerIdx]
		for _, idx := range statesIdxs {
			final, err := l.Playout(states[idx], players)
			if err != nil {
				errCh <- err
				return
			}
			finals[idx] = final
		}
		errCh <- nil
	}

	for workerIdx, statesIdxs := range parallel.DistributeIndicesEvenly(n, p) {
		go worker(workerIdx, statesIdxs)
	}

	for i := 0; i < p; i++ {
		if err := <-errCh; err != nil {
			return nil, err
		}
	}
	return finals, nil
}

func (l *Logic[S, As, A, G]) NewRandActionPlayer(rng *rand.Rand) Player[S, As, A] {
	return func(_ *S, legalActions As) (A, error) {
		return omwrand.Choice(legalActions, rng), nil
	}
}

func (l *Logic[S, As, A, G]) MakePlayerByAgent() PlayerByAgent[S, As, A, G] {
	return PlayerByAgent[S, As, A, G]{}
}

type Player[S any, As ~[]A, A comparable] func(*S, As) (A, error)
type PlayerByAgent[S any, As ~[]A, A, G comparable] map[G]Player[S, As, A]