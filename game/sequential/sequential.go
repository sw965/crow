package sequential

import (
	"fmt"
	"sort"
	"math/rand"
	omwrand "github.com/sw965/omw/math/rand"
)

type LegalActionsProvider[S any, As ~[]A, A comparable] func(*S) As
type Transitioner[S any, A comparable] func(S, *A) (S, error)
type Comparator[S any] func(*S, *S) bool
type CurrentTurnAgentGetter[S any, Ag comparable] func(*S) Ag

// ゲームが終了していない場合は、空である事を想定。
type AgentPlacements[Ag comparable] map[Ag]int

func NewAgentPlacements[Agss ~[]Ags, Ags ~[]Ag, Ag comparable](agentTable Agss) (AgentPlacements[Ag], error) {
	placements := AgentPlacements[Ag]{}
	rank := 1
	for _, agents := range agentTable {
		if len(agents) == 0 {
			return AgentPlacements[Ag]{}, fmt.Errorf("順位 %d に対応するエージェントが存在しません", rank)
		}

		for _, agent := range agents {
			if _, ok := placements[agent]; ok {
				return AgentPlacements[Ag]{}, fmt.Errorf("エージェント %v が複数回出現しています", agent)
			}
			placements[agent] = rank
		}
		rank += len(agents)
	}
	return placements, nil
}

func (p AgentPlacements[G]) Validate() error {
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

type PlacementsJudger[S any, Ag comparable] func(*S) (AgentPlacements[Ag], error)
type AgentResultScores[Ag comparable] map[Ag]float64

func (ss AgentResultScores[Ag]) DivScalar(a float64) {
	for k := range ss {
		ss[k] /= a
	}
}

type ResultScoresEvaluator[Ag comparable] func(AgentPlacements[Ag]) (AgentResultScores[Ag], error)

type Logic[S any, As ~[]A, A, Ag comparable] struct {
	LegalActionsProvider     LegalActionsProvider[S, As, A]
	Transitioner             Transitioner[S, A]
	Comparator               Comparator[S]
	CurrentTurnAgentGetter   CurrentTurnAgentGetter[S, Ag]
	PlacementsJudger         PlacementsJudger[S, Ag]
	ResultScoresEvaluator    ResultScoresEvaluator[Ag]
}

func (l *Logic[S, As, A, Ag]) IsEnd(state *S) (bool, error) {
	placements, err := l.PlacementsJudger(state)
	return len(placements) != 0, err
}

func (l *Logic[S, As, A, Ag]) SetStandardResultScoresEvaluator() {
	l.ResultScoresEvaluator = func(placements AgentPlacements[Ag]) (AgentResultScores[Ag], error) {
		if err := placements.Validate(); err != nil {
			return AgentResultScores[Ag]{} , err
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

		scores := AgentResultScores[Ag]{}

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

func (l *Logic[S, As, A, Ag]) EvaluateAgentResultScores(state *S) (AgentResultScores[Ag], error) {
	placements, err := l.PlacementsJudger(state)
	if err != nil {
		return AgentResultScores[Ag]{}, err
	}
	return l.ResultScoresEvaluator(placements)
}

func (l *Logic[S, As, A, Ag]) Playout(state S, players AgentPlayers[S, As, A, Ag]) (S, error) {
	for {
		isEnd, err := l.IsEnd(&state)
		if err != nil {
			var s S
			return s, err
		}

		if isEnd {
			break
		}

		agent := l.CurrentTurnAgentGetter(&state)
		player := players[agent]
		legalActions := l.LegalActionsProvider(&state)

		action, err := player(&state, legalActions)
		if err != nil {
			var s S
			return s, err
		}

		state, err = l.Transitioner(state, &action)
		if err != nil {
			var s S
			return s, err
		}
	}
	return state, nil
}

func (l *Logic[S, As, A, Ag]) ComparePlayerStrength(state S, players AgentPlayers[S, As, A, Ag], n int) (AgentResultScores[Ag], error) {
	avgs := AgentResultScores[Ag]{}
	for i := 0; i < n; i++ {
		final, err := l.Playout(state, players)
		if err != nil {
			return nil, err
		}

		scores, err := l.EvaluateAgentResultScores(&final)
		if err != nil {
			return nil, err
		}
		for k, v := range scores {
			avgs[k] += v
		}
	}
	avgs.DivScalar(float64(n))
	return avgs, nil
}

type Player[S any, As ~[]A, A comparable] func(*S, As) (A, error)

func NewRandActionPlayer[S any, As ~[]A, A comparable](r *rand.Rand) Player[S, As, A] {
	return func(_ *S, legalActions As) (A, error) {
		return omwrand.Choice(legalActions, r), nil
	}
}

type AgentPlayers[S any, As ~[]A, A, Ag comparable] map[Ag]Player[S, As, A]