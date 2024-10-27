package sequential

import (
	"fmt"
	"sort"
	"math/rand"
	omwrand "github.com/sw965/omw/math/rand"
)

type Player[S any, A comparable] func(*S) (A, error)
type AgentPlayers[S any, A, Agent comparable] map[Agent]Player[S, A]

type LegalActionsProvider[S any, As ~[]A, A comparable] func(*S) As
type Transitioner[S any, A comparable] func(S, *A) (S, error)
type Comparator[S any] func(*S, *S) bool
type CurrentTurnAgentGetter[S any, Agent comparable] func(*S) Agent

// ゲームが終了していない場合は、空である事を想定。
type AgentPlacements[Agent comparable] map[Agent]int

func NewAgentPlacements[Ass ~[]As, As ~[]Agent, Agent comparable](ass Ass) (AgentPlacements[Agent], error) {
	placements := AgentPlacements[Agent]{}
	rank := 1
	for _, agents := range ass {
		if len(agents) == 0 {
			return AgentPlacements[Agent]{}, fmt.Errorf("順位 %d に対応するエージェントが存在しません", rank+1)
		}

		for _, agent := range agents {
			if _, ok := placements[agent]; ok {
				return AgentPlacements[Agent]{}, fmt.Errorf("エージェント %v が複数回出現しています", agent)
			}
			placements[agent] = rank
		}
		rank += len(agents)
	}
	return placements, nil
}

func (p AgentPlacements[Agent]) Validate() error {
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

type AgentPlacementsJudger[S any, Agent comparable] func(*S) (AgentPlacements[Agent], error)
type AgentResultScores[Agent comparable] map[Agent]float64
type AgentResultScoresEvaluator[Agent comparable] func(AgentPlacements[Agent]) (AgentResultScores[Agent], error)

type Logic[S any, As ~[]A, A, Agent comparable] struct {
	LegalActionsProvider           LegalActionsProvider[S, As, A]
	Transitioner                   Transitioner[S, A]
	Comparator                     Comparator[S]
	CurrentTurnAgentGetter         CurrentTurnAgentGetter[S, Agent]
	AgentPlacementsJudger          AgentPlacementsJudger[S, Agent]
	AgentResultScoresEvaluator     AgentResultScoresEvaluator[Agent]
}

func (l *Logic[S, As, A, Agent]) IsEnd(s *S) bool {
	placements, _ := l.AgentPlacementsJudger(s)
	return len(placements) != 0
}

func (l *Logic[S, As, A, Agent]) SetStandardResultScoresEvaluator() {
	l.AgentResultScoresEvaluator = func(placements AgentPlacements[Agent]) (AgentResultScores[Agent], error) {
		if err := placements.Validate(); err != nil {
			return AgentResultScores[Agent]{} , err
		}

		n := len(placements)
		rankCounts := map[int]int{}
		for _, rank := range placements {
			if _, ok := rankCounts[rank]; !ok {
				rankCounts[rank] = 1
			} else {
				rankCounts[rank] += 1
			}
		}

		scores := AgentResultScores[Agent]{}
		for agent, rank := range placements {
			v := 1.0 - ((float64(rank) - 1.0) / (float64(n) - 1.0))
			scores[agent] = v / float64(rankCounts[rank])
		}
		return scores, nil
	}
}

func (l *Logic[S, As, A, Agent]) EvaluateAgentResultScores(s *S) (AgentResultScores[Agent], error) {
	placements, err := l.AgentPlacementsJudger(s)
	if err != nil {
		return AgentResultScores[Agent]{}, err
	}
	return l.AgentResultScoresEvaluator(placements)
}

func (l *Logic[S, As, A, Agent]) NewRandActionPlayer(r *rand.Rand) Player[S, A] {
	return func(state *S) (A, error) {
		as := l.LegalActionsProvider(state)
		return omwrand.Choice(as, r), nil
	}
}

func (l *Logic[S, As, A, Agent]) Play(players AgentPlayers[S, A, Agent], state S, f func(*S) bool) (S, error) {
	for {
		isEnd := l.IsEnd(&state)
		if isEnd || f(&state) {
			break
		}

		agent := l.CurrentTurnAgentGetter(&state)
		player := players[agent]

		action, err := player(&state)
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

func (l *Logic[S, As, A, Agent]) Playout(players AgentPlayers[S, A, Agent], state S) (S, error) {
	return l.Play(players, state, func(_ *S) bool { return false })
}