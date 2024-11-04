package sequential

import (
	"fmt"
	"sort"
	"math/rand"
	"golang.org/x/exp/maps"
	omwmath "github.com/sw965/omw/math"
	omwrand "github.com/sw965/omw/math/rand"
	omwslices "github.com/sw965/omw/slices"
)

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

func (r ResultScorePerAgent[G]) ToEval() EvalPerAgent[G] {
	es := EvalPerAgent[G]{}
	for k, v := range r {
		es[k] = Eval(v)
	}
	return es
}

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

func (l *Logic[S, As, A, G]) IsEnd(state *S) bool {
	placements, _ := l.PlacementsJudger(state)
	return len(placements) != 0
}

func (l *Logic[S, As, A, G]) EvaluateResultScorePerAgent(state *S) (ResultScorePerAgent[G], error) {
	placements, err := l.PlacementsJudger(state)
	if err != nil {
		return ResultScorePerAgent[G]{}, err
	}
	return l.ResultScoresEvaluator(placements)
}

type Policy[A comparable] map[A]float64
type PolicyProvider[S any, As ~[]A, A comparable] func(*S, As) Policy[A]

func UniformPolicyProvider[S any, As ~[]A, A comparable](state *S, legalActions As) Policy[A] {
	n := len(legalActions)
	p := 1.0 / float64(n)
	policy := Policy[A]{}
	for _, a := range legalActions {
		policy[a] = p
	}
	return policy
}

type Player[S any, As ~[]A, A comparable] func(*S, As) (A, error)

func NewRandActionPlayer[S any, As ~[]A, A comparable](r *rand.Rand) Player[S, As, A] {
	return func(_ *S, legalActions As) (A, error) {
		return omwrand.Choice(legalActions, r), nil
	}
}

type PlayerPerAgent[S any, As ~[]A, A, G comparable] map[G]Player[S, As, A]

type Eval float64
type EvalPerAgent[G comparable] map[G]Eval

type ActorCritic[S any, As ~[]A, A comparable] func(*S, As) (Policy[A], Eval, error)
type Selector[A comparable] func(Policy[A]) A

func NewMaxSelector[A comparable](r *rand.Rand) Selector[A] {
	return func(policy Policy[A]) A {
		ks := maps.Keys(policy)
		vs := maps.Values(policy)
		idxs := omwslices.MaxIndices(vs)
		idx := omwrand.Choice(idxs, r)
		return ks[idx]
	}
}

func NewThresholdWeightedSelector[A comparable](t float64, r *rand.Rand) Selector[A] {
	return func(policy Policy[A]) A {
		max := omwmath.Max(maps.Values(policy)...)
		threshold := max * t
		n := len(policy)
		options := make([]A, 0, n)
		weights := make([]float64, 0, n)
		for action, p := range policy {
			if p >= threshold {
				options = append(options, action)
				weights = append(weights, p)
			}
		}
		idx := omwrand.IntByWeight(weights, r)
		return options[idx]
	}
}

type Solver[S any, As ~[]A, A comparable] struct {
	ActorCritic ActorCritic[S, As, A]
	Selector    Selector[A]
}

type SolverPerAgent[S any, As ~[]A, A, G comparable] map[G]*Solver[S, As, A]

type SolverEpisode[S any, As ~[]A, A, G comparable] struct {
	States   []S
	Agents   []G
	Actions  As
	Policies []Policy[A]
	Evals    []Eval
}

func NewSolverEpisode[S any, As ~[]A, A, G comparable](capacity int) SolverEpisode[S, As, A, G] {
	return SolverEpisode[S, As, A, G]{
		States:make([]S, 0, capacity),
		Agents:make([]G, 0, capacity),
		Actions:make(As, 0, capacity),
		Policies:make([]Policy[A], 0, capacity),
		Evals:make([]Eval, 0, capacity),
	}
}

func (e *SolverEpisode[S, As, A, G]) Append(other *SolverEpisode[S, As, A, G]) {
	e.States = append(e.States, other.States...)
	e.Agents = append(e.Agents, other.Agents...)
	e.Actions = append(e.Actions, other.Actions...)
	e.Policies = append(e.Policies, other.Policies...)
	e.Evals = append(e.Evals, other.Evals...)
}

type Game[S any, As ~[]A, A, G comparable] struct {
	Logic          Logic[S, As, A, G]
	PlayerPerAgent PlayerPerAgent[S, As, A, G]
	SolverPerAgent SolverPerAgent[S, As, A, G]
	Rand           *rand.Rand
}

func (g *Game[S, As, A, G]) GetCurrentPlayer(state *S) Player[S, As, A] {
	agent := g.Logic.CurrentTurnAgentGetter(state)
	return g.PlayerPerAgent[agent]
}

func (g *Game[S, As, A, G]) Play(state S, f func(*S) bool) (S, error) {
	for {
		isEnd := g.Logic.IsEnd(&state)
		if isEnd || f(&state) {
			break
		}

		player := g.GetCurrentPlayer(&state)
		legalActions := g.Logic.LegalActionsProvider(&state)

		action, err := player(&state, legalActions)
		if err != nil {
			var zero S
			return zero, err
		}

		state, err = g.Logic.Transitioner(state, &action)
		if err != nil {
			var zero S
			return zero, err
		}
	}
	return state, nil
}

func (g *Game[S, As, A, G]) Playout(state S) (S, error) {
	return g.Play(state, func(_ *S) bool { return false })
}

func (g *Game[S, As, A, G]) ComparePlayerStrength(state S, gameNum int) (AverageResultScorePerAgent[G], error) {
	totals := TotalResultScorePerAgent[G]{}
	for i := 0; i < gameNum; i++ {
		final, err := g.Playout(state)
		if err != nil {
			return nil, err
		}

		scores, err := g.Logic.EvaluateResultScorePerAgent(&final)
		if err != nil {
			return nil, err
		}
		totals.Add(scores)
	}
	return totals.ToAverage(gameNum), nil
}

func (g *Game[S, As, A, G]) GenerateSolverEpisode(state S, capacity int) (S, SolverEpisode[S, As, A, G], error) {
	episode := NewSolverEpisode[S, As, A, G](capacity)
	for {
		isEnd := g.Logic.IsEnd(&state)
		if isEnd {
			break
		}

		agent := g.Logic.CurrentTurnAgentGetter(&state)
		solver := g.SolverPerAgent[agent]
		legalActions := g.Logic.LegalActionsProvider(&state)

		policy, eval, err := solver.ActorCritic(&state, legalActions)
		if err != nil {
			var zero S
			return zero, SolverEpisode[S, As, A, G]{}, err
		}
		action := solver.Selector(policy)

		episode.States = append(episode.States, state)
		episode.Agents = append(episode.Agents, agent)
		episode.Actions = append(episode.Actions, action)
		episode.Policies = append(episode.Policies, policy)
		episode.Evals = append(episode.Evals, eval)

		state, err = g.Logic.Transitioner(state, &action)
		if err != nil {
			var zero S
			return zero, SolverEpisode[S, As, A, G]{}, err
		}
	}
	return state, episode, nil
}