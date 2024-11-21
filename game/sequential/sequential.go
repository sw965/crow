package sequential

import (
	"fmt"
	"sort"
	"math/rand"
	omwrand "github.com/sw965/omw/math/rand"
	omwslices "github.com/sw965/omw/slices"
	omwmath "github.com/sw965/omw/math"
	"golang.org/x/exp/maps"
	"github.com/sw965/crow/tensor"
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

func (ss AgentResultScores[Ag]) ToEvalPerAgent() AgentEvals[Ag] {
	es := AgentEvals[Ag]{}
	for k, v := range ss {
		es[k] = Eval(v)
	}
	return es
}

func (ss AgentResultScores[Ag]) Add(other AgentResultScores[Ag]) {
	for k := range ss {
		ss[k] += other[k]
	}
}

func (ss AgentResultScores[Ag]) DivScalar(a float64) {
	for k := range ss {
		ss[k] /= a
	}
}

type ResultScoresEvaluator[Ag comparable] func(AgentPlacements[Ag]) (AgentResultScores[Ag], error)

type AverageAgentResultScores[Ag comparable] map[Ag]float64

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

type Player[S any, As ~[]A, A comparable] func(*S, As) (A, error)

func NewRandActionPlayer[S any, As ~[]A, A comparable](r *rand.Rand) Player[S, As, A] {
	return func(_ *S, legalActions As) (A, error) {
		return omwrand.Choice(legalActions, r), nil
	}
}

type AgentPlayers[S any, As ~[]A, A, Ag comparable] map[Ag]Player[S, As, A]

type Engine[S any, As ~[]A, A, Ag comparable] struct {
	Logic        Logic[S, As, A, Ag]
	AgentPlayers AgentPlayers[S, As, A, Ag]
}

func (e *Engine[S, As, A, Ag]) GetCurrentPlayer(state *S) Player[S, As, A] {
	agent := e.Logic.CurrentTurnAgentGetter(state)
	return e.AgentPlayers[agent]
}

func (e *Engine[S, As, A, Ag]) Play(state S, f func(*S) bool) (S, error) {
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

func (e *Engine[S, As, A, Ag]) Playout(state S) (S, error) {
	return e.Play(state, func(_ *S) bool { return false })
}

func (e *Engine[S, As, A, Ag]) ComparePlayerStrength(state S, n int) (AgentResultScores[Ag], error) {
	avgs := AgentResultScores[Ag]{}
	for i := 0; i < n; i++ {
		final, err := e.Playout(state)
		if err != nil {
			return nil, err
		}

		scores, err := e.Logic.EvaluateAgentResultScores(&final)
		if err != nil {
			return nil, err
		}
		avgs.Add(scores)
	}
	avgs.DivScalar(float64(n))
	return avgs, nil
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

type Eval float64
type AgentEvals[Ag comparable] map[Ag]Eval

func (es AgentEvals[Ag]) Add(other AgentEvals[Ag]) {
	for k, v := range other {
		es[k] += v
	}
}

func (es AgentEvals[Ag]) DivScalar(e Eval) {
	for k := range es {
		es[k] /= e
	}
}

type ActorCritic[S any, As ~[]A, A, Ag comparable] func(*S, As) (Policy[A], AgentEvals[Ag], error)
type Selector[A comparable] func(Policy[A]) A

func NewEpsilonGreedySelector[A comparable](e float64, r *rand.Rand) Selector[A] {
	return func(p Policy[A]) A {
		ks := maps.Keys(p)
		if e > r.Float64() {
			return omwrand.Choice(ks, r)
		}
		vs := maps.Values(p)
		idxs := omwslices.MaxIndices(vs)
		idx := omwrand.Choice(idxs, r)
		return ks[idx]
	}
}

func NewMaxSelector[A comparable](r *rand.Rand) Selector[A] {
	return NewEpsilonGreedySelector[A](0.0, r)
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

type Solver[S any, As ~[]A, A, Ag comparable] struct {
	ActorCritic ActorCritic[S, As, A, Ag]
	Selector Selector[A]
}

type AgentSolvers[S any, As ~[]A, A, Ag comparable] map[Ag]*Solver[S, As, A, Ag]

type SolverEngine[S any, As ~[]A, A, Ag comparable] struct {
	GameLogic    Logic[S, As, A, Ag]
	AgentSolvers AgentSolvers[S, As, A, Ag]
}

func (e SolverEngine[S, As, A, Ag]) Play(state S, f func(*S) bool) (S, error) {
	for {
		isEnd, err := e.GameLogic.IsEnd(&state)
		if err != nil {
			var zero S
			return zero, err
		}

		if isEnd || f(&state) {
			break
		}

		agent := e.GameLogic.CurrentTurnAgentGetter(&state)
		solver := e.AgentSolvers[agent]
		legalActions := e.GameLogic.LegalActionsProvider(&state)

		policy, _, err := solver.ActorCritic(&state, legalActions)
		if err != nil {
			var zero S
			return zero, err
		}
		action := solver.Selector(policy)

		state, err = e.GameLogic.Transitioner(state, &action)
		if err != nil {
			var zero S
			return zero, err
		}
	}
	return state, nil
}

func (e SolverEngine[S, As, A, Ag]) MakeTrainingInitStates(state S, n int, f func(*S) bool, c int) ([]S, error) {
	states := make([]S, 0, c)
	init := state
	for i := 0; i < n; i++ {
		state, err := e.Play(state, f)
		if err != nil {
			return []S{}, err
		}

		isEnd, err := e.GameLogic.IsEnd(&state)
		if err != nil {
			return []S{}, err
		}

		if !isEnd {
			states = append(states, state)
		}
		state = init
	}
	return states, nil
}

func(e SolverEngine[S, As, A, Ag]) GenerateEpisode(states []S, totalCap, oneGameCap int) (SolverEpisode[S, As, A, Ag], error) {
	episode := SolverEpisode[S, As, A, Ag]{
		States:make([]S, 0, totalCap),
		Agents:make([]Ag, 0, totalCap),
		Actions:make(As, 0, totalCap),
		Policies:make([]Policy[A], 0, totalCap),
		AgentEvalTable:make([]AgentEvals[Ag], 0, totalCap),
		ResultScores:make([]float64, 0, totalCap),
	}

	oneGameAgents := make([]Ag, 0, oneGameCap)
	for _, state := range states {
		for {
			isEnd, err := e.GameLogic.IsEnd(&state)
			if err != nil {
				return SolverEpisode[S, As, A, Ag]{}, err
			}
	
			if isEnd {
				scores, err := e.GameLogic.EvaluateAgentResultScores(&state)
				if err != nil {
					return SolverEpisode[S, As, A, Ag]{}, err
				}
				for _, agent := range oneGameAgents {
					score := scores[agent]
					episode.ResultScores = append(episode.ResultScores, score)
				}
				episode.Agents = append(episode.Agents, oneGameAgents...)
				oneGameAgents = oneGameAgents[:0] //空にする(容量はそのまま)
				break
			}
	
			agent := e.GameLogic.CurrentTurnAgentGetter(&state)
			solver := e.AgentSolvers[agent]
			legalActions := e.GameLogic.LegalActionsProvider(&state)
	
			policy, evals, err := solver.ActorCritic(&state, legalActions)
			if err != nil {
				return SolverEpisode[S, As, A, Ag]{}, err
			}
			action := solver.Selector(policy)

			episode.States = append(episode.States, state)
			episode.Agents = append(episode.Agents, agent)
			episode.Policies = append(episode.Policies, policy)
			episode.AgentEvalTable = append(episode.AgentEvalTable, evals)
	
			state, err = e.GameLogic.Transitioner(state, &action)
			if err != nil {
				return SolverEpisode[S, As, A, Ag]{}, err
			}
		}
	}
	return SolverEpisode[S, As, A, Ag]{}, nil
}

type SolverEpisode[S any, As ~[]A, A, Ag comparable] struct {
	States []S
	Agents []Ag
	Actions As
	Policies []Policy[A]
	AgentEvalTable []AgentEvals[Ag]
	ResultScores []float64
}

func (e *SolverEpisode[S, As, A, Ag]) MakeValueLabels(agents []Ag, resultRatio float64) (tensor.D2, error) {
	if resultRatio < 0.0 || resultRatio > 1.0 {
		return tensor.D2{}, fmt.Errorf("引数のゲーム結果の比率は、0.0～1.0でなければならない。")
	}
	evalRatio := 1.0 - resultRatio

	labels := make(tensor.D2, len(e.States))
	agentsN := len(agents)
	for i, evals := range e.AgentEvalTable {
		label := make(tensor.D1, agentsN)
		for _, agent := range agents {
			eval := evals[agent]
			label[i] = evalRatio * float64(eval) + (resultRatio * e.ResultScores[i])
		}
		labels[i] = label
	}
	return labels, nil
}

func NewPlayer() {
	
}