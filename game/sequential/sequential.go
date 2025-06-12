package sequential

import (
	"fmt"
	"sort"
	"github.com/sw965/omw/parallel"
	"math/rand"
	orand "github.com/sw965/omw/math/rand"
	oslices "github.com/sw965/omw/slices"
)

type LegalActionsProvider[S any, Ac comparable] func(S) []Ac
type Transitioner[S any, Ac comparable] func(S, Ac) (S, error)
type Comparator[S any] func(S, S) bool
type CurrentAgentGetter[S any, Ag comparable] func(S) Ag

// ゲームが終了していない場合は、戻り値は空あるいはnilである事を想定。
type PlacementByAgent[Ag comparable] map[Ag]int

func NewPlacementByAgent[Ag comparable](agentsByPlacement [][]Ag) (PlacementByAgent[Ag], error) {
	placements := PlacementByAgent[Ag]{}
	rank := 1
	for _, agents := range agentsByPlacement {
		if len(agents) == 0 {
			return nil, fmt.Errorf("順位 %d に対応するエージェントが存在しません", rank)
		}

		for _, agent := range agents {
			if _, ok := placements[agent]; ok {
				return nil, fmt.Errorf("エージェント %v が複数回出現しています", agent)
			}
			placements[agent] = rank
		}
		rank += len(agents)
	}
	return placements, nil
}

func (ps PlacementByAgent[Ag]) Validate() error {
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

type PlacementsJudger[S any, Ag comparable] func(S) (PlacementByAgent[Ag], error)
type ResultScoreByAgent[Ag comparable] map[Ag]float32

type ResultScoresEvaluator[Ag comparable] func(PlacementByAgent[Ag]) (ResultScoreByAgent[Ag], error)

type Logic[S any, Ac, Ag comparable] struct {
	LegalActionsProvider  LegalActionsProvider[S, Ac]
	Transitioner          Transitioner[S, Ac]
	Comparator            Comparator[S]
	CurrentAgentGetter    CurrentAgentGetter[S, Ag]
	PlacementsJudger      PlacementsJudger[S, Ag]
	ResultScoresEvaluator ResultScoresEvaluator[Ag]
	Agents                []Ag
}

func (l Logic[S, Ac, Ag]) IsEnd(state S) (bool, error) {
	placements, err := l.PlacementsJudger(state)
	return len(placements) != 0, err
}

func (l *Logic[S, Ac, Ag]) SetStandardResultScoresEvaluator() {
	l.ResultScoresEvaluator = func(placements PlacementByAgent[Ag]) (ResultScoreByAgent[Ag], error) {
		if err := placements.Validate(); err != nil {
			return ResultScoreByAgent[Ag]{} , err
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

		scores := ResultScoreByAgent[Ag]{}

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

func (l Logic[S, Ac, Ag]) EvaluateResultScoreByAgent(state S) (ResultScoreByAgent[Ag], error) {
	placements, err := l.PlacementsJudger(state)
	if err != nil {
		return ResultScoreByAgent[Ag]{}, err
	}
	return l.ResultScoresEvaluator(placements)
}

func (l Logic[S, Ac, Ag]) Playouts(inits []S, actor Actor[S, Ac, Ag], rngByWorker []*rand.Rand) ([]S, error) {
	n := len(inits)
	finals := make([]S, n)
	p := len(rngByWorker)
	errCh := make(chan error, p)

	worker := func(workerIdx int, statesIdxs []int) {
		rng := rngByWorker[workerIdx]
		for _, idx := range statesIdxs {
			state := inits[idx]
			for {
				isEnd, err := l.IsEnd(state)
				if err != nil {
					errCh <- err
					return
				}

				if isEnd {
					break
				}

				legalActions := l.LegalActionsProvider(state)
				policy := actor.PolicyProvider(state, legalActions)

				legalPolicy := Policy[Ac]{}
				for _, a := range legalActions {
					p, ok := policy[a]
					if !ok {
						errCh <- fmt.Errorf("Policyは、全ての合法行動を含む必要があります。")
						return
					}
					legalPolicy[a] = p
				}

				agent := l.CurrentAgentGetter(state)
				action, err := actor.Selector(legalPolicy, agent, rng)
				if err != nil {
					errCh <- err
					return
				}

				state, err = l.Transitioner(state, action)
				if err != nil {
					errCh <- err
					return
				}
			}
			finals[idx] = state
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

func (l Logic[S, Ac, Ag]) PlayoutsWithHistory(inits []S, actor Actor[S, Ac, Ag], rngByWorker []*rand.Rand) (History[S, Ac, Ag], error) {
	n := len(inits)
	history := NewHistory[S, Ac, Ag](n)
	p := len(rngByWorker)
	errCh := make(chan error, p)

	worker := func(workerIdx int, statesIdxs []int) {
		rng := rngByWorker[workerIdx]
		for _, idx := range statesIdxs {
			state := inits[idx]
			for {
				isEnd, err := l.IsEnd(state)
				if err != nil {
					errCh <- err
					return
				}

				if isEnd {
					history.FinalStateByGame[idx] = state
					break
				}

				legalActions := l.LegalActionsProvider(state)
				policy := actor.PolicyProvider(state, legalActions)

				legalPolicy := Policy[Ac]{}
				for _, a := range legalActions {
					p, ok := policy[a]
					if !ok {
						errCh <- fmt.Errorf("Policyは、全ての合法行動を含む必要があります。")
						return
					}
					legalPolicy[a] = p
				}

				agent := l.CurrentAgentGetter(state)
				action, err := actor.Selector(legalPolicy, agent, rng)
				if err != nil {
					errCh <- err
					return
				}

				history.IntermediateStatesByGame[idx] = append(history.IntermediateStatesByGame[idx], state)
				history.PoliciesByGame[idx] = append(history.PoliciesByGame[idx], policy)
				history.ActionsByGame[idx] = append(history.ActionsByGame[idx], action)
				history.AgentsByGame[idx] = append(history.AgentsByGame[idx], agent)

				state, err = l.Transitioner(state, action)
				if err != nil {
					errCh <- err
					return
				}
			}
		}
		errCh <- nil
	}

	for workerIdx, statesIdxs := range parallel.DistributeIndicesEvenly(n, p) {
		go worker(workerIdx, statesIdxs)
	}

	for i := 0; i < p; i++ {
		if err := <-errCh; err != nil {
			return History[S, Ac, Ag]{}, err
		}
	}
	return history, nil	
}

func (l Logic[S, Ac, Ag]) CrossPlayouts(inits []S, actors []Actor[S, Ac, Ag], rngByWorker []*rand.Rand) ([][]S, [][]Ag, error) {
	agentN := len(l.Agents)
	if len(actors) != agentN {
		return nil, nil, fmt.Errorf("len(actors) != len(Logic.Agents)")
	}

	actorsByOrder := oslices.Permutation[[][]Actor[S, Ac, Ag]](actors, agentN)
	agentsByOrder := oslices.Permutation[[][]Ag](l.Agents, agentN)
	finalsByOrder := make([][]S, len(actorsByOrder))

	for i, actors := range actorsByOrder {
		ppByAgent := map[Ag]PolicyProvider[S, Ac]{}
		selectorByAgent := map[Ag]Selector[Ac, Ag]{}
		for j, actor := range actors {
			agent := agentsByOrder[i][j]
			ppByAgent[agent] = actor.PolicyProvider
			selectorByAgent[agent] = actor.Selector
		}

		pp := func(state S, legalActions []Ac) Policy[Ac] {
			ag := l.CurrentAgentGetter(state)
			return ppByAgent[ag](state, legalActions)
		}

		selector := func(p Policy[Ac], ag Ag, rng *rand.Rand) (Ac, error) {
			return selectorByAgent[ag](p, ag, rng)
		}

		newActor := Actor[S, Ac, Ag]{
			PolicyProvider:pp,
			Selector:selector,
		}

		finals, err := l.Playouts(inits, newActor, rngByWorker)
		if err != nil {
			return nil, nil, err
		}
		finalsByOrder[i] = finals
	}
	return finalsByOrder, agentsByOrder, nil
}

type Policy[Ac comparable] map[Ac]float32
type PolicyProvider[S any, Ac comparable] func(S, []Ac) Policy[Ac]

func UniformLegalPolicyProvider[S any, Ac comparable](state S, legalActions []Ac) Policy[Ac] {
	n := len(legalActions)
	p := 1.0 / float32(n)
	m := Policy[Ac]{}
	for _, a := range legalActions {
		m[a] = p
	}
	return m
}

type Selector[Ac, Ag comparable] func(Policy[Ac], Ag, *rand.Rand) (Ac, error)

func WeightedRandomSelector[Ac, Ag comparable](policy Policy[Ac], agent Ag, rng *rand.Rand) (Ac, error) {
	n := len(policy)
	actions := make([]Ac, 0, n)
	percents := make([]float32, 0, n)
	for a, p := range policy {
		actions = append(actions, a)
		percents = append(percents, p)
	}
	idx, err := orand.IntByWeight(percents, rng)
	if err != nil {
		var a Ac
		return a, err
	}
	return actions[idx], nil
}

type Actor[S any, Ac, Ag comparable] struct {
	PolicyProvider PolicyProvider[S, Ac]
	Selector       Selector[Ac, Ag]
}

type History[S any, Ac, Ag comparable] struct {
	IntermediateStatesByGame [][]S
	FinalStateByGame         []S
	PoliciesByGame           [][]Policy[Ac]
	ActionsByGame            [][]Ac
	AgentsByGame             [][]Ag
}

func NewHistory[S any, Ac, Ag comparable](n int) History[S, Ac, Ag] {
	h := History[S, Ac, Ag]{
    	IntermediateStatesByGame: make([][]S, n),
    	FinalStateByGame:         make([]S, n),
    	PoliciesByGame:           make([][]Policy[Ac], n),
    	ActionsByGame:            make([][]Ac, n),
		AgentsByGame:             make([][]Ag, n),
	}

	for i := 0; i < n; i++ {
    	h.IntermediateStatesByGame[i] = make([]S, 0, oneGameCap)
    	h.PoliciesByGame[i] = make([]Policy[Ac], 0, oneGameCap)
    	h.ActionsByGame[i] = make([]Ac, 0, oneGameCap)
		h.AgentsByGame[i] = make([]Ag, 0, oneGameCap)
	}
	return h
}

func (h History[S, Ac, Ag]) ToExperiences(logic Logic[S, Ac, Ag]) (Experiences[S, Ac, Ag], error) {
	experiences := make(Experiences[S, Ac, Ag], 0, len(h.FinalStateByGame) * oneGameCap)
	for gameI, final := range h.FinalStateByGame {
		states := h.IntermediateStatesByGame[gameI]
		policies := h.PoliciesByGame[gameI]
		actions := h.ActionsByGame[gameI]
		agents := h.AgentsByGame[gameI]
		scores, err := logic.EvaluateResultScoreByAgent(final)
		if err != nil {
			return nil, err
		}
		for i, state := range states {
			agent := agents[i]
			experiences = append(experiences, Experience[S, Ac, Ag]{
				State:state,
				Policy:policies[i],
				Action:actions[i],
				ResultScore:scores[agent],
				Agent:agent,
			})
		}
	}
	return experiences, nil
}

type Experience[S any, Ac, Ag comparable] struct {
	State       S
	Policy      Policy[Ac]
	Action      Ac
	ResultScore float32
	Agent       Ag
}

type Experiences[S any, Ac, Ag comparable] []Experience[S, Ac, Ag]

func (es Experiences[S, Ac, Ag]) Split() ([]S, []Policy[Ac], []Ac, []float32, []Ag) {
	n := len(es)
	states := make([]S, n)
	policies := make([]Policy[Ac], n)
	actions := make([]Ac, n)
	scores := make([]float32, n)
	agents := make([]Ag, n)

	for i, e := range es {
		states[i] = e.State
		policies[i] = e.Policy
		actions[i] = e.Action
		scores[i] = e.ResultScore
		agents[i] = e.Agent
	}
	return states, policies, actions, scores, agents
}

var oneGameCap int = 256

func GetOneGameCap() int {
	return oneGameCap
}

func SetOneGameCap(c int) {
	oneGameCap = c
}