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
	p := len(rngByWorker)
	finals := make([]S, n)
	err := parallel.For(p, n, func(workerId, idx int) error {
		rng := rngByWorker[workerId]
		state := inits[idx]
		for {
			isEnd, err := l.IsEnd(state)
			if err != nil {
				return err
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
					return fmt.Errorf("Policyは、全ての合法行動を含む必要があります。")
				}
				legalPolicy[a] = p
			}

			agent := l.CurrentAgentGetter(state)
			action, err := actor.Selector(legalPolicy, agent, rng)
			if err != nil {
				return err
			}

			state, err = l.Transitioner(state, action)
			if err != nil {
				return err
			}
		}
		finals[idx] = state
		return nil
	})
	return finals, err
}

func (l Logic[S, Ac, Ag]) CrossPlayouts(inits []S, actors []Actor[S, Ac, Ag], rngByWorker []*rand.Rand) ([][]S, error) {
	agentsN := len(l.Agents)
	if len(actors) != agentsN {
		return nil, fmt.Errorf("len(actors) != len(Logic.Agents)")
	}

	actorPerms := oslices.Permutations[[]Actor[S, Ac, Ag]](actors, agentsN)
	permsN := len(actorPerms)
	finalsByAgPerm := make([][]S, permsN)

	for i, actorPerm := range actorPerms {
		ppByAgent := map[Ag]PolicyProvider[S, Ac]{}
		selectorByAgent := map[Ag]Selector[Ac, Ag]{}
		for j, actor := range actorPerm {
			agent := l.Agents[j]
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
			return nil, err
		}
		finalsByAgPerm[i] = finals
	}
	return finalsByAgPerm, nil
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