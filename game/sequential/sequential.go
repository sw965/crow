package sequential

import (
	"fmt"
	"sort"
	"github.com/sw965/omw/parallel"
	"math/rand"
	orand "github.com/sw965/omw/math/rand"
	"math"
)

type LegalActionsProvider[S any, As ~[]A, A comparable] func(S) As
type Transitioner[S any, A comparable] func(S, A) (S, error)
type Comparator[S any] func(S, S) bool
type CurrentAgentGetter[S any, G comparable] func(S) G

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
	LegalActionsProvider  LegalActionsProvider[S, As, A]
	Transitioner          Transitioner[S, A]
	Comparator            Comparator[S]
	CurrentAgentGetter    CurrentAgentGetter[S, G]
	PlacementsJudger      PlacementsJudger[S, G]
	ResultScoresEvaluator ResultScoresEvaluator[G]
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

func (l Logic[S, As, A, G]) Playouts(initStates []S, pp PolicyProvider[S, As, A], rngByWorker []*rand.Rand) ([]S, error) {
	n := len(initStates)
	finals := make([]S, n)

	p := len(rngByWorker)
	errCh := make(chan error, p)
	worker := func(workerIdx int, statesIdxs []int) {
		rng := rngByWorker[workerIdx]
		for _, idx := range statesIdxs {
			state := initStates[idx]
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
				policy := pp(state, legalActions)
				legalPolicy := Policy[A]{}
				pSum := float32(0.0)
				for _, a := range legalActions {
					p, ok := policy[a]
					if !ok {
						errCh <- fmt.Errorf("Selectorの戻り値には、必ず全ての合法行動を含む必要があります。")
						return
					}

					if p <= 0 || math.IsNaN(float64(p)) {
        				errCh <- fmt.Errorf("確率 p が 0 より小さいまたは NaN です: p = %v", p)
        				return
    				}

					legalPolicy[a] = p
					pSum += p
				}

				if pSum <= 0.0 {
					errCh <- fmt.Errorf("合法行動の確率分布の和が0以下")
					return
				}

				action := legalPolicy.Sample(rng)

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

func (l Logic[S, As, A, G]) PlayoutsWithHistory(initStates []S, pp PolicyProvider[S, As, A], rngByWorker []*rand.Rand) (History[S, As, A, G], error) {
	n := len(initStates)

	history := History[S, As, A, G]{
    	IntermediateStatesByGame: make([][]S, n),
    	FinalStateByGame:         make([]S, n),
    	PoliciesByGame:           make([][]Policy[A], n),
    	ActionsByGame:            make([]As, n),
		AgentsByGame:             make([][]G, n),
	}

	for i := 0; i < n; i++ {
    	history.IntermediateStatesByGame[i] = make([]S, 0, oneGameCap)
    	history.PoliciesByGame[i] = make([]Policy[A], 0, oneGameCap)
    	history.ActionsByGame[i] = make(As, 0, oneGameCap)
		history.AgentsByGame[i] = make([]G, 0, oneGameCap)
    	// FinalStateByGame[i] はゲーム終了時に書き込まれるのでここでは不要
	}

	p := len(rngByWorker)
	errCh := make(chan error, p)
	worker := func(workerIdx int, statesIdxs []int) {
		rng := rngByWorker[workerIdx]
		for _, idx := range statesIdxs {
			state := initStates[idx]
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
				policy := pp(state, legalActions)
				legalPolicy := Policy[A]{}
				pSum := float32(0.0)
				for _, a := range legalActions {
					p, ok := policy[a]
					if !ok {
						errCh <- fmt.Errorf("Selectorの戻り値には、必ず全ての合法行動を含む必要があります。")
						return
					}

					if p <= 0 || math.IsNaN(float64(p)) {
        				errCh <- fmt.Errorf("確率 p が 0より小さい または NaN です: p = %v", p)
        				return
    				}

					legalPolicy[a] = p
					pSum += p
				}

				if pSum <= 0.0 {
					errCh <- fmt.Errorf("合法行動の確率分布の和が0以下")
					return
				}

				action := legalPolicy.Sample(rng)
				agent := l.CurrentAgentGetter(state)
				state, err = l.Transitioner(state, action)
				if err != nil {
					errCh <- err
					return
				}

				history.IntermediateStatesByGame[idx] = append(history.IntermediateStatesByGame[idx], state)
				history.PoliciesByGame[idx] = append(history.PoliciesByGame[idx], policy)
				history.ActionsByGame[idx] = append(history.ActionsByGame[idx], action)
				history.AgentsByGame[idx] = append(history.AgentsByGame[idx], agent)
			}
		}
		errCh <- nil
	}

	for workerIdx, statesIdxs := range parallel.DistributeIndicesEvenly(n, p) {
		go worker(workerIdx, statesIdxs)
	}

	for i := 0; i < p; i++ {
		if err := <-errCh; err != nil {
			return History[S, As, A, G]{}, err
		}
	}
	return history, nil	
}

/*
	Selector は「state → 各合法手とその重みを返す」関数。
	返却される Policy[A] は以下を必ず満たすこと：
	1) キーに legalActions のすべての要素を含むこと。
	2) 各 value (float32) は NaN でもマイナスでもなく、合計が 0 より大きいこと。
	これらを満たしていない場合、Playouts 系メソッド呼び出し時にエラーになる。

	※合法手以外をキーに含めても問題なし。(Playouts系メソッドでは、内部でそぎ落とされる)
	よってソフトマックスなどを出力するモデルの確率分布とそれに対応する行動からPolicyを作っても問題ない。
	また確率分布は内部(orand.IntByWeight)で1に正規化される為、合計が1である必要はなし。
*/
type Policy[A comparable] map[A]float32

func (p Policy[A]) Sample(rng *rand.Rand) A {
	n := len(p)
	actions := make([]A, 0, n)
	percents := make([]float32, 0, n)
	for a, v := range p {
		actions = append(actions, a)
		percents = append(percents, v)
	}
	idx := orand.IntByWeight(percents, rng)
	return actions[idx]
}

type PolicyProvider[S any, As ~[]A, A comparable] func(S, As) Policy[A]

func UniformPolicyProvider[S any, As ~[]A, A comparable](state S, legalActions As) Policy[A] {
	n := len(legalActions)
	p := 1.0 / float32(n)
	m := Policy[A]{}
	for _, a := range legalActions {
		m[a] = p
	}
	return m
}

type History[S any, As ~[]A, A, G comparable] struct {
	IntermediateStatesByGame [][]S
	FinalStateByGame         []S
	PoliciesByGame           [][]Policy[A]
	ActionsByGame            []As
	AgentsByGame             [][]G
}

func (h History[S, As, A, G]) ToExperiences(logic Logic[S, As, A, G]) (Experiences[S, A, G], error) {
	experiences := make(Experiences[S, A, G], 0, len(h.FinalStateByGame) * oneGameCap)
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
			experiences = append(experiences, Experience[S, A, G]{
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

type Experience[S any, A, G comparable] struct {
	State       S
	Policy      Policy[A]
	Action      A
	ResultScore float32
	Agent       G
}

type Experiences[S any, A, G comparable] []Experience[S, A, G]

func (es Experiences[S, A, G]) Split() ([]S, []Policy[A], []A, []float32, []G) {
	n := len(es)
	states := make([]S, n)
	policies := make([]Policy[A], n)
	actions := make([]A, n)
	scores := make([]float32, n)
	agents := make([]G, n)

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