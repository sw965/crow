package sequential

import (
	"fmt"
	"sort"
	"github.com/sw965/omw/parallel"
	"math/rand"
	orand "github.com/sw965/omw/math/rand"
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

func (l Logic[S, As, A, G]) Playouts(initStates []S, selector Selector[S, As, A], rngs []*rand.Rand) ([]S, error) {
	n := len(initStates)
	finals := make([]S, n)

	p := len(rngs)
	errCh := make(chan error, p)
	worker := func(workerIdx int, statesIdxs []int) {
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
				percentByAction, err := selector(state, legalActions)
				if err != nil {
					errCh <- err
					return
				}

				percentByLegalAction := map[A]float32{}
				pSum := float32(0.0)
				for _, a := range legalActions {
					p, ok := percentByAction[a]
					if !ok {
						errCh <- fmt.Errorf("Selectorの戻り値には、必ず全ての合法行動を含む必要があります。")
						return
					}
					percentByLegalAction[a] = p
					pSum += p
				}

				if len(percentByLegalAction) == 0 {
					errCh <- fmt.Errorf("合法行動が存在しない")
					return
				}

				if pSum <= 0.0 {
					errCh <- fmt.Errorf("合法行動の確率分布の和が0以下")
					return
				}

				actions := make(As, 0, len(percentByLegalAction))
				percents := make([]float32, 0, len(percentByLegalAction))

				for a, p := range percentByLegalAction {
					actions = append(actions, a)
					percents = append(percents, p)
				}

				rng := rngs[workerIdx]
				actionIdx := orand.IntByWeight(percents, rng)
				action := actions[actionIdx]

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

func (l Logic[S, As, A, G]) PlayoutsWithHistory(initStates []S, selector Selector[S, As, A], rngs []*rand.Rand) (History[S, As, A], error) {
	n := len(initStates)
	history := History[S, As, A]{
		IntermediateStatesByGame:make([][]S, n),
		FinalStateByGame:make([]S, n),
		ActionsByGame:make([]As, n),
	}

	for i := range history.IntermediateStatesByGame {
		history.IntermediateStatesByGame[i] = make([]S, 0, oneGameCap)
		history.ActionsByGame[i] = make(As, 0, oneGameCap)
	}

	p := len(rngs)
	errCh := make(chan error, p)
	worker := func(workerIdx int, statesIdxs []int) {
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
				percentByAction, err := selector(state, legalActions)
				if err != nil {
					errCh <- err
					return
				}

				percentByLegalAction := map[A]float32{}
				pSum := float32(0.0)
				for _, a := range legalActions {
					p, ok := percentByAction[a]
					if !ok {
						errCh <- fmt.Errorf("Selectorの戻り値には、必ず全ての合法行動を含む必要があります。")
						return
					}
					percentByLegalAction[a] = p
					pSum += p
				}

				if len(percentByLegalAction) == 0 {
					errCh <- fmt.Errorf("合法行動が存在しない")
					return
				}

				if pSum <= 0.0 {
					errCh <- fmt.Errorf("合法行動の確率分布の和が0以下")
					return
				}

				actions := make(As, 0, len(percentByLegalAction))
				percents := make([]float32, 0, len(percentByLegalAction))

				for a, p := range percentByLegalAction {
					actions = append(actions, a)
					percents = append(percents, p)
				}

				rng := rngs[workerIdx]
				actionIdx := orand.IntByWeight(percents, rng)
				action := actions[actionIdx]

				state, err = l.Transitioner(state, action)
				if err != nil {
					errCh <- err
					return
				}

				history.IntermediateStatesByGame[idx] = append(history.IntermediateStatesByGame[idx], state)
				history.ActionsByGame[idx] = append(history.ActionsByGame[idx], action)
			}
		}
		errCh <- nil
	}

	for workerIdx, statesIdxs := range parallel.DistributeIndicesEvenly(n, p) {
		go worker(workerIdx, statesIdxs)
	}

	for i := 0; i < p; i++ {
		if err := <-errCh; err != nil {
			return History[S, As, A]{}, err
		}
	}
	return history, nil	
}

type Selector[S any, As ~[]A, A comparable] func(S, As) (map[A]float32, error)

func UniformSelector[S any, As ~[]A, A comparable](state S, legalActions As) (map[A]float32, error) {
	n := len(legalActions)
	p := 1.0 / float32(n)
	m := map[A]float32{}
	for _, a := range legalActions {
		m[a] = p
	}
	return m, nil
}

type History[S any, As ~[]A, A comparable] struct {
	IntermediateStatesByGame [][]S
	FinalStateByGame         []S
	ActionsByGame            []As
}

var oneGameCap int = 256

func GetOneGameCap() int {
	return oneGameCap
}

func SetOneGameCap(c int) {
	oneGameCap = c
}