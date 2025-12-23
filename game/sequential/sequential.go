// Package sequential provides utilities to run game playouts in a sequential-game setting.
// Policy consistency validation is centralized in Engine.Playouts.
//
// Package sequential は逐次（ターン制）ゲームのプレイアウト実行ユーティリティを提供します。
// Policy の整合性チェックは Engine.Playouts に集約されています。
package sequential

import (
	"errors"
	"fmt"
	"github.com/sw965/omw/mathx/randx"
	"github.com/sw965/omw/parallel"
	"github.com/sw965/omw/slicesx"
	"maps"
	"math"
	"math/rand/v2"
	"slices"
	"sort"
)

var (
	ErrEmptySlice     = errors.New("空スライスエラー")

    ErrNilLogicFunc = errors.New("Logicエラー: フィールドの関数がnilです")
    ErrNilEngineFunc = errors.New("Engineエラー: フィールドの関数がnilです")

	ErrDuplicateAgent = errors.New("エージェント重複エラー")
	ErrAgentNotFound = errors.New("Agentエラー: Agentsに存在しません")

	ErrInvalidRankValue  = errors.New("順位エラー: 1以上の正の整数である必要があります")
	ErrMinRankNotOne     = errors.New("最小順位エラー: 1から始まる必要があります")
	ErrRankNotContiguous = errors.New("順位不連続エラー: 順位が連続していません")

	ErrEmptyLegalMoves     = errors.New("legalMovesエラー: 要素数が0です")
	ErrNotUniqueLegalMoves = errors.New("legalMovesエラー: 重複した要素があります")

	ErrPolicySizeMismatch     = errors.New("Policyエラー: legalMoves と同じ要素数である必要があります")
	ErrPolicyMissingLegalMove = errors.New("Policyエラー: 全ての合法手を含む必要があります")
	ErrPolicyBadValue         = errors.New("Policyエラー: 値が不正です（負数/NaN/Inf）")
	ErrPolicyZeroSum          = errors.New("Policyエラー: 合計値が0です")

	ErrNilActorFunc = errors.New("Actorエラー: フィールドの関数がnilです")
)

type LegalMovesFunc[S any, M comparable] func(S) []M
type MoveFunc[S any, M comparable] func(S, M) (S, error)
type EqualFunc[S any] func(S, S) bool
type CurrentAgentFunc[S any, A comparable] func(S) A

type Logic[S any, M, A comparable] struct {
	LegalMovesFunc   LegalMovesFunc[S, M]
	MoveFunc         MoveFunc[S, M]
	EqualFunc        EqualFunc[S]
	CurrentAgentFunc CurrentAgentFunc[S, A]
}

func (l Logic[S, M, A]) Validate() error {
    if l.LegalMovesFunc == nil {
        return fmt.Errorf("%w: LegalMovesFunc", ErrNilLogicFunc)
    }
    if l.MoveFunc == nil {
        return fmt.Errorf("%w: MoveFunc", ErrNilLogicFunc)
    }
    if l.EqualFunc == nil {
        return fmt.Errorf("%w: EqualFunc", ErrNilLogicFunc)
    }
    if l.CurrentAgentFunc == nil {
        return fmt.Errorf("%w: CurrentAgentFunc", ErrNilLogicFunc)
    }
    return nil
}

// ゲームが終了していない場合は、戻り値は空あるいはnilである事を想定。
type RankByAgent[A comparable] map[A]int

func NewRankByAgent[A comparable](agentsPerRank [][]A) (RankByAgent[A], error) {
	ranks := RankByAgent[A]{}
	rank := 1
	for _, agents := range agentsPerRank {
		if len(agents) == 0 {
			return nil, fmt.Errorf("順位 %d に対応するエージェントが存在しません: %w", rank, ErrEmptySlice)
		}

		for _, agent := range agents {
			if _, ok := ranks[agent]; ok {
				return nil, fmt.Errorf("エージェント %v が複数回出現しています: %w", agent, ErrDuplicateAgent)
			}
			ranks[agent] = rank
		}
		rank += len(agents)
	}
	return ranks, nil
}

func (r RankByAgent[A]) Validate() error {
	n := len(r)
	if n == 0 {
		return nil
	}

	ranks := make([]int, 0, n)
	for _, rank := range r {
		if rank < 1 {
			return fmt.Errorf("%w", ErrInvalidRankValue)
		}
		ranks = append(ranks, rank)
	}
	sort.Ints(ranks)

	current := ranks[0]
	if current != 1 {
		return fmt.Errorf("%w: 入力された最小順位: %d", ErrMinRankNotOne, current)
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
			return fmt.Errorf("%w", ErrRankNotContiguous)
		}
	}
	return nil
}

type RankByAgentFunc[S any, A comparable] func(S) (RankByAgent[A], error)
type ResultScoreByAgent[A comparable] map[A]float32
type ResultScoreByAgentFunc[A comparable] func(RankByAgent[A]) (ResultScoreByAgent[A], error)

type Engine[S any, M, A comparable] struct {
	Logic                  Logic[S, M, A]
	RankByAgentFunc        RankByAgentFunc[S, A]
	ResultScoreByAgentFunc ResultScoreByAgentFunc[A]
	Agents                 []A
}

func (e Engine[S, M, A]) Validate() error {
    if err := e.Logic.Validate(); err != nil {
        return err
    }

    if e.RankByAgentFunc == nil {
        return fmt.Errorf("%w: RankByAgentFunc", ErrNilEngineFunc)
    }

    if e.ResultScoreByAgentFunc == nil {
        return fmt.Errorf("%w: ResultScoreByAgentFunc", ErrNilEngineFunc)
    }

    if len(e.Agents) == 0 {
		return fmt.Errorf("%w: Engine.Agents が空です", ErrEmptySlice)
	}
    return nil
}

func (e Engine[S, M, A]) IsEnd(state S) (bool, error) {
	rankByAgent, err := e.RankByAgentFunc(state)
	return len(rankByAgent) != 0, err
}

func (e *Engine[S, M, A]) SetStandardResultScoreByAgentFunc() {
	e.ResultScoreByAgentFunc = func(ranks RankByAgent[A]) (ResultScoreByAgent[A], error) {
		if err := ranks.Validate(); err != nil {
			return nil, err
		}

		n := len(ranks)
		scores := map[A]float32{}

		// エージェントが1人だけなら 1.0 固定
		if n == 1 {
			for agent := range ranks {
				scores[agent] = 1.0
			}
			return scores, nil
		}

		counts := map[int]int{}
		for _, rank := range ranks {
			counts[rank]++
		}

		den := float32(n - 1)

		tieScore := func(r, k int) float32 {
			return 1.0 - float32(2*r+k-3)/(2.0*den)
		}

		for agent, r := range ranks {
			k := counts[r]
			scores[agent] = tieScore(r, k)
		}
		return scores, nil
	}
}

func (e Engine[S, M, A]) EvaluateResultScoreByAgent(state S) (map[A]float32, error) {
	rankByAgent, err := e.RankByAgentFunc(state)
	if err != nil {
		return nil, err
	}
	return e.ResultScoreByAgentFunc(rankByAgent)
}

func (e Engine[S, M, A]) Playouts(inits []S, actor Actor[S, M, A], rngs []*rand.Rand) ([]S, error) {
	if err := e.Validate(); err != nil {
		return nil, err
	}

	if err := actor.Validate(); err != nil {
		return nil, err
	}

	n := len(inits)
	p := len(rngs)
	finals := make([]S, n)

	err := parallel.For(n, p, func(workerId, idx int) error {
		rng := rngs[workerId]
		state := inits[idx]
		for {
			isEnd, err := e.IsEnd(state)
			if err != nil {
				return err
			}

			if isEnd {
				break
			}

			legalMoves := e.Logic.LegalMovesFunc(state)
			// policy.ValidateForLegalMovesでもlegalMovesの空チェックをするが、PolicyFuncを安全に呼ぶ為に、ここでもチェックする
			if len(legalMoves) == 0 {
				return ErrEmptyLegalMoves
			}
			policy := actor.PolicyFunc(state, legalMoves)

			// legalMovesがユニークならば、policyは合法手のみを持つ事が保障される
			// 一手毎に、legalMovesがユニークであるかをチェックするのは、計算コストの観点から見送る
			err = policy.ValidateForLegalMoves(legalMoves)
			if err != nil {
				return err
			}

			agent := e.Logic.CurrentAgentFunc(state)
			move := actor.SelectFunc(policy, agent, rng)

			state, err = e.Logic.MoveFunc(state, move)
			if err != nil {
				return err
			}
		}
		finals[idx] = state
		return nil
	})
	return finals, err
}

func (e Engine[S, M, A]) CrossPlayouts(inits []S, actors []Actor[S, M, A], rngs []*rand.Rand) ([]CrossPlayoutResult[S, M, A], error) {
	agentsN := len(e.Agents)
	if len(actors) < agentsN {
		return nil, fmt.Errorf("insufficient actors: have %d, need %d to fill all agent roles", len(actors), agentsN)
	}

	actorPermsSeq := slicesx.Permutations[[]Actor[S, M, A]](actors, agentsN)
	actorPerms := slices.Collect(actorPermsSeq)

	permsN := len(actorPerms)
	results := make([]CrossPlayoutResult[S, M, A], permsN)

	for pi, actorPerm := range actorPerms {
		actorByAgent := map[A]Actor[S, M, A]{}
		policyFuncByAgent := map[A]PolicyFunc[S, M]{}
		selectFuncByAgent := map[A]SelectFunc[M, A]{}

		for ai, agent := range e.Agents {
			actor := actorPerm[ai]
			actorByAgent[agent] = actor
			policyFuncByAgent[agent] = actor.PolicyFunc
			selectFuncByAgent[agent] = actor.SelectFunc
		}

		policyFunc := func(state S, legalMoves []M) Policy[M] {
			agent := e.Logic.CurrentAgentFunc(state)
			return policyFuncByAgent[agent](state, legalMoves)
		}

		selectFunc := func(p Policy[M], agent A, rng *rand.Rand) M {
			return selectFuncByAgent[agent](p, agent, rng)
		}

		newActor := Actor[S, M, A]{
			PolicyFunc: policyFunc,
			SelectFunc: selectFunc,
		}

		finals, err := e.Playouts(inits, newActor, rngs)
		if err != nil {
			return nil, err
		}

		results[pi] = CrossPlayoutResult[S, M, A]{
			ActorByAgent: actorByAgent,
			Finals:       finals,
		}
	}
	return results, nil
}

type Policy[M comparable] map[M]float32

func (p Policy[M]) ValidateForLegalMoves(legalMoves []M) error {
	if len(legalMoves) == 0 {
		return ErrEmptyLegalMoves
	}
	if len(p) != len(legalMoves) {
		return fmt.Errorf("%w: policy=%d legalMoves=%d", ErrPolicySizeMismatch, len(p), len(legalMoves))
	}

	var sum float32
	for i, m := range legalMoves {
		v, ok := p[m]
		if !ok {
			return fmt.Errorf("%w: idx=%d move=%v", ErrPolicyMissingLegalMove, i, m)
		}

		f64 := float64(v)
		if v < 0 || math.IsNaN(f64) || math.IsInf(f64, 0) {
			return fmt.Errorf("%w: idx=%d move=%v value=%v", ErrPolicyBadValue, i, m, v)
		}
		sum += v
	}

	if sum == 0 {
		return ErrPolicyZeroSum
	}
	return nil
}

type PolicyFunc[S any, M comparable] func(S, []M) Policy[M]

func UniformPolicyFunc[S any, M comparable](state S, legalMoves []M) Policy[M] {
	n := len(legalMoves)
	if n == 0 {
		panic("BUG: len(legalMoves) == 0 である為、UniformPolicyFuncが実行出来ません")
	}

	p := 1.0 / float32(n)
	policy := Policy[M]{}
	for _, a := range legalMoves {
		policy[a] = p
	}
	return policy
}

type SelectFunc[M, A comparable] func(Policy[M], A, *rand.Rand) M

func MaxSelectFunc[M, A comparable](policy Policy[M], agent A, rng *rand.Rand) M {
	keys := slices.Collect(maps.Keys(policy))
	max := policy[keys[0]]
	moves := []M{keys[0]}

	for _, k := range keys[1:] {
		v := policy[k]
		switch {
		case v > max:
			max = v
			moves = []M{k}
		case v == max:
			moves = append(moves, k)
		}
	}

	move, err := randx.Choice(moves, rng)
	if err != nil {
		panic(fmt.Sprintf("BUG: %v", err))
	}
	return move
}

func WeightedRandomSelectFunc[M, A comparable](policy Policy[M], agent A, rng *rand.Rand) M {
	n := len(policy)
	moves := make([]M, 0, n)
	ws := make([]float32, 0, n)
	for m, p := range policy {
		moves = append(moves, m)
		ws = append(ws, p)
	}

	idx, err := randx.IntByWeight(ws, rng)
	if err != nil {
		panic(fmt.Sprintf("BUG: %v", err))
	}
	return moves[idx]
}

type Actor[S any, M, A comparable] struct {
	Name       string
	PolicyFunc PolicyFunc[S, M]
	SelectFunc SelectFunc[M, A]
}

func NewRandomActor[S any, M, A comparable](name string) Actor[S, M, A] {
	return Actor[S, M, A]{
		Name:name,
		PolicyFunc:UniformPolicyFunc[S, M],
		SelectFunc:WeightedRandomSelectFunc[M, A],
	}
}

func (a Actor[S, M, A]) Validate() error {
	if a.PolicyFunc == nil {
		return fmt.Errorf("%w: PolicyFunc", ErrNilActorFunc)
	}
	if a.SelectFunc == nil {
		return fmt.Errorf("%w: SelectFunc", ErrNilActorFunc)
	}
	return nil
}

type CrossPlayoutResult[S any, M, A comparable] struct {
	ActorByAgent map[A]Actor[S, M, A]
	Finals       []S
}
