package simultaneous

import (
    "fmt"
    "sort"
    "math/rand"
    omwrand "github.com/sw965/omw/math/rand"
    "github.com/sw965/crow/game/sequential"
)

type LegalActionTableProvider[S any, Ass ~[]As, As ~[]A, A comparable] func(*S) Ass
type Transitioner[S any, As ~[]A, A comparable] func(S, As) (S, error)
type Comparator[S any] func(*S, *S) bool

type Placements []int

func (p Placements) Validate() error {
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

type PlacementsJudger[S any] func(*S) (Placements, error)
type ResultScores []float64

func (ss ResultScores) ToEvals() Evals {
	es := make(Evals, len(ss))
	for i, s := range ss {
		es[i] = Eval(s)
	}
	return es
}

func (ss ResultScores) DivScalar(a float64) {
	for i := range ss {
		ss[i] /= a
	}
}

type ResultScoresEvaluator func(Placements) (ResultScores, error)

type Logic[S any, Ass ~[]As, As ~[]A, A comparable] struct {
    LegalActionTableProvider LegalActionTableProvider[S, Ass, As, A]
    Transitioner             Transitioner[S, As, A]
    Comparator               Comparator[S]
    PlacementsJudger         PlacementsJudger[S]
    ResultScoresEvaluator    ResultScoresEvaluator
}

func (l Logic[S, Ass, As, A]) IsEnd(s *S) (bool, error) {
    placements, err := l.PlacementsJudger(s)
    return len(placements) != 0, err
}

func (l *Logic[S, Ass, As, A]) SetStandardResultScoresEvaluator() {
    l.ResultScoresEvaluator = func(placements Placements) (ResultScores, error) {
        if err := placements.Validate(); err != nil {
            return ResultScores{}, err
        }

        n := len(placements)
        scores := make(ResultScores, n)

        if n == 1 {
            scores[0] = 1.0
            return scores, nil
        }

        counts := map[int]int{}
        for _, rank := range placements {
            counts[rank]++
        }

        for i, rank := range placements {
            score := 1.0 - ((float64(rank) - 1.0) / (float64(n) - 1.0))
            // 同順位の人数で割る
            scores[i] = score / float64(counts[rank])
        }
        return scores, nil
    }
}

func (l Logic[S, Ass, As, A]) EvaluateResultScores(s *S) (ResultScores, error) {
    placements, err := l.PlacementsJudger(s)
    if err != nil {
        return ResultScores{}, err
    }
    return l.ResultScoresEvaluator(placements)
}

type Player[S any, Ass ~[]As, As ~[]A, A comparable] func(*S, Ass) (As, error)

func NewRandActionPlayer[S any, Ass ~[]As, As ~[]A, A comparable](r *rand.Rand) Player[S, Ass, As, A] {
    return func(state *S, legalActionTable Ass) (As, error) {
        jointAction := make(As, len(legalActionTable))
		for i, legalActions := range legalActionTable {
			jointAction[i] = omwrand.Choice(legalActions, r)
		}
		return jointAction, nil
    }
}

type Engine[S any, Ass ~[]As, As ~[]A, A comparable] struct {
    Logic   Logic[S, Ass, As, A]
    Player Player[S, Ass, As, A]
}

func (e *Engine[S, Ass, As, A]) Playout(state S) (S, error) {
    for {
        isEnd, err := e.Logic.IsEnd(&state)
        if err != nil {
            var s S
            return s, err
        }

        if isEnd {
            break
        }

        legalActionTable := e.Logic.LegalActionTableProvider(&state)
        jointAction, err := e.Player(&state, legalActionTable)
		if err != nil {
			var s S
			return s, err
		}

        state, err = e.Logic.Transitioner(state, jointAction)
        if err != nil {
            var s S
            return s, err
        }
    }
    return state, nil
}

func (e *Engine[S, Ass, As, A]) ComparePlayerStrength(state S, playerNum, gameNum int) (ResultScores, error) {
    avgs := make(ResultScores, playerNum)
    for i := 0; i < gameNum; i++ {
        final, err := e.Playout(state)
        if err != nil {
            return nil, err
        }

        scores, err := e.Logic.EvaluateResultScores(&final)
        if err != nil {
            return nil, err
        }
        for k, v := range scores {
            avgs[k] += v
        }
    }
	avgs.DivScalar(float64(gameNum))
    return avgs, nil
}

func (e *Engine[S, Ass, As, A]) NewStates(init S, n int, r *rand.Rand) ([]S, error) {
    if n <= 0 {
        return []S{}, fmt.Errorf("引数のnが0以下です。0より大きい値にしてください。")
    }

    states := make([]S, 0, n)
    c := 0

    for {
        state := init
        for {
            isEnd, err := e.Logic.IsEnd(&state)
            if err != nil {
                return []S{}, err
            }
    
            if isEnd {
                break
            }

            states = append(states, state)
            c += 1

            if n == c {
                return states, nil
            }

            legalActionTable := e.Logic.LegalActionTableProvider(&state)
            jointAction, err := e.Player(&state, legalActionTable)
            if err != nil {
                return []S{}, err
            }
    
            state, err = e.Logic.Transitioner(state, jointAction)
            if err != nil {
                return []S{}, err
            }
        }
    }
    return states, nil
}

type Policy[A comparable] map[A]float64
type Policies[A comparable] []Policy[A]

type PoliciesProvider[S any, Ass ~[]As, As ~[]A, A comparable] func(*S, Ass) Policies[A]

func UniformPoliciesProvider[S any, Ass ~[]As, As ~[]A, A comparable](state *S, legalActionTable Ass) Policies[A] {
	policies := make(Policies[A], len(legalActionTable))
	for i, legalActions := range legalActionTable {
		n := len(legalActions)
		p := 1.0 / float64(n)
		policy := Policy[A]{}
		for _, a := range legalActions {
			policy[a] = p
		}
		policies[i] = policy
	}
	return policies
}

type Eval float64
type Evals []Eval

type Selector[As ~[]A, A comparable] func(Policies[A]) As

func NewThresholdWeightedSelector[As ~[]A, A comparable](t float64, r *rand.Rand) Selector[As, A] {
    return func(ps Policies[A]) As {
        jointAction := make(As, len(ps))
        for i, p := range ps {
            sp := sequential.Policy[A]{}
            for k, v := range p {
                sp[k] = v
            }
            jointAction[i] = sequential.NewThresholdWeightedSelector[A](t, r)(sp)
        }
        return jointAction
    }
}