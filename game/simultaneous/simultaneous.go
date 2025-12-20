package simultaneous

import (
    "fmt"
    "sort"
    "math/rand"
    "github.com/sw965/omw/mathx/randx"
)

type LegalActionTableProvider[S any, Ass ~[]As, As ~[]A, A comparable] func(*S) Ass
type Transitioner[S any, As ~[]A, A comparable] func(S, As) (S, error)
type Comparator[S any] func(*S, *S) bool

type Placements []int

func (ps Placements) Validate() error {
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

type PlacementsJudger[S any] func(*S) (Placements, error)
type ResultScores []float32

func (ss ResultScores) DivScalar(a float32) {
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
            score := 1.0 - ((float32(rank) - 1.0) / (float32(n) - 1.0))
            // 同順位の人数で割る
            scores[i] = score / float32(counts[rank])
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

func (l *Logic[S, Ass, As, A]) Playout(state S, players Players[S, Ass, As, A]) (S, error) {
    n := len(players)
    for {
        isEnd, err := l.IsEnd(&state)
        if err != nil {
            var s S
            return s, err
        }

        if isEnd {
            break
        }

        legalActionTable := l.LegalActionTableProvider(&state)
        jointAction := make(As, n)
        for i, player := range players {
            ja, err := player(&state, legalActionTable)
            if err != nil {
                var s S
                return s, err
            }
            jointAction[i] = ja[i]
        }

        state, err = l.Transitioner(state, jointAction)
        if err != nil {
            var s S
            return s, err
        }
    }
    return state, nil
}

func (l *Logic[S, Ass, As, A]) ComparePlayerStrength(state S, players Players[S, Ass, As, A], gameNum int) (ResultScores, error) {
    avgs := make(ResultScores, len(players))
    for i := 0; i < gameNum; i++ {
        final, err := l.Playout(state, players)
        if err != nil {
            return nil, err
        }

        scores, err := l.EvaluateResultScores(&final)
        if err != nil {
            return nil, err
        }
        for k, v := range scores {
            avgs[k] += v
        }
    }
	avgs.DivScalar(float32(gameNum))
    return avgs, nil
}

func (l *Logic[S, Ass, As, A]) NewRandActionPlayer(r *rand.Rand) Player[S, Ass, As, A] {
    return func(state *S, legalActionTable Ass) (As, error) {
        jointAction := make(As, len(legalActionTable))
        for i, actions := range legalActionTable {
            idx := r.IntN(len(actions))
            jointAction[i] = actions[idx]
        }
        return jointAction, nil
    }
}

func (l *Logic[S, Ass, As, A]) MakePlayers(n int) Players[S, Ass, As, A] {
    return make(Players[S, Ass, As, A], n)
}

type Player[S any, Ass ~[]As, As ~[]A, A comparable] func(*S, Ass) (As, error)
type Players[S any, Ass ~[]As, As ~[]A, A comparable] []Player[S, Ass, As, A]