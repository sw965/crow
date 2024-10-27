package simultaneous

import (
	"fmt"
	"sort"
	omwrand "github.com/sw965/omw/math/rand"
	"math/rand"
)

type Player[S any, As ~[]A, A comparable] func(*S) (As, error)
type Players[S any, As ~[]A, A comparable] []Player[S, As, A]

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

func (l *Logic[S, As, A, Agent]) SetStandardResultScoresEvaluator() {
	l.ResultScoresEvaluator = func(placements Placements) (ResultScores, error) {
		if err := placements.Validate(); err != nil {
			return ResultScores{} , err
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

		scores := make(ResultScores, n)
		for i, rank := range placements {
			v := 1.0 - ((float64(rank) - 1.0) / (float64(n) - 1.0))
			scores[i] = v / float64(counts[rank])
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

func (l *Logic[S, Ass, As, A]) NewRandActionPlayer(r *rand.Rand) Player[S, As, A] {
	return func(state *S) (As, error) {
		ass := l.LegalActionTableProvider(state)
		joint := make(As, len(ass))
		for playerI, as := range ass {
			joint[playerI] = omwrand.Choice(as, r)
		}
		return joint, nil
	}
}

func (l *Logic[S, Ass, As, A]) Play(players Players[S, As, A], state S, f func(*S) bool) (S, error) {
	n := len(players)
	for {
		isEnd, err := l.IsEnd(&state)
		if err != nil {
			var s S
			return s, err
		}

		if isEnd || f(&state) {
			break
		}

		jointAction := make(As, n)
		for i, player := range players {
			jointA, err := player(&state)
			if err != nil {
				var s S
				return s, err
			}
			jointAction[i] = jointA[i]
		}

		state, err = l.Transitioner(state, jointAction)
		if err != nil {
			var s S
			return s, err
		}
	}
	return state, nil
}

func (l *Logic[S, Ass, As, A]) Playout(players Players[S, As, A], state S) (S, error) {
	return l.Play(players, state, func(_ *S) bool { return false })
}
