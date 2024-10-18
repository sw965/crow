package simultaneous

import (
	omwrand "github.com/sw965/omw/math/rand"
	"math/rand"
)

type Player[S any, As ~[]A, A comparable] func(*S) (As, error)
type SeparateLegalActionsProvider[S any, Ass ~[]As, As ~[]A, A comparable] func(*S) Ass
type Transitioner[S any, As ~[]A, A comparable] func(S, As) (S, error)
type Comparator[S any] func(*S, *S) bool
type EndChecker[S any] func(*S) bool

type Logic[S any, Ass ~[]As, As ~[]A, A comparable] struct {
	SeparateLegalActionsProvider SeparateLegalActionsProvider[S, Ass, As, A]
	Transitioner                 Transitioner[S, As, A]
	Comparator                   Comparator[S]
	EndChecker                   EndChecker[S]
}

func (l *Logic[S, Ass, As, A]) NewRandActionPlayer(r *rand.Rand) Player[S, As, A] {
	return func(state *S) (As, error) {
		ass := l.SeparateLegalActionsProvider(state)
		as := make(As, len(ass))
		for playerI, legalAs := range ass {
			as[playerI] = omwrand.Choice(legalAs, r)
		}
		return as, nil
	}
}

func (l *Logic[S, Ass, As, A]) Play(player Player[S, As, A], state S, f func(*S) bool) (S, error) {
	for {
		isEnd := l.EndChecker(&state)
		if isEnd || f(&state) {
			break
		}

		jointAction, err := player(&state)
		if err != nil {
			var s S
			return s, err
		}

		state, err = l.Transitioner(state, jointAction)
		if err != nil {
			var s S
			return s, err
		}
	}
	return state, nil
}

func (l *Logic[S, Ass, As, A]) Playout(player Player[S, As, A], state S) (S, error) {
	return l.Play(player, state, func(_ *S) bool { return false })
}

type ResultJointEval []float64
type ResultJointEvaluator[S any] func(*S) (ResultJointEval, error)
