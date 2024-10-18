package sequential

import (
	omwrand "github.com/sw965/omw/math/rand"
	"math/rand"
)

type Player[S any, A comparable] func(*S) (A, error)
type LegalActionsProvider[S any, As ~[]A, A comparable] func(*S) As
type Transitioner[S any, A comparable] func(S, *A) (S, error)
type Comparator[S any] func(*S, *S) bool
type EndChecker[S any] func(*S) bool
type CurrentTurnAgentProvider[S any, Agent comparable] func(*S) Agent

type Logic[S any, As ~[]A, A, Agent comparable] struct {
	LegalActionsProvider     LegalActionsProvider[S, As, A]
	Transitioner             Transitioner[S, A]
	Comparator               Comparator[S]
	EndChecker               EndChecker[S]
	CurrentTurnAgentProvider CurrentTurnAgentProvider[S, Agent]
}

func (l *Logic[S, As, A, Agent]) NewRandActionPlayer(r *rand.Rand) Player[S, A] {
	return func(state *S) (A, error) {
		as := l.LegalActionsProvider(state)
		return omwrand.Choice(as, r), nil
	}
}

func (l *Logic[S, As, A, Agent]) Play(player Player[S, A], state S, f func(*S) bool) (S, error) {
	for {
		isEnd := l.EndChecker(&state)
		if isEnd || f(&state) {
			break
		}

		action, err := player(&state)
		if err != nil {
			var s S
			return s, err
		}

		state, err = l.Transitioner(state, &action)
		if err != nil {
			var s S
			return s, err
		}
	}
	return state, nil
}

func (l *Logic[S, As, A, Agent]) Playout(player Player[S, A], state S) (S, error) {
	return l.Play(player, state, func(_ *S) bool { return false })
}

type ResultEval[Agent comparable] map[Agent]float64
type ResultEvaluator[S any, Agent comparable] func(*S) (ResultEval[Agent], error)
