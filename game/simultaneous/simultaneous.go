package simultaneous

import (
	"math/rand"
	omwrand "github.com/sw965/omw/math/rand"
)

type JointEval []float64
type JointEvals []JointEval
type Player[S any, AS ~[]A, A comparable] func(*S) (AS, JointEval, error)
type SeparateLegalActionsFunc[S any, ASS ~[]AS, AS ~[]A, A comparable] func(*S) ASS
type PushFunc[S any, AS ~[]A, A comparable] func(S, AS) (S, error)
type EqualFunc[S any] func(*S, *S) bool
type IsEndFunc[S any] func(*S) (bool, JointEval)

type Game[S any, ASS ~[]AS, AS ~[]A, A comparable] struct {
	SeparateLegalActions SeparateLegalActionsFunc[S, ASS, AS, A]
	Push PushFunc[S, AS, A]
	Equal EqualFunc[S]
	IsEnd IsEndFunc[S]
}

func (g *Game[S, ASS, AS, A]) NewRandActionPlayer(r *rand.Rand) Player[S, AS, A] {
	return func(state *S) (AS, JointEval, error) {
		ass := g.SeparateLegalActions(state)
		as := make(AS, len(ass))
		for playerI, legalAs := range ass {
			as[playerI] = omwrand.Choice(legalAs, r)
		} 
		return as, JointEval{}, nil
	}
}

func (g *Game[S, ASS, AS, A]) Play(player Player[S, AS, A], state S, f func(*S, int) bool) (S, error) {
	i := 0
	for {
		isEnd, _ := g.IsEnd(&state)
		if isEnd || f(&state, i) {
			break
		}

		jointAction, _, err := player(&state)
		if err != nil {
			var s S
			return s, err
		}
		state, err = g.Push(state, jointAction)
		if err != nil {
			var s S
			return s, err
		}
		i += 1
	}
	return state, nil
}

func (g *Game[S, ASS, AS, A]) PlayWithHistory(player Player[S, AS, A], state S, f func(*S, int) bool, c int) (S, []S, ASS, JointEvals, error) {
	i := 0
	stateHistory := make([]S, 0, c)
	jointActionHistory := make(ASS, 0, c)
	jointEvalHistory := make(JointEvals, 0, c)

	for {
		isEnd, _ := g.IsEnd(&state)
		if isEnd || f(&state, i) {
			break
		}

		jointAction, jointEval, err := player(&state)
		if err != nil {
			var s S
			return s, []S{}, ASS{}, JointEvals{}, err
		}

		stateHistory = append(stateHistory, state)
		jointActionHistory = append(jointActionHistory, jointAction)
		jointEvalHistory = append(jointEvalHistory, jointEval)

		state, err = g.Push(state, jointAction)
		if err != nil {
			var s S
			return s, []S{}, ASS{}, JointEvals{}, err
		}
		i += 1
	}
	return state, stateHistory, jointActionHistory, jointEvalHistory, nil
}

func (g *Game[S, ASS, AS, A]) Playout(player Player[S, AS, A], state S) (S, error) {
	return g.Play(player, state, func(_ *S, _ int) bool { return false })
}

func (g *Game[S, ASS, AS, A]) PlayoutWithHistory(player Player[S, AS, A], state S, capacity int) (S, []S, ASS, JointEvals, error) {
	return g.PlayWithHistory(player, state, func(_ *S, _ int) bool { return false }, capacity)
}