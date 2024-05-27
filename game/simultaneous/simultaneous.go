package simultaneous

import (
	"math/rand"
	omwrand "github.com/sw965/omw/math/rand"
)

type Player[S any, AS ~[]A, A comparable] func(*S) (AS, error)
type LegalActionssFunc[S any, ASS ~[]AS, AS ~[]A, A comparable] func(*S) ASS
type PushFunc[S any, AS ~[]A, A comparable] func(S, AS) (S, error)
type EqualFunc[S any] func(*S, *S) bool
type IsEndFunc[S any] func(*S) bool

type Game[S any, ASS ~[]AS, AS ~[]A, A comparable] struct {
	Player Player[S, AS, A]
	LegalActionss LegalActionssFunc[S, ASS, AS, A]
	Push PushFunc[S, AS, A]
	Equal EqualFunc[S]
	IsEnd IsEndFunc[S]
}

func (g *Game[S, ASS, AS, A]) Clone() Game[S, ASS, AS, A] {
	return Game[S, ASS, AS, A]{
		Player:g.Player,
		LegalActionss:g.LegalActionss,
		Push:g.Push,
		Equal:g.Equal,
		IsEnd:g.IsEnd,
	}
}

func (g *Game[S, ASS, AS, A]) SetRandActionPlayer(r *rand.Rand) {
	g.Player = func(state *S) (AS, error) {
		ass := g.LegalActionss(state)
		ret := make(AS, len(ass))
		for playerI, as := range ass {
			ret[playerI] = omwrand.Choice(as, r)
		} 
		return ret, nil
	}
}

func (g *Game[S, ASS, AS, A]) Play(state S, f func(*S, int) bool) (S, error) {
	i := 0
	for {
		if g.IsEnd(&state) || f(&state, i) {
			break
		}

		jointAction, err := g.Player(&state)
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

func (g *Game[S, ASS, AS, A]) PlayWithHistory(state S, f func(*S, int) bool, c int) (S, []S, ASS, error) {
	i := 0
	stateHistory := make([]S, 0, c)
	jointActionHistory := make(ASS, 0, c)
	for {
		if g.IsEnd(&state) || f(&state, i) {
			break
		}

		jointAction, err := g.Player(&state)
		if err != nil {
			var s S
			return s, []S{}, ASS{}, err
		}

		stateHistory = append(stateHistory, state)
		jointActionHistory = append(jointActionHistory, jointAction)

		state, err = g.Push(state, jointAction)
		if err != nil {
			var s S
			return s, []S{}, ASS{}, err
		}
		i += 1
	}
	return state, stateHistory, jointActionHistory, nil
}

func (g *Game[S, ASS, AS, A]) Playout(state S) (S, error) {
	return g.Play(state, func(_ *S, _ int) bool { return false })
}

func (g *Game[S, ASS, AS, A]) PlayoutWithHistory(state S, c int) (S, []S, ASS, error) {
	return g.PlayWithHistory(state, func(_ *S, _ int) bool { return false }, c)
}