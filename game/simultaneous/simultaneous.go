package simultaneous

import (
	"math/rand"
	orand "github.com/sw965/omw/rand"
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
			ret[playerI] = orand.Choice(as, r)
		} 
		return ret, nil
	}
}

func (g *Game[S, ASS, AS, A]) Play(state S, n int) (S, error) {
	for i := 0; i < n; i++ {
		isEnd := g.IsEnd(&state)
		if isEnd {
			break
		}
		actions, err := g.Player(&state)
		if err != nil {
			var s S
			return s, err
		}
		state, err = g.Push(state, actions)
		if err != nil {
			var s S
			return s, err
		}
	}
	return state, nil
}

func (g *Game[S, ASS, AS, A]) Playout(state S) (S, error) {
	return g.Play(state, -1)
}

func (g *Game[S, ASS, AS, A]) RepeatedPlayout(state S, n int) ([]S, error) {
	ret := make([]S, n)
	for i := 0; i < n; i++ {
		endS, err := g.Playout(state)
		if err != nil {
			return []S{}, err
		}
		ret[i] = endS
	}
	return ret, nil
}

func (g *Game[S, ASS, AS, A]) PlayWithHistory(state S, n, c int) ([]S, error) {
	stateHistory := make([]S, 0, c)
	for i := 0; i < n; i++ {
		isEnd := g.IsEnd(&state)
		if isEnd {
			break
		}
		actions, err := g.Player(&state)
		if err != nil {
			return []S{}, err
		}
		state, err = g.Push(state, actions)
		if err != nil {
			return []S{}, err
		}
		stateHistory = append(stateHistory, state)
	}
	return stateHistory, nil
}

func (g *Game[S, ASS, AS, A]) PlayoutWithHistory(state S, c int) ([]S, error) {
	return g.PlayWithHistory(state, -1, c)
}