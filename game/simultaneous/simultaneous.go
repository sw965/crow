package simultaneous

import (
	"math/rand"
	"github.com/sw965/omw"
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

func (g *Game[S, ASS, AS, A]) SetRandomActionPlayer(r *rand.Rand) {
	g.Player = func(state *S) (AS, error) {
		actionss := g.LegalActionss(state)
		result := make(AS, len(actionss))
		for playerI, as := range actionss {
			result[playerI] = omw.RandChoice(as, r)
		}
		return result, nil
	}
}

func (g *Game[S, ASS, AS, A]) Playout(state S) (S, error) {
	for {
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