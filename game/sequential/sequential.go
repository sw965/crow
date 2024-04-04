package sequential

import (
	"math/rand"

	"github.com/sw965/omw"
)

type Player[S any, A comparable] func(*S) A
type LegalActionsFunc[S any, AS ~[]A, A comparable] func(*S) AS
type PushFunc[S any, A comparable] func(S, A) S
type EqualFunc[S any] func(*S, *S) bool
type IsEndFunc[S any] func(*S) bool

type Game[S any, AS ~[]A, A comparable] struct {
	Player Player[S, A]
	LegalActions LegalActionsFunc[S, AS, A]
	Push PushFunc[S, A]
	Equal EqualFunc[S]
	IsEnd IsEndFunc[S]
}

func (g *Game[S, AS, A]) Clone() Game[S, AS, A] {
	return Game[S, AS, A]{
		Player:g.Player,
		LegalActions:g.LegalActions,
		Push:g.Push,
		Equal:g.Equal,
		IsEnd:g.IsEnd,
	}
}

func (g *Game[S, AS, A]) SetRandomActionPlayer(r *rand.Rand) {
	g.Player = func(state *S) A {
		actions := g.LegalActions(state)
		return omw.RandChoice(actions, r)
	}
}

func (g *Game[S, AS, A]) Playout(state S) S {
	for {
		isEnd := g.IsEnd(&state)
		if isEnd {
			break
		}
		action := g.Player(&state)
		state = g.Push(state, action)
	}
	return state
}

func (g *Game[S, AS, A]) PlayoutWithHistory(state S, cap_ int) (S, []S, AS) {
	ss := make([]S, 0, cap_)
	as := make(AS, 0, cap_)

	for {
		isEnd := g.IsEnd(&state)
		if isEnd {
			break
		}
		action := g.Player(&state)
		ss = append(ss, state)
		as = append(as, action)
		state = g.Push(state, action)
	}
	return state, ss, as
}