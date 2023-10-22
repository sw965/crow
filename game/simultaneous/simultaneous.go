package simultaneous

import (
	"math/rand"
	omwrand "github.com/sw965/omw/rand"
)

type Player[S any, AS ~[]A, A comparable] func(*S) AS
type LegalActionssFunc[S any, ASS ~[]AS, AS ~[]A, A comparable] func(*S) ASS
type PushFunc[S any, A comparable] func(S, ...A) S
type EqualFunc[S any] func(*S, *S) bool
type IsEndFunc[S any] func(*S) bool

type Game[S any, ASS ~[]AS, AS ~[]A, A comparable] struct {
	Player Player[S, AS, A]
	LegalActionss LegalActionssFunc[S, ASS, AS, A]
	Push PushFunc[S, A]
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

func (g *Game[S, ASS, AS, A]) PadLegalActionss(state *S, noAction A) ASS {
	actionss := g.LegalActionss(state)
	ass := make(ASS, len(actionss))
	for playerI, actions := range actionss {
		if len(actions) == 0 {
			ass[playerI] = AS{noAction}
		} else {
			ass[playerI] = actionss[playerI]
		}
	}
	return ass
}

func (g *Game[S, ASS, AS, A]) SetRandomActionPlayer(r *rand.Rand, noAction A) {
	g.Player = func(state *S) AS {
		actionss := g.PadLegalActionss(state, noAction)
		y := make([]A, len(actionss))
		for playerI, actions := range actionss {
			y[playerI] = omwrand.Choice(actions, r)
		}
		return y
	}
}

func (g *Game[S, ASS, AS, A]) Playout(state S) S {
	for {
		isEnd := g.IsEnd(&state)
		if isEnd {
			break
		}
		actions := g.Player(&state)
		state = g.Push(state, actions...)
	}
	return state
}

func (g *Game[S, ASS, AS, A]) PlayoutWithHistory(state S, cap_ int) (S, []S, ASS) {
	ss := make([]S, 0, cap_)
	ass := make(ASS, 0, cap_)
	for {
		isEnd := g.IsEnd(&state)
		if isEnd {
			break
		}
		actions := g.Player(&state)
		ss = append(ss, state)
		ass = append(ass, actions)
		state = g.Push(state, actions...)
	}
	return state, ss, ass
}