package solver

type PushState[S any, A comparable] func (S, ...A) S
type EqualState[S any] func(*S, *S) bool
type IsEndState[S any] func(*S) bool

type StateFunc[S any, A comparable] struct {
	Push PushState[S, A]
	Equal EqualState[S]
	IsEnd IsEndState[S]
}

type Player[S any, A comparable] func(S) []A

func Playout[S any, A comparable](player Player[S, A], state S, f StateFunc[S, A]) S {
	for {
		if f.IsEnd(&state) {
			break
		}
		actions := player(state)
		state = f.Push(state, actions...)
	}
	return state
}