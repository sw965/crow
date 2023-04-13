package mcts

type StatePushFn[S any, A comparable] func(S, A) S
type StateEqualFn[S any] func(*S, *S) bool
type IsEndStateFn[S any] func(*S) bool