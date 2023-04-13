package mcts

type PolicY[A comparable] map[A]float64
type PolicyFn[S any, A comparable] func(*S) PolicY[A]