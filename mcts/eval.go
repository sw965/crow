package mcts

type LeafEvalY float64
type LeafEvalFn[S any] func(*S) LeafEvalY

type BackwardEvalY float64
type BackwardEvalFn[S any] func(LeafEvalY, *S) BackwardEvalY