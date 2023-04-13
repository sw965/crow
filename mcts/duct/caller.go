package duct

import (
	"github.com/sw965/crow/mcts"
)

type StateFnCaller[S any, A comparable] struct {
	Push StatePushFn[S, A]
	Equal mcts.StateEqualFn[S]
	IsEnd mcts.IsEndStateFn[S]
}

type FnCaller[S any, A comparable] struct {
	LeafEvals LeafEvalsFn[S]
	Policies PoliciesFn[S, A]
	State StateFnCaller[S, A]
}

func (f *FnCaller[S, A]) NewNode(state *S) *Node[S, A] {
	policYs := f.Policies(state)
	ms := make(PUCBMapManagers[A], len(policYs))

	for playerI, policY := range policYs {
		ms[playerI] = mcts.PUCBMapManager[A]{}
		for action, p := range policY {
			ms[playerI][action].P = p
		}
	}
	return &Node[S, A]{State:*state, PUCBManagers:ms}
}