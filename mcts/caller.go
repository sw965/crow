package mcts

type StateFnCaller[S any, A comparable] struct {
	Push StatePushFn[S, A]
	Equal StateEqualFn[S]
	IsEnd IsEndStateFn[S]
}

type EvalFnCaller[S any] struct {
	Leaf LeafEvalFn[S]
	Backward BackwardEvalFn[S]
}

type FnCaller[S any, A comparable] struct {
	State StateFnCaller[S, A]
	Eval EvalFnCaller[S]
	Policy PolicyFn[S, A]
}

func (f *FnCaller[S, A]) NewNode(state *S) *Node[S, A] {
	policY := f.Policy(state)
	m := PUCBMapManager[A]{}
	for action, p := range policY {
		m[action] = &PUCB{P:p}
	}
	return &Node[S, A]{State:*state, PUCBManager:m}
}