package duct

import (
	"math/rand"
)

type Runner[S any, A comparable] struct {
	Simulation int
	FnCaller FnCaller[S, A]
	C float64
	Rand *rand.Rand
}

func (r *Runner[S, A]) Run(rootState S) (Nodes[S, A], error) {
	rootNode := r.FnCaller.NewNode(&rootState)
	allNodes := Nodes[S, A]{rootNode}
	simultaneous := len(rootNode.PUCBManagers)

	var leafState S
	var selects Selects[S, A]
	var err error
	selector := Selector[S, A]{Simultaneous:simultaneous, FnCaller:r.FnCaller, C:r.C, Rand:r.Rand, Cap:1}

	for i := 0; i < r.Simulation; i++ {
		leafState, allNodes, selects, err = selector.SelectAndExpansion(rootNode, allNodes)
		if err != nil {
			return Nodes[S, A]{}, err
		}

		ys := r.FnCaller.LeafEvals(&leafState)
		if err != nil {
			return Nodes[S, A]{}, err
		}

		selects.Backward(ys)
	}
	return allNodes, nil
}