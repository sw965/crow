package mcts

import (
	mrand "math/rand"
	orand "github.com/sw965/omw/math/rand"
)

type Select[S any, A comparable] struct {
	Node *Node[S, A]
	Action A
}

type Selects[S any, A comparable] []Select[S, A]

func (ss Selects[S, A]) Backward(y LeafEvalY, eval BackwardEvalFn[S]) {
	for _, s := range ss {
		node := s.Node
		action := s.Action
		node.PUCBManager[action].AccumReward += float64(eval(y, &node.State))
		node.PUCBManager[action].Trial += 1
		node.SelectCount = 0
	}
}

type Selector[S any, A comparable] struct {
	FnCaller *FnCaller[S, A]
	C float64
	Rand *mrand.Rand
	Cap int
}

func (s *Selector[S, A])SelectAndExpansion(node *Node[S, A], allNodes Nodes[S, A]) (S, Nodes[S, A], Selects[S, A], error) {
	selects := make(Selects[S, A], 0, s.Cap)
	state := node.State
	f := s.FnCaller

	for {
		action := orand.Choice(node.PUCBManager.MaxKeys(s.C), s.Rand)
		selects = append(selects, Select[S, A]{Node:node, Action:action})

		node.SelectCount += 1

		state = f.State.Push(state, action)
		stateP := &state

		if f.State.IsEnd(stateP) {
			break
		}

		//nextNodesの中に、同じstateが存在するならば、それを次のNodeとする
		//nextNodesの中に、同じstateが存在しないなら、allNodesの中から同じstateが存在しないかを調べる。
		//allNodesの中に、同じstateが存在するならば、次回から高速に探索出来るように、nextNodesに追加して、次のnodeとする。
		//nextNodesにもallNodesにも同じstateが存在しないなら、新しくnodeを作り、
		//nextNodesと、allNodesに追加し、新しく作ったnodeを次のnodeとし、select処理を終了する。

		nextNode, err := node.NextNodes.Find(stateP, f.State.Equal)
		if err != nil {
			nextNode, err = allNodes.Find(stateP, f.State.Equal)
			if err == nil {
				node.NextNodes = append(node.NextNodes, nextNode)
			} else {
				nextNode = f.NewNode(stateP)
				allNodes = append(allNodes, nextNode)
				node.NextNodes = append(node.NextNodes, nextNode)
				//新しくノードを作成したら、処理を終了する
				break
			}
		}

		if nextNode.SelectCount == 1 {
			break
		}
		node = nextNode
	}
	s.Cap = len(selects) + 1
	return state, allNodes, selects, nil
}