package duct

import (
	"fmt"
	"github.com/sw965/crow/mcts"
	mrand "math/rand"
	orand "github.com/sw965/omw/math/rand"
)

type Node[S any, A comparable] struct {
	State S
	PUCBManagers PUCBMapManagers[A]
	NextNodes Nodes[S, A]
	SelectCount     int
}

func (node *Node[S, A])SelectAndExpansion(allNodes Nodes[S, A], f *FnCaller[S, A], c float64, r *mrand.Rand, simultaneous, cap int) (S, Nodes[S, A], Selects[S, A], error) {
	selects := make(Selects[S, A], 0, cap)
	state := node.State

	for {
		actions := make([]A, simultaneous)
		for playerI, m := range node.PUCBManagers {
			actions[playerI] = orand.Choice(m.MaxKeys(c), r)
		}

		selects = append(selects, Select[S, A]{Node:node, Actions:actions})

		node.SelectCount += 1

		state = f.State.Push(state, actions...)
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
	return state, allNodes, selects, nil
}

type Nodes[S any, A comparable] []*Node[S, A]

func (nodes Nodes[S, A]) Find(state *S, eq mcts.StateEqualFn[S]) (*Node[S, A], error) {
	for _, node := range nodes {
		if eq(&node.State, state) {
			return node, nil
		}
	}
	return &Node[S, A]{}, fmt.Errorf("一致するNodeが見つからなかった")
}