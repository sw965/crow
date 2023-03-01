package mcts

import (
	"fmt"
	"math/rand"
	"github.com/sw965/omw"
	"github.com/sw965/crow/pucb"
)

type Policy[S any, A comparable] func(*S) map[A]float64
type LeafEval[S any] func(*S) float64
type BackwardEval[S any, A comparable] func(float64, *Node[S, A]) float64

type Eval[S any, A comparable]  struct {
	Leaf LeafEval[S]
	Backward BackwardEval[S, A]
}

type StatePush[S any, A comparable] func(S, A) (S, error)
type StateEqual[S any] func(*S, *S) bool
type IsEndState[S any] func(*S) bool


type StateFunc[S any, A comparable] struct {
	Push StatePush[S, A]
	Equal StateEqual[S]
	IsEnd IsEndState[S]
}

type NewNode[S any, A comparable] func(*S, Policy[S, A]) *Node[S, A]

type NodeFunc[S any, A comparable] struct {
	New NewNode[S, A]
}

type Func[S any, A comparable] struct {
	State StateFunc[S, A]
	Node NodeFunc[S, A]
}

type Node[S any, A comparable] struct {
	State S
	LegalActions []A
	PUCBByAction pucb.ManagerByKey[A]
	NextNodes Nodes[S, A]
	SelectCount     int
	ID int
}

func (node *Node[S, A])SelectAndExpansion(allNodes Nodes[S, A], policy Policy[S, A], X float64, r *rand.Rand, f *Func[S, A], capSize int) (S, Nodes[S, A], Selects[S, A], error) {
	selects := make(Selects[S, A], 0, capSize)
	state := node.State
	var err error

	for {
		actions := node.PUCBByAction.MaxKeys(X)
		action := omw.RandomChoice(actions, r)
		selects = append(selects, Select[S, A]{Node:node, Action:action})
		node.SelectCount += 1

		state, err = f.State.Push(state, action)

		if err != nil {
			return state, Nodes[S, A]{}, Selects[S, A]{}, err
		}

		if f.State.IsEnd(&state) {
			break
		}

		//nextNodesの中に、同じstateが存在するならば、それを次のNodeとする
		//nextNodesの中に、同じstateが存在しないなら、allNodesの中から同じstateが存在しないかを調べる。
		//allNodesの中に、同じstateが存在するならば、次回から高速に探索出来るように、nextNodesに追加して、次のnodeとする。
		//nextNodesにもallNodesにも同じstateが存在しないなら、新しくnodeを作り、
		//nextNodesと、allNodesに追加し、新しく作ったnodeを次のnodeとし、select処理を終了する。

		nextNode, err := node.NextNodes.Find(&state, f.State.Equal)
		if err != nil {
			nextNode, err = allNodes.Find(&state, f.State.Equal)
			if err == nil {
				node.NextNodes = append(node.NextNodes, nextNode)
			} else {
				nextNode = f.Node.New(&state, policy)
				allNodes = append(allNodes, nextNode)
				node.NextNodes = append(node.NextNodes, nextNode)
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

func (nodes Nodes[S, A]) Find(state *S, equal StateEqual[S]) (*Node[S, A], error) {
	for _, v := range nodes {
		if equal(&v.State, state) {
			return v, nil
		}
	}
	return &Node[S, A]{}, fmt.Errorf("一致するNodeが見つからなかった")
}

type Select[S any, A comparable] struct {
	Node *Node[S, A]
	Action A
}

type Selects[S any, A comparable] []Select[S, A]

func (ss Selects[S, A]) Backward(leafEvalY float64, eval BackwardEval[S, A]) {
	for _, s := range ss {
		node := s.Node
		action := s.Action
		node.PUCBByAction[action].AccumReward += eval(leafEvalY, node)
		node.PUCBByAction[action].Trial += 1
		node.SelectCount = 0
	}
}

func Run[S any, A comparable](simulation int, rootState S, policy Policy[S, A], eval *Eval[S, A], X float64, r *rand.Rand, f *Func[S, A]) (Nodes[S, A], error) {
	rootNode := f.Node.New(&rootState, policy)
	allNodes := Nodes[S, A]{rootNode}

	node := rootNode
	var leafState S
	var selects Selects[S, A]
	capSize := 0
	var err error

	for i := 0; i < simulation; i++ {
		leafState, allNodes, selects, err = node.SelectAndExpansion(allNodes, policy, X, r, f, capSize + 1)
		if err != nil {
			return Nodes[S, A]{}, err
		}
		capSize = len(selects)

		leafEvalY := eval.Leaf(&leafState)
		if err != nil {
			return Nodes[S, A]{}, err
		}

		selects.Backward(leafEvalY, eval.Backward)
		node = rootNode
	}
	return allNodes, nil
}