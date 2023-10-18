package dpuct

import (
	"github.com/sw965/crow"
	"github.com/sw965/crow/game/simultaneous"
	omwrand "github.com/sw965/omw/rand"
	"math/rand"
)

type LeafEvalY float64
type LeafEvalYs []LeafEvalY
type LeafEvalsFunc[S any] func(*S) LeafEvalYs

type Node[S any, ASS ~[]AS, AS ~[]A, A comparable] struct {
	State        S
	PUCBManagers crow.PUCBManagers[AS, A]
	NextNodes    Nodes[S, ASS, AS, A]
	Trial        int
	SelectCount  int
}

func (node *Node[S, ASS, AS, A]) ActionPrediction(r *rand.Rand, cap_ int) ASS {
	y := make(ASS, 0, cap_)
	for {
		if len(node.PUCBManagers) == 0 {
			break
		}

		actions := make(AS, len(node.PUCBManagers))
		for playerI, m := range node.PUCBManagers {
			actions[playerI] = omwrand.Choice(m.MaxTrialKeys(), r)
		}
		y = append(y, actions)

		if len(node.NextNodes) == 0 {
			break
		}

		max := node.NextNodes[0].Trial
		nextNode := node.NextNodes[0]

		for _, n := range node.NextNodes[1:] {
			trial := n.Trial
			if trial > max {
				max = trial
				nextNode = n
			}
		}
		node = nextNode
	}
	return y
}

type Nodes[S any, ASS ~[]AS, AS ~[]A, A comparable] []*Node[S, ASS, AS, A]

func (nodes Nodes[S, ASS, AS, A]) Find(state *S, eq simultaneous.EqualFunc[S]) (*Node[S, ASS, AS, A], bool) {
	for _, node := range nodes {
		if eq(&node.State, state) {
			return node, true
		}
	}
	return &Node[S, ASS, AS, A]{}, false
}

type Select[S any, ASS ~[]AS, AS ~[]A, A comparable] struct {
	Node    *Node[S, ASS, AS, A]
	Actions AS
}

type Selects[S any, ASS ~[]AS, AS ~[]A, A comparable] []Select[S, ASS, AS, A]

func (ss Selects[S, ASS, AS, A]) Backward(ys LeafEvalYs) {
	for _, s := range ss {
		node := s.Node
		actions := s.Actions
		for playerI, action := range actions {
			node.PUCBManagers[playerI][action].AccumReward += float64(ys[playerI])
			node.PUCBManagers[playerI][action].Trial += 1
		}
		node.SelectCount = 0
	}
}

type MCTS[S any, ASS ~[]AS, AS ~[]A, A comparable] struct {
	Game      simultaneous.Game[S, ASS, AS, A]
	ActionPolicies  crow.ActionPoliciesFunc[S, A]
	LeafEvals LeafEvalsFunc[S]
}

func (mcts *MCTS[S, ASS, AS, A]) SetNoPolicies() {
	f := func(state *S) crow.ActionPolicYs[A] {
		actionss := mcts.Game.PadLegalActionss(state)
		ys := make(crow.ActionPolicYs[A], len(actionss))
		for playerI, as := range actionss {
			y := map[A]float64{}
			p := 1.0 / float64(len(as))
			for _, a := range as {
				y[a] = p
			}
			ys[playerI] = y
		}
		return ys
	}
	mcts.ActionPolicies = f
}

func (mcts *MCTS[S, ASS, AS, A]) NewNode(state *S) *Node[S, ASS, AS, A] {
	apys := mcts.ActionPolicies(state)
	ms := make(crow.PUCBManagers[AS, A], len(apys))
	for playerI, apy := range apys {
		m := crow.PUCBManager[AS, A]{}
		for a, p := range apy {
			m[a] = &crow.PUCBCalculator{P: p}
		}
		ms[playerI] = m
	}
	return &Node[S, ASS, AS, A]{State: *state, PUCBManagers: ms}
}

func (mcts *MCTS[S, ASS, AS, A]) SelectExpansionBackward(simultaneous int, node *Node[S, ASS, AS, A], allNodes Nodes[S, ASS, AS, A], c float64, r *rand.Rand, cap_ int) (Nodes[S, ASS, AS, A], int) {
	state := node.State
	selects := make(Selects[S, ASS, AS, A], 0, cap_)

	for {
		actions := make(AS, simultaneous)
		for playerI, m := range node.PUCBManagers {
			//select
			actions[playerI] = omwrand.Choice(m.MaxKeys(c), r)
		}

		selects = append(selects, Select[S, ASS, AS, A]{Node: node, Actions: actions})
		node.SelectCount += 1
		node.Trial += 1

		state = mcts.Game.Push(state, actions...)
		stateP := &state

		if isEnd := mcts.Game.IsEnd(stateP); isEnd {
			break
		}

		//nextNodesの中に、同じstateが存在するならば、それを次のNodeとする
		//nextNodesの中に、同じstateが存在しないなら、allNodesの中から同じstateが存在しないかを調べる。
		//allNodesの中に、同じstateが存在するならば、次回から高速に探索出来るように、nextNodesに追加して、次のnodeとする。
		//nextNodesにもallNodesにも同じstateが存在しないなら、新しくnodeを作り、
		//nextNodesと、allNodesに追加し、新しく作ったnodeを次のnodeとし、select処理を終了する。

		nextNode, ok := node.NextNodes.Find(stateP, mcts.Game.Equal)
		if !ok {
			nextNode, ok = allNodes.Find(stateP, mcts.Game.Equal)
			if ok {
				node.NextNodes = append(node.NextNodes, nextNode)
			} else {
				//expansion
				nextNode = mcts.NewNode(stateP)
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

	ys := mcts.LeafEvals(&state)
	selects.Backward(ys)
	return allNodes, len(selects)
}

func (mcts *MCTS[S, ASS, AS, A]) Run(simulation int, rootState S, c float64, r *rand.Rand) Nodes[S, ASS, AS, A] {
	rootNode := mcts.NewNode(&rootState)
	allNodes := Nodes[S, ASS, AS, A]{rootNode}
	simultaneous := len(rootNode.PUCBManagers)
	selectNum := 0
	for i := 0; i < simulation; i++ {
		allNodes, selectNum = mcts.SelectExpansionBackward(simultaneous, rootNode, allNodes, c, r, selectNum+1)
	}
	return allNodes
}
