package puct

import (
	"math/rand"
	"github.com/sw965/omw"
	"github.com/sw965/crow"
	"github.com/sw965/crow/game/sequential"
	"golang.org/x/exp/maps"
)

type LeafEvalY float64
type LeafEvalFunc[S any] func(*S) LeafEvalY

type BackwardEvalY float64
type BackwardEvalFunc[S any] func(LeafEvalY, *S) BackwardEvalY

type Eval[S any] struct {
	Leaf LeafEvalFunc[S]
	Backward BackwardEvalFunc[S]
}

type Node[S any, AS ~[]A, A comparable] struct {
	State S
	PUCBManager crow.PUCBMapManager[AS, A]
	NextNodes Nodes[S, AS, A]
	SelectCount     int
}

func (node *Node[S, AS, A]) Trial() int {
	trials := node.PUCBManager.Trials()
	return omw.Sum(trials...)
}

func (node *Node[S, AS, A]) ActionPrediction(r *rand.Rand, cap_ int) AS {
	y := make([]A, 0, cap_)
	for {
		if len(node.PUCBManager) == 0 {
			break
		}

		action := omw.RandChoice(node.PUCBManager.MaxTrialKeys(), r)
		y = append(y, action)

		if len(node.NextNodes) == 0 {
			break
		}

		max := node.NextNodes[0].Trial()
		nextNode := node.NextNodes[0]

		for _, nn := range node.NextNodes[1:] {
			trial := nn.Trial()
			if trial > max {
				max = trial
				nextNode = nn
			}
		}
		node = nextNode
	}
	return y
}

type Nodes[S any, AS ~[]A, A comparable] []*Node[S, AS, A]

func (nodes Nodes[S, AS, A]) Find(state *S, eq sequential.EqualFunc[S]) (*Node[S, AS, A], bool) {
	for _, node := range nodes {
		if eq(&node.State, state) {
			return node, true
		}
	}
	return &Node[S, AS, A]{}, false
}

type Select[S any, AS ~[]A,  A comparable] struct {
	Node *Node[S, AS, A]
	Action A
}

type Selects[S any, AS ~[]A, A comparable] []Select[S, AS, A]

func (ss Selects[S, AS, A]) Backward(y LeafEvalY, eval BackwardEvalFunc[S]) {
	for _, s := range ss {
		node := s.Node
		action := s.Action
		node.PUCBManager[action].AccumReward += float64(eval(y, &node.State))
		node.PUCBManager[action].Trial += 1
		node.SelectCount = 0
	}
}

type MCTS[S any, AS ~[]A, A comparable] struct {
	Game sequential.Game[S, AS, A]
	Policy crow.ActionPolicyFunc[S, A]
	Eval Eval[S]
}

func (mcts *MCTS[S, AS, A]) NewNode(state *S) *Node[S, AS, A] {
	py := mcts.Policy(state)
	m := crow.PUCBMapManager[AS, A]{}
	for a, p := range py {
		m[a] = &crow.UtilPUCB{P:p}
	}
	return &Node[S, AS, A]{State:*state, PUCBManager:m}
}

func (mcts *MCTS[S, AS, A]) SetNoPolicy() {
	var f crow.ActionPolicyFunc[S, A]
	f = func(state *S) crow.ActionPolicY[A] {
		actions := mcts.Game.LegalActions(state)
		p := 1.0 / float64(len(actions))
		y := crow.ActionPolicY[A]{}
		for _, a := range actions {
			y[a] = p
		}
		return y
	}
	mcts.Policy = f
}

func (mcts *MCTS[S, AS, A]) SelectAndExpansion(node *Node[S, AS, A], allNodes Nodes[S, AS, A], c float64, r *rand.Rand, cap_ int) (S, Nodes[S, AS, A], Selects[S, AS, A]) {
	state := node.State
	selects := make(Selects[S, AS, A], 0, cap_)

	for {
		action := omw.RandChoice(node.PUCBManager.MaxKeys(c), r)
		selects = append(selects, Select[S, AS, A]{Node:node, Action:action})

		node.SelectCount += 1

		state = mcts.Game.Push(state, action)
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
	return state, allNodes, selects
}

func (mcts *MCTS[S, AS, A]) Run(simulation int, rootState S, c float64, r *rand.Rand) Nodes[S, AS, A] {
	rootNode := mcts.NewNode(&rootState)
	allNodes := Nodes[S, AS, A]{rootNode}

	var leafState S
	var selects Selects[S, AS, A]

	for i := 0; i < simulation; i++ {
		leafState, allNodes, selects = mcts.SelectAndExpansion(rootNode, allNodes, c, r, len(selects) + 1)
		y := mcts.Eval.Leaf(&leafState)
		selects.Backward(y, mcts.Eval.Backward)
	}
	return allNodes
}

func NewMCTSPlayer[S any, AS ~[]A, A comparable](mcts *MCTS[S, AS, A], simulation int, c float64, random *rand.Rand, r float64) sequential.Player[S, A] {
	player := func(state *S) A {
		allNodes := mcts.Run(simulation, *state, c, random)
		rootNode := allNodes[0]

		percents := rootNode.PUCBManager.TrialPercents()
		max := omw.Max(maps.Values(percents)...)

		n := len(percents)
		actions := make(AS, 0, n)
		ws := make([]float64, 0, n)

		for action, p := range percents {
			if max*r <= p {
				actions = append(actions, action)
				ws = append(ws, p)
			}
		}

		idx := omw.RandIntWithWeight(ws, random)
		return actions[idx]
	}
	return player
}