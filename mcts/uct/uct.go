package uct

import (
	"math/rand"
	"github.com/sw965/crow/ucb"
	"github.com/sw965/omw"
	"github.com/sw965/crow/game/sequential"
	"golang.org/x/exp/maps"
)

type ActionPolicy[A comparable] map[A]float64
type ActionPolicyFunc[S any, A comparable] func(*S) ActionPolicy[A]

type LeafNodeEvalY float64
type LeafNodeEvalFunc[S any] func(*S) LeafNodeEvalY

type EachPlayerEvalY float64
type EachPlayerEvalFunc[S any] func(LeafNodeEvalY, *S) EachPlayerEvalY

type EvalFunc[S any] struct {
	LeafNode LeafNodeEvalFunc[S]
	EachPlayer EachPlayerEvalFunc[S]
}

type Node[S any, AS ~[]A, A comparable] struct {
	State S
	UCBManager ucb.Manager[AS, A]
	NextNodes Nodes[S, AS, A]
	SelectCount     int
}

func (node *Node[S, AS, A]) Trial() int {
	return node.UCBManager.TotalTrial()
}

func (node *Node[S, AS, A]) MaxTrialActionPath(r *rand.Rand, n int) AS {
	ret := make([]A, 0, n)
	for i := 0; i < n; i++ {
		if len(node.UCBManager) == 0 {
			break
		}

		action := omw.RandChoice(node.UCBManager.MaxTrialKeys(), r)
		ret = append(ret, action)

		if len(node.NextNodes) == 0 {
			break
		}

		maxTrial := node.NextNodes[0].Trial()
		nextNode := node.NextNodes[0]

		for _, nn := range node.NextNodes[1:] {
			trial := nn.Trial()
			if trial > maxTrial {
				maxTrial = trial
				nextNode = nn
			}
		}
		node = nextNode
	}
	return ret
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

type nodeSelect[S any, AS ~[]A,  A comparable] struct {
	node *Node[S, AS, A]
	action A
}

type Selects[S any, AS ~[]A, A comparable] []nodeSelect[S, AS, A]

func (ss Selects[S, AS, A]) Backward(y LeafNodeEvalY, eval EachPlayerEvalFunc[S]) {
	for _, s := range ss {
		node := s.node
		action := s.action
		node.UCBManager[action].TotalValue += float64(eval(y, &node.State))
		node.UCBManager[action].Trial += 1
		node.SelectCount = 0
	}
}

type MCTS[S any, AS ~[]A, A comparable] struct {
	Game sequential.Game[S, AS, A]
	UCBFunc ucb.Func
	ActionPolicy ActionPolicyFunc[S, A]
	EvalFunc EvalFunc[S]
}

func (mcts *MCTS[S, AS, A]) NewNode(state *S) *Node[S, AS, A] {
	policy := mcts.ActionPolicy(state)
	m := ucb.Manager[AS, A]{}
	for a, p := range policy {
		m[a] = &ucb.Calculator{Func:mcts.UCBFunc, P:p}
	}
	return &Node[S, AS, A]{State:*state, UCBManager:m}
}

func (mcts *MCTS[S, AS, A]) SetUniformActionPolicy() {
	var f ActionPolicyFunc[S, A]
	f = func(state *S) ActionPolicy[A] {
		legals := mcts.Game.LegalActions(state)
		n := len(legals)
		p := 1.0 / float64(n)
		policy := ActionPolicy[A]{}
		for _, a := range legals {
			policy[a] = p
		}
		return policy
	}
	mcts.ActionPolicy = f
}

func (mcts *MCTS[S, AS, A]) SelectExpansionBackward(node *Node[S, AS, A], allNodes Nodes[S, AS, A], r *rand.Rand, capacity int) (Nodes[S, AS, A], int, error) {
	state := node.State
	selects := make(Selects[S, AS, A], 0, capacity)
	var err error
	for {
		action := omw.RandChoice(node.UCBManager.MaxKeys(), r)
		selects = append(selects, nodeSelect[S, AS, A]{node:node, action:action})
		node.SelectCount += 1

		state, err = mcts.Game.Push(state, &action)
		if err != nil {
			return Nodes[S, AS, A]{}, 0, err
		}
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
	y := mcts.EvalFunc.LeafNode(&state)
	selects.Backward(y, mcts.EvalFunc.EachPlayer)
	return allNodes, len(selects), nil
}

func (mcts *MCTS[S, AS, A]) Run(simulation int, rootState S, r *rand.Rand) (Nodes[S, AS, A], error) {
	rootNode := mcts.NewNode(&rootState)
	allNodes := Nodes[S, AS, A]{rootNode}
	selectNum := 0
	var err error
	for i := 0; i < simulation; i++ {
		allNodes, selectNum, err = mcts.SelectExpansionBackward(rootNode, allNodes, r, selectNum + 1)
		if err != nil {
			return Nodes[S, AS, A]{}, err
		}
	}
	return allNodes, nil
}

func NewPlayer[S any, AS ~[]A, A comparable](mcts *MCTS[S, AS, A], simulation int, r float64, rng *rand.Rand) sequential.Player[S, A] {
	player := func(state *S) (A, error) {
		allNodes, err:= mcts.Run(simulation, *state, rng)
		if err != nil {
			var a A
			return a, err
		}
		rootNode := allNodes[0]
	
		ps := rootNode.UCBManager.TrialPercents()
		max := omw.Max(maps.Values(ps)...)
		n := len(ps)
		ws := make([]float64, 0, n)
		actions := make(AS, 0, n)

		for a, p := range ps {
			if max*r <= p {
				actions = append(actions, a)
				ws = append(ws, p)
			}
		}

		idx := omw.RandIntByWeight(ws, rng)
		return actions[idx], nil
	}
	return player
}