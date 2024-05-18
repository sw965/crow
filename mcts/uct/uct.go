package uct

import (
	"math/rand"
	"github.com/sw965/crow/ucb"
	oslices "github.com/sw965/omw/slices"
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

type selects[S any, AS ~[]A, A comparable] []nodeSelect[S, AS, A]

func (ss selects[S, AS, A]) Backward(y LeafNodeEvalY, eval EachPlayerEvalFunc[S]) {
	for _, s := range ss {
		node := s.node
		action := s.action
		node.UCBManager[action].TotalValue += float64(eval(y, &node.State))
		node.UCBManager[action].Trial += 1
	}
}

type MCTS[S any, AS ~[]A, A comparable] struct {
	Game sequential.Game[S, AS, A]
	UCBFunc ucb.Func
	ActionPolicyFunc ActionPolicyFunc[S, A]
	EvalFunc EvalFunc[S]
	NextNodesCap int
}

func (mcts *MCTS[S, AS, A]) NewNode(state *S) *Node[S, AS, A] {
	policy := mcts.ActionPolicyFunc(state)
	m := ucb.Manager[AS, A]{}
	for a, p := range policy {
		m[a] = &ucb.Calculator{Func:mcts.UCBFunc, P:p}
	}
	nextNodes := make(Nodes[S, AS, A], 0, mcts.NextNodesCap)
	return &Node[S, AS, A]{State:*state, UCBManager:m, NextNodes:nextNodes}
}

func (mcts *MCTS[S, AS, A]) SetUniformActionPolicy() {
	mcts.ActionPolicyFunc = func(state *S) ActionPolicy[A] {
		as := mcts.Game.LegalActions(state)
		n := len(as)
		p := 1.0 / float64(n)
		policy := ActionPolicy[A]{}
		for _, a := range as {
			policy[a] = p
		}
		return policy
	}
}

func (mcts *MCTS[S, AS, A]) SelectExpansionBackward(node *Node[S, AS, A], r *rand.Rand, capacity int) (int, error) {
	state := node.State
	selects := make(selects[S, AS, A], 0, capacity)
	var err error
	for {
		action := omw.RandChoice(node.UCBManager.MaxKeys(), r)
		selects = append(selects, nodeSelect[S, AS, A]{node:node, action:action})

		state, err = mcts.Game.Push(state, &action)
		if err != nil {
			return 0, err
		}

		if isEnd := mcts.Game.IsEnd(&state); isEnd {
			break
		}

		//nextNodesの中に、同じstateが存在するならば、それを次のNodeとする
		//nextNodesの中に、同じstateが存在しないなら、allNodesの中から同じstateが存在しないかを調べる。
		//allNodesの中に、同じstateが存在するならば、次回から高速に探索出来るように、nextNodesに追加して、次のnodeとする。
		//nextNodesにもallNodesにも同じstateが存在しないなら、新しくnodeを作り、
		//nextNodesと、allNodesに追加し、新しく作ったnodeを次のnodeとし、select処理を終了する。

		nextNode, ok := node.NextNodes.Find(&state, mcts.Game.Equal)
		if !ok {
			//expansion
			nextNode = mcts.NewNode(&state)
			node.NextNodes = append(node.NextNodes, nextNode)
			//新しくノードを作成したら、処理を終了する
			break
		}
		node = nextNode
	}
	y := mcts.EvalFunc.LeafNode(&state)
	selects.Backward(y, mcts.EvalFunc.EachPlayer)
	return len(selects), nil
}

func (mcts *MCTS[S, AS, A]) Run(simulation int, rootNode *Node[S, AS, A], r *rand.Rand) error {
	selectCount := 0
	var err error
	for i := 0; i < simulation; i++ {
		selectCount, err = mcts.SelectExpansionBackward(rootNode, r, selectCount+1)
		if err != nil {
			return err
		}
	}
	return nil
}

func NewPlayer[S any, AS ~[]A, A comparable](mcts *MCTS[S, AS, A], simulation int, r float64, rng *rand.Rand) sequential.Player[S, A] {
	player := func(state *S) (A, error) {
		rootNode := mcts.NewNode(state)
		err := mcts.Run(simulation, rootNode, rng)
		if err != nil {
			var a A
			return a, err
		}
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
		idx := randomw.IntByWeight(ws, rng)
		return actions[idx], nil
	}
	return player
}