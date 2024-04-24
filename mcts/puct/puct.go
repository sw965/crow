package puct

import (
	"math/rand"
	"github.com/sw965/crow/pucb"
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
	PUCBManager pucb.Manager[AS, A]
	NextNodes Nodes[S, AS, A]
	SelectCount     int
}

func (node *Node[S, AS, A]) Trial() int {
	return node.PUCBManager.TotalTrial()
}

func (node *Node[S, AS, A]) MaxTrialActionPath(r *rand.Rand, n int) AS {
	result := make([]A, 0, n)
	for i := 0; i < n; i++ {
		if len(node.PUCBManager) == 0 {
			break
		}

		maxTrialAction := omw.RandChoice(node.PUCBManager.MaxTrialKeys(), r)
		result = append(result, maxTrialAction)

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
	return result
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

func (ss Selects[S, AS, A]) Backward(y LeafNodeEvalY, eval EachPlayerEvalFunc[S]) {
	for _, s := range ss {
		node := s.Node
		action := s.Action
		node.PUCBManager[action].TotalValue += float64(eval(y, &node.State))
		node.PUCBManager[action].Trial += 1
		node.SelectCount = 0
	}
}

type MCTS[S any, AS ~[]A, A comparable] struct {
	Game sequential.Game[S, AS, A]
	ActionPolicy ActionPolicyFunc[S, A]
	EvalFunc EvalFunc[S]
}

func (mcts *MCTS[S, AS, A]) NewNode(state *S) *Node[S, AS, A] {
	apy := mcts.ActionPolicy(state)
	m := pucb.Manager[AS, A]{}
	for a, p := range apy {
		m[a] = &pucb.Calculator{P:p}
	}
	return &Node[S, AS, A]{State:*state, PUCBManager:m}
}

func (mcts *MCTS[S, AS, A]) SetUniformActionPolicy() {
	var f ActionPolicyFunc[S, A]
	f = func(state *S) ActionPolicy[A] {
		actions := mcts.Game.LegalActions(state)
		n := len(actions)
		p := 1.0 / float64(n)
		policy := ActionPolicy[A]{}
		for _, a := range actions {
			policy[a] = p
		}
		return policy
	}
	mcts.ActionPolicy = f
}

func (mcts *MCTS[S, AS, A]) SelectExpansionBackward(node *Node[S, AS, A], allNodes Nodes[S, AS, A], c float64, r *rand.Rand, cap_ int) (Nodes[S, AS, A], int, error) {
	state := node.State
	selects := make(Selects[S, AS, A], 0, cap_)
	var err error
	for {
		action := omw.RandChoice(node.PUCBManager.MaxKeys(c), r)
		selects = append(selects, Select[S, AS, A]{Node:node, Action:action})

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

func (mcts *MCTS[S, AS, A]) Run(simulation int, rootState S, c float64, r *rand.Rand) (Nodes[S, AS, A], error) {
	rootNode := mcts.NewNode(&rootState)
	allNodes := Nodes[S, AS, A]{rootNode}
	selectNum := 0
	var err error
	for i := 0; i < simulation; i++ {
		allNodes, selectNum, err = mcts.SelectExpansionBackward(rootNode, allNodes, c, r, selectNum + 1)
		if err != nil {
			return Nodes[S, AS, A]{}, err
		}
	}
	return allNodes, nil
}

func NewPlayer[S any, AS ~[]A, A comparable](mcts *MCTS[S, AS, A], simulation int, c float64, random *rand.Rand, r float64) sequential.Player[S, A] {
	player := func(state *S) (A, error) {
		allNodes, err:= mcts.Run(simulation, *state, c, random)
		if err != nil {
			var a A
			return a, err
		}
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

		idx := omw.RandIntByWeight(ws, random)
		return actions[idx], nil
	}
	return player
}