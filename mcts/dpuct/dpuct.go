package dpuct

import (
	"github.com/sw965/crow/game/simultaneous"
	"math/rand"
	"github.com/sw965/crow/pucb"
)

type ActionPolicy[A comparable] map[A]float64
type ActionPolicyFunc[S any, A comparable] func(*S) ActionPolicy[A]

type ActionPolicies[A comparable] []ActionPolicy[A]
type ActionPoliciesFunc[S any, A comparable] func(*S) ActionPolicies[A]

type LeafNodeEvalY float64
type LeafNodeEvalYs []LeafNodeEvalY
type LeafNodeEvalsFunc[S any] func(*S) LeafNodeEvalYs

type Node[S any, ASS ~[]AS, AS ~[]A, A comparable] struct {
	State        S
	PUCBManagers pucb.Managers[AS, A]
	NextNodes    Nodes[S, ASS, AS, A]
	Trial        int
	SelectCount  int
}

func (node *Node[S, ASS, AS, A]) PredictActionss(r *rand.Rand, cap_ int) ASS {
	result := make(ASS, 0, cap_)
	for {
		if len(node.PUCBManagers) == 0 {
			break
		}

		maxTrialActions := node.PUCBManagers.MaxTrialKeys(r)
		result = append(result, maxTrialActions)

		if len(node.NextNodes) == 0 {
			break
		}

		maxTrial := node.NextNodes[0].Trial
		nextNode := node.NextNodes[0]

		for _, nn := range node.NextNodes[1:] {
			trial := nn.Trial
			if trial > maxTrial {
				maxTrial = trial
				nextNode = nn
			}
		}
		node = nextNode
	}
	return result
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

func (ss Selects[S, ASS, AS, A]) Backward(ys LeafNodeEvalYs) {
	for _, s := range ss {
		node := s.Node
		actions := s.Actions
		for playerI, action := range actions {
			node.PUCBManagers[playerI][action].TotalValue += float64(ys[playerI])
			node.PUCBManagers[playerI][action].Trial += 1
		}
		node.SelectCount = 0
	}
}

type MCTS[S any, ASS ~[]AS, AS ~[]A, A comparable] struct {
	Game      simultaneous.Game[S, ASS, AS, A]
	ActionPoliciesFunc  ActionPoliciesFunc[S, A]
	LeafNodeEvalsFunc LeafNodeEvalsFunc[S]
}

func (mcts *MCTS[S, ASS, AS, A]) SetUniformActionPoliciesFunc() {
	mcts.ActionPoliciesFunc = func(state *S) ActionPolicies[A] {
		actionss := mcts.Game.LegalActionss(state)
		policies := make(ActionPolicies[A], len(actionss))
		for playerI, as := range actionss {
			policy := ActionPolicy[A]{}
			n := len(as)
			p := 1.0 / float64(n)
			for _, a := range as {
				policy[a] = p
			}
			policies[playerI] = policy
		}
		return policies
	}
}

func (mcts *MCTS[S, ASS, AS, A]) NewNode(state *S) *Node[S, ASS, AS, A] {
	policies := mcts.ActionPoliciesFunc(state)
	ms := make(pucb.Managers[AS, A], len(policies))
	for playerI, policy := range policies {
		m := pucb.Manager[AS, A]{}
		for a, p := range policy {
			m[a] = &pucb.Calculator{P: p}
		}
		ms[playerI] = m
	}
	return &Node[S, ASS, AS, A]{State: *state, PUCBManagers: ms}
}

func (mcts *MCTS[S, ASS, AS, A]) SelectExpansionBackward(node *Node[S, ASS, AS, A], allNodes Nodes[S, ASS, AS, A], c float64, r *rand.Rand, cap_ int) (Nodes[S, ASS, AS, A], int, error) {
	state := node.State
	selects := make(Selects[S, ASS, AS, A], 0, cap_)
	var err error
	for {
		actions := node.PUCBManagers.MaxKeys(c, r)
		selects = append(selects, Select[S, ASS, AS, A]{Node: node, Actions: actions})
		node.Trial += 1
		node.SelectCount += 1

		state, err = mcts.Game.Push(state, actions)
		if err != nil {
			return Nodes[S, ASS, AS, A]{}, 0, err
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

	ys := mcts.LeafNodeEvalsFunc(&state)
	selects.Backward(ys)
	return allNodes, len(selects), nil
}

func (mcts *MCTS[S, ASS, AS, A]) Run(simulation int, rootState S, c float64, r *rand.Rand) (Nodes[S, ASS, AS, A], error) {
	rootNode := mcts.NewNode(&rootState)
	allNodes := Nodes[S, ASS, AS, A]{rootNode}
	selectNum := 0
	var err error
	for i := 0; i < simulation; i++ {
		allNodes, selectNum, err = mcts.SelectExpansionBackward(rootNode, allNodes, c, r, selectNum+1)
		if err != nil {
			return Nodes[S, ASS, AS, A]{}, err
		}
	}
	return allNodes, nil
}
