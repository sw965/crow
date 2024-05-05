package duct

import (
	"github.com/sw965/crow/game/simultaneous"
	"github.com/sw965/crow/ucb"
	"github.com/sw965/omw"
	"math/rand"
)

type ActionPolicy[A comparable] map[A]float64
type ActionPolicies[A comparable] []ActionPolicy[A]
type ActionPoliciesFunc[S any, A comparable] func(*S) ActionPolicies[A]

type LeafNodeEvalY float64
type LeafNodeEvalYs []LeafNodeEvalY
type LeafNodeEvalsFunc[S any] func(*S) LeafNodeEvalYs

type Node[S any, ASS ~[]AS, AS ~[]A, A comparable] struct {
	State       S
	UCBManagers ucb.Managers[AS, A]
	NextNodes   Nodes[S, ASS, AS, A]
	Trial       int
	LastUCBSelectJoinAction AS
}

func (node *Node[S, ASS, AS, A]) MaxTrialJointActionPath(r *rand.Rand, n int) ([]S, ASS) {
	ret := make(ASS, 0, n)
	states := make([]S, 0, n)
	for i := 0; i < n; i++ {
		simultaneousN := len(node.UCBManagers)
		jointAction := make(AS, simultaneousN)
		isBreak := false
		for playerI, m := range node.UCBManagers {
			as := m.MaxTrialKeys()
			if !isBreak && len(as) != 1 {
				isBreak = true
			}
			jointAction[playerI] = omw.RandChoice(as, r)
		}
		states = append(states, node.State)
		ret = append(ret, jointAction)

		if len(node.NextNodes) == 0 || isBreak {
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
	return states, ret
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

func (nodes Nodes[S, ASS, AS, A]) Trials() []int {
	ret := make([]int, len(nodes))
	for i, node := range nodes {
		ret[i] = node.Trial
	}
	return ret
}

func (nodes Nodes[S, ASS, AS, A]) MaxTrial() int {
	return omw.Max(nodes.Trials()...)
}

func (nodes Nodes[S, ASS, AS, A]) MaxTrialNodes() Nodes[S, ASS, AS, A] {
	max := nodes.MaxTrial()
	ret := make(Nodes[S, ASS, AS, A], 0, len(nodes))
	for _, node := range nodes {
		if node.Trial == max {
			ret = append(ret, node)
		}
	}
	return ret
}

type nodeSelect[S any, ASS ~[]AS, AS ~[]A, A comparable] struct {
	node        *Node[S, ASS, AS, A]
	jointAction AS
}

type selects[S any, ASS ~[]AS, AS ~[]A, A comparable] []nodeSelect[S, ASS, AS, A]

func (ss selects[S, ASS, AS, A]) Backward(ys LeafNodeEvalYs) {
	for _, s := range ss {
		node := s.node
		jointAction := s.jointAction
		for playerI, action := range jointAction {
			node.UCBManagers[playerI][action].TotalValue += float64(ys[playerI])
			node.UCBManagers[playerI][action].Trial += 1
		}
	}
}

type MCTS[S any, ASS ~[]AS, AS ~[]A, A comparable] struct {
	Game               simultaneous.Game[S, ASS, AS, A]
	UCBFunc            ucb.Func
	ActionPoliciesFunc ActionPoliciesFunc[S, A]
	LeafNodeEvalsFunc  LeafNodeEvalsFunc[S]
	NextNodesCap int
}

func (mcts *MCTS[S, ASS, AS, A]) SetUniformActionPoliciesFunc() {
	mcts.ActionPoliciesFunc = func(state *S) ActionPolicies[A] {
		ass := mcts.Game.LegalActionss(state)
		policies := make(ActionPolicies[A], len(ass))
		for playerI, as := range ass {
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
	ms := make(ucb.Managers[AS, A], len(policies))
	for playerI, policy := range policies {
		m := ucb.Manager[AS, A]{}
		for a, p := range policy {
			m[a] = &ucb.Calculator{Func: mcts.UCBFunc, P: p}
		}
		ms[playerI] = m
	}
	nextNodes := make(Nodes[S, ASS, AS, A], 0, mcts.NextNodesCap)
	return &Node[S, ASS, AS, A]{State: *state, UCBManagers: ms, NextNodes:nextNodes}
}

func (mcts *MCTS[S, ASS, AS, A]) SelectExpansionBackward(node *Node[S, ASS, AS, A], r *rand.Rand, capacity int) (int, error) {
	state := node.State
	selects := make(selects[S, ASS, AS, A], 0, capacity)
	var err error
	for {
		jointAction := make(AS, len(node.UCBManagers))
		for playerI, m := range node.UCBManagers {
			jointAction[playerI] = omw.RandChoice(m.MaxKeys(), r)
		}
		selects = append(selects, nodeSelect[S, ASS, AS, A]{node: node, jointAction: jointAction})
		node.Trial += 1
		node.LastUCBSelectJoinAction = jointAction

		state, err = mcts.Game.Push(state, jointAction)
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
	ys := mcts.LeafNodeEvalsFunc(&state)
	selects.Backward(ys)
	return len(selects), nil
}

func (mcts *MCTS[S, ASS, AS, A]) Run(simulation int, rootNode *Node[S, ASS, AS, A], r *rand.Rand) error {
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
