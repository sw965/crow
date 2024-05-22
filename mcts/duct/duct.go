package duct

import (
	"github.com/sw965/crow/game/simultaneous"
	"github.com/sw965/crow/ucb"
	oslices "github.com/sw965/omw/slices"
	orand "github.com/sw965/omw/rand"
	"math/rand"
	"golang.org/x/exp/slices"
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
	LastJointActions ASS
	LastJointActionsTrials []int
}

func (node *Node[S, ASS, AS, A]) MaxTrialJointActionPath(r *rand.Rand, limit int) (ASS, [][]float64) {
	ass := make(ASS, 0, limit)
	avgss := make([][]float64, 0, limit)

	for i := 0; i < limit; i++ {
		jointAction := node.UCBManagers.JointActionByMaxTrial(r)
		ass = append(ass, jointAction)
		avgss = append(avgss, node.UCBManagers.AverageValues())

		n := len(node.NextNodes)
		if n == 0 {
			return ass, avgss
		}

		eqToJointAction := oslices.Equal(jointAction)
		maxTrial := 0
		nextNodes := make(Nodes[S, ASS, AS, A], 0, n)
		for _, nextNode := range node.NextNodes {
			idx := slices.IndexFunc(nextNode.LastJointActions, eqToJointAction)
			if idx != -1 {
				trial := nextNode.LastJointActionsTrials[idx]
				if trial > maxTrial {
					nextNodes = make(Nodes[S, ASS, AS, A], 0, n)
					nextNodes = append(nextNodes, nextNode)
					maxTrial = trial
				} else if trial == maxTrial {
					nextNodes = append(nextNodes, nextNode)
				}
			}
		}

		if len(nextNodes) == 0 {
			break
		}
		node = orand.Choice(nextNodes, r)
	}
	return ass, avgss
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

type nodeSelect[S any, ASS ~[]AS, AS ~[]A, A comparable] struct {
	node        *Node[S, ASS, AS, A]
	jointAction AS
}

type selects[S any, ASS ~[]AS, AS ~[]A, A comparable] []nodeSelect[S, ASS, AS, A]

func (ss selects[S, ASS, AS, A]) backward(ys LeafNodeEvalYs) {
	for _, s := range ss {
		node := s.node
		jointAction := s.jointAction
		for playerI, a := range jointAction {
			node.UCBManagers[playerI][a].TotalValue += float64(ys[playerI])
			node.UCBManagers[playerI][a].Trial += 1
		}
	}
}

type MCTS[S any, ASS ~[]AS, AS ~[]A, A comparable] struct {
	Game               simultaneous.Game[S, ASS, AS, A]
	UCBFunc            ucb.Func
	ActionPoliciesFunc ActionPoliciesFunc[S, A]
	LeafNodeEvalsFunc  LeafNodeEvalsFunc[S]
	NextNodesCap int
	LastJointActionsCap int
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
	lastJointActions := make(ASS, 0, mcts.LastJointActionsCap)
	lastJointActionsTrials := make([]int, 0, mcts.LastJointActionsCap)

	return &Node[S, ASS, AS, A]{
		State: *state,
		UCBManagers: ms,
		NextNodes:nextNodes,
		LastJointActions:lastJointActions,
		LastJointActionsTrials:lastJointActionsTrials,
	}
}

func (mcts *MCTS[S, ASS, AS, A]) SelectExpansionBackward(node *Node[S, ASS, AS, A], r *rand.Rand, capacity int) (int, error) {
	state := node.State
	selects := make(selects[S, ASS, AS, A], 0, capacity)
	var err error
	for {
		jointAction := node.UCBManagers.JointActionByMax(r)
		selects = append(selects, nodeSelect[S, ASS, AS, A]{node: node, jointAction: jointAction})
		node.Trial += 1

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
		}

		eqToJointAction := oslices.Equal(jointAction)
		if !slices.ContainsFunc(nextNode.LastJointActions, eqToJointAction) {
			nextNode.LastJointActions = append(nextNode.LastJointActions, jointAction)
			nextNode.LastJointActionsTrials = append(nextNode.LastJointActionsTrials, 0)
		}

		idx := slices.IndexFunc(nextNode.LastJointActions, eqToJointAction)
		nextNode.LastJointActionsTrials[idx] += 1

		//新しくノードを作成したら、処理を終了する
		if !ok {
			break
		}
		node = nextNode
	}
	ys := mcts.LeafNodeEvalsFunc(&state)
	selects.backward(ys)
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