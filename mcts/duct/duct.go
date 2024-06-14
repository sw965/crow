package duct

import (
	"fmt"
	"github.com/sw965/crow/game/simultaneous"
	"github.com/sw965/crow/ucb"
	omwslices "github.com/sw965/omw/slices"
	omwrand "github.com/sw965/omw/math/rand"
	"math/rand"
	"golang.org/x/exp/slices"
)

type ActionPolicy[A comparable] map[A]float64
type SeparateActionPolicy[A comparable] []ActionPolicy[A]
type SeparateActionPolicyFunc[S any, A comparable] func(*S) SeparateActionPolicy[A]

type LeafNodeJointEvalY []float64
type LeafNodeJointEvalFunc[S any] func(*S) (LeafNodeJointEvalY, error)

type Node[S any, ASS ~[]AS, AS ~[]A, A comparable] struct {
	State       S
	SeparateUCBManager ucb.SeparateManager[AS, A]
	NextNodes   Nodes[S, ASS, AS, A]
	Trial       int
	LastJointActions ASS
	LastJointActionsTrials []int
}

func (node *Node[S, ASS, AS, A]) Predict(r *rand.Rand, limit int) (ASS, [][]float64) {
	jointActions := make(ASS, 0, limit)
	jointAvgs := make([][]float64, 0, limit)

	for i := 0; i < limit; i++ {
		fmt.Println("sekect1", node.SeparateUCBManager)
		jointAction := node.SeparateUCBManager.JointActionByMaxTrial(r)
		fmt.Println("slect2", jointAction)
		jointActions = append(jointActions, jointAction)
		jointAvgs = append(jointAvgs, node.SeparateUCBManager.JointAverageValue())

		n := len(node.NextNodes)
		if n == 0 {
			return jointActions, jointAvgs
		}

		eqToJointAction := omwslices.Equal(jointAction)
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
		node = omwrand.Choice(nextNodes, r)
	}
	return jointActions, jointAvgs
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

func (ss selects[S, ASS, AS, A]) backward(jointEval LeafNodeJointEvalY) {
	for _, s := range ss {
		node := s.node
		jointAction := s.jointAction
		for playerI, action := range jointAction {
			node.SeparateUCBManager[playerI][action].TotalValue += float64(jointEval[playerI])
			node.SeparateUCBManager[playerI][action].Trial += 1
		}
	}
}

type MCTS[S any, ASS ~[]AS, AS ~[]A, A comparable] struct {
	Game               simultaneous.Game[S, ASS, AS, A]
	UCBFunc            ucb.Func
	SeparateActionPolicyFunc SeparateActionPolicyFunc[S, A]
	LeafNodeJointEvalFunc  LeafNodeJointEvalFunc[S]
	NextNodesCap int
	LastJointActionsCap int
}

func (mcts *MCTS[S, ASS, AS, A]) SetUniformSeparateActionPolicyFunc() {
	mcts.SeparateActionPolicyFunc = func(state *S) SeparateActionPolicy[A] {
		ass := mcts.Game.LegalSeparateActions(state)
		fmt.Println("ass = ", ass)
		policies := make(SeparateActionPolicy[A], len(ass))
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
	policies := mcts.SeparateActionPolicyFunc(state)
	fmt.Println("policies", policies)
	ms := make(ucb.SeparateManager[AS, A], len(policies))
	for playerI, policy := range policies {
		m := ucb.Manager[AS, A]{}
		for a, p := range policy {
			m[a] = &ucb.Calculator{Func: mcts.UCBFunc, P: p}
		}
		ms[playerI] = m
	}
	fmt.Println("ms = ", ms)
	nextNodes := make(Nodes[S, ASS, AS, A], 0, mcts.NextNodesCap)
	lastJointActions := make(ASS, 0, mcts.LastJointActionsCap)
	lastJointActionsTrials := make([]int, 0, mcts.LastJointActionsCap)

	return &Node[S, ASS, AS, A]{
		State: *state,
		SeparateUCBManager: ms,
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
		fmt.Println("koko1", node.SeparateUCBManager)
		jointAction := node.SeparateUCBManager.JointActionByMax(r)
		fmt.Println("koko2", jointAction)
		selects = append(selects, nodeSelect[S, ASS, AS, A]{node: node, jointAction: jointAction})
		node.Trial += 1

		state, err = mcts.Game.Push(state, jointAction)
		if err != nil {
			return 0, err
		}

		if isEnd, _ := mcts.Game.IsEnd(&state); isEnd {
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

		eqToJointAction := omwslices.Equal(jointAction)
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
	jointEval, err := mcts.LeafNodeJointEvalFunc(&state)
	selects.backward(jointEval)
	return len(selects), err
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

func (mcts *MCTS[S, ASS, AS, A]) NewPlayer(simulation int, r *rand.Rand) simultaneous.Player[S, AS, A] {
	return func(state *S) (AS, []float64, error) {
		rootNode := mcts.NewNode(state)
		err := mcts.Run(simulation, rootNode, r)
		jointAction := rootNode.SeparateUCBManager.JointActionByMaxTrial(r)
		jointAvg := rootNode.SeparateUCBManager.JointAverageValue()
		return jointAction, jointAvg, err
	}
}