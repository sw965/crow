package duct

import (
	"fmt"
	"github.com/sw965/crow/game/simultaneous"
	"github.com/sw965/crow/ucb"
	"math/rand"
)

type ActionPolicy[A comparable] map[A]float64
type SeparateActionPolicy[A comparable] []ActionPolicy[A]
type SeparateActionPolicyProvider[S any, A comparable] func(*S) SeparateActionPolicy[A]

type LeafNodeJointEval []float64
type LeafNodeJointEvaluator[S any] func(*S) (LeafNodeJointEval, error)

type Node[S any, Ass ~[]As, As ~[]A, A comparable] struct {
	State              S
	SeparateUCBManager ucb.SeparateManager[As, A]
	NextNodes          Nodes[S, Ass, As, A]
}

type Nodes[S any, Ass ~[]As, As ~[]A, A comparable] []*Node[S, Ass, As, A]

func (nodes Nodes[S, Ass, As, A]) FindByState(state *S, eq simultaneous.Comparator[S]) (*Node[S, Ass, As, A], bool) {
	for _, node := range nodes {
		if eq(&node.State, state) {
			return node, true
		}
	}
	return nil, false
}

type selectionInfo[S any, Ass ~[]As, As ~[]A, A comparable] struct {
	node        *Node[S, Ass, As, A]
	jointAction As
}

type selectionInfoSlice[S any, Ass ~[]As, As ~[]A, A comparable] []selectionInfo[S, Ass, As, A]

func (ss selectionInfoSlice[S, Ass, As, A]) backward(jointEval LeafNodeJointEval) {
	for _, s := range ss {
		node := s.node
		jointAction := s.jointAction
		for playerI, action := range jointAction {
			node.SeparateUCBManager[playerI][action].TotalValue += float64(jointEval[playerI])
			node.SeparateUCBManager[playerI][action].Trial += 1
		}
	}
}

type MCTS[S any, Ass ~[]As, As ~[]A, A comparable] struct {
	GameLogic                    simultaneous.Logic[S, Ass, As, A]
	UCBFunc                      ucb.Func
	SeparateActionPolicyProvider SeparateActionPolicyProvider[S, A]
	LeafNodeJointEvaluator       LeafNodeJointEvaluator[S]
	NextNodesCap                 int
}

func (mcts *MCTS[S, Ass, As, A]) SetSeparateUniformActionPolicyProvider() {
	mcts.SeparateActionPolicyProvider = func(state *S) SeparateActionPolicy[A] {
		ass := mcts.GameLogic.SeparateLegalActionsProvider(state)
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

func (mcts *MCTS[S, Ass, As, A]) SetPlayout(player simultaneous.Player[S, As, A], evaluator simultaneous.ResultJointEvaluator[S]) {
	mcts.LeafNodeJointEvaluator = func(sp *S) (LeafNodeJointEval, error) {
		s, err := mcts.GameLogic.Playout(player, *sp)
		if err != nil {
			return LeafNodeJointEval{}, err
		}
		jointEval, err := evaluator(&s)
		return LeafNodeJointEval(jointEval), err
	}
}

func (mcts *MCTS[S, Ass, As, A]) SetRandPlayout(eval simultaneous.ResultJointEvaluator[S], r *rand.Rand) {
	player := mcts.GameLogic.NewRandActionPlayer(r)
	mcts.SetPlayout(player, eval)
}

func (mcts *MCTS[S, Ass, As, A]) NewNode(state *S) (*Node[S, Ass, As, A], error) {
	policies := mcts.SeparateActionPolicyProvider(state)
	if len(policies) == 0 {
		return &Node[S, Ass, As, A]{}, fmt.Errorf("len(SeparateActionPolicy) == 0 である為、新しくNodeを生成出来ません。")
	}

	ms := make(ucb.SeparateManager[As, A], len(policies))
	for playerI, policy := range policies {
		m := ucb.Manager[As, A]{}
		if len(policy) == 0 {
			return &Node[S, Ass, As, A]{}, fmt.Errorf("%d番目のプレイヤーのActionPolicyが空である為、新しくNodeを生成出来ません。", playerI)
		}
		for a, p := range policy {
			m[a] = &ucb.Calculator{Func: mcts.UCBFunc, P: p}
		}
		ms[playerI] = m
	}

	nextNodes := make(Nodes[S, Ass, As, A], 0, mcts.NextNodesCap)
	return &Node[S, Ass, As, A]{
		State:              *state,
		SeparateUCBManager: ms,
		NextNodes:          nextNodes,
	}, nil
}

func (mcts *MCTS[S, Ass, As, A]) SelectExpansionBackward(node *Node[S, Ass, As, A], r *rand.Rand, capacity int) (int, error) {
	state := node.State
	selections := make(selectionInfoSlice[S, Ass, As, A], 0, capacity)
	var err error
	for {
		jointAction := node.SeparateUCBManager.JointActionByMax(r)
		selections = append(selections, selectionInfo[S, Ass, As, A]{node: node, jointAction: jointAction})

		state, err = mcts.GameLogic.Transitioner(state, jointAction)
		if err != nil {
			return 0, err
		}

		if isEnd := mcts.GameLogic.EndChecker(&state); isEnd {
			break
		}

		nextNode, ok := node.NextNodes.FindByState(&state, mcts.GameLogic.Comparator)
		if !ok {
			//expansion
			nextNode, err = mcts.NewNode(&state)
			if err != nil {
				return 0, err
			}
			node.NextNodes = append(node.NextNodes, nextNode)
			//新しくノードを作成したら、selectを終了する
			break
		}
		//nextNodesの中に、同じstateのNodeが存在するならば、それを次のNodeとする
		node = nextNode
	}
	jointEval, err := mcts.LeafNodeJointEvaluator(&state)
	selections.backward(jointEval)
	return len(selections), err
}

func (mcts *MCTS[S, Ass, As, A]) Run(simulation int, rootNode *Node[S, Ass, As, A], r *rand.Rand) error {
	if mcts.NextNodesCap <= 0 {
		return fmt.Errorf("MCTS.NextNodesCap > 0 でなければなりません。")
	}
	depth := 0
	var err error
	for i := 0; i < simulation; i++ {
		capacity := depth + 1
		depth, err = mcts.SelectExpansionBackward(rootNode, r, capacity)
		if err != nil {
			return err
		}
	}
	return nil
}
