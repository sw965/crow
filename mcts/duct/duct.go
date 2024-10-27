package duct

import (
	"fmt"
	"github.com/sw965/crow/game/simultaneous"
	"github.com/sw965/crow/ucb"
	"math/rand"
)

// https://www.terry-u16.net/entry/decoupled-uct

type ActionPolicy[A comparable] map[A]float64
type ActionPolicies[A comparable] []ActionPolicy[A]
type ActionPoliciesProvider[S any, Ass ~[]As, As ~[]A, A comparable] func(*S, Ass) ActionPolicies[A]

type LeafNodeEvals []float64
type LeafNodeEvaluator[S any] func(*S) (LeafNodeEvals, error)

type Node[S any, Ass ~[]As, As ~[]A, A comparable] struct {
	State       S
	UCBManagers ucb.Managers[As, A]
	NextNodes   Nodes[S, Ass, As, A]
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

func (ss selectionInfoSlice[S, Ass, As, A]) backward(evals LeafNodeEvals) {
	for _, s := range ss {
		node := s.node
		jointAction := s.jointAction
		for playerI, action := range jointAction {
			node.UCBManagers[playerI][action].TotalValue += float64(evals[playerI])
			node.UCBManagers[playerI][action].Trial += 1
		}
	}
}

type MCTS[S any, Ass ~[]As, As ~[]A, A comparable] struct {
	GameLogic              simultaneous.Logic[S, Ass, As, A]
	UCBFunc                ucb.Func
	ActionPoliciesProvider ActionPoliciesProvider[S, Ass, As, A]
	LeafNodeEvaluator      LeafNodeEvaluator[S]
	NextNodesCap           int
}

func (mcts *MCTS[S, Ass, As, A]) SetUniformActionPoliciesProvider() {
	mcts.ActionPoliciesProvider = func(state *S, legalActionTable Ass) ActionPolicies[A] {
		policies := make(ActionPolicies[A], len(legalActionTable))
		for playerI, as := range legalActionTable {
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

func (mcts *MCTS[S, Ass, As, A]) SetPlayout(players simultaneous.Players[S, As, A]) {
	mcts.LeafNodeEvaluator = func(sp *S) (LeafNodeEvals, error) {
		s, err := mcts.GameLogic.Playout(players, *sp)
		if err != nil {
			return LeafNodeEvals{}, err
		}
		scores, err := mcts.GameLogic.EvaluateResultScores(&s)
		return LeafNodeEvals(scores), err
	}
}

func (mcts *MCTS[S, Ass, As, A]) SetRandPlayout(playerNum int, r *rand.Rand) {
	players := make(simultaneous.Players[S, As, A], playerNum)
	for i := 0; i < playerNum; i++ {
		players[i] = mcts.GameLogic.NewRandActionPlayer(r)
	}
	mcts.SetPlayout(players)
}

func (mcts *MCTS[S, Ass, As, A]) NewNode(state *S) (*Node[S, Ass, As, A], error) {
	ass := mcts.GameLogic.LegalActionTableProvider(state)
	policies := mcts.ActionPoliciesProvider(state, ass)
	if len(policies) == 0 {
		return &Node[S, Ass, As, A]{}, fmt.Errorf("len(SeparateActionPolicy) == 0 である為、新しくNodeを生成出来ません。")
	}

	ms := make(ucb.Managers[As, A], len(policies))
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
		State:       *state,
		UCBManagers: ms,
		NextNodes:   nextNodes,
	}, nil
}

func (mcts *MCTS[S, Ass, As, A]) SelectExpansionBackward(node *Node[S, Ass, As, A], r *rand.Rand, capacity int) (int, error) {
	state := node.State
	selections := make(selectionInfoSlice[S, Ass, As, A], 0, capacity)
	var err error
	for {
		jointAction := node.UCBManagers.RandMaxKeys(r)
		selections = append(selections, selectionInfo[S, Ass, As, A]{node: node, jointAction: jointAction})

		state, err = mcts.GameLogic.Transitioner(state, jointAction)
		if err != nil {
			return 0, err
		}

		if isEnd, err := mcts.GameLogic.IsEnd(&state); err != nil {
			if err != nil {
				return 0, err
			}
			if isEnd {
				break
			}
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

	evals, err := mcts.LeafNodeEvaluator(&state)
	//selections.backwardの前にエラー処理をしない場合、index out of range が起きる事がある。
	if err != nil {
		return 0, err
	}
	selections.backward(evals)
	return len(selections), nil
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