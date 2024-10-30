package duct

import (
	"fmt"
	"github.com/sw965/crow/game/simultaneous"
	"github.com/sw965/crow/game/simultaneous/solver"
	"github.com/sw965/crow/ucb"
	"math/rand"
)

// https://www.terry-u16.net/entry/decoupled-uct

type LeafNodeEvaluator[S any] func(*S) (solver.Evals, error)

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

func (ss selectionInfoSlice[S, Ass, As, A]) backward(evals solver.Evals) {
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
	gameLogic              simultaneous.Logic[S, Ass, As, A]
	UCBFunc                ucb.Func
	PoliciesProvider       solver.PoliciesProvider[S, Ass, As, A]
	LeafNodeEvaluator      LeafNodeEvaluator[S]
	NextNodesCap           int
}

func (m *MCTS[S, Ass, As, A]) GetGameLogic() simultaneous.Logic[S, Ass, As, A] {
	return m.gameLogic
}

func (m *MCTS[S, Ass, As, A]) SetGameLogic(gl simultaneous.Logic[S, Ass, As, A]) {
	m.gameLogic = gl
}

func (m *MCTS[S, As, A, G]) SetUniformPoliciesProvider() {
	m.PoliciesProvider = solver.UniformPoliciesProvider[S, As, A]
}

func (m *MCTS[S, Ass, As, A]) SetPlayout(players simultaneous.Players[S, Ass, As, A]) {
	m.LeafNodeEvaluator = func(state *S) (solver.Evals, error) {
		final, err := m.gameLogic.Playout(players, *state)
		if err != nil {
			return solver.Evals{}, err
		}
		scores, err := m.gameLogic.EvaluateResultScores(&final)
		return solver.ResultScoresToEvals(scores), err
	}
}

func (m *MCTS[S, Ass, As, A]) SetRandPlayout(playerNum int, r *rand.Rand) {
	players := make(simultaneous.Players[S, Ass, As, A], playerNum)
	for i := 0; i < playerNum; i++ {
		players[i] = m.gameLogic.NewRandActionPlayer(r)
	}
	m.SetPlayout(players)
}

func (m *MCTS[S, Ass, As, A]) NewNode(state *S) (*Node[S, Ass, As, A], error) {
	legalActionTable := m.gameLogic.LegalActionTableProvider(state)
	policies := m.PoliciesProvider(state, legalActionTable)
	if len(policies) == 0 {
		return &Node[S, Ass, As, A]{}, fmt.Errorf("len(SeparateActionPolicy) == 0 である為、新しくNodeを生成出来ません。")
	}

	ums := make(ucb.Managers[As, A], len(policies))
	for playerI, policy := range policies {
		um := ucb.Manager[As, A]{}
		if len(policy) == 0 {
			return &Node[S, Ass, As, A]{}, fmt.Errorf("%d番目のプレイヤーのActionPolicyが空である為、新しくNodeを生成出来ません。", playerI)
		}
		for a, p := range policy {
			um[a] = &ucb.Calculator{Func: m.UCBFunc, P: p}
		}
		ums[playerI] = um
	}

	nextNodes := make(Nodes[S, Ass, As, A], 0, m.NextNodesCap)
	return &Node[S, Ass, As, A]{
		State:       *state,
		UCBManagers: ums,
		NextNodes:   nextNodes,
	}, nil
}

func (m *MCTS[S, Ass, As, A]) SelectExpansionBackward(node *Node[S, Ass, As, A], r *rand.Rand, capacity int) (int, error) {
	state := node.State
	selections := make(selectionInfoSlice[S, Ass, As, A], 0, capacity)
	var err error
	for {
		jointAction := node.UCBManagers.RandMaxKeys(r)
		selections = append(selections, selectionInfo[S, Ass, As, A]{node: node, jointAction: jointAction})

		state, err = m.gameLogic.Transitioner(state, jointAction)
		if err != nil {
			return 0, err
		}

		if isEnd, err := m.gameLogic.IsEnd(&state); err != nil {
			if err != nil {
				return 0, err
			}
			if isEnd {
				break
			}
		}

		nextNode, ok := node.NextNodes.FindByState(&state, m.gameLogic.Comparator)
		if !ok {
			//expansion
			nextNode, err = m.NewNode(&state)
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

	evals, err := m.LeafNodeEvaluator(&state)
	//selections.backwardの前にエラー処理をしない場合、index out of range が起きる事がある。
	if err != nil {
		return 0, err
	}
	selections.backward(evals)
	return len(selections), nil
}

func (m *MCTS[S, Ass, As, A]) Run(simulation int, rootNode *Node[S, Ass, As, A], r *rand.Rand) error {
	if m.NextNodesCap <= 0 {
		return fmt.Errorf("MCTS.NextNodesCap > 0 でなければなりません。")
	}
	depth := 0
	var err error
	for i := 0; i < simulation; i++ {
		capacity := depth + 1
		depth, err = m.SelectExpansionBackward(rootNode, r, capacity)
		if err != nil {
			return err
		}
	}
	return nil
}