package duct

import (
	"fmt"
	game "github.com/sw965/crow/game/simultaneous"
	"github.com/sw965/crow/game/solver"
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

func (nodes Nodes[S, Ass, As, A]) FindByState(state *S, eq game.Comparator[S]) (*Node[S, Ass, As, A], bool) {
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

type Engine[S any, Ass ~[]As, As ~[]A, A comparable] struct {
	gameEngine             game.Engine[S, Ass, As, A]
	UCBFunc                ucb.Func
	PoliciesProvider       solver.PoliciesProvider[S, Ass, As, A]
	LeafNodeEvaluator      LeafNodeEvaluator[S]
	NextNodesCap           int
}

func (e *Engine[S, Ass, As, A]) GetGameEngine() game.Engine[S, Ass, As, A] {
	return e.gameEngine
}

func (e *Engine[S, Ass, As, A]) SetGameEngine(ge game.Engine[S, Ass, As, A]) {
	e.gameEngine = ge
}

func (e *Engine[S, As, A, G]) SetUniformPoliciesProvider() {
	e.PoliciesProvider = solver.UniformPoliciesProvider[S, As, A]
}

func (e *Engine[S, Ass, As, A]) SetPlayout() {
	e.LeafNodeEvaluator = func(state *S) (solver.Evals, error) {
		final, err := e.gameEngine.Playout(*state)
		if err != nil {
			return solver.Evals{}, err
		}
		scores, err := e.gameEngine.Logic.EvaluateResultScores(&final)
		return scores.ToEvals(), err
	}
}

func (e *Engine[S, Ass, As, A]) NewNode(state *S) (*Node[S, Ass, As, A], error) {
	legalActionTable := e.gameEngine.Logic.LegalActionTableProvider(state)
	policies := e.PoliciesProvider(state, legalActionTable)
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
			m[a] = &ucb.Calculator{Func: e.UCBFunc, P: p}
		}
		ms[playerI] = m
	}

	nextNodes := make(Nodes[S, Ass, As, A], 0, e.NextNodesCap)
	return &Node[S, Ass, As, A]{
		State:       *state,
		UCBManagers: ms,
		NextNodes:   nextNodes,
	}, nil
}

func (e *Engine[S, Ass, As, A]) SelectExpansionBackward(node *Node[S, Ass, As, A], r *rand.Rand, capacity int) (int, error) {
	state := node.State
	selections := make(selectionInfoSlice[S, Ass, As, A], 0, capacity)
	var err error
	for {
		jointAction := node.UCBManagers.RandMaxKeys(r)
		selections = append(selections, selectionInfo[S, Ass, As, A]{node: node, jointAction: jointAction})

		state, err = e.gameEngine.Logic.Transitioner(state, jointAction)
		if err != nil {
			return 0, err
		}

		if isEnd, err := e.gameEngine.Logic.IsEnd(&state); err != nil {
			if err != nil {
				return 0, err
			}
			if isEnd {
				break
			}
		}

		nextNode, ok := node.NextNodes.FindByState(&state, e.gameEngine.Logic.Comparator)
		if !ok {
			//expansion
			nextNode, err = e.NewNode(&state)
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

	evals, err := e.LeafNodeEvaluator(&state)
	//selections.backwardの前にエラー処理をしない場合、index out of range が起きる事がある。
	if err != nil {
		return 0, err
	}
	selections.backward(evals)
	return len(selections), nil
}

func (e *Engine[S, Ass, As, A]) Run(simulation int, rootNode *Node[S, Ass, As, A], r *rand.Rand) error {
	if e.NextNodesCap <= 0 {
		return fmt.Errorf("MCTS.NextNodesCap > 0 でなければなりません。")
	}
	depth := 0
	var err error
	for i := 0; i < simulation; i++ {
		capacity := depth + 1
		depth, err = e.SelectExpansionBackward(rootNode, r, capacity)
		if err != nil {
			return err
		}
	}
	return nil
}