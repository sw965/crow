package duct

import (
	"fmt"
	game "github.com/sw965/crow/game/simultaneous"
	"github.com/sw965/crow/ucb"
	"math/rand"
)

// https://www.terry-u16.net/entry/decoupled-uct

type LeafNodeEvaluator[S any] func(*S) (game.Evals, error)

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

func (ss selectionInfoSlice[S, Ass, As, A]) backward(evals game.Evals) {
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
	gameLogic         game.Logic[S, Ass, As, A]
	UCBFunc           ucb.Func
	PoliciesProvider  game.PoliciesProvider[S, Ass, As, A]
	LeafNodeEvaluator LeafNodeEvaluator[S]
	NextNodesCap      int
}

func (e *Engine[S, Ass, As, A]) GetGameLogic() game.Logic[S, Ass, As, A] {
	return e.gameLogic
}

func (e *Engine[S, Ass, As, A]) SetGameLogic(gl game.Logic[S, Ass, As, A]) {
	e.gameLogic = gl
}

func (e *Engine[S, Ass, As, A]) SetUniformPoliciesProvider() {
	e.PoliciesProvider = game.UniformPoliciesProvider[S, Ass, As, A]
}

func (e *Engine[S, Ass, As, A]) SetPlayout(player game.Player[S, Ass, As, A]) {
	e.LeafNodeEvaluator = func(state *S) (game.Evals, error) {
		final, err := e.gameLogic.Playout(*state, player)
		if err != nil {
			return game.Evals{}, err
		}
		scores, err := e.gameLogic.EvaluateResultScores(&final)
		return scores.ToEvals(), err
	}
}

func (e *Engine[S, Ass, As, A]) NewNode(state *S) (*Node[S, Ass, As, A], error) {
	legalActionTable := e.gameLogic.LegalActionTableProvider(state)
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
	var isEnd bool

	for {
		jointAction := node.UCBManagers.RandMaxKeys(r)
		selections = append(selections, selectionInfo[S, Ass, As, A]{node: node, jointAction: jointAction})

		state, err = e.gameLogic.Transitioner(state, jointAction)
		if err != nil {
			return 0, err
		}

		isEnd, err = e.gameLogic.IsEnd(&state)
		if err != nil {
			return 0, err
		}

		if isEnd {
			break
		}

		nextNode, ok := node.NextNodes.FindByState(&state, e.gameLogic.Comparator)
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

	var evals game.Evals
	if isEnd {
		scores, err := e.gameLogic.EvaluateResultScores(&state)
		if err != nil {
			return 0, err
		}
		evals = scores.ToEvals()
	} else {
		evals, err = e.LeafNodeEvaluator(&state)
		//selections.backwardの前にエラー処理をしない場合、index out of range が起きる事がある。
		if err != nil {
			return 0, err
		}
	}

	selections.backward(evals)
	return len(selections), nil
}

func (e *Engine[S, Ass, As, A]) Search(rootNode *Node[S, Ass, As, A], simulation int, r *rand.Rand) error {
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

func (e *Engine[S, Ass, As, A]) NewPlayer(simulation int, selector game.Selector[As, A], r *rand.Rand) game.Player[S, Ass, As, A] {
	return func(state *S, _ Ass) (As, error) {
		rootNode, err := e.NewNode(state)
		if err != nil {
			return As{}, err
		}

		err = e.Search(rootNode, simulation, r)
		if err != nil {
			return As{}, err
		}

		ps := make(game.Policies[A], len(rootNode.UCBManagers))
		for i, m := range rootNode.UCBManagers {
			trialPercents := m.TrialPercentPerKey()
			p := game.Policy[A]{}
			for k, v := range trialPercents {
				p[k] = v
			}
			ps[i] = p		
		}
		jointAction := selector(ps)
		return jointAction, nil
	}
}