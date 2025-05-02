package dpuct

import (
	"fmt"
	game "github.com/sw965/crow/game/simultaneous"
	"github.com/sw965/crow/pucb"
	"math/rand"
	orand "github.com/sw965/omw/math/rand"
)

// https://www.terry-u16.net/entry/decoupled-uct

type LeafNodeEvals []float32
type LeafNodeEvaluator[S any] func(*S) (LeafNodeEvals, error)

type Node[S any, Ass ~[]As, As ~[]A, A comparable] struct {
	State       S
	UCBManagers pucb.Managers[As, A]
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

func (ss selectionInfoSlice[S, Ass, As, A]) backward(evals LeafNodeEvals) {
	for _, s := range ss {
		node := s.node
		jointAction := s.jointAction
		for playerI, action := range jointAction {
			node.UCBManagers[playerI][action].TotalValue += float32(evals[playerI])
			node.UCBManagers[playerI][action].Trial += 1
		}
	}
}

type Policy[A comparable] map[A]float32
type Policies[A comparable] []Policy[A]
type PoliciesProvider[S any, Ass ~[]As, As ~[]A, A comparable] func(*S, Ass) Policies[A]

func UniformPoliciesProvider[S any, Ass ~[]As, As ~[]A, A comparable](state *S, legalActionTable Ass) Policies[A] {
	policies := make(Policies[A], len(legalActionTable))
	for i, actions := range legalActionTable {
		n := len(actions)
		p := 1.0 / float32(n)
		policy := Policy[A]{}
		for _, a := range actions {
			policy[a] = p
		}
		policies[i] = policy
	}
	return policies
}

type Engine[S any, Ass ~[]As, As ~[]A, A comparable] struct {
	GameLogic         game.Logic[S, Ass, As, A]
	UCBFunc           pucb.Func
	PoliciesProvider  PoliciesProvider[S, Ass, As, A]
	LeafNodeEvaluator LeafNodeEvaluator[S]
	NextNodesCap      int
}

func (e *Engine[S, Ass, As, A]) SetUniformPoliciesProvider() {
	e.PoliciesProvider = UniformPoliciesProvider[S, Ass, As, A]
}

func (e *Engine[S, Ass, As, A]) SetPlayout(player game.Players[S, Ass, As, A]) {
	e.LeafNodeEvaluator = func(state *S) (LeafNodeEvals, error) {
		final, err := e.GameLogic.Playout(*state, player)
		if err != nil {
			return LeafNodeEvals{}, err
		}
		scores, err := e.GameLogic.EvaluateResultScores(&final)
		evals := make(LeafNodeEvals, len(scores))
		for i, s := range scores {
			evals[i] = s
		}
		return evals, err
	}
}

func (e *Engine[S, Ass, As, A]) NewNode(state *S) (*Node[S, Ass, As, A], error) {
	legalActionTable := e.GameLogic.LegalActionTableProvider(state)
	policies := e.PoliciesProvider(state, legalActionTable)
	if len(policies) == 0 {
		return &Node[S, Ass, As, A]{}, fmt.Errorf("len(Policies) == 0 である為、新しくNodeを生成出来ません。")
	}

	ms := make(pucb.Managers[As, A], len(policies))
	for playerI, policy := range policies {
		m := pucb.Manager[As, A]{}
		if len(policy) == 0 {
			return &Node[S, Ass, As, A]{}, fmt.Errorf("%d番目のプレイヤーのPolicyが空である為、新しくNodeを生成出来ません。", playerI)
		}
		for a, p := range policy {
			m[a] = &pucb.Calculator{Func: e.UCBFunc, P: p}
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
		jointAction := make(As, len(node.UCBManagers))
		for playerI, m := range node.UCBManagers {
			ks := m.MaxKeys()
			jointAction[playerI] = orand.Choice(ks, r)
		}

		selections = append(selections, selectionInfo[S, Ass, As, A]{node: node, jointAction: jointAction})

		state, err = e.GameLogic.Transitioner(state, jointAction)
		if err != nil {
			return 0, err
		}

		isEnd, err = e.GameLogic.IsEnd(&state)
		if err != nil {
			return 0, err
		}

		if isEnd {
			break
		}

		nextNode, ok := node.NextNodes.FindByState(&state, e.GameLogic.Comparator)
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

	var evals LeafNodeEvals
	if isEnd {
		scores, err := e.GameLogic.EvaluateResultScores(&state)
		if err != nil {
			return 0, err
		}
		evals = make(LeafNodeEvals, len(scores))
		for i, s := range scores {
			evals[i] = s
		}
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

func (e *Engine[S, Ass, As, A]) NewPlayer(simulation int, t float32, r *rand.Rand) game.Player[S, Ass, As, A] {
	return func(state *S, _ Ass) (As, error) {
		rootNode, err := e.NewNode(state)
		if err != nil {
			return As{}, err
		}

		err = e.Search(rootNode, simulation, r)
		if err != nil {
			return As{}, err
		}

		jointAction := make(As, len(rootNode.UCBManagers))
		for i, m := range rootNode.UCBManagers {
			action, err := m.SelectKeyByTrialPercentAboveFractionOfMax(t, r)
			if err != nil {
				return As{}, err
			}
			jointAction[i] = action
		}
		return jointAction, nil
	}
}