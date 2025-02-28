package uct

import (
	"fmt"
	"math/rand"
	"github.com/sw965/crow/ucb"
	game "github.com/sw965/crow/game/sequential"
	omwrand "github.com/sw965/omw/math/rand"
)

type RootNodeEvalByAgent[G comparable] map[G]float64

func (es RootNodeEvalByAgent[G]) DivScalar(s float64) {
	for k := range es {
		es[k] /= s
	}
}

type LeafNodeEvalByAgent[G comparable] map[G]float64
type LeafNodeEvaluator[S any, G comparable] func(*S) (LeafNodeEvalByAgent[G], error)

type Node[S any, As ~[]A, A, G comparable] struct {
	State      S
	Agent      G
	UCBManager ucb.Manager[As, A]
	NextNodes  Nodes[S, As, A, G]
}

func (node *Node[S, As, A, G]) Trial() int {
	return node.UCBManager.TotalTrial()
}

type Nodes[S any, As ~[]A, A, G comparable] []*Node[S, As, A, G]

func (nodes Nodes[S, As, A, G]) FindByState(state *S, eq game.Comparator[S]) (*Node[S, As, A, G], bool) {
	for _, node := range nodes {
		if eq(&node.State, state) {
			return node, true
		}
	}
	return nil, false
}

type selectionInfo[S any, As ~[]A, A, G comparable] struct {
	node   *Node[S, As, A, G]
	action A
}

type selectionInfoSlice[S any, As ~[]A, A, G comparable] []selectionInfo[S, As, A, G]

func (ss selectionInfoSlice[S, As, A, G]) backward(evals LeafNodeEvalByAgent[G]) {
	for _, s := range ss {
		node := s.node
		action := s.action
		eval := evals[node.Agent]
		node.UCBManager[action].TotalValue += float64(eval)
		node.UCBManager[action].Trial += 1
	}
}

type Policy[A comparable] map[A]float64
type PolicyProvider[S any, As ~[]A, A comparable] func(*S, As) Policy[A]

func UniformPolicyProvider[S any, As ~[]A, A comparable](state *S, legalActions As) Policy[A] {
	n := len(legalActions)
	p := 1.0 / float64(n)
	policy := Policy[A]{}
	for _, a := range legalActions {
		policy[a] = p
	}
	return policy
}

type Engine[S any, As ~[]A, A, G comparable] struct {
	GameLogic         game.Logic[S, As, A, G]
	UCBFunc           ucb.Func
	PolicyProvider    PolicyProvider[S, As, A]
	LeafNodeEvaluator LeafNodeEvaluator[S, G]
	NextNodesCap      int
}

func (e *Engine[S, As, A, G]) SetUniformPolicyProvider() {
	e.PolicyProvider = UniformPolicyProvider[S, As, A]
}

func (e *Engine[S, As, A, G]) SetPlayout(players game.PlayerByAgent[S, As, A, G]) {
	e.LeafNodeEvaluator = func(state *S) (LeafNodeEvalByAgent[G], error) {
		final, err := e.GameLogic.Playout(*state, players)
		if err != nil {
			return LeafNodeEvalByAgent[G]{}, err
		}
		scores, err := e.GameLogic.EvaluateResultScoreByAgent(&final)
		evals := LeafNodeEvalByAgent[G]{}
		for k, v := range scores {
			evals[k] = v
		}
		return evals, err
	}
}

func (e *Engine[S, As, A, G]) NewNode(state *S) (*Node[S, As, A, G], error) {
	policy := e.PolicyProvider(state, e.GameLogic.LegalActionsProvider(state))
	if len(policy) == 0 {
		return &Node[S, As, A, G]{}, fmt.Errorf("len(Policy) == 0 である為、新しくNodeを生成出来ません。")
	}

	u := ucb.Manager[As, A]{}
	for a, p := range policy {
		u[a] = &ucb.Calculator{Func: e.UCBFunc, P: p}
	}

	agent := e.GameLogic.CurrentTurnAgentGetter(state)
	nextNodes := make(Nodes[S, As, A, G], 0, e.NextNodesCap)
	return &Node[S, As, A, G]{State: *state, Agent: agent, UCBManager: u, NextNodes: nextNodes}, nil
}

func (e *Engine[S, As, A, G]) SelectExpansionBackward(node *Node[S, As, A, G], capacity int, r *rand.Rand) (LeafNodeEvalByAgent[G], int, error) {
	state := node.State
	selections := make(selectionInfoSlice[S, As, A, G], 0, capacity)
	var err error
	var isEnd bool

	for {
		action := omwrand.Choice(node.UCBManager.MaxKeys(), r)
		selections = append(selections, selectionInfo[S, As, A, G]{node: node, action: action})

		state, err = e.GameLogic.Transitioner(state, &action)
		if err != nil {
			return LeafNodeEvalByAgent[G]{}, 0, err
		}

		isEnd, err = e.GameLogic.IsEnd(&state)
		if err != nil {
			return LeafNodeEvalByAgent[G]{}, 0, err
		}

		if isEnd {
			break
		}

		nextNode, ok := node.NextNodes.FindByState(&state, e.GameLogic.Comparator)
		if !ok {
			// expansion
			nextNode, err = e.NewNode(&state)
			if err != nil {
				return LeafNodeEvalByAgent[G]{}, 0, err
			}
			node.NextNodes = append(node.NextNodes, nextNode)
			// 新しくノードを作成したら、selectを終了する
			break
		}
		// nextNodes の中に、同じ state の Node が存在するならば、それを次の Node とする
		node = nextNode
	}

	evals := LeafNodeEvalByAgent[G]{}
	if isEnd {
		scores, err := e.GameLogic.EvaluateResultScoreByAgent(&state)
		if err != nil {
			return LeafNodeEvalByAgent[G]{}, 0, err
		}
		for k, v := range scores {
			evals[k] = v
		}
	} else {
		evals, err = e.LeafNodeEvaluator(&state)
		if err != nil {
			return LeafNodeEvalByAgent[G]{}, 0, err
		}
	}

	selections.backward(evals)
	return evals, len(selections), err
}

func (e *Engine[S, As, A, G]) Search(rootNode *Node[S, As, A, G], simulation int, r *rand.Rand) (RootNodeEvalByAgent[G], error) {
	if e.NextNodesCap <= 0 {
		return RootNodeEvalByAgent[G]{}, fmt.Errorf("Engine.NextNodesCap > 0 でなければなりません。")
	}

	rootEvals := RootNodeEvalByAgent[G]{}
	capacity := 0
	for i := 0; i < simulation; i++ {
		leafEvals, depth, err := e.SelectExpansionBackward(rootNode, capacity, r)
		if err != nil {
			return RootNodeEvalByAgent[G]{}, err
		}
		for k, v := range leafEvals {
			rootEvals[k] += v
		}
		capacity = depth + 1
	}

	rootEvals.DivScalar(float64(simulation))
	return rootEvals, nil
}

func (e *Engine[S, As, A, G]) NewPlayer(simulation int, t float64, r *rand.Rand) game.Player[S, As, A] {
	return func(state *S, _ As) (A, error) {
		rootNode, err := e.NewNode(state)
		if err != nil {
			var a A
			return a, err
		}

		_, err = e.Search(rootNode, simulation, r)
		if err != nil {
			var a A
			return a, err
		}

		action, err := rootNode.UCBManager.SelectKeyByTrialPercentAboveFractionOfMax(t, r)
		return action, err
	}
}