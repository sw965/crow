package uct

import (
	"fmt"
	"math/rand"
	omwrand "github.com/sw965/omw/math/rand"
	"github.com/sw965/crow/ucb"
	game "github.com/sw965/crow/game/sequential"
)

type LeafNodeEvaluator[S any, G comparable] func(*S) (game.AgentEvals[G], error)

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

func (ss selectionInfoSlice[S, As, A, G]) backward(evals game.AgentEvals[G]) {
	for _, s := range ss {
		node := s.node
		action := s.action
		eval := evals[node.Agent]
		node.UCBManager[action].TotalValue += float64(eval)
		node.UCBManager[action].Trial += 1
	}
}

type Engine[S any, As ~[]A, A, Ag comparable] struct {
	game              game.Engine[S, As, A, Ag]
	UCBFunc           ucb.Func
	PolicyProvider    game.PolicyProvider[S, As, A]
	LeafNodeEvaluator LeafNodeEvaluator[S, Ag]
	NextNodesCap      int
}

func (e *Engine[S, As, A, G]) GetGame() game.Engine[S, As, A, G]{
	return e.game
}

func (e *Engine[S, As, A, G]) SetGame(g game.Engine[S, As, A, G]) {
	e.game = g
}

func (e *Engine[S, As, A, G]) SetUniformPolicyProvider() {
	e.PolicyProvider = game.UniformPolicyProvider[S, As, A]
}

func (e *Engine[S, As, A, G]) SetPlayout() {
	e.LeafNodeEvaluator = func(state *S) (game.AgentEvals[G], error) {
		final, err := e.game.Playout(*state)
		if err != nil {
			return game.AgentEvals[G]{}, err
		}
		scores, err := e.game.Logic.EvaluateAgentResultScores(&final)
		return scores.ToEvalPerAgent(), err
	}
}

func (e *Engine[S, As, A, G]) NewNode(state *S) (*Node[S, As, A, G], error) {
	policy := e.PolicyProvider(state, e.game.Logic.LegalActionsProvider(state))
	if len(policy) == 0 {
		return &Node[S, As, A, G]{}, fmt.Errorf("len(Policy) == 0 である為、新しくNodeを生成出来ません。")
	}

	u := ucb.Manager[As, A]{}
	for a, p := range policy {
		u[a] = &ucb.Calculator{Func: e.UCBFunc, P: p}
	}

	agent := e.game.Logic.CurrentTurnAgentGetter(state)
	nextNodes := make(Nodes[S, As, A, G], 0, e.NextNodesCap)
	return &Node[S, As, A, G]{State: *state, Agent: agent, UCBManager: u, NextNodes: nextNodes}, nil
}

func (e *Engine[S, As, A, Ag]) SelectExpansionBackward(node *Node[S, As, A, Ag], capacity int, r *rand.Rand) (game.AgentEvals[Ag], int, error) {
	state := node.State
	selections := make(selectionInfoSlice[S, As, A, Ag], 0, capacity)
	var err error

	for {
		action := omwrand.Choice(node.UCBManager.MaxKeys(), r)
		selections = append(selections, selectionInfo[S, As, A, Ag]{node: node, action: action})

		state, err = e.game.Logic.Transitioner(state, &action)
		if err != nil {
			return game.AgentEvals[Ag]{}, 0, err
		}

		isEnd, err := e.game.Logic.IsEnd(&state)
		if err != nil {
			return game.AgentEvals[Ag]{}, 0, err
		}

		if isEnd {
			break
		}

		nextNode, ok := node.NextNodes.FindByState(&state, e.game.Logic.Comparator)
		if !ok {
			//expansion
			nextNode, err = e.NewNode(&state)
			if err != nil {
				return game.AgentEvals[Ag]{}, 0, err
			}
			node.NextNodes = append(node.NextNodes, nextNode)
			//新しくノードを作成したら、selectを終了する
			break
		}
		//nextNodesの中に、同じstateのNodeが存在するならば、それを次のNodeとする
		node = nextNode
	}

	evals, err := e.LeafNodeEvaluator(&state)
	selections.backward(evals)
	return evals, len(selections), err
}

func (e *Engine[S, As, A, Ag]) Search(rootNode *Node[S, As, A, Ag], simulation int, r *rand.Rand) (game.AgentEvals[Ag], error) {
	if e.NextNodesCap <= 0 {
		return game.AgentEvals[Ag]{}, fmt.Errorf("Engine.NextNodesCap > 0 でなければなりません。")
	}

	avgs := game.AgentEvals[Ag]{}
	capacity := 0
	for i := 0; i < simulation; i++ {
		evals, depth, err := e.SelectExpansionBackward(rootNode, capacity, r)
		if err != nil {
			return game.AgentEvals[Ag]{}, err
		}
		avgs.Add(evals)
		capacity = depth + 1
	}

	avgs.DivScalar(game.Eval(simulation))
	return avgs, nil
}

func (e *Engine[S, As, A, Ag]) NewPlayer(simulation int, selector game.Selector[A], r *rand.Rand) game.Player[S, As, A] {
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
		
		trialPercents := rootNode.UCBManager.TrialPercentPerKey()
		policy := game.Policy[A]{}
		for k, v := range trialPercents {
			trialPercents[k] = v
		}

		action := selector(policy)
		return action, nil
	}
}