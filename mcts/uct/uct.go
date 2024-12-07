package uct

import (
	"fmt"
	"math/rand"
	omwrand "github.com/sw965/omw/math/rand"
	"github.com/sw965/crow/ucb"
	game "github.com/sw965/crow/game/sequential"
)

type LeafNodeEvaluator[S any, Ag comparable] func(*S) (game.AgentEvals[Ag], error)

type Node[S any, As ~[]A, A, Ag comparable] struct {
	State      S
	Agent      Ag
	UCBManager ucb.Manager[As, A]
	NextNodes  Nodes[S, As, A, Ag]
}

func (node *Node[S, As, A, Ag]) Trial() int {
	return node.UCBManager.TotalTrial()
}

type Nodes[S any, As ~[]A, A, Ag comparable] []*Node[S, As, A, Ag]

func (nodes Nodes[S, As, A, Ag]) FindByState(state *S, eq game.Comparator[S]) (*Node[S, As, A, Ag], bool) {
	for _, node := range nodes {
		if eq(&node.State, state) {
			return node, true
		}
	}
	return nil, false
}

type selectionInfo[S any, As ~[]A, A, Ag comparable] struct {
	node   *Node[S, As, A, Ag]
	action A
}

type selectionInfoSlice[S any, As ~[]A, A, Ag comparable] []selectionInfo[S, As, A, Ag]

func (ss selectionInfoSlice[S, As, A, Ag]) backward(evals game.AgentEvals[Ag]) {
	for _, s := range ss {
		node := s.node
		action := s.action
		eval := evals[node.Agent]
		node.UCBManager[action].TotalValue += float64(eval)
		node.UCBManager[action].Trial += 1
	}
}

type Engine[S any, As ~[]A, A, Ag comparable] struct {
	gameLogic         game.Logic[S, As, A, Ag]
	UCBFunc           ucb.Func
	PolicyProvider    game.PolicyProvider[S, As, A]
	LeafNodeEvaluator LeafNodeEvaluator[S, Ag]
	NextNodesCap      int
}

func (e *Engine[S, As, A, Ag]) GetGameLogic() game.Logic[S, As, A, Ag]{
	return e.gameLogic
}

func (e *Engine[S, As, A, Ag]) SetGameLogic(gl game.Logic[S, As, A, Ag]) {
	e.gameLogic = gl
}

func (e *Engine[S, As, A, Ag]) SetUniformPolicyProvider() {
	e.PolicyProvider = game.UniformPolicyProvider[S, As, A]
}

func (e *Engine[S, As, A, Ag]) SetPlayout(players game.AgentPlayers[S, As, A, Ag]) {
	e.LeafNodeEvaluator = func(state *S) (game.AgentEvals[Ag], error) {
		final, err := e.gameLogic.Playout(*state, players)
		if err != nil {
			return game.AgentEvals[Ag]{}, err
		}
		scores, err := e.gameLogic.EvaluateAgentResultScores(&final)
		return scores.ToAgentEvals(), err
	}
}

func (e *Engine[S, As, A, Ag]) NewNode(state *S) (*Node[S, As, A, Ag], error) {
	policy := e.PolicyProvider(state, e.gameLogic.LegalActionsProvider(state))
	if len(policy) == 0 {
		return &Node[S, As, A, Ag]{}, fmt.Errorf("len(Policy) == 0 である為、新しくNodeを生成出来ません。")
	}

	u := ucb.Manager[As, A]{}
	for a, p := range policy {
		u[a] = &ucb.Calculator{Func: e.UCBFunc, P: p}
	}

	agent := e.gameLogic.CurrentTurnAgentGetter(state)
	nextNodes := make(Nodes[S, As, A, Ag], 0, e.NextNodesCap)
	return &Node[S, As, A, Ag]{State: *state, Agent: agent, UCBManager: u, NextNodes: nextNodes}, nil
}

func (e *Engine[S, As, A, Ag]) SelectExpansionBackward(node *Node[S, As, A, Ag], capacity int, r *rand.Rand) (game.AgentEvals[Ag], int, error) {
	state := node.State
	selections := make(selectionInfoSlice[S, As, A, Ag], 0, capacity)
	var err error
	var isEnd bool

	for {
		action := omwrand.Choice(node.UCBManager.MaxKeys(), r)
		selections = append(selections, selectionInfo[S, As, A, Ag]{node: node, action: action})

		state, err = e.gameLogic.Transitioner(state, &action)
		if err != nil {
			return game.AgentEvals[Ag]{}, 0, err
		}

		isEnd, err = e.gameLogic.IsEnd(&state)
		if err != nil {
			return game.AgentEvals[Ag]{}, 0, err
		}

		if isEnd {
			break
		}

		nextNode, ok := node.NextNodes.FindByState(&state, e.gameLogic.Comparator)
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

	var evals game.AgentEvals[Ag]
	if isEnd {
		scores, err := e.gameLogic.EvaluateAgentResultScores(&state)
		if err != nil {
			return game.AgentEvals[Ag]{}, 0, err
		}
		evals = scores.ToAgentEvals()
	} else {
		evals, err = e.LeafNodeEvaluator(&state)
		if err != nil {
			return game.AgentEvals[Ag]{}, 0, err
		}		
	}

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
			policy[k] = v
		}

		action := selector(policy)
		return action, nil
	}
}