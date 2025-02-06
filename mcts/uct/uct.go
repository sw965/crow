package uct

import (
	"fmt"
	"math/rand"
	omwrand "github.com/sw965/omw/math/rand"
	"github.com/sw965/crow/ucb"
	game "github.com/sw965/crow/game/sequential"
)

type AgentLeafNodeEvals[Ag comparable] map[Ag]float64

type LeafNodeEvaluator[S any, Ag comparable] func(*S) (AgentLeafNodeEvals[Ag], error)

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

func (ss selectionInfoSlice[S, As, A, Ag]) backward(evals AgentLeafNodeEvals[Ag]) {
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

type AgentTotalEvals[Ag comparable] map[Ag]float64

func (e AgentTotalEvals[Ag]) Add(evals AgentLeafNodeEvals[Ag]) {
	for k, v := range evals {
		e[k] += v
	}
}

func (e AgentTotalEvals[Ag]) ToAverage(s float64) AgentAverageEvals[Ag] {
	avgs := AgentAverageEvals[Ag]{}
	for k, v := range e {
		avgs[k] = v / s
	}
	return avgs
}

type AgentAverageEvals[Ag comparable] map[Ag]float64

type Engine[S any, As ~[]A, A, Ag comparable] struct {
	gameLogic         game.Logic[S, As, A, Ag]
	UCBFunc           ucb.Func
	PolicyProvider    PolicyProvider[S, As, A]
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
	e.PolicyProvider = UniformPolicyProvider[S, As, A]
}

func (e *Engine[S, As, A, Ag]) SetPlayout(players game.AgentPlayers[S, As, A, Ag]) {
	e.LeafNodeEvaluator = func(state *S) (AgentLeafNodeEvals[Ag], error) {
		final, err := e.gameLogic.Playout(*state, players)
		if err != nil {
			return AgentLeafNodeEvals[Ag]{}, err
		}
		scores, err := e.gameLogic.EvaluateAgentResultScores(&final)
		evals := make(AgentLeafNodeEvals[Ag], len(scores))
		for k, v := range scores {
			evals[k] = v
		}
		return evals, err
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

func (e *Engine[S, As, A, Ag]) SelectExpansionBackward(node *Node[S, As, A, Ag], capacity int, r *rand.Rand) (AgentLeafNodeEvals[Ag], int, error) {
	state := node.State
	selections := make(selectionInfoSlice[S, As, A, Ag], 0, capacity)
	var err error
	var isEnd bool

	for {
		action := omwrand.Choice(node.UCBManager.MaxKeys(), r)
		selections = append(selections, selectionInfo[S, As, A, Ag]{node: node, action: action})

		state, err = e.gameLogic.Transitioner(state, &action)
		if err != nil {
			return AgentLeafNodeEvals[Ag]{}, 0, err
		}

		isEnd, err = e.gameLogic.IsEnd(&state)
		if err != nil {
			return AgentLeafNodeEvals[Ag]{}, 0, err
		}

		if isEnd {
			break
		}

		nextNode, ok := node.NextNodes.FindByState(&state, e.gameLogic.Comparator)
		if !ok {
			//expansion
			nextNode, err = e.NewNode(&state)
			if err != nil {
				return AgentLeafNodeEvals[Ag]{}, 0, err
			}
			node.NextNodes = append(node.NextNodes, nextNode)
			//新しくノードを作成したら、selectを終了する
			break
		}
		//nextNodesの中に、同じstateのNodeが存在するならば、それを次のNodeとする
		node = nextNode
	}

	var evals AgentLeafNodeEvals[Ag]
	if isEnd {
		scores, err := e.gameLogic.EvaluateAgentResultScores(&state)
		if err != nil {
			return AgentLeafNodeEvals[Ag]{}, 0, err
		}
		for k, v := range scores {
			evals[k] = v
		}
	} else {
		evals, err = e.LeafNodeEvaluator(&state)
		if err != nil {
			return AgentLeafNodeEvals[Ag]{}, 0, err
		}		
	}

	selections.backward(evals)
	return evals, len(selections), err
}

func (e *Engine[S, As, A, Ag]) Search(rootNode *Node[S, As, A, Ag], simulation int, r *rand.Rand) (map[Ag]float64, error) {
	if e.NextNodesCap <= 0 {
		return AgentLeafNodeEvals[Ag]{}, fmt.Errorf("Engine.NextNodesCap > 0 でなければなりません。")
	}

	totals := AgentTotalEvals[Ag]{}
	capacity := 0
	for i := 0; i < simulation; i++ {
		evals, depth, err := e.SelectExpansionBackward(rootNode, capacity, r)
		if err != nil {
			return AgentLeafNodeEvals[Ag]{}, err
		}
		totals.Add(evals)
		capacity = depth + 1
	}

	avgs := totals.ToAverage(float64(simulation))
	return avgs, nil
}

func (e *Engine[S, As, A, Ag]) NewPlayer(simulation int, t float64, r *rand.Rand) game.Player[S, As, A] {
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
		
		action := rootNode.UCBManager.SelectKeyByTrialPercent(t, r)
		return action, nil
	}
}