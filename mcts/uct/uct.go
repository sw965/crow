package uct

import (
	"fmt"
	"math/rand"
	omwrand "github.com/sw965/omw/math/rand"
	"github.com/sw965/crow/ucb"
	game "github.com/sw965/crow/game/sequential"
)

type LeafNodeEvaluator[S any, G comparable] func(*S) (game.EvalPerAgent[G], error)

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

func (ss selectionInfoSlice[S, As, A, G]) Backward(evals game.EvalPerAgent[G]) {
	for _, s := range ss {
		node := s.node
		action := s.action
		eval := evals[node.Agent]
		node.UCBManager[action].TotalValue += float64(eval)
		node.UCBManager[action].Trial += 1
	}
}

type Engine[S any, As ~[]A, A, G comparable] struct {
	game              game.Game[S, As, A, G]
	UCBFunc           ucb.Func
	PolicyProvider    game.PolicyProvider[S, As, A]
	LeafNodeEvaluator LeafNodeEvaluator[S, G]
	NextNodesCap      int
}

func (e *Engine[S, As, A, G]) GetGame() game.Game[S, As, A, G]{
	return e.game
}

func (e *Engine[S, As, A, G]) SetGame(g game.Game[S, As, A, G]) {
	e.game = g
}

func (e *Engine[S, As, A, G]) SetUniformPolicyProvider() {
	e.PolicyProvider = game.UniformPolicyProvider[S, As, A]
}

func (e *Engine[S, As, A, G]) SetPlayout() {
	e.LeafNodeEvaluator = func(state *S) (game.EvalPerAgent[G], error) {
		final, err := e.game.Playout(*state)
		if err != nil {
			return game.EvalPerAgent[G]{}, err
		}
		scores, err := e.game.Logic.EvaluateResultScorePerAgent(&final)
		return scores.ToEval(), err
	}
}

func (e *Engine[S, As, A, G]) NewNode(state *S) (*Node[S, As, A, G], error) {
	policy := e.PolicyProvider(state, e.game.Logic.LegalActionsProvider(state))
	if len(policy) == 0 {
		return &Node[S, As, A, G]{}, fmt.Errorf("len(Policy) == 0 である為、新しくNodeを生成出来ません。")
	}

	um := ucb.Manager[As, A]{}
	for a, p := range policy {
		um[a] = &ucb.Calculator{Func: e.UCBFunc, P: p}
	}

	agent := e.game.Logic.CurrentTurnAgentGetter(state)
	nextNodes := make(Nodes[S, As, A, G], 0, e.NextNodesCap)
	return &Node[S, As, A, G]{State: *state, Agent: agent, UCBManager: um, NextNodes: nextNodes}, nil
}

func (e *Engine[S, As, A, G]) SelectExpansionBackward(node *Node[S, As, A, G], capacity int, r *rand.Rand) (int, error) {
	state := node.State
	selections := make(selectionInfoSlice[S, As, A, G], 0, capacity)
	var err error

	for {
		action := omwrand.Choice(node.UCBManager.MaxKeys(), r)
		selections = append(selections, selectionInfo[S, As, A, G]{node: node, action: action})

		state, err = e.game.Logic.Transitioner(state, &action)
		if err != nil {
			return 0, err
		}

		if isEnd := e.game.Logic.IsEnd(&state); isEnd {
			break
		}

		nextNode, ok := node.NextNodes.FindByState(&state, e.game.Logic.Comparator)
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
	selections.Backward(evals)
	return len(selections), err
}

func (e *Engine[S, As, A, G]) Search(rootNode *Node[S, As, A, G], simulation int, r *rand.Rand) error {
	if e.NextNodesCap <= 0 {
		return fmt.Errorf("Engine.NextNodesCap > 0 でなければなりません。")
	}
	depth := 0
	var err error
	for i := 0; i < simulation; i++ {
		capacity := depth + 1
		depth, err = e.SelectExpansionBackward(rootNode, capacity, r)
		if err != nil {
			return err
		}
	}
	return nil
}

func (e *Engine[S, As, A, G]) NewPlayer(simulation int, r *rand.Rand) game.Player[S, As, A] {
	return func(state *S, _ As) (A, error) {
		rootNode, err := e.NewNode(state)
		if err != nil {
			var zero A
			return zero, err
		}

		err = e.Search(rootNode, simulation, r)
		if err != nil {
			var zero A
			return zero, err
		}

		action := rootNode.UCBManager.RandMaxTrialKey(r)
		return action, nil
	}
}

func (e *Engine[S, As, A, G]) NewActorCritic(simulation int, r *rand.Rand) game.ActorCritic[S, As, A] {
	return func(state *S, _ As) (game.Policy[A], game.Eval, error) {
		rootNode, err := e.NewNode(state)
		if err != nil {
			return nil, 0.0, err			
		}

		err = e.Search(rootNode, simulation, r)
		if err != nil {
			return nil, 0.0, err
		}

		posterior := game.Policy[A](rootNode.UCBManager.TrialPercentPerKey())
		avg := rootNode.UCBManager.AverageValue()
		return posterior, game.Eval(avg), nil
	}
}

func (e *Engine[S, As, A, G]) NewSolver(simulation int, t float64, r *rand.Rand) game.Solver[S, As, A] {
	return game.Solver[S, As, A]{
		ActorCritic:e.NewActorCritic(simulation, r),
		Selector:game.NewThresholdWeightedSelector[A](t, r),
	}
}