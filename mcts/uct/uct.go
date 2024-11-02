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
	gameLogic         game.Logic[S, As, A, G]
	UCBFunc           ucb.Func
	PolicyProvider    game.PolicyProvider[S, As, A]
	LeafNodeEvaluator LeafNodeEvaluator[S, G]
	NextNodesCap      int
}

func (e *Engine[S, As, A, G]) GetGameLogic() game.Logic[S, As, A, G]{
	return e.gameLogic
}

func (e *Engine[S, As, A, G]) SetGameLogic(gl game.Logic[S, As, A, G]) {
	e.gameLogic = gl
}

func (e *Engine[S, As, A, G]) SetUniformPolicyProvider() {
	e.PolicyProvider = game.UniformPolicyProvider[S, As, A]
}

func (e *Engine[S, As, A, G]) SetPlayout(players game.PlayerPerAgent[S, As, A, G]) {
	e.LeafNodeEvaluator = func(state *S) (game.EvalPerAgent[G], error) {
		final, err := e.gameLogic.Playout(players, *state)
		if err != nil {
			return game.EvalPerAgent[G]{}, err
		}
		scores, err := e.gameLogic.EvaluateResultScorePerAgent(&final)
		return scores.ToEval(), err
	}
}

func (e *Engine[S, As, A, G]) SetRandPlayout(agents []G, r *rand.Rand) {
	players := game.PlayerPerAgent[S, As, A, G]{}
	for _, agent := range agents {
		players[agent] = game.NewRandActionPlayer[S, As](r)
	}
	e.SetPlayout(players)
}

func (e *Engine[S, As, A, G]) NewNode(state *S) (*Node[S, As, A, G], error) {
	policy := e.PolicyProvider(state, e.gameLogic.LegalActionsProvider(state))
	if len(policy) == 0 {
		return &Node[S, As, A, G]{}, fmt.Errorf("len(Policy) == 0 である為、新しくNodeを生成出来ません。")
	}

	um := ucb.Manager[As, A]{}
	for a, p := range policy {
		um[a] = &ucb.Calculator{Func: e.UCBFunc, P: p}
	}

	agent := e.gameLogic.CurrentTurnAgentGetter(state)
	nextNodes := make(Nodes[S, As, A, G], 0, e.NextNodesCap)
	return &Node[S, As, A, G]{State: *state, Agent: agent, UCBManager: um, NextNodes: nextNodes}, nil
}

func (e *Engine[S, As, A, G]) SelectExpansionBackward(node *Node[S, As, A, G], r *rand.Rand, capacity int) (int, error) {
	state := node.State
	selections := make(selectionInfoSlice[S, As, A, G], 0, capacity)
	var err error

	for {
		action := omwrand.Choice(node.UCBManager.MaxKeys(), r)
		selections = append(selections, selectionInfo[S, As, A, G]{node: node, action: action})

		state, err = e.gameLogic.Transitioner(state, &action)
		if err != nil {
			return 0, err
		}

		if isEnd := e.gameLogic.IsEnd(&state); isEnd {
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
	evals, err := e.LeafNodeEvaluator(&state)
	selections.Backward(evals)
	return len(selections), err
}

func (e *Engine[S, As, A, G]) Search(simulation int, rootNode *Node[S, As, A, G], r *rand.Rand) error {
	if e.NextNodesCap <= 0 {
		return fmt.Errorf("Engine.NextNodesCap > 0 でなければなりません。")
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

func (e *Engine[S, As, A, G]) NewPlayer(simulation int, r *rand.Rand) game.Player[S, As, A] {
	return func(state *S, _ As) (A, error) {
		rootNode, err := e.NewNode(state)
		if err != nil {
			var zero A
			return zero, err
		}

		err = e.Search(simulation, rootNode, r)
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

		err = e.Search(simulation, rootNode, r)
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