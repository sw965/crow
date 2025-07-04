package puct

import (
	"fmt"
	"math/rand"
	"github.com/sw965/crow/pucb"
	game "github.com/sw965/crow/game/sequential"
	orand "github.com/sw965/omw/math/rand"
)

type RootNodeEvalByAgent[G comparable] map[G]float32

func (es RootNodeEvalByAgent[G]) DivScalar(s float32) {
	for k := range es {
		es[k] /= s
	}
}

type LeafNodeEvalByAgent[G comparable] map[G]float32
type LeafNodeEvaluator[S any, G comparable] func(S) (LeafNodeEvalByAgent[G], error)

type Node[S any, A, G comparable] struct {
	State      S
	Agent      G
	UCBManager pucb.Manager[A]
	NextNodes  Nodes[S, A, G]
}

func (node *Node[S, A, G]) Trial() int {
	return node.UCBManager.TotalTrial()
}

type Nodes[S any, A, G comparable] []*Node[S, A, G]

func (nodes Nodes[S, A, G]) FindByState(state S, eq game.Comparator[S]) (*Node[S, A, G], bool) {
	for _, node := range nodes {
		if eq(node.State, state) {
			return node, true
		}
	}
	return nil, false
}

type selectionInfo[S any, A, G comparable] struct {
	node   *Node[S, A, G]
	action A
}

type selectionInfoSlice[S any, A, G comparable] []selectionInfo[S, A, G]

func (ss selectionInfoSlice[S, A, G]) backward(evals LeafNodeEvalByAgent[G]) {
	for _, s := range ss {
		node := s.node
		action := s.action
		eval := evals[node.Agent]
		node.UCBManager[action].TotalValue += float32(eval)
		node.UCBManager[action].Trial += 1
	}
}

type Engine[S any, A, G comparable] struct {
	GameLogic         game.Logic[S, A, G]
	UCBFunc           pucb.Func
	PolicyProvider    game.PolicyProvider[S, A]
	LeafNodeEvaluator LeafNodeEvaluator[S, G]
	NextNodesCap      int
}

func (e *Engine[S, A, G]) SetUniformPolicyProvider() {
	e.PolicyProvider = game.UniformPolicyProvider[S, A]
}

func (e *Engine[S, A, G]) SetPlayout(popp game.PolicyProvider[S, A], rng *rand.Rand) {
	e.LeafNodeEvaluator = func(state S) (LeafNodeEvalByAgent[G], error) {
		finals, err := e.GameLogic.Playouts([]S{state}, popp, []*rand.Rand{rng})
		if err != nil {
			return nil, err
		}
		final := finals[0]

		scores, err := e.GameLogic.EvaluateResultScoreByAgent(final)
		evals := LeafNodeEvalByAgent[G]{}
		for k, v := range scores {
			evals[k] = v
		}
		return evals, err
	}
}

func (e Engine[S, A, G]) NewNode(state S) (*Node[S, A, G], error) {
	policy := e.PolicyProvider(state, e.GameLogic.LegalActionsProvider(state))
	if len(policy) == 0 {
		return &Node[S, A, G]{}, fmt.Errorf("len(Policy) == 0 である為、新しくNodeを生成出来ません。")
	}

	u := pucb.Manager[A]{}
	for a, p := range policy {
		u[a] = &pucb.Calculator{Func: e.UCBFunc, P: p}
	}

	agent := e.GameLogic.CurrentAgentGetter(state)
	nextNodes := make(Nodes[S, A, G], 0, e.NextNodesCap)
	return &Node[S, A, G]{State: state, Agent: agent, UCBManager: u, NextNodes: nextNodes}, nil
}

func (e Engine[S, A, G]) SelectExpansionBackward(node *Node[S, A, G], capacity int, rng *rand.Rand) (LeafNodeEvalByAgent[G], int, error) {
	state := node.State
	selections := make(selectionInfoSlice[S, A, G], 0, capacity)
	var err error
	var isEnd bool

	for {
		action, err := orand.Choice(node.UCBManager.MaxKeys(), rng)
		if err != nil {
			return nil, 0, err
		}

		selections = append(selections, selectionInfo[S, A, G]{node: node, action: action})

		state, err = e.GameLogic.Transitioner(state, action)
		if err != nil {
			return nil, 0, err
		}

		isEnd, err = e.GameLogic.IsEnd(state)
		if err != nil {
			return nil, 0, err
		}

		if isEnd {
			break
		}

		nextNode, ok := node.NextNodes.FindByState(state, e.GameLogic.Comparator)
		if !ok {
			// expansion
			nextNode, err = e.NewNode(state)
			if err != nil {
				return nil, 0, err
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
		scores, err := e.GameLogic.EvaluateResultScoreByAgent(state)
		if err != nil {
			return nil, 0, err
		}
		for k, v := range scores {
			evals[k] = v
		}
	} else {
		evals, err = e.LeafNodeEvaluator(state)
		if err != nil {
			return nil, 0, err
		}
	}

	selections.backward(evals)
	return evals, len(selections), err
}

func (e Engine[S, A, G]) Search(rootNode *Node[S, A, G], simulation int, rng *rand.Rand) (RootNodeEvalByAgent[G], error) {
	if e.NextNodesCap <= 0 {
		return nil, fmt.Errorf("Engine.NextNodesCap > 0 でなければなりません。")
	}

	rootEvals := RootNodeEvalByAgent[G]{}
	capacity := 0
	for i := 0; i < simulation; i++ {
		leafEvals, depth, err := e.SelectExpansionBackward(rootNode, capacity, rng)
		if err != nil {
			return nil, err
		}
		for k, v := range leafEvals {
			rootEvals[k] += v
		}
		capacity = depth + 1
	}

	rootEvals.DivScalar(float32(simulation))
	return rootEvals, nil
}