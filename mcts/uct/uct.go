package uct

import (
	"fmt"
	"github.com/sw965/crow/game/sequential"
	"github.com/sw965/crow/ucb"
	omwrand "github.com/sw965/omw/math/rand"
	"math/rand"
)

type ActionPolicy[A comparable] map[A]float64
type ActionPolicyProvider[S any, As ~[]A, A comparable] func(*S, As) ActionPolicy[A]

type AgentLeafNodeEvals[Agent comparable] map[Agent]float64
type AgentLeafNodeEvaluator[S any, Agent comparable] func(*S) (AgentLeafNodeEvals[Agent], error)

type Node[S any, As ~[]A, A, Agent comparable] struct {
	State      S
	Agent      Agent
	UCBManager ucb.Manager[As, A]
	NextNodes  Nodes[S, As, A, Agent]
}

func (node *Node[S, As, A, Agent]) Trial() int {
	return node.UCBManager.TotalTrial()
}

type Nodes[S any, As ~[]A, A, Agent comparable] []*Node[S, As, A, Agent]

func (nodes Nodes[S, As, A, Agent]) FindByState(state *S, eq sequential.Comparator[S]) (*Node[S, As, A, Agent], bool) {
	for _, node := range nodes {
		if eq(&node.State, state) {
			return node, true
		}
	}
	return nil, false
}

type selectionInfo[S any, As ~[]A, A, Agent comparable] struct {
	node   *Node[S, As, A, Agent]
	action A
}

type selectionInfoSlice[S any, As ~[]A, A, Agent comparable] []selectionInfo[S, As, A, Agent]

func (ss selectionInfoSlice[S, As, A, Agent]) Backward(evals AgentLeafNodeEvals[Agent]) {
	for _, s := range ss {
		node := s.node
		action := s.action
		v := evals[node.Agent]
		node.UCBManager[action].TotalValue += float64(v)
		node.UCBManager[action].Trial += 1
	}
}

type MCTS[S any, As ~[]A, A, Agent comparable] struct {
	GameLogic                 sequential.Logic[S, As, A, Agent]
	UCBFunc                   ucb.Func
	ActionPolicyProvider      ActionPolicyProvider[S, As, A]
	AgentLeafNodeEvaluator    AgentLeafNodeEvaluator[S, Agent]
	NextNodesCap              int
}

func (mcts *MCTS[S, As, A, Agent]) NewNode(state *S) (*Node[S, As, A, Agent], error) {
	policy := mcts.ActionPolicyProvider(state, mcts.GameLogic.LegalActionsProvider(state))
	if len(policy) == 0 {
		return &Node[S, As, A, Agent]{}, fmt.Errorf("len(ActionPolicy) == 0 である為、新しくNodeを生成出来ません。")
	}

	m := ucb.Manager[As, A]{}
	for a, p := range policy {
		m[a] = &ucb.Calculator{Func: mcts.UCBFunc, P: p}
	}

	agent := mcts.GameLogic.CurrentTurnAgentGetter(state)
	nextNodes := make(Nodes[S, As, A, Agent], 0, mcts.NextNodesCap)
	return &Node[S, As, A, Agent]{State: *state, Agent: agent, UCBManager: m, NextNodes: nextNodes}, nil
}

func (mcts *MCTS[S, As, A, Agent]) SetUniformActionPolicyProvider() {
	mcts.ActionPolicyProvider = func(state *S, legalActions As) ActionPolicy[A] {
		n := len(legalActions)
		p := 1.0 / float64(n)
		policy := ActionPolicy[A]{}
		for _, a := range legalActions {
			policy[a] = p
		}
		return policy
	}
}

func (mcts *MCTS[S, As, A, Agent]) SetPlayout(players sequential.AgentPlayers[S, A, Agent]) {
	mcts.AgentLeafNodeEvaluator = func(sp *S) (AgentLeafNodeEvals[Agent], error) {
		s, err := mcts.GameLogic.Playout(players, *sp)
		if err != nil {
			return AgentLeafNodeEvals[Agent]{}, err
		}
		scores, err := mcts.GameLogic.EvaluateAgentResultScores(&s)
		return AgentLeafNodeEvals[Agent](scores), err
	}
}

func (mcts *MCTS[S, As, A, Agent]) SetRandPlayout(agents []Agent, r *rand.Rand) {
	players := sequential.AgentPlayers[S, A, Agent]{}
	for _, agent := range agents {
		players[agent] = mcts.GameLogic.NewRandActionPlayer(r)
	}
	mcts.SetPlayout(players)
}

func (mcts *MCTS[S, As, A, Agent]) SelectExpansionBackward(node *Node[S, As, A, Agent], r *rand.Rand, capacity int) (int, error) {
	state := node.State
	selections := make(selectionInfoSlice[S, As, A, Agent], 0, capacity)
	var err error
	for {
		action := omwrand.Choice(node.UCBManager.MaxKeys(), r)
		selections = append(selections, selectionInfo[S, As, A, Agent]{node: node, action: action})

		state, err = mcts.GameLogic.Transitioner(state, &action)
		if err != nil {
			return 0, err
		}

		if isEnd := mcts.GameLogic.IsEnd(&state); isEnd {
			break
		}

		nextNode, ok := node.NextNodes.FindByState(&state, mcts.GameLogic.Comparator)
		if !ok {
			//expansion
			nextNode, err = mcts.NewNode(&state)
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
	eval, err := mcts.AgentLeafNodeEvaluator(&state)
	selections.Backward(eval)
	return len(selections), err
}

func (mcts *MCTS[S, As, A, Agent]) Run(simulation int, rootNode *Node[S, As, A, Agent], r *rand.Rand) error {
	if mcts.NextNodesCap <= 0 {
		return fmt.Errorf("MCTS.NextNodesCap > 0 でなければなりません。")
	}
	depth := 0
	var err error
	for i := 0; i < simulation; i++ {
		capacity := depth + 1
		depth, err = mcts.SelectExpansionBackward(rootNode, r, capacity)
		if err != nil {
			return err
		}
	}
	return nil
}