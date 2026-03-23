package puct

import (
	"fmt"
	"math/rand/v2"

	"errors"
	"github.com/sw965/crow/game"
	"github.com/sw965/crow/game/sequential"
	"github.com/sw965/crow/pucb"
	"github.com/sw965/omw/parallel"
	"maps"
	"slices"
	"sync"
)

var (
	ErrNilEngineFunc = errors.New("puct.Engineエラー: フィールドの関数がnilです")
	ErrInvalidConfig = errors.New("puct.Engineエラー: 設定値が不正です")
)

// TODO このモンテカルロ木探索は、ユーザー側のゲーム設定次第では、無限ループするので直しておく。

type RootNodeEvalByAgent[Ag comparable] map[Ag]float32

func (es RootNodeEvalByAgent[Ag]) DivScalar(s float32) {
	for k := range es {
		es[k] /= s
	}
}

type LeafNodeEvalByAgent[Ag comparable] map[Ag]float32
type LeafNodeEvalByAgentFunc[S any, Ag comparable] func(S) (LeafNodeEvalByAgent[Ag], error)

type Node[S any, Ac, Ag comparable] struct {
	State             S
	Agent             Ag
	virtualSelector   pucb.VirtualSelector[Ac]
	nextNodesByAction map[Ac]Nodes[S, Ac, Ag]
	sync.Mutex
}

func (n *Node[S, Ac, Ag]) VirtualSelector() pucb.VirtualSelector[Ac] {
	return maps.Clone(n.virtualSelector)
}

type Nodes[S any, Ac, Ag comparable] []*Node[S, Ac, Ag]

func (nodes Nodes[S, Ac, Ag]) FindByState(state S, eq sequential.EqualFunc[S]) (*Node[S, Ac, Ag], bool) {
	for _, node := range nodes {
		if eq(node.State, state) {
			return node, true
		}
	}
	return nil, false
}

type selectBuffer[S any, Ac, Ag comparable] struct {
	node   *Node[S, Ac, Ag]
	action Ac
}

type selectBuffers[S any, Ac, Ag comparable] []selectBuffer[S, Ac, Ag]

func (ss selectBuffers[S, Ac, Ag]) backward(evals LeafNodeEvalByAgent[Ag]) {
	for _, s := range ss {
		node := s.node
		action := s.action
		eval, ok := evals[node.Agent]
		if !ok {
			msg := fmt.Sprintf(
				"BUG: LeafNodeEvalByAgentに存在しないキー(Agent)でアクセスしようとした為、backwardを実行出来ませんでした。leafNode.Agent = %v, LeafNodeEvalByAgent.Keys() = %v",
				node.Agent, slices.Collect(maps.Keys(evals)),
			)
			panic(msg)
		}

		node.Lock()
		// 未観測のカウントを消す
		node.virtualSelector[action].DecrementO()
		node.virtualSelector[action].AddW(eval)
		node.virtualSelector[action].IncrementVisits()
		node.Unlock()
	}
}

func (ss selectBuffers[S, Ac, Ag]) rollbackO() {
	for _, s := range ss {
		s.node.Lock()
		s.node.virtualSelector[s.action].DecrementO()
		s.node.Unlock()
	}
}

type Engine[S any, Ac, Ag comparable] struct {
	Game                    sequential.Engine[S, Ac, Ag]
	PUCBFunc                pucb.Func
	PolicyFunc              sequential.PolicyFunc[S, Ac]
	LeafNodeEvalByAgentFunc LeafNodeEvalByAgentFunc[S, Ag]
	NextNodesCap            int
	VirtualValue            float32
}

func (e Engine[S, Ac, Ag]) Validate() error {
	if err := e.Game.Validate(); err != nil {
		return err
	}

	if e.PUCBFunc == nil {
		return fmt.Errorf("%w: PUCBFunc", ErrNilEngineFunc)
	}

	if e.PolicyFunc == nil {
		return fmt.Errorf("%w: PolicyFunc", ErrNilEngineFunc)
	}

	if e.LeafNodeEvalByAgentFunc == nil {
		return fmt.Errorf("%w: LeafNodeEvalByAgentFunc", ErrNilEngineFunc)
	}

	if e.NextNodesCap <= 0 {
		return fmt.Errorf("%w: NextNodesCap=%d(0より大きい必要があります)", ErrInvalidConfig, e.NextNodesCap)
	}
	return nil
}

func (e *Engine[S, Ac, Ag]) SetUniformPolicyFunc() {
	e.PolicyFunc = sequential.UniformPolicyFunc[S, Ac]
}

func (e *Engine[S, Ac, Ag]) SetPlayout(accr sequential.ActorCritic[S, Ac, Ag], rng *rand.Rand) {
	e.LeafNodeEvalByAgentFunc = func(state S) (LeafNodeEvalByAgent[Ag], error) {
		finals, err := e.Game.Playouts([]S{state}, accr, []*rand.Rand{rng})
		if err != nil {
			return nil, err
		}
		final := finals[0]

		scores, err := e.Game.EvaluateResultScoreByAgent(final)
		if err != nil {
			return nil, err
		}

		evals := LeafNodeEvalByAgent[Ag]{}
		for k, v := range scores {
			evals[k] = v
		}
		return evals, nil
	}
}

func (e Engine[S, Ac, Ag]) NewNode(state S) (*Node[S, Ac, Ag], error) {
	legalActions := e.Game.Logic.LegalActionsFunc(state)

	// policy.ValidateForLegalActionsでもlegalActionsの空チェックはするが、PolicyFuncを安全に呼ぶ為に、ここでもチェックする
	if len(legalActions) == 0 {
		return nil, fmt.Errorf("後でエラーメッセージを書く")
	}

	policy, err := e.PolicyFunc(state, legalActions)
	if err != nil {
		return nil, err
	}

	err = policy.ValidateForLegalActions(legalActions, true)
	if err != nil {
		return nil, err
	}

	s := pucb.VirtualSelector[Ac]{}
	for _, action := range legalActions {
		p := policy[action]
		s[action] = &pucb.Calculator{Func: e.PUCBFunc, P: p, VirtualValue: e.VirtualValue}
	}

	agent := e.Game.Logic.CurrentAgentFunc(state)

	found := false
	for _, ag := range e.Game.Agents {
		if ag == agent {
			found = true
			break
		}
	}

	if !found {
		return nil, fmt.Errorf("後でエラーメッセージを書く")
	}

	return &Node[S, Ac, Ag]{
		State:             state,
		Agent:             agent,
		virtualSelector:   s,
		nextNodesByAction: make(map[Ac]Nodes[S, Ac, Ag], e.NextNodesCap),
	}, nil
}

func (e Engine[S, Ac, Ag]) SelectExpansionBackward(node *Node[S, Ac, Ag], capacity int, rng *rand.Rand) (LeafNodeEvalByAgent[Ag], int, error) {
	state := node.State
	buffers := make(selectBuffers[S, Ac, Ag], 0, capacity)
	var err error
	var isEnd bool

	// 途中でエラーが起きた場合、 o を元に戻す（成功時は backward が o--するので不要）
	defer func() {
		if err != nil {
			buffers.rollbackO()
		}
	}()

	for {
		node.Lock()

		var action Ac
		action, err := node.virtualSelector.Select(rng)
		if err != nil {
			node.Unlock()
			return nil, 0, err
		}
		// 選択した行動のノードの未観測の数をインクリメントする
		node.virtualSelector[action].IncrementO()

		node.Unlock()
		buffers = append(buffers, selectBuffer[S, Ac, Ag]{node: node, action: action})

		state, err = e.Game.Logic.TransitionFunc(state, action)
		if err != nil {
			return nil, 0, err
		}

		isEnd, err = e.Game.IsEnd(state)
		if err != nil {
			return nil, 0, err
		}

		if isEnd {
			break
		}

		var expand bool

		// node.nextNodesByActionはmap型 node.nextNodesByAction[action]はslice型
		// この処理はデータを読むだけだが、他のワーカーが、書き込む処理をすると、破綻する為、Lockが必要
		node.Lock()
		nextNode, ok := node.nextNodesByAction[action].FindByState(state, e.Game.Logic.EqualFunc)
		node.Unlock()

		if ok {
			node = nextNode
			expand = false
		} else {
			var newNode *Node[S, Ac, Ag]
			newNode, err = e.NewNode(state)
			if err != nil {
				return nil, 0, err
			}

			// Unlockして NewNodeを作ってる間に、別のワーカーがノードを追加した可能性がある為、再度Lockして調べる
			node.Lock()
			// nextNodesの中に、一致するstateが見つかれば、それを次のノードとする
			if nn, ok := node.nextNodesByAction[action].FindByState(state, e.Game.Logic.EqualFunc); ok {
				nextNode = nn
				expand = false
				// nextNodesの中に、一致するstateが見つからなければ、newNodeをnextNodesに追加し、selectを終了する
			} else {
				node.nextNodesByAction[action] = append(node.nextNodesByAction[action], newNode)
				expand = true
			}
			node.Unlock()
		}

		if expand {
			break
		}
		node = nextNode
	}

	evals := LeafNodeEvalByAgent[Ag]{}
	// ゲームが終了した場合、ゲームエンジンの結果スコアを、リーフノードの評価値とする
	// ゲームが終了していなかった場合、リーフノードの評価関数を呼び出す
	if isEnd {
		var scores game.ResultScoreByAgent[Ag]
		scores, err = e.Game.EvaluateResultScoreByAgent(state)
		if err != nil {
			return nil, 0, err
		}
		for k, v := range scores {
			evals[k] = v
		}
	} else {
		evals, err = e.LeafNodeEvalByAgentFunc(state)
		if err != nil {
			return nil, 0, err
		}
	}

	buffers.backward(evals)
	return evals, len(buffers), err
}

func (e Engine[S, Ac, Ag]) Search(rootNode *Node[S, Ac, Ag], n int, workerRngs []*rand.Rand) (RootNodeEvalByAgent[Ag], error) {
	if err := e.Validate(); err != nil {
		return nil, err
	}

	if rootNode == nil {
		return nil, fmt.Errorf("rootNode が nil です")
	}

	if n <= 0 {
		return nil, fmt.Errorf("シミュレーション数は0より大きい必要があります。")
	}

	p := len(workerRngs)
	rootEvalsPerWorker := make([]RootNodeEvalByAgent[Ag], p)
	for i := range p {
		rootEvalsPerWorker[i] = RootNodeEvalByAgent[Ag]{}
	}

	workerBuffCaps := make([]int, p)
	err := parallel.For(n, p, func(workerId, idx int) error {
		rng := workerRngs[workerId]
		leafEvals, depth, err := e.SelectExpansionBackward(rootNode, workerBuffCaps[workerId], rng)
		if err != nil {
			return err
		}

		for k, v := range leafEvals {
			rootEvalsPerWorker[workerId][k] += v
		}

		workerBuffCaps[workerId] = depth + 1
		return nil
	})

	if err != nil {
		return nil, err
	}

	sum := RootNodeEvalByAgent[Ag]{}
	for i := range rootEvalsPerWorker {
		for k, v := range rootEvalsPerWorker[i] {
			sum[k] += v
		}
	}

	sum.DivScalar(float32(n))
	return sum, nil
}

func (e Engine[S, Ac, Ag]) NewPolicyNoValueFunc(simulations int, rngs []*rand.Rand) sequential.PolicyValueFunc[S, Ac] {
	return func(state S, legalActions []Ac) (game.Policy[Ac], float32, error) {
		rootNode, err := e.NewNode(state)
		if err != nil {
			return nil, 0.0, err
		}

		_, err = e.Search(rootNode, simulations, rngs)
		if err != nil {
			return nil, 0.0, err
		}

		visitPercents := rootNode.VirtualSelector().VisitPercentByKey()
		policy := game.Policy[Ac]{}
		for _, action := range legalActions {
			if p, ok := visitPercents[action]; !ok {
				return nil, 0.0, fmt.Errorf("後でエラーメッセージを書く")
			} else {
				policy[action] = p
			}
		}
		return policy, 0.0, nil
	}
}

func (e Engine[S, Ac, Ag]) NewPolicyValueFunc(simulations int, rngs []*rand.Rand) sequential.PolicyValueFunc[S, Ac] {
	return func(state S, legalActions []Ac) (game.Policy[Ac], float32, error) {
		rootNode, err := e.NewNode(state)
		if err != nil {
			return nil, 0.0, err
		}

		evals, err := e.Search(rootNode, simulations, rngs)
		if err != nil {
			return nil, 0.0, err
		}

		visitPercents := rootNode.VirtualSelector().VisitPercentByKey()
		policy := game.Policy[Ac]{}
		for _, action := range legalActions {
			if p, ok := visitPercents[action]; !ok {
				return nil, 0.0, fmt.Errorf("後でエラーメッセージを書く")
			} else {
				policy[action] = p
			}
		}

		eval, ok := evals[rootNode.Agent]
		if !ok {
			return nil, 0.0, fmt.Errorf("テスト失敗")
		}
		return policy, eval, nil
	}
}
