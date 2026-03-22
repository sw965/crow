// https://www.terry-u16.net/entry/decoupled-uct

package dpuct

import (
	"errors"
	"fmt"
	"maps"
	"math/rand/v2"
	"slices"
	"sync"

	"github.com/sw965/crow/game"
	"github.com/sw965/crow/game/simultaneous"
	"github.com/sw965/crow/pucb"
	"github.com/sw965/omw/parallel"
)

// TODO 後で削除を検討
var (
	ErrNilEngineFunc = errors.New("dpuct.Engineエラー: フィールドの関数がnilです")
	ErrInvalidConfig = errors.New("dpuct.Engineエラー: 設定値が不正です")
)

type RootNodeEvalByAgent[Ag comparable] map[Ag]float32

func (es RootNodeEvalByAgent[Ag]) DivScalar(s float32) {
	for k := range es {
		es[k] /= s
	}
}

type LeafNodeEvalByAgent[Ag comparable] map[Ag]float32
type LeafNodeEvalByAgentFunc[S any, Ag comparable] func(S) (LeafNodeEvalByAgent[Ag], error)

type PolicyFunc[S any, Ac, Ag comparable] func(S, simultaneous.LegalActionsByAgent[Ac, Ag]) (simultaneous.PolicyByAgent[Ac, Ag], error)

type Node[S any, Ac, Ag comparable] struct {
	State            S
	virtualSelectors map[Ag]pucb.VirtualSelector[Ac]
	nextNodes        Nodes[S, Ac, Ag]
	sync.Mutex
}

func (n *Node[S, Ac, Ag]) VirtualSelectors() map[Ag]pucb.VirtualSelector[Ac] {
	cloned := make(map[Ag]pucb.VirtualSelector[Ac], len(n.virtualSelectors))
	for agent, vs := range n.virtualSelectors {
		cloned[agent] = maps.Clone(vs)
	}
	return cloned
}

type Nodes[S any, Ac, Ag comparable] []*Node[S, Ac, Ag]

func (nodes Nodes[S, Ac, Ag]) FindByState(state S, eq simultaneous.EqualFunc[S]) (*Node[S, Ac, Ag], bool) {
	for _, node := range nodes {
		if eq(node.State, state) {
			return node, true
		}
	}
	return nil, false
}

type selectBuffer[S any, Ac, Ag comparable] struct {
	node        *Node[S, Ac, Ag]
	actionByAgent map[Ag]Ac
}

type selectBuffers[S any, Ac, Ag comparable] []selectBuffer[S, Ac, Ag]

func (ss selectBuffers[S, Ac, Ag]) backward(evals LeafNodeEvalByAgent[Ag]) {
	for _, s := range ss {
		node := s.node
		actionByAgent := s.actionByAgent

		node.Lock()
		for agent, action := range actionByAgent {
			eval, ok := evals[agent]
			if !ok {
				msg := fmt.Sprintf(
					"BUG: LeafNodeEvalByAgentに存在しないキー(Agent)でアクセスしようとした為、backwardを実行出来ませんでした。Agent = %v, LeafNodeEvalByAgent.Keys() = %v",
					agent, slices.Collect(maps.Keys(evals)),
				)
				panic(msg)
			}
			
			// 未観測のカウントを消し、実観測データを反映する
			node.virtualSelectors[agent][action].DecrementO()
			node.virtualSelectors[agent][action].AddW(eval)
			node.virtualSelectors[agent][action].IncrementVisits()
		}
		node.Unlock()
	}
}

func (ss selectBuffers[S, Ac, Ag]) rollbackO() {
	for _, s := range ss {
		s.node.Lock()
		for agent, action := range s.actionByAgent {
			s.node.virtualSelectors[agent][action].DecrementO()
		}
		s.node.Unlock()
	}
}

type Engine[S any, Ac, Ag comparable] struct {
	Game                    simultaneous.Engine[S, Ac, Ag]
	PUCBFunc                pucb.Func
	PolicyFunc              PolicyFunc[S, Ac, Ag]
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
	e.PolicyFunc = func(state S, legalActionsByAgent simultaneous.LegalActionsByAgent[Ac, Ag]) (simultaneous.PolicyByAgent[Ac, Ag], error) {
		p, _, err := simultaneous.UniformPolicyNoValueFunc[S, Ac, Ag](state, legalActionsByAgent)
		return p, err
	}
}

func (e *Engine[S, Ac, Ag]) SetPlayout(accr simultaneous.ActorCritic[S, Ac, Ag], rng *rand.Rand) {
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
	legalActionsByAgent := e.Game.Logic.LegalActionsByAgentFunc(state)
	if len(legalActionsByAgent) == 0 {
		return nil, fmt.Errorf("ゲームが終了していないのに合法手がありません")
	}

	policyByAgent, err := e.PolicyFunc(state, legalActionsByAgent)
	if err != nil {
		return nil, err
	}

	selectors := make(map[Ag]pucb.VirtualSelector[Ac], len(e.Game.Agents))

	for _, agent := range e.Game.Agents {
		legalActions := legalActionsByAgent[agent]
		policy, ok := policyByAgent[agent]
		if !ok {
			return nil, fmt.Errorf("エージェント %v の Policy が見つかりません", agent)
		}

		if err := policy.ValidateForLegalActions(legalActions, true); err != nil {
			return nil, err
		}

		s := pucb.VirtualSelector[Ac]{}
		for _, action := range legalActions {
			p := policy[action]
			s[action] = &pucb.Calculator{Func: e.PUCBFunc, P: p, VirtualValue: e.VirtualValue}
		}
		selectors[agent] = s
	}

	return &Node[S, Ac, Ag]{
		State:            state,
		virtualSelectors: selectors,
		nextNodes:        make(Nodes[S, Ac, Ag], 0, e.NextNodesCap),
	}, nil
}

func (e Engine[S, Ac, Ag]) SelectExpansionBackward(node *Node[S, Ac, Ag], capacity int, rng *rand.Rand) (LeafNodeEvalByAgent[Ag], int, error) {
	state := node.State
	buffers := make(selectBuffers[S, Ac, Ag], 0, capacity)
	var err error
	var isEnd bool

	// 途中でエラーが起きた場合、 o を元に戻す
	defer func() {
		if err != nil {
			buffers.rollbackO()
		}
	}()

	for {
		node.Lock()
		actionByAgent := make(map[Ag]Ac, len(e.Game.Agents))
		for _, agent := range e.Game.Agents {
			vs := node.virtualSelectors[agent]
			action, errSelect := vs.Select(rng)
			if errSelect != nil {
				node.Unlock()
				err = errSelect
				return nil, 0, err
			}
			// 選択した行動の未観測カウントをインクリメント
			vs[action].IncrementO()
			actionByAgent[agent] = action
		}
		node.Unlock()

		buffers = append(buffers, selectBuffer[S, Ac, Ag]{node: node, actionByAgent: actionByAgent})

		state, err = e.Game.Logic.ActionFunc(state, actionByAgent)
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

		node.Lock()
		nextNode, ok := node.nextNodes.FindByState(state, e.Game.Logic.EqualFunc)
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

			node.Lock()
			// 生成中に他のスレッドが追加した可能性があるため再度確認
			if nn, ok := node.nextNodes.FindByState(state, e.Game.Logic.EqualFunc); ok {
				nextNode = nn
				expand = false
			} else {
				node.nextNodes = append(node.nextNodes, newNode)
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

func (e Engine[S, Ac, Ag]) NewPolicyNoValueFunc(simulations int, rngs []*rand.Rand) simultaneous.PolicyValueFunc[S, Ac, Ag] {
	return func(state S, legalActionsByAgent simultaneous.LegalActionsByAgent[Ac, Ag]) (simultaneous.PolicyByAgent[Ac, Ag], simultaneous.ValueByAgent[Ag], error) {
		rootNode, err := e.NewNode(state)
		if err != nil {
			return nil, nil, err
		}

		_, err = e.Search(rootNode, simulations, rngs)
		if err != nil {
			return nil, nil, err
		}

		policyByAgent := make(simultaneous.PolicyByAgent[Ac, Ag], len(e.Game.Agents))
		valueByAgent := make(simultaneous.ValueByAgent[Ag], len(e.Game.Agents))
		vSelectors := rootNode.VirtualSelectors()

		for _, agent := range e.Game.Agents {
			visitPercents := vSelectors[agent].VisitPercentByKey()
			policy := game.Policy[Ac]{}
			for _, action := range legalActionsByAgent[agent] {
				if p, ok := visitPercents[action]; !ok {
					return nil, nil, fmt.Errorf("エラー: actionの観測確率が存在しません")
				} else {
					policy[action] = p
				}
			}
			policyByAgent[agent] = policy
			valueByAgent[agent] = 0.0
		}
		return policyByAgent, valueByAgent, nil
	}
}

func (e Engine[S, Ac, Ag]) NewPolicyValueFunc(simulations int, rngs []*rand.Rand) simultaneous.PolicyValueFunc[S, Ac, Ag] {
	return func(state S, legalActionsByAgent simultaneous.LegalActionsByAgent[Ac, Ag]) (simultaneous.PolicyByAgent[Ac, Ag], simultaneous.ValueByAgent[Ag], error) {
		rootNode, err := e.NewNode(state)
		if err != nil {
			return nil, nil, err
		}

		evals, err := e.Search(rootNode, simulations, rngs)
		if err != nil {
			return nil, nil, err
		}

		policyByAgent := make(simultaneous.PolicyByAgent[Ac, Ag], len(e.Game.Agents))
		valueByAgent := make(simultaneous.ValueByAgent[Ag], len(e.Game.Agents))
		vSelectors := rootNode.VirtualSelectors()

		for _, agent := range e.Game.Agents {
			visitPercents := vSelectors[agent].VisitPercentByKey()
			policy := game.Policy[Ac]{}
			for _, action := range legalActionsByAgent[agent] {
				if p, ok := visitPercents[action]; !ok {
					return nil, nil, fmt.Errorf("エラー: actionの観測確率が存在しません")
				} else {
					policy[action] = p
				}
			}
			policyByAgent[agent] = policy

			eval, ok := evals[agent]
			if !ok {
				return nil, nil, fmt.Errorf("エラー: エージェントの評価値が存在しません")
			}
			valueByAgent[agent] = eval
		}
		return policyByAgent, valueByAgent, nil
	}
}