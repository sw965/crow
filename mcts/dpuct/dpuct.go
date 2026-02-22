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

var (
	ErrNilEngineFunc = errors.New("dpuct.Engineエラー: フィールドの関数がnilです")
	ErrInvalidConfig = errors.New("dpuct.Engineエラー: 設定値が不正です")
)

type RootNodeEvalByAgent[A comparable] map[A]float32

func (es RootNodeEvalByAgent[A]) DivScalar(s float32) {
	for k := range es {
		es[k] /= s
	}
}

type LeafNodeEvalByAgent[A comparable] map[A]float32
type LeafNodeEvalByAgentFunc[S any, A comparable] func(S) (LeafNodeEvalByAgent[A], error)

type PolicyFunc[S any, M, A comparable] func(S, simultaneous.LegalMovesByAgent[M, A]) (simultaneous.PolicyByAgent[M, A], error)

type Node[S any, M, A comparable] struct {
	State            S
	virtualSelectors map[A]pucb.VirtualSelector[M]
	nextNodes        Nodes[S, M, A]
	sync.Mutex
}

func (n *Node[S, M, A]) VirtualSelectors() map[A]pucb.VirtualSelector[M] {
	cloned := make(map[A]pucb.VirtualSelector[M], len(n.virtualSelectors))
	for agent, vs := range n.virtualSelectors {
		cloned[agent] = maps.Clone(vs)
	}
	return cloned
}

type Nodes[S any, M, A comparable] []*Node[S, M, A]

func (nodes Nodes[S, M, A]) FindByState(state S, eq simultaneous.EqualFunc[S]) (*Node[S, M, A], bool) {
	for _, node := range nodes {
		if eq(node.State, state) {
			return node, true
		}
	}
	return nil, false
}

type selectBuffer[S any, M, A comparable] struct {
	node        *Node[S, M, A]
	moveByAgent map[A]M
}

type selectBuffers[S any, M, A comparable] []selectBuffer[S, M, A]

func (ss selectBuffers[S, M, A]) backward(evals LeafNodeEvalByAgent[A]) {
	for _, s := range ss {
		node := s.node
		moveByAgent := s.moveByAgent

		node.Lock()
		for agent, move := range moveByAgent {
			eval, ok := evals[agent]
			if !ok {
				msg := fmt.Sprintf(
					"BUG: LeafNodeEvalByAgentに存在しないキー(Agent)でアクセスしようとした為、backwardを実行出来ませんでした。Agent = %v, leftNodeEvalByAgent.Keys() = %v",
					agent, slices.Collect(maps.Keys(evals)),
				)
				panic(msg)
			}
			
			// 未観測のカウントを消し、実観測データを反映する
			node.virtualSelectors[agent][move].DecrementO()
			node.virtualSelectors[agent][move].AddW(eval)
			node.virtualSelectors[agent][move].IncrementVisits()
		}
		node.Unlock()
	}
}

func (ss selectBuffers[S, M, A]) rollbackO() {
	for _, s := range ss {
		s.node.Lock()
		for agent, move := range s.moveByAgent {
			s.node.virtualSelectors[agent][move].DecrementO()
		}
		s.node.Unlock()
	}
}

type Engine[S any, M, A comparable] struct {
	Game                    simultaneous.Engine[S, M, A]
	PUCBFunc                pucb.Func
	PolicyFunc              PolicyFunc[S, M, A]
	LeafNodeEvalByAgentFunc LeafNodeEvalByAgentFunc[S, A]
	NextNodesCap            int
	VirtualValue            float32
}

func (e Engine[S, M, A]) Validate() error {
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

func (e *Engine[S, M, A]) SetUniformPolicyFunc() {
	e.PolicyFunc = func(state S, legalMovesByAgent simultaneous.LegalMovesByAgent[M, A]) (simultaneous.PolicyByAgent[M, A], error) {
		p, _, err := simultaneous.UniformPolicyNoValueFunc[S, M, A](state, legalMovesByAgent)
		return p, err
	}
}

func (e *Engine[S, M, A]) SetPlayout(ac simultaneous.ActorCritic[S, M, A], rng *rand.Rand) {
	e.LeafNodeEvalByAgentFunc = func(state S) (LeafNodeEvalByAgent[A], error) {
		finals, err := e.Game.Playouts([]S{state}, ac, []*rand.Rand{rng})
		if err != nil {
			return nil, err
		}
		final := finals[0]

		scores, err := e.Game.EvaluateResultScoreByAgent(final)
		if err != nil {
			return nil, err
		}

		evals := LeafNodeEvalByAgent[A]{}
		for k, v := range scores {
			evals[k] = v
		}
		return evals, nil
	}
}

func (e Engine[S, M, A]) NewNode(state S) (*Node[S, M, A], error) {
	legalMovesByAgent := e.Game.Logic.LegalMovesByAgentFunc(state)
	if len(legalMovesByAgent) == 0 {
		return nil, fmt.Errorf("ゲームが終了していないのに合法手がありません")
	}

	policyByAgent, err := e.PolicyFunc(state, legalMovesByAgent)
	if err != nil {
		return nil, err
	}

	selectors := make(map[A]pucb.VirtualSelector[M], len(e.Game.Agents))

	for _, agent := range e.Game.Agents {
		legalMoves := legalMovesByAgent[agent]
		policy, ok := policyByAgent[agent]
		if !ok {
			return nil, fmt.Errorf("エージェント %v の Policy が見つかりません", agent)
		}

		if err := policy.ValidateForLegalMoves(legalMoves, true); err != nil {
			return nil, err
		}

		s := pucb.VirtualSelector[M]{}
		for _, move := range legalMoves {
			p := policy[move]
			s[move] = &pucb.Calculator{Func: e.PUCBFunc, P: p, VirtualValue: e.VirtualValue}
		}
		selectors[agent] = s
	}

	return &Node[S, M, A]{
		State:            state,
		virtualSelectors: selectors,
		nextNodes:        make(Nodes[S, M, A], 0, e.NextNodesCap),
	}, nil
}

func (e Engine[S, M, A]) SelectExpansionBackward(node *Node[S, M, A], capacity int, rng *rand.Rand) (LeafNodeEvalByAgent[A], int, error) {
	state := node.State
	buffers := make(selectBuffers[S, M, A], 0, capacity)
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
		moveByAgent := make(map[A]M, len(e.Game.Agents))
		for _, agent := range e.Game.Agents {
			vs := node.virtualSelectors[agent]
			move, errSelect := vs.Select(rng)
			if errSelect != nil {
				node.Unlock()
				err = errSelect
				return nil, 0, err
			}
			// 選択した行動の未観測カウントをインクリメント
			vs[move].IncrementO()
			moveByAgent[agent] = move
		}
		node.Unlock()

		buffers = append(buffers, selectBuffer[S, M, A]{node: node, moveByAgent: moveByAgent})

		state, err = e.Game.Logic.MoveFunc(state, moveByAgent)
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
			var newNode *Node[S, M, A]
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

	evals := LeafNodeEvalByAgent[A]{}
	if isEnd {
		var scores game.ResultScoreByAgent[A]
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

func (e Engine[S, M, A]) Search(rootNode *Node[S, M, A], n int, workerRngs []*rand.Rand) (RootNodeEvalByAgent[A], error) {
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
	rootEvalsPerWorker := make([]RootNodeEvalByAgent[A], p)
	for i := range p {
		rootEvalsPerWorker[i] = RootNodeEvalByAgent[A]{}
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

	sum := RootNodeEvalByAgent[A]{}
	for i := range rootEvalsPerWorker {
		for k, v := range rootEvalsPerWorker[i] {
			sum[k] += v
		}
	}

	sum.DivScalar(float32(n))
	return sum, nil
}

func (e Engine[S, M, A]) NewPolicyNoValueFunc(simulations int, rngs []*rand.Rand) simultaneous.PolicyValueFunc[S, M, A] {
	return func(state S, legalMovesByAgent simultaneous.LegalMovesByAgent[M, A]) (simultaneous.PolicyByAgent[M, A], simultaneous.ValueByAgent[A], error) {
		rootNode, err := e.NewNode(state)
		if err != nil {
			return nil, nil, err
		}

		_, err = e.Search(rootNode, simulations, rngs)
		if err != nil {
			return nil, nil, err
		}

		policyByAgent := make(simultaneous.PolicyByAgent[M, A], len(e.Game.Agents))
		valueByAgent := make(simultaneous.ValueByAgent[A], len(e.Game.Agents))
		vSelectors := rootNode.VirtualSelectors()

		for _, agent := range e.Game.Agents {
			visitPercents := vSelectors[agent].VisitPercentByKey()
			policy := game.Policy[M]{}
			for _, move := range legalMovesByAgent[agent] {
				if p, ok := visitPercents[move]; !ok {
					return nil, nil, fmt.Errorf("エラー: moveの観測確率が存在しません")
				} else {
					policy[move] = p
				}
			}
			policyByAgent[agent] = policy
			valueByAgent[agent] = 0.0
		}
		return policyByAgent, valueByAgent, nil
	}
}

func (e Engine[S, M, A]) NewPolicyValueFunc(simulations int, rngs []*rand.Rand) simultaneous.PolicyValueFunc[S, M, A] {
	return func(state S, legalMovesByAgent simultaneous.LegalMovesByAgent[M, A]) (simultaneous.PolicyByAgent[M, A], simultaneous.ValueByAgent[A], error) {
		rootNode, err := e.NewNode(state)
		if err != nil {
			return nil, nil, err
		}

		evals, err := e.Search(rootNode, simulations, rngs)
		if err != nil {
			return nil, nil, err
		}

		policyByAgent := make(simultaneous.PolicyByAgent[M, A], len(e.Game.Agents))
		valueByAgent := make(simultaneous.ValueByAgent[A], len(e.Game.Agents))
		vSelectors := rootNode.VirtualSelectors()

		for _, agent := range e.Game.Agents {
			visitPercents := vSelectors[agent].VisitPercentByKey()
			policy := game.Policy[M]{}
			for _, move := range legalMovesByAgent[agent] {
				if p, ok := visitPercents[move]; !ok {
					return nil, nil, fmt.Errorf("エラー: moveの観測確率が存在しません")
				} else {
					policy[move] = p
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