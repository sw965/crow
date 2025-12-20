package puct

import (
	"fmt"
	"math/rand/v2"

	"errors"
	game "github.com/sw965/crow/game/sequential"
	"github.com/sw965/crow/pucb"
	"github.com/sw965/omw/slicesx"
	"maps"
	"slices"
	"sync"
	"github.com/sw965/omw/parallel"
)

var (
	ErrNilEngineFunc = errors.New("puct.Engineエラー: フィールドの関数がnilです")
	ErrInvalidConfig = errors.New("puct.Engineエラー: 設定値が不正です")
)

type RootNodeEvalByAgent[A comparable] map[A]float32

func (es RootNodeEvalByAgent[A]) DivScalar(s float32) {
	for k := range es {
		es[k] /= s
	}
}

type LeafNodeEvalByAgent[A comparable] map[A]float32
type LeafNodeEvalByAgentFunc[S any, A comparable] func(S) (LeafNodeEvalByAgent[A], error)

type Node[S any, M, A comparable] struct {
	State           S
	Agent           A
	virtualSelector pucb.VirtualSelector[M]
	nextNodes       Nodes[S, M, A]
	sync.Mutex
}

// FindOrAppendNextNode は、state に一致する子ノードがあればそれを返し、なければ newNodeFunc で生成して追加します。
// 追加したかどうか（appended）も返します。
// NOTE: 二重生成レースを避けるため、生成前後でダブルチェックします。
// newNodeFuncが支配的になる為、一回Unlockする。
func (node *Node[S, M, A]) FindOrAppendNextNode(state S, eq game.EqualFunc[S], newNodeFunc func(S) (*Node[S, M, A], error)) (next *Node[S, M, A], appended bool, err error) {
	if newNodeFunc == nil {
		return nil, false, fmt.Errorf("newNodeFunc が nil です")
	}

	// 1st check (lock)
	node.Lock()
	if n, ok := node.nextNodes.FindByState(state, eq); ok {
		node.Unlock()
		return n, false, nil
	}
	node.Unlock()

	// Create outside lock (expensive work is outside)
	created, err := newNodeFunc(state)
	if err != nil {
		return nil, false, err
	}

	// 2nd check + append (lock)
	node.Lock()
	defer node.Unlock()

	if n, ok := node.nextNodes.FindByState(state, eq); ok {
		// Someone else appended while we were creating.
		return n, false, nil
	}

	node.nextNodes = append(node.nextNodes, created)
	return created, true, nil
}

type Nodes[S any, M, A comparable] []*Node[S, M, A]

func (nodes Nodes[S, M, A]) FindByState(state S, eq game.EqualFunc[S]) (*Node[S, M, A], bool) {
	for _, node := range nodes {
		if eq(node.State, state) {
			return node, true
		}
	}
	return nil, false
}

type selectBuffer[S any, M, A comparable] struct {
	node *Node[S, M, A]
	move M
}

type selectBuffers[S any, M, A comparable] []selectBuffer[S, M, A]

func (ss selectBuffers[S, M, A]) backward(evals LeafNodeEvalByAgent[A]) {
	for _, s := range ss {
		node := s.node
		move := s.move
		eval, ok := evals[node.Agent]
		if !ok {
			msg := fmt.Sprintf(
				"BUG: LeafNodeEvalByAgentに存在しないキー(Agent)でアクセスしようとした為、backwardを実行出来ませんでした。leafNode.Agent = %v, leftNodeEvalByAgent.Keys() = %v",
				node.Agent, slices.Collect(maps.Keys(evals)),
			)
			panic(msg)
		}

		node.Lock()
		// 未観測のカウントを消す
		node.virtualSelector[move].DecrementPending()
		node.virtualSelector[move].AddTotalValue(eval)
		node.virtualSelector[move].IncrementTrial()
		node.Unlock()
	}
}

func (ss selectBuffers[S, M, A]) rollbackPending() {
    for _, s := range ss {
		s.node.Lock()
        s.node.virtualSelector[s.move].DecrementPending()
		s.node.Unlock()
    }
}

type Engine[S any, M, A comparable] struct {
	Game                    game.Engine[S, M, A]
	PUCBFunc                pucb.Func
	PolicyFunc              game.PolicyFunc[S, M]
	LeafNodeEvalByAgentFunc LeafNodeEvalByAgentFunc[S, A]
	NextNodesCap            int
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
	e.PolicyFunc = game.UniformPolicyFunc[S, M]
}

func (e *Engine[S, M, A]) SetPlayout(actor game.Actor[S, M, A], rng *rand.Rand) {
	e.LeafNodeEvalByAgentFunc = func(state S) (LeafNodeEvalByAgent[A], error) {
		finals, err := e.Game.Playouts([]S{state}, actor, []*rand.Rand{rng})
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
	legalMoves := e.Game.Logic.LegalMovesFunc(state)

	// policy.ValidateForLegalMovesでもlegalMovesの空チェックはするが、PolicyFuncを安全に呼ぶ為に、ここでもチェックする
	if len(legalMoves) == 0 {
		return nil, game.ErrEmptyLegalMoves
	}

	// policy.ValidateForLegalMovesは、legalMovesがユニークならば、policyが合法手のみを持つ事が保障される
	// ノード生成時に、legalMovesのユニーク性をチェックするのは、オーバーヘッドの比率が低いと判断した為、チェックする。
	if !slicesx.IsUnique(legalMoves) {
		return nil, game.ErrNotUniqueLegalMoves
	}

	policy := e.PolicyFunc(state, legalMoves)
	err := policy.ValidateForLegalMoves(legalMoves)
	if err != nil {
		return nil, err
	}

	s := pucb.VirtualSelector[M]{}
	for _, move := range legalMoves {
		p := policy[move]
		s[move] = &pucb.Calculator{Func: e.PUCBFunc, P: p}
	}

	agent := e.Game.Logic.CurrentAgentFunc(state)

	found := false
	for _, a := range e.Game.Agents {
		if a == agent {
			found = true
			break
		}
	}

	if !found {
		return nil, fmt.Errorf("%w: Engine.Game.Logic.CurrentAgentFunc(state)=%v", game.ErrAgentNotFound, agent)
	}

	nextNodes := make(Nodes[S, M, A], 0, e.NextNodesCap)
	return &Node[S, M, A]{
		State:           state,
		Agent:           agent,
		virtualSelector: s,
		nextNodes:       nextNodes,
	}, nil
}

func (e Engine[S, M, A]) SelectExpansionBackward(node *Node[S, M, A], capacity int, rng *rand.Rand) (LeafNodeEvalByAgent[A], int, error) {
	state := node.State
	buffers := make(selectBuffers[S, M, A], 0, capacity)
	var err error
	var isEnd bool
	var move M
	var nextNode *Node[S, M, A]
	var expand bool

    // 途中エラーなら pending を元に戻す（成功時は backward が pending--するので不要）
    defer func() {
        if err != nil {
            buffers.rollbackPending()
        }
    }()

	for {
		node.Lock()
		// := だと err が内側でシャドウイングされて、 defer func の err != nil が var err error を参照してくれなくなる
		move, err = node.virtualSelector.Select(rng)
		if err != nil {
			node.Unlock()
			return nil, 0, err
		}

        // ノードを選択した直後に、未観測(まだプレイアウトや評価が確定していない)をインクリメントする
        node.virtualSelector[move].IncrementPending()
		node.Unlock()

		buffers = append(buffers, selectBuffer[S, M, A]{node: node, move: move})

		state, err = e.Game.Logic.MoveFunc(state, move)
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

		nextNode, expand, err = node.FindOrAppendNextNode(state, e.Game.Logic.EqualFunc, e.NewNode)
		if err != nil {
			return nil, 0, err
		}

		if expand {
			break
		}

		node = nextNode
	}

	evals := LeafNodeEvalByAgent[A]{}

	// ゲームが終了した場合、ゲームエンジンの結果スコアを、リーフノードの評価値とする
	// ゲームが終了していなかった場合、リーフノードの評価関数を呼び出す
	if isEnd {
		var scores game.ResultScoreByAgent[A]
		// := を使うと、defer func() の err != nil が var err error を参照してくれなくなる
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

func (e Engine[S, M, A]) Search(rootNode *Node[S, M, A], n int, rngs []*rand.Rand) (RootNodeEvalByAgent[A], error) {
	if err := e.Validate(); err != nil {
		return nil, err
	}

	if rootNode == nil {
		return nil, fmt.Errorf("rootNode が nil です")
	}

	if n <= 0 {
		return nil, fmt.Errorf("シミュレーション数は0より大きい必要があります。")
	}

	capacity := 128
	p := len(rngs)
	rootEvalsPerWorker := make([]RootNodeEvalByAgent[A], p)
	for i := range p {
		rootEvalsPerWorker[i] = RootNodeEvalByAgent[A]{}
	}

	err := parallel.For(n, p, func(workerId, idx int) error {
		rng := rngs[workerId]
		leafEvals, _, err := e.SelectExpansionBackward(rootNode, capacity, rng)
		if err != nil {
			return err
		}

		for k, v := range leafEvals {
			rootEvalsPerWorker[workerId][k] += v
		}
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
