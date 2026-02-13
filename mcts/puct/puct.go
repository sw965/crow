package puct

import (
	"fmt"
	"math/rand/v2"

	"slices"
	"errors"
	game "github.com/sw965/crow/game/sequential"
	"github.com/sw965/crow/pucb"
	"github.com/sw965/omw/parallel"
	"maps"
	"sync"
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
	nextNodesByMove map[M]Nodes[S, M, A]
	sync.Mutex
}

func (n *Node[S, M, A]) VirtualSelector() pucb.VirtualSelector[M] {
	return maps.Clone(n.virtualSelector)
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
		node.virtualSelector[move].DecrementO()
		node.virtualSelector[move].AddW(eval)
		node.virtualSelector[move].IncrementVisits()
		node.Unlock()
	}
}

func (ss selectBuffers[S, M, A]) rollbackO() {
	for _, s := range ss {
		s.node.Lock()
		s.node.virtualSelector[s.move].DecrementO()
		s.node.Unlock()
	}
}

type Engine[S any, M, A comparable] struct {
	Game                    game.Engine[S, M, A]
	PUCBFunc                pucb.Func
	PolicyFunc              game.PolicyFunc[S, M]
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
		return nil, fmt.Errorf("後でエラーメッセージを書く")
	}

	policy, err := e.PolicyFunc(state, legalMoves)
	if err != nil {
		return nil, err
	}

	err = policy.ValidateForLegalMoves(legalMoves, true)
	if err != nil {
		return nil, err
	}

	s := pucb.VirtualSelector[M]{}
	for _, move := range legalMoves {
		p := policy[move]
		s[move] = &pucb.Calculator{Func: e.PUCBFunc, P: p, VirtualValue: e.VirtualValue}
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
		return nil, fmt.Errorf("後でエラーメッセージを書く")
	}

	return &Node[S, M, A]{
		State:           state,
		Agent:           agent,
		virtualSelector: s,
		nextNodesByMove: make(map[M]Nodes[S, M, A], e.NextNodesCap),
	}, nil
}

func (e Engine[S, M, A]) SelectExpansionBackward(node *Node[S, M, A], capacity int, rng *rand.Rand) (LeafNodeEvalByAgent[A], int, error) {
	state := node.State
	buffers := make(selectBuffers[S, M, A], 0, capacity)
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

		var move M
		move, err := node.virtualSelector.Select(rng)
		if err != nil {
			node.Unlock()
			return nil, 0, err
		}
		// 選択した行動のノードの未観測の数をインクリメントする
		node.virtualSelector[move].IncrementO()

		node.Unlock()
		buffers = append(buffers, selectBuffer[S, M, A]{node: node, move: move})

		state, err = e.Game.Logic.MoveFunc(state, move)
		if err != nil {
			return nil, 0, err
		}

		var isEnd bool
		isEnd, err = e.Game.IsEnd(state)
		if err != nil {
			return nil, 0, err
		}

		if isEnd {
			break
		}

		var expand bool

		// node.nextNodesByMoveはmap型 node.nextNodesByMove[move]はslice型
		// この処理はデータを読むだけだが、他のワーカーが、書き込む処理をすると、破綻する為、Lockが必要
		node.Lock()
		nextNode, ok := node.nextNodesByMove[move].FindByState(state, e.Game.Logic.EqualFunc)
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

			// Unlockして NewNodeを作ってる間に、別のワーカーがノードを追加した可能性がある為、再度Lockして調べる
			node.Lock()
			// nextNodesの中に、一致するstateが見つかれば、それを次のノードとする
			if nn, ok := node.nextNodesByMove[move].FindByState(state, e.Game.Logic.EqualFunc); ok {
				nextNode = nn
				expand = false
				// nextNodesの中に、一致するstateが見つからなければ、newNodeをnextNodesに追加し、selectを終了する
			} else {
				node.nextNodesByMove[move] = append(node.nextNodesByMove[move], newNode)
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
	// ゲームが終了した場合、ゲームエンジンの結果スコアを、リーフノードの評価値とする
	// ゲームが終了していなかった場合、リーフノードの評価関数を呼び出す
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

func (e Engine[S, M, A]) NewPolicy(simulations int, rngs []*rand.Rand) game.PolicyFunc[S, M] {
	return func(state S, legalMoves []M) (game.Policy[M], error) {
		rootNode, err := e.NewNode(state)
		if err != nil {
			return nil, err
		}

		_, err = e.Search(rootNode, simulations, rngs)
		if err != nil {
			return nil, err
		}

		visitPercents := rootNode.VirtualSelector().VisitPercentByKey()
		policy := game.Policy[M]{}
		for _, move := range legalMoves {
			// 未訪問の手は0.0になるようにフォールバック
			policy[move] = visitPercents[move]
		}
		return policy, nil
	}
}
