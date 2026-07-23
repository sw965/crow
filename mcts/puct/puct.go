package puct

import (
	"fmt"
	"math/rand/v2"

	"errors"
	"maps"
	"slices"
	"sync"

	"github.com/sw965/crow/game"
	"github.com/sw965/crow/game/sequential"
	"github.com/sw965/crow/pucb"
	"github.com/sw965/omw/parallel"
)

var (
	ErrNilEngineFunc = errors.New("puct.Engineエラー: フィールドの関数がnilです")
	ErrInvalidConfig = errors.New("puct.Engineエラー: 設定値が不正です")
)

type RootNodeEvalByAgent[Ag comparable] map[Ag]float32

func (es RootNodeEvalByAgent[Ag]) DivScalar(s float32) {
	for k := range es {
		es[k] /= s
	}
}

type LeafNodeEvalByAgent[Ag comparable] map[Ag]float32

// LeafNodeEvalByAgentFunc は、リーフノードの状態を評価する。
// 探索は複数のワーカーから並行に呼び出す為、乱数が必要な評価（プレイアウト等）は、
// 引数で渡されるワーカー毎の乱数器を使う事。
type LeafNodeEvalByAgentFunc[S any, Ag comparable] func(S, *rand.Rand) (LeafNodeEvalByAgent[Ag], error)

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

// backward は、リーフノードの評価値を、経路上の全ノードに反映する。
// 途中でエラーが起きても、pending の解放は全ての経路に対して必ず行い、
// 発生したエラーはまとめて返す。
func (ss selectBuffers[S, Ac, Ag]) backward(evals LeafNodeEvalByAgent[Ag]) error {
	var errs []error
	for _, s := range ss {
		node := s.node
		action := s.action

		node.Lock()
		c := node.virtualSelector[action]
		// 未観測のカウントを消す
		if err := c.DecrementPending(); err != nil {
			errs = append(errs, err)
		}

		eval, ok := evals[node.Agent]
		if !ok {
			errs = append(errs, fmt.Errorf(
				"LeafNodeEvalByAgentに存在しないキー(Agent)でアクセスしようとした為、backwardを実行出来ませんでした。node.Agent = %v, LeafNodeEvalByAgent.Keys() = %v",
				node.Agent, slices.Collect(maps.Keys(evals)),
			))
			node.Unlock()
			continue
		}

		if err := c.AddW(eval); err != nil {
			errs = append(errs, err)
			node.Unlock()
			continue
		}
		c.IncrementVisits()
		node.Unlock()
	}
	return errors.Join(errs...)
}

// rollbackPending は、backward を実行しない場合に、経路上の pending を解放する。
func (ss selectBuffers[S, Ac, Ag]) rollbackPending() error {
	var errs []error
	for _, s := range ss {
		s.node.Lock()
		if err := s.node.virtualSelector[s.action].DecrementPending(); err != nil {
			errs = append(errs, err)
		}
		s.node.Unlock()
	}
	return errors.Join(errs...)
}

type Engine[S any, Ac, Ag comparable] struct {
	Game                    sequential.Engine[S, Ac, Ag]
	PUCBFunc                pucb.Func
	PolicyFunc              sequential.PolicyFunc[S, Ac]
	LeafNodeEvalByAgentFunc LeafNodeEvalByAgentFunc[S, Ag]
	NextNodesCap            int
	VirtualValue            float32
	// MaxDepth は1回のシミュレーションで辿るノード数の上限。
	// 状態が循環し得るゲームでは、展開もゲーム終了も起きずに探索が無限ループする恐れがある為、
	// そのようなゲームでは必ず設定する事。上限に達した場合、その状態をリーフノードとして評価する。
	// 0の場合は無制限。
	MaxDepth int
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

// SetPlayout は、リーフノードの評価関数として、ゲーム終了までのプレイアウトを設定する。
// 乱数器は探索の呼び出し側からワーカー毎に渡される為、ここでは受け取らない。
func (e *Engine[S, Ac, Ag]) SetPlayout(accr sequential.ActorCritic[S, Ac, Ag]) {
	e.LeafNodeEvalByAgentFunc = func(state S, rng *rand.Rand) (LeafNodeEvalByAgent[Ag], error) {
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
		return nil, fmt.Errorf("ゲームが終了していないのに合法手がありません")
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
		return nil, fmt.Errorf("CurrentAgentFuncが返したエージェントがAgentsに含まれていません: agent = %v", agent)
	}

	return &Node[S, Ac, Ag]{
		State:             state,
		Agent:             agent,
		virtualSelector:   s,
		nextNodesByAction: make(map[Ac]Nodes[S, Ac, Ag], e.NextNodesCap),
	}, nil
}

func (e Engine[S, Ac, Ag]) SelectExpansionBackward(node *Node[S, Ac, Ag], capacity int, rng *rand.Rand) (evals LeafNodeEvalByAgent[Ag], depth int, err error) {
	state := node.State
	buffers := make(selectBuffers[S, Ac, Ag], 0, capacity)
	var isEnd bool

	// バッファ積み上げ中にエラーが起きた場合、pending を元に戻す。
	// backward 実行後は backward 側が pending を解放する為、ここでは戻さない。
	backwardStarted := false
	defer func() {
		if err != nil && !backwardStarted {
			if rbErr := buffers.rollbackPending(); rbErr != nil {
				err = errors.Join(err, rbErr)
			}
		}
	}()

	for {
		node.Lock()

		var action Ac
		action, err = node.virtualSelector.Select(rng)
		if err != nil {
			node.Unlock()
			return nil, 0, err
		}
		// 選択した行動のノードの未観測の数をインクリメントする
		node.virtualSelector[action].IncrementPending()

		node.Unlock()
		buffers = append(buffers, selectBuffer[S, Ac, Ag]{node: node, action: action})

		state, err = e.Game.Logic.TransitionFunc(state, action)
		if err != nil {
			return nil, 0, err
		}

		isEnd, err = e.Game.IsTerminal(state)
		if err != nil {
			return nil, 0, err
		}

		if isEnd {
			break
		}

		// 深さが上限に達した場合、この状態をリーフノードとして評価する
		if e.MaxDepth > 0 && len(buffers) >= e.MaxDepth {
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

	evals = LeafNodeEvalByAgent[Ag]{}
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
		evals, err = e.LeafNodeEvalByAgentFunc(state, rng)
		if err != nil {
			return nil, 0, err
		}
	}

	backwardStarted = true
	if err = buffers.backward(evals); err != nil {
		return nil, 0, err
	}
	return evals, len(buffers), nil
}

func (e Engine[S, Ac, Ag]) Search(rootNode *Node[S, Ac, Ag], n int, workerRngs []*rand.Rand) (RootNodeEvalByAgent[Ag], error) {
	if err := e.Validate(); err != nil {
		return nil, err
	}

	if rootNode == nil {
		return nil, fmt.Errorf("rootNode が nil です")
	}

	if n <= 0 {
		return nil, fmt.Errorf("シミュレーション数が不正: n = %d: n > 0 であるべき", n)
	}

	p := len(workerRngs)
	rootEvalsPerWorker := make([]RootNodeEvalByAgent[Ag], p)
	for i := range p {
		rootEvalsPerWorker[i] = RootNodeEvalByAgent[Ag]{}
	}

	workerBuffCaps := make([]int, p)
	err := parallel.For(n, p, func(workerID, idx int) error {
		rng := workerRngs[workerID]
		leafEvals, depth, err := e.SelectExpansionBackward(rootNode, workerBuffCaps[workerID], rng)
		if err != nil {
			return err
		}

		for k, v := range leafEvals {
			rootEvalsPerWorker[workerID][k] += v
		}

		workerBuffCaps[workerID] = depth + 1
		return nil
	})

	if err != nil {
		return nil, err
	}

	rootEvals := RootNodeEvalByAgent[Ag]{}
	for i := range rootEvalsPerWorker {
		for k, v := range rootEvalsPerWorker[i] {
			rootEvals[k] += v
		}
	}

	rootEvals.DivScalar(float32(n))
	return rootEvals, nil
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

		visitRatios := rootNode.VirtualSelector().VisitRatioByKey()
		policy := game.Policy[Ac]{}
		for _, action := range legalActions {
			if p, ok := visitRatios[action]; !ok {
				return nil, 0.0, fmt.Errorf("actionの訪問比率が存在しません: action = %v", action)
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

		visitRatios := rootNode.VirtualSelector().VisitRatioByKey()
		policy := game.Policy[Ac]{}
		for _, action := range legalActions {
			if p, ok := visitRatios[action]; !ok {
				return nil, 0.0, fmt.Errorf("actionの訪問比率が存在しません: action = %v", action)
			} else {
				policy[action] = p
			}
		}

		eval, ok := evals[rootNode.Agent]
		if !ok {
			return nil, 0.0, fmt.Errorf("ルートノードのエージェントの評価値が存在しません: agent = %v", rootNode.Agent)
		}
		return policy, eval, nil
	}
}
