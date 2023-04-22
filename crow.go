package crow

import (
	"math"
	"math/rand"
	"github.com/sw965/omw"
	"golang.org/x/exp/constraints"
)

func NumericalGradient[X constraints.Float](xs []X, f func([]X) X) []X {
	h := X(0.0001)
	n := len(xs)
	grad := make([]X, n)
	for i := 0; i < n; i++ {
		tmp := xs[i]
		xs[i] = tmp + h
		y1 := f(xs)

		xs[i] = tmp - h
		y2 := f(xs)

		grad[i] = (y1 - y2) / (h * 2)
		xs[i] = tmp
	}
	return grad
}

func PolicyUpperConfidenceBound(c, p, v float64, n, a int) float64 {
	fn := float64(n)
	return v + (c * p * math.Sqrt(fn) / float64(a+1))
}

type Reward float64

type AlternatelyMoveGamePlayer[S any, A comparable] func(*S) A
type SimultaneousMoveGamePlayer[S any, A comparable] func(*S) []A

type AlternatelyMoveGameLegalActionsFunc[S any, A comparable] func(*S) []A
type SimultaneousMoveGameLegalActionssFunc[S any, A comparable] func(*S) [][]A

type StateAlternatelyPushFunc[S any, A comparable] func(S, A) S
type StateSimultaneousPushFunc[S any, A comparable] func(S, ...A) S

type EqualStateFunc[S any] func(*S, *S) bool
type IsEndStateWithRewardFunc[S any] func(*S) (bool, Reward)

func playout[S any, A comparable](state S, pushMethod func(S) S, isEndWithReward func(*S) (bool, Reward)) Reward {
	for {
		isEnd, reward := isEndWithReward(&state)
		if isEnd {
			return reward
		}
		state = pushMethod(state)
	}
	return -128.0
}

type AlternatelyMoveGameFunCaller[S any, A comparable] struct {
	Player AlternatelyMoveGamePlayer[S, A]
	LegalActions AlternatelyMoveGameLegalActionsFunc[S, A]
	Push StateAlternatelyPushFunc[S, A]
	EqualState EqualStateFunc[S]
	IsEndWithReward IsEndStateWithRewardFunc[S]
}

func (f *AlternatelyMoveGameFunCaller[S, A]) SetRandomActionPlayer(r *rand.Rand) {
	f.Player = func(state *S) A {
		return omw.RandChoice(f.LegalActions(state), r)
	}
}

func (f *AlternatelyMoveGameFunCaller[S, A]) PushMethod(state S) S {
	action := f.Player(&state)
	return f.Push(state, action)
}

func (f *AlternatelyMoveGameFunCaller[S, A]) PlayoutMethod(state S) Reward {
	return playout[S, A](state, f.PushMethod, f.IsEndWithReward)
}

type SimultaneousMoveGameFunCaller[S any, A comparable] struct {
	Player SimultaneousMoveGamePlayer[S, A]
	LegalActions SimultaneousMoveGameLegalActionssFunc[S, A]
	Push StateSimultaneousPushFunc[S, A]
	EqualState EqualStateFunc[S]
	IsEndWithReward IsEndStateWithRewardFunc[S]
}

func (f *SimultaneousMoveGameFunCaller[S, A]) PushMethod(state S) S {
	actions := f.Player(&state)
	return f.Push(state, actions...)
}

func (f *SimultaneousMoveGameFunCaller[S, A]) PlayoutMethod(state S) Reward {
	return playout[S, A](state, f.PushMethod, f.IsEndWithReward)
}

type UtilPUCB struct {
	AccumReward Reward
	Trial       int
	C float64
	P float64
}

func (p *UtilPUCB) AverageReward() float64 {
	return float64(p.AccumReward) / float64(p.Trial+1)
}

func (p *UtilPUCB) Get(totalTrial int) float64 {
	v := p.AverageReward()
	return PolicyUpperConfidenceBound(p.C, p.P, v, totalTrial, p.Trial)
}

type PUCBMapManager[K comparable] map[K]*UtilPUCB

func (m PUCBMapManager[K]) Trials() []int {
	y := make([]int, 0, len(m))
	for _, v := range m {
		y = append(y, v.Trial)
	}
	return y
}
func (m PUCBMapManager[K]) Max() float64 {
	total := omw.Sum(m.Trials()...)
	y := make([]float64, 0, len(m))
	for _, v := range m {
		y = append(y, v.Get(total))
	}
	return omw.Max(y...)
}

func (m PUCBMapManager[K]) MaxKeys() []K {
	max := m.Max()
	total := omw.Sum(m.Trials()...)
	ks := make([]K, 0, len(m))
	for k, v := range m {
		if v.Get(total) == max {
			ks = append(ks, k)
		}
	}
	return ks
}

func (m PUCBMapManager[K]) MaxTrialKeys() []K {
	max := omw.Max(m.Trials()...)
	ks := make([]K, 0, len(m))
	for k, v := range m {
		if v.Trial == max {
			ks = append(ks, k)
		}
	}
	return ks
}

func (m PUCBMapManager[K]) TrialPercent() map[K]float64 {
	total := omw.Sum(m.Trials()...)
	y := map[K]float64{}
	for k, v := range m {
		y[k] = float64(v.Trial) / float64(total)
	}
	return y
}

type PUCBMapManagers[K comparable] []PUCBMapManager[K]

type ActionPolicY[A comparable] map[A]float64
type ActionPolicyFunc[S any, A comparable] func(*S) ActionPolicY[A]

type PUCT_LeafEvalY Reward
type PUCT_LeafEvalFunc[S any] func(*S) PUCT_LeafEvalY

func NewPUCTPlayoutLeafEvalFunc[S any, A comparable](f *AlternatelyMoveGameFunCaller[S, A]) PUCT_LeafEvalFunc[S] {
	return func(state *S) PUCT_LeafEvalY {
		return PUCT_LeafEvalY(f.PlayoutMethod(*state))
	}
}

type PUCT_BackwardEvalY float64
type PUCT_BackwardEvalFunc[S any] func(PUCT_LeafEvalY, *S) PUCT_BackwardEvalY

type PUCT_EvalFunCaller[S any] struct {
	Leaf PUCT_LeafEvalFunc[S]
	Backward PUCT_BackwardEvalFunc[S]
}

type PUCT_FunCaller[S any, A comparable] struct {
	Game AlternatelyMoveGameFunCaller[S, A]
	Policy ActionPolicyFunc[S, A]
	Eval PUCT_EvalFunCaller[S]
}

func (f *PUCT_FunCaller[S, A]) NewNodeMethod(state *S, c float64) *PUCT_Node[S, A] {
	py := f.Policy(state)
	m := PUCBMapManager[A]{}
	for a, p := range py {
		m[a] = &UtilPUCB{C:c, P:p}
	}
	return &PUCT_Node[S, A]{State:*state, PUCBManager:m}
}

func (f *PUCT_FunCaller[S, A]) SetNoPolicy() {
	var policy ActionPolicyFunc[S, A]
	policy = func(state *S) ActionPolicY[A] {
		legalActions := f.Game.LegalActions(state)
		p := 1.0 / float64(len(legalActions))
		y := ActionPolicY[A]{}
		for _, a := range legalActions {
			y[a] = p
		}
		return y
	}
	f.Policy = policy
}

func (f *PUCT_FunCaller[S, A]) SetPlayoutLeafEval() {
	var leaf PUCT_LeafEvalFunc[S]
	leaf = func(state *S) PUCT_LeafEvalY {
		return PUCT_LeafEvalY(f.Game.PlayoutMethod(*state))
	}
	f.Eval.Leaf = leaf
}

type PUCT_Node[S any, A comparable] struct {
	State S
	PUCBManager PUCBMapManager[A]
	NextNodes PUCT_Nodes[S, A]
	SelectCount     int
}

type PUCT_Nodes[S any, A comparable] []*PUCT_Node[S, A]

func (nodes PUCT_Nodes[S, A]) Find(state *S, eq EqualStateFunc[S]) (*PUCT_Node[S, A], bool) {
	for _, node := range nodes {
		if eq(&node.State, state) {
			return node, true
		}
	}
	return &PUCT_Node[S, A]{}, false
}

type PUCT_Select[S any, A comparable] struct {
	Node *PUCT_Node[S, A]
	Action A
}

type PUCT_Selects[S any, A comparable] []PUCT_Select[S, A]

func (ss PUCT_Selects[S, A]) Backward(y PUCT_LeafEvalY, eval PUCT_BackwardEvalFunc[S]) {
	for _, s := range ss {
		node := s.Node
		action := s.Action
		node.PUCBManager[action].AccumReward += Reward(eval(y, &node.State))
		node.PUCBManager[action].Trial += 1
		node.SelectCount = 0
	}
}

type PUCT_Selector[S any, A comparable] struct {
	FunCaller PUCT_FunCaller[S, A]
	Cap int
}

func (s *PUCT_Selector[S, A]) SelectAndExpansion(node *PUCT_Node[S, A], allNodes PUCT_Nodes[S, A], c float64, r *rand.Rand) (S, PUCT_Nodes[S, A], PUCT_Selects[S, A]) {
	f := s.FunCaller
	state := node.State
	selects := make(PUCT_Selects[S, A], 0, s.Cap)

	for {
		// for a, pucb := range node.PUCBManager {
		// 	fmt.Println(a, pucb.Get(omw.Sum(node.PUCBManager.Trials()...), len(node.PUCBManager)))
		// }
		action := omw.RandChoice(node.PUCBManager.MaxKeys(), r)
		selects = append(selects, PUCT_Select[S, A]{Node:node, Action:action})

		node.SelectCount += 1

		state = f.Game.Push(state, action)
		stateP := &state

		if isEnd, _ := f.Game.IsEndWithReward(stateP); isEnd {
			break
		}

		//nextNodesの中に、同じstateが存在するならば、それを次のNodeとする
		//nextNodesの中に、同じstateが存在しないなら、allNodesの中から同じstateが存在しないかを調べる。
		//allNodesの中に、同じstateが存在するならば、次回から高速に探索出来るように、nextNodesに追加して、次のnodeとする。
		//nextNodesにもallNodesにも同じstateが存在しないなら、新しくnodeを作り、
		//nextNodesと、allNodesに追加し、新しく作ったnodeを次のnodeとし、select処理を終了する。

		nextNode, ok := node.NextNodes.Find(stateP, f.Game.EqualState)
		if !ok {
			nextNode, ok = allNodes.Find(stateP, f.Game.EqualState)
			if ok {
				node.NextNodes = append(node.NextNodes, nextNode)
			} else {
				nextNode = f.NewNodeMethod(stateP, c)
				allNodes = append(allNodes, nextNode)
				node.NextNodes = append(node.NextNodes, nextNode)
				//新しくノードを作成したら、処理を終了する
				break
			}
		}

		if nextNode.SelectCount == 1 {
			break
		}
		node = nextNode
	}

	s.Cap = len(selects) + 1
	return state, allNodes, selects
}

type PUCT[S any, A comparable] struct {
	FunCaller PUCT_FunCaller[S, A]
}

func (p *PUCT[S, A]) Run(simulation int, rootState S, c float64, r *rand.Rand) PUCT_Nodes[S, A] {
	f := p.FunCaller
	rootNode := f.NewNodeMethod(&rootState, c)
	allNodes := PUCT_Nodes[S, A]{rootNode}

	var leafState S
	var selects PUCT_Selects[S, A]
	selector := PUCT_Selector[S, A]{FunCaller:f, Cap:1}

	for i := 0; i < simulation; i++ {
		leafState, allNodes, selects = selector.SelectAndExpansion(rootNode, allNodes, c, r)
		y := f.Eval.Leaf(&leafState)
		selects.Backward(y, f.Eval.Backward)
	}
	return allNodes
}

type ActionPolicYs[A comparable] []ActionPolicY[A]
type ActionPoliciesFunc[S any, A comparable] func(*S) ActionPolicYs[A]

type DPUCT_LeafEvalY float64
type DPUCT_LeafEvalYs []DPUCT_LeafEvalY
type DPUCT_LeafEvalsFunc[S any] func(*S) DPUCT_LeafEvalYs

type DPUCT_FunCaller[S any, A comparable] struct {
	Game SimultaneousMoveGameFunCaller[S, A]
	Policies ActionPoliciesFunc[S, A]
	LeafEvals DPUCT_LeafEvalsFunc[S]
}

func (f *DPUCT_FunCaller[S, A]) NewNodeMethod(state *S) *DPUCT_Node[S, A] {
	pys := f.Policies(state)
	ms := make(PUCBMapManagers[A], len(pys))

	for playerI, py := range pys {
		ms[playerI] = PUCBMapManager[A]{}
		for a, p := range py {
			ms[playerI][a].P = p
		}
	}
	return &DPUCT_Node[S, A]{State:*state, PUCBManagers:ms}
}

type DPUCT_Node[S any, A comparable] struct {
	State S
	PUCBManagers PUCBMapManagers[A]
	NextNodes DPUCT_Nodes[S, A]
	SelectCount     int
}

type DPUCT_Nodes[S any, A comparable] []*DPUCT_Node[S, A]

func (nodes DPUCT_Nodes[S, A]) Find(state *S, eq EqualStateFunc[S]) (*DPUCT_Node[S, A], bool) {
	for _, node := range nodes {
		if eq(&node.State, state) {
			return node, true
		}
	}
	return &DPUCT_Node[S, A]{}, false
}

type DPUCT_Select[S any, A comparable] struct {
	Node *DPUCT_Node[S, A]
	Actions []A
}

type DPUCT_Selects[S any, A comparable] []DPUCT_Select[S, A]

func (ss DPUCT_Selects[S, A]) Backward(ys DPUCT_LeafEvalYs) {
	for _, s := range ss {
		node := s.Node
		actions := s.Actions
		for playerI, action := range actions {
			node.PUCBManagers[playerI][action].AccumReward += Reward(ys[playerI])
			node.PUCBManagers[playerI][action].Trial += 1
		}
		node.SelectCount = 0
	}
}

type DPUCT_Selector[S any, A comparable] struct {
	FunCaller DPUCT_FunCaller[S, A]
	Cap int
}

func (s *DPUCT_Selector[S, A])SelectAndExpansion(simultaneous int, node *DPUCT_Node[S, A], allNodes DPUCT_Nodes[S, A], c float64, r *rand.Rand) (S, DPUCT_Nodes[S, A], DPUCT_Selects[S, A]) {
	f := s.FunCaller
	state := node.State
	selects := make(DPUCT_Selects[S, A], 0, s.Cap)

	for {
		actions := make([]A, simultaneous)
		for playerI, m := range node.PUCBManagers {
			actions[playerI] = omw.RandChoice(m.MaxKeys(), r)
		}

		selects = append(selects, DPUCT_Select[S, A]{Node:node, Actions:actions})
		node.SelectCount += 1

		state = f.Game.Push(state, actions...)
		stateP := &state

		if isEnd, _ := f.Game.IsEndWithReward(stateP); isEnd {
			break
		}

		//nextNodesの中に、同じstateが存在するならば、それを次のNodeとする
		//nextNodesの中に、同じstateが存在しないなら、allNodesの中から同じstateが存在しないかを調べる。
		//allNodesの中に、同じstateが存在するならば、次回から高速に探索出来るように、nextNodesに追加して、次のnodeとする。
		//nextNodesにもallNodesにも同じstateが存在しないなら、新しくnodeを作り、
		//nextNodesと、allNodesに追加し、新しく作ったnodeを次のnodeとし、select処理を終了する。

		nextNode, ok := node.NextNodes.Find(stateP, f.Game.EqualState)
		if !ok {
			nextNode, ok = allNodes.Find(stateP, f.Game.EqualState)
			if ok {
				node.NextNodes = append(node.NextNodes, nextNode)
			} else {
				nextNode = f.NewNodeMethod(stateP)
				allNodes = append(allNodes, nextNode)
				node.NextNodes = append(node.NextNodes, nextNode)
				//新しくノードを作成したら、処理を終了する
				break
			}
		}

		if nextNode.SelectCount == 1 {
			break
		}
		node = nextNode
	}
	s.Cap = len(selects) + 1
	return state, allNodes, selects
}

type DPUCT[S any, A comparable] struct {
	FunCaller DPUCT_FunCaller[S, A]
}

func (d *DPUCT[S, A]) Run(simulation int, rootState S, c float64, r *rand.Rand) (DPUCT_Nodes[S, A], error) {
	rootNode := d.FunCaller.NewNodeMethod(&rootState)
	allNodes := DPUCT_Nodes[S, A]{rootNode}
	simultaneous := len(rootNode.PUCBManagers)

	var leafState S
	var selects DPUCT_Selects[S, A]
	selector := DPUCT_Selector[S, A]{FunCaller:d.FunCaller, Cap:1}

	for i := 0; i < simulation; i++ {
		leafState, allNodes, selects = selector.SelectAndExpansion(simultaneous, rootNode, allNodes, c, r)
		ys := d.FunCaller.LeafEvals(&leafState)
		selects.Backward(ys)
	}
	return allNodes, nil
}