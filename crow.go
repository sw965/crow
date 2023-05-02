package crow

import (
	"math"
	"math/rand"
	"github.com/sw965/omw"
	"golang.org/x/exp/constraints"
	"golang.org/x/exp/maps"
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

type SequentialGamePlayer[S any, A comparable] func(*S) A
type SimultaneousGamePlayer[S any, AS ~[]A, A comparable] func(*S) AS

type SequentialGameLegalActionsFunc[S any, AS ~[]A, A comparable] func(*S) AS
type SimultaneousGameLegalActionssFunc[S any, ASS ~[]AS, AS ~[]A, A comparable] func(*S) ASS

type StateAlternatelyPushFunc[S any, A comparable] func(S, A) S
type StateSimultaneousPushFunc[S any, A comparable] func(S, ...A) S

type EqualStateFunc[S any] func(*S, *S) bool
type IsEndStateFunc[S any] func(*S) bool

type SequentialGameFunCaller[S any, AS ~[]A, A comparable] struct {
	Player SequentialGamePlayer[S, A]
	LegalActions SequentialGameLegalActionsFunc[S, AS, A]
	Push StateAlternatelyPushFunc[S, A]
	EqualState EqualStateFunc[S]
	IsEnd IsEndStateFunc[S]
}

func (f *SequentialGameFunCaller[S, AS, A]) Clone() SequentialGameFunCaller[S, AS, A] {
	return SequentialGameFunCaller[S, AS, A]{
		Player:f.Player,
		LegalActions:f.LegalActions,
		Push:f.Push,
		EqualState:f.EqualState,
		IsEnd:f.IsEnd,
	}
}

func (f *SequentialGameFunCaller[S, AS, A]) SetRandomActionPlayer(r *rand.Rand) {
	f.Player = func(state *S) A {
		return omw.RandChoice(f.LegalActions(state), r)
	}
}

func (f *SequentialGameFunCaller[S, AS, A]) SetPUCTPlayer(puct *PUCT[S, AS, A], simulation int, c float64, random *rand.Rand, r float64) {
	f.Player = func(state *S) A {
		allNodes := puct.Run(simulation, *state, c, random)
		node := allNodes[0]
		percent := node.PUCBManager.TrialPercent()
		max := omw.Max(maps.Values(percent)...)

		n := len(percent)
		actions := make([]A, 0, n)
		ws := make([]float64, 0, n)

		for a, p := range percent {
			if max*r <= p {
				actions = append(actions, a)
				ws = append(ws, p)
			}
		}

		idx := omw.RandIntWithWeight(ws, random)
		return actions[idx]
	}
}

func (f *SequentialGameFunCaller[S, AS, A]) Playout(state S) S {
	for {
		isEnd := f.IsEnd(&state)
		if isEnd {
			break
		}
		action := f.Player(&state)
		state = f.Push(state, action)
	}
	return state
}

type SimultaneousGameFunCaller[S any, ASS ~[]AS, AS ~[]A, A comparable] struct {
	Player SimultaneousGamePlayer[S, AS, A]
	LegalActionss SimultaneousGameLegalActionssFunc[S, ASS, AS, A]
	Push StateSimultaneousPushFunc[S, A]
	EqualState EqualStateFunc[S]
	IsEnd IsEndStateFunc[S]
}

func (f *SimultaneousGameFunCaller[S, ASS, AS, A]) Clone() SimultaneousGameFunCaller[S, ASS, AS, A] {
	return SimultaneousGameFunCaller[S, ASS, AS, A]{
		Player:f.Player,
		LegalActionss:f.LegalActionss,
		Push:f.Push,
		EqualState:f.EqualState,
		IsEnd:f.IsEnd,
	}
}

func (f *SimultaneousGameFunCaller[S, ASS, AS, A]) SetRandomActionPlayer(r *rand.Rand) {
	f.Player = func(state *S) AS {
		actionss := f.LegalActionss(state)
		y := make([]A, len(actionss))
		for playerI, actions := range actionss {
			y[playerI] = omw.RandChoice(actions, r)
		}
		return y
	}
}

func (f *SimultaneousGameFunCaller[S, ASS, AS, A]) Playout(state S) S {
	for {
		isEnd := f.IsEnd(&state)
		if isEnd {
			break
		}
		actions := f.Player(&state)
		state = f.Push(state, actions...)
	}
	return state
}

type UtilPUCB struct {
	AccumReward float64
	Trial       int
	P float64
}

func (p *UtilPUCB) AverageReward() float64 {
	return float64(p.AccumReward) / float64(p.Trial+1)
}

func (p *UtilPUCB) Get(totalTrial int, c float64) float64 {
	v := p.AverageReward()
	return PolicyUpperConfidenceBound(c, p.P, v, totalTrial, p.Trial)
}

type PUCBMapManager[K comparable] map[K]*UtilPUCB

func (m PUCBMapManager[K]) Trials() []int {
	y := make([]int, 0, len(m))
	for _, v := range m {
		y = append(y, v.Trial)
	}
	return y
}
func (m PUCBMapManager[K]) Max(c float64) float64 {
	total := omw.Sum(m.Trials()...)
	y := make([]float64, 0, len(m))
	for _, v := range m {
		y = append(y, v.Get(total, c))
	}
	return omw.Max(y...)
}

func (m PUCBMapManager[K]) MaxKeys(c float64) []K {
	max := m.Max(c)
	total := omw.Sum(m.Trials()...)
	ks := make([]K, 0, len(m))
	for k, v := range m {
		if v.Get(total, c) == max {
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

type PUCT_LeafEvalY float64
type PUCT_LeafEvalFunc[S any] func(*S) PUCT_LeafEvalY

type PUCT_BackwardEvalY float64
type PUCT_BackwardEvalFunc[S any] func(PUCT_LeafEvalY, *S) PUCT_BackwardEvalY

type PUCT_EvalFunCaller[S any] struct {
	Leaf PUCT_LeafEvalFunc[S]
	Backward PUCT_BackwardEvalFunc[S]
}

type PUCT_FunCaller[S any, AS ~[]A, A comparable] struct {
	Game SequentialGameFunCaller[S, AS, A]
	Policy ActionPolicyFunc[S, A]
	Eval PUCT_EvalFunCaller[S]
}

func (f *PUCT_FunCaller[S, AS, A]) NewNode(state *S) *PUCT_Node[S, A] {
	py := f.Policy(state)
	m := PUCBMapManager[A]{}
	for a, p := range py {
		m[a] = &UtilPUCB{P:p}
	}
	return &PUCT_Node[S, A]{State:*state, PUCBManager:m}
}

func (f *PUCT_FunCaller[S, AS, A]) SetNoPolicy() {
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

type PUCT_Node[S any, A comparable] struct {
	State S
	PUCBManager PUCBMapManager[A]
	NextNodes PUCT_Nodes[S, A]
	SelectCount     int
}

func (node *PUCT_Node[S, A]) Trial() int {
	return omw.Sum(node.PUCBManager.Trials()...)
}

func (node *PUCT_Node[S, A]) ActionPrediction(r *rand.Rand, cap_ int) []A {
	y := make([]A, 0, cap_)
	for {
		if len(node.PUCBManager) == 0 {
			break
		}

		action := omw.RandChoice(node.PUCBManager.MaxTrialKeys(), r)
		y = append(y, action)

		if len(node.NextNodes) == 0 {
			break
		}

		max := node.NextNodes[0].Trial()
		nextNode := node.NextNodes[0]

		for _, nn := range node.NextNodes[1:] {
			trial := nn.Trial()
			if trial > max {
				max = trial
				nextNode = nn
			}
		}
		node = nextNode
	}
	return y
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
		node.PUCBManager[action].AccumReward += float64(eval(y, &node.State))
		node.PUCBManager[action].Trial += 1
		node.SelectCount = 0
	}
}

type PUCT_Selector[S any, AS ~[]A, A comparable] struct {
	FunCaller PUCT_FunCaller[S, AS, A]
	Cap int
}

func (s *PUCT_Selector[S, AS, A]) SelectAndExpansion(node *PUCT_Node[S, A], allNodes PUCT_Nodes[S, A], c float64, r *rand.Rand) (S, PUCT_Nodes[S, A], PUCT_Selects[S, A]) {
	f := s.FunCaller
	state := node.State
	selects := make(PUCT_Selects[S, A], 0, s.Cap)

	for {
		action := omw.RandChoice(node.PUCBManager.MaxKeys(c), r)
		selects = append(selects, PUCT_Select[S, A]{Node:node, Action:action})

		node.SelectCount += 1

		state = f.Game.Push(state, action)
		stateP := &state

		if isEnd := f.Game.IsEnd(stateP); isEnd {
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
				nextNode = f.NewNode(stateP)
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

type PUCT[S any, AS ~[]A, A comparable] struct {
	FunCaller PUCT_FunCaller[S, AS, A]
}

func (p *PUCT[S, AS, A]) Run(simulation int, rootState S, c float64, r *rand.Rand) PUCT_Nodes[S, A] {
	f := p.FunCaller
	rootNode := f.NewNode(&rootState)
	allNodes := PUCT_Nodes[S, A]{rootNode}

	var leafState S
	var selects PUCT_Selects[S, A]
	selector := PUCT_Selector[S, AS, A]{FunCaller:f, Cap:1}

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

type DPUCT_FunCaller[S any, ASS ~[]AS, AS ~[]A, A comparable] struct {
	Game SimultaneousGameFunCaller[S, ASS, AS, A]
	Policies ActionPoliciesFunc[S, A]
	LeafEvals DPUCT_LeafEvalsFunc[S]
}

func (f *DPUCT_FunCaller[S, ASS, AS, A]) NewNode(state *S) *DPUCT_Node[S, A] {
	pys := f.Policies(state)
	ms := make(PUCBMapManagers[A], len(pys))

	for playerI, py := range pys {
		m := PUCBMapManager[A]{}
		for a, p := range py {
			m[a] = &UtilPUCB{P:p}
		}
		ms[playerI] = m
	}
	return &DPUCT_Node[S, A]{State:*state, PUCBManagers:ms}
}

func (f *DPUCT_FunCaller[S, ASS, AS, A]) SetNoPolicies() {
	f.Policies = func(state *S) ActionPolicYs[A] {
		legalActionss := f.Game.LegalActionss(state)
		ys := make(ActionPolicYs[A], len(legalActionss))
		for playerI, actions := range legalActionss {
			y := map[A]float64{}
			p := 1.0 / float64(len(actions))
			for _, action := range actions {
				y[action] = p
			}
			ys[playerI] = y
		}
		return ys
	}
}

type DPUCT_Node[S any, A comparable] struct {
	State S
	PUCBManagers PUCBMapManagers[A]
	NextNodes DPUCT_Nodes[S, A]
	Trial int
	SelectCount     int
}

func (node *DPUCT_Node[S, A]) ActionPrediction(r *rand.Rand, cap_ int) [][]A {
	y := make([][]A, 0, cap_)
	for {
		if len(node.PUCBManagers) == 0 {
			break
		}

		actions := make([]A, len(node.PUCBManagers))
		for playerI, m := range node.PUCBManagers {
			actions[playerI] = omw.RandChoice(m.MaxTrialKeys(), r)
		}
		y = append(y, actions)

		if len(node.NextNodes) == 0 {
			break
		}

		max := node.NextNodes[0].Trial
		nextNode := node.NextNodes[0]

		for _, nn := range node.NextNodes[1:] {
			trial := nn.Trial
			if trial > max {
				max = trial
				nextNode = nn
			}
		}
		node = nextNode
	}
	return y
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

type DPUCT_Select[S any, AS ~[]A,  A comparable] struct {
	Node *DPUCT_Node[S, A]
	Actions AS
}

type DPUCT_Selects[S any, AS ~[]A, A comparable] []DPUCT_Select[S, AS, A]

func (ss DPUCT_Selects[S, AS, A]) Backward(ys DPUCT_LeafEvalYs) {
	for _, s := range ss {
		node := s.Node
		actions := s.Actions
		for playerI, action := range actions {
			node.PUCBManagers[playerI][action].AccumReward += float64(ys[playerI])
			node.PUCBManagers[playerI][action].Trial += 1
		}
		node.SelectCount = 0
	}
}

type DPUCT_Selector[S any, ASS ~[]AS, AS ~[]A, A comparable] struct {
	FunCaller DPUCT_FunCaller[S, ASS, AS, A]
	Cap int
}

func (s *DPUCT_Selector[S, ASS, AS, A])SelectAndExpansion(simultaneous int, node *DPUCT_Node[S, A], allNodes DPUCT_Nodes[S, A], c float64, r *rand.Rand) (S, DPUCT_Nodes[S, A], DPUCT_Selects[S, AS, A]) {
	f := s.FunCaller
	state := node.State
	selects := make(DPUCT_Selects[S, AS, A], 0, s.Cap)

	for {
		actions := make([]A, simultaneous)
		for playerI, m := range node.PUCBManagers {
			actions[playerI] = omw.RandChoice(m.MaxKeys(c), r)
		}

		selects = append(selects, DPUCT_Select[S, AS, A]{Node:node, Actions:actions})
		node.SelectCount += 1
		node.Trial += 1

		state = f.Game.Push(state, actions...)
		stateP := &state

		if isEnd := f.Game.IsEnd(stateP); isEnd {
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
				nextNode = f.NewNode(stateP)
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

type DPUCT[S any, ASS ~[]AS, AS ~[]A, A comparable] struct {
	FunCaller DPUCT_FunCaller[S, ASS, AS, A]
}

func (d *DPUCT[S, ASS, AS, A]) Run(simulation int, rootState S, c float64, r *rand.Rand) DPUCT_Nodes[S, A] {
	rootNode := d.FunCaller.NewNode(&rootState)
	allNodes := DPUCT_Nodes[S, A]{rootNode}
	simultaneous := len(rootNode.PUCBManagers)

	var leafState S
	var selects DPUCT_Selects[S, AS, A]
	selector := DPUCT_Selector[S, ASS, AS, A]{FunCaller:d.FunCaller, Cap:1}

	for i := 0; i < simulation; i++ {
		leafState, allNodes, selects = selector.SelectAndExpansion(simultaneous, rootNode, allNodes, c, r)
		ys := d.FunCaller.LeafEvals(&leafState)
		selects.Backward(ys)
	}
	return allNodes
}