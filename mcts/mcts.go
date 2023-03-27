package mcts6

import (
	"fmt"
	"math"
	"math/rand"
	"github.com/sw965/omw"
)

type StatePushFunc[S any, A comparable] func (S, ...A) S
type StateEqualFunc[S any] func(*S, *S) bool
type IsEndStateFunc[S any] func(*S) bool
type LegalActionsByPlayerFunc[S any, A comparable] func(*S) [][]A
type EndStateScoreFunc[S any] func(*S) LeafEvalY

type StateFunc[S any, A comparable] struct {
	Push StatePushFunc[S, A]
	Equal StateEqualFunc[S]
	IsEnd IsEndStateFunc[S]
	LegalActionsByPlayer LegalActionsByPlayerFunc[S, A]
	EndScore EndStateScoreFunc[S]
}

type ActionByPlayerFunc[S any, A comparable] func(*S) []A

func NewRandomActionByPlayer[S any, A comparable](f *StateFunc[S, A], r *rand.Rand) ActionByPlayerFunc[S, A] {
	return func(state *S) []A {
		actionss := f.LegalActionsByPlayer(state)
		n := len(actionss)
		y := make([]A, n)
		for playerI, actions := range actionss {
			y[playerI] = omw.RandomChoice(actions, r)
		}
		return y
	}
}

func (player ActionByPlayerFunc[S, A]) Playout(state S, f *StateFunc[S, A]) S {
	for {
		if f.IsEnd(&state) {
			break
		}
		actions := player(&state)
		state = f.Push(state, actions...)
	}
	return state
}

type UCBFunc func(float64, int, int) float64

func UpperConfidenceBound1(c float64) UCBFunc {
	return func(v float64, n, a int) float64 {
		fn := float64(n)
		return v + (c * math.Sqrt(fn) / float64(a))
	}
}

type UCB struct {
	AccumReward float64
	Trial       int
	Func UCBFunc
}

func (u *UCB) AverageReward() float64 {
	return float64(u.AccumReward) / float64(u.Trial+1)
}

func (u *UCB) Get(totalTrial int) float64 {
	v := u.AverageReward()
	return u.Func(v, totalTrial, u.Trial)
}

type UCBByKey[K comparable] map[K]*UCB

func (u UCBByKey[K]) Trials() []int {
	y := make([]int, 0, len(u))
	for _, v := range u {
		y = append(y, v.Trial)
	}
	return y
}
func (u UCBByKey[K]) Max() float64 {
	total := omw.Sum(u.Trials()...)
	y := make([]float64, 0, len(u))
	for _, v := range u {
		y = append(y, v.Get(total))
	}
	return omw.Max(y...)
}

func (u UCBByKey[K]) MaxKeys() []K {
	max := u.Max()
	total := omw.Sum(u.Trials()...)
	ks := make([]K, 0, len(u))
	for k, v := range u {
		a := v.Get(total)
		if a == max {
			ks = append(ks, k)
		}
	}
	return ks
}

func (u UCBByKey[K]) MaxTrialKeys() []K {
	max := omw.Max(u.Trials()...)
	ks := make([]K, 0, len(u))
	for k, v := range u {
		if v.Trial == max {
			ks = append(ks, k)
		}
	}
	return ks 
}

type UCBByKeyByPlayer[K comparable] []UCBByKey[K]

func NewUCBByKeyByPlayer[A comparable](pss PolicYByActionByPlayer[A], c float64) UCBByKeyByPlayer[A] {
	y := make(UCBByKeyByPlayer[A], len(pss))
	for playerI, ps := range pss {
		y[playerI] = UCBByKey[A]{}
		for a, p := range ps {
			y[playerI][a] = &UCB{Func:UpperConfidenceBound1(p * c)}
		}
	}
	return y
}

type NodeID int
type LeafEvalY float64
type LeafEvalFunc[S any] func(*S) LeafEvalY

func NewPlayoutLeafEvalFunc[S any, A comparable](player ActionByPlayerFunc[S, A], f *StateFunc[S, A]) LeafEvalFunc[S] {
	eval := func(state *S) LeafEvalY {
		endState := player.Playout(*state, f)
		return f.EndScore(&endState)
	}
	return eval
}

type BackwardEvalYByNodeID map[NodeID]float64
type BackwardEvalYByNodeIDByPlayer []BackwardEvalYByNodeID
type BackwardEvalFuncByNodeIDByPlayer func(LeafEvalY) BackwardEvalYByNodeIDByPlayer

type PolicYByAction[A comparable] map[A]float64
type PolicYByActionByPlayer[A comparable] []PolicYByAction[A]
type PolicyFuncByActionByPlayer[S any, A comparable] func(*S) PolicYByActionByPlayer[A]

type EvalFunc[S any]  struct {
	Leaf LeafEvalFunc[S]
	BackwardByNodeIDByPlayer BackwardEvalFuncByNodeIDByPlayer
}

type SetNodeID[S any, A comparable] func(*Node[S, A])

type Func[S any, A comparable] struct {
	Eval EvalFunc[S]
	PolicyByActionByPlayer PolicyFuncByActionByPlayer[S, A]
	State StateFunc[S, A]
	SetNodeID SetNodeID[S, A]
}

func (f *Func[S, A]) NewNode(state *S, c float64) *Node[S, A] {
	pss := f.PolicyByActionByPlayer(state)
	y := &Node[S, A]{State:*state, UCBByKeyByPlayer:NewUCBByKeyByPlayer(pss, c)}
	f.SetNodeID(y)
	return y
}

type Node[S any, A comparable] struct {
	State S
	UCBByKeyByPlayer UCBByKeyByPlayer[A]
	NextNodes Nodes[S, A]
	SelectCount     int
	ID NodeID
}

func (node *Node[S, A])SelectAndExpansion(allNodes Nodes[S, A], f *Func[S, A], c float64, r *rand.Rand, simultaneous, cap int) (S, Nodes[S, A], SelectByPlayerByOrder[S, A], error) {
	selectss := make(SelectByPlayerByOrder[S, A], 0, cap)
	state := node.State
	var err error

	for {
		actions := make([]A, simultaneous)
		for playerI, u := range node.UCBByKeyByPlayer {
			actions[playerI] = omw.RandomChoice(u.MaxKeys(), r)
		}

		selects := make(SelectByPlayer[S, A], simultaneous)
		for playerI, action := range actions {
			selects[playerI] = Select[S, A]{Node:node, Action:action}
		}
		selectss = append(selectss, selects)

		node.SelectCount += 1

		state = f.State.Push(state, actions...)
		if err != nil {
			var zero S
			return zero, Nodes[S, A]{}, SelectByPlayerByOrder[S, A]{}, err
		}

		stateP := &state
		if f.State.IsEnd(stateP) {
			break
		}

		//nextNodesの中に、同じstateが存在するならば、それを次のNodeとする
		//nextNodesの中に、同じstateが存在しないなら、allNodesの中から同じstateが存在しないかを調べる。
		//allNodesの中に、同じstateが存在するならば、次回から高速に探索出来るように、nextNodesに追加して、次のnodeとする。
		//nextNodesにもallNodesにも同じstateが存在しないなら、新しくnodeを作り、
		//nextNodesと、allNodesに追加し、新しく作ったnodeを次のnodeとし、select処理を終了する。

		nextNode, err := node.NextNodes.Find(stateP, f.State.Equal)
		if err != nil {
			nextNode, err = allNodes.Find(stateP, f.State.Equal)
			if err == nil {
				node.NextNodes = append(node.NextNodes, nextNode)
			} else {
				nextNode = f.NewNode(&state, c)
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
	return state, allNodes, selectss, nil
}

type Nodes[S any, A comparable] []*Node[S, A]

func (nodes Nodes[S, A]) Find(state *S, eq StateEqualFunc[S]) (*Node[S, A], error) {
	for _, node := range nodes {
		if eq(&node.State, state) {
			return node, nil
		}
	}
	return &Node[S, A]{}, fmt.Errorf("一致するNodeが見つからなかった")
}

type Select[S any, A comparable] struct {
	Node *Node[S, A]
	Action A
}

type SelectByPlayer[S any, A comparable] []Select[S, A]

func (ss SelectByPlayer[S, A]) Backward(ys BackwardEvalYByNodeIDByPlayer) {
	for playerI, s := range ss {
		node := s.Node
		action := s.Action
		node.UCBByKeyByPlayer[playerI][action].AccumReward += float64(ys[playerI][node.ID])
		node.UCBByKeyByPlayer[playerI][action].Trial += 1
		node.SelectCount = 0
	}
}

type SelectByPlayerByOrder[S any, A comparable] []SelectByPlayer[S, A]

func (sss SelectByPlayerByOrder[S, A]) Backward(y LeafEvalY, eval BackwardEvalFuncByNodeIDByPlayer) {
	for _, ss := range sss {
		ys := eval(y)
		ss.Backward(ys)
	}
}

func Run[S any, A comparable](simulation int, rootState S, f *Func[S, A], c float64, r *rand.Rand) (Nodes[S, A], error) {
	rootNode := f.NewNode(&rootState, c)
	allNodes := Nodes[S, A]{rootNode}
	simultaneous := len(rootNode.UCBByKeyByPlayer)

	var leafState S
	var selectss SelectByPlayerByOrder[S, A]
	cap := 0
	var err error

	for i := 0; i < simulation; i++ {
		leafState, allNodes, selectss, err = rootNode.SelectAndExpansion(allNodes, f, c, r, simultaneous, cap)
		if err != nil {
			return Nodes[S, A]{}, err
		}
		cap = len(selectss)

		y := f.Eval.Leaf(&leafState)
		if err != nil {
			return Nodes[S, A]{}, err
		}

		selectss.Backward(y, f.Eval.BackwardByNodeIDByPlayer)
	}
	return allNodes, nil
}