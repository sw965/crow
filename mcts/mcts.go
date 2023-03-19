package mcts

import (
	"fmt"
	"math/rand"
	"github.com/sw965/omw"
	"github.com/sw965/crow"
)

type UCB struct {
	AccumReward float64
	Trial       int
	Func crow.UCBFunc
}

func (u *UCB) AverageReward() float64 {
	return float64(u.AccumReward) / float64(u.Trial+1)
}

func (u *UCB) Get(totalTrial int) float64 {
	v := u.AverageReward()
	return u.Func(v, totalTrial, u.Trial)
}

type KUCB[K comparable] map[K]*UCB

func (m KUCB[K]) Trials() []int {
	y := make([]int, 0, len(m))
	for _, v := range m {
		y = append(y, v.Trial)
	}
	return y
}
func (m KUCB[K]) Max() float64 {
	total := omw.Sum(m.Trials()...)
	y := make([]float64, 0, len(m))
	for _, v := range m {
		y = append(y, v.Get(total))
	}
	return omw.Max(y...)
}

func (m KUCB[K]) MaxKeys() []K {
	max := m.Max()
	total := omw.Sum(m.Trials()...)
	ks := make([]K, 0, len(m))
	for k, v := range m {
		a := v.Get(total)
		if a == max {
			ks = append(ks, k)
		}
	}
	return ks
}

func (m KUCB[K]) MaxTrialKeys() []K {
	max := omw.Max(m.Trials()...)
	ks := make([]K, 0, len(m))
	for k, v := range m {
		if v.Trial == max {
			ks = append(ks, k)
		}
	}
	return ks 
}

type KUCBs[K comparable] []KUCB[K]

func NewKUCBs[A comparable](policies Policies[A], c float64) KUCBs[A] {
	y := make(KUCBs[A], len(policies))
	for i, policy := range policies {
		y[i] = KUCB[A]{}
		for a, p := range policy {
			y[i][a] = &UCB{Func:crow.UpperConfidenceBound1(p * c)}
		}
	}
	return y
}

type NodeID int
type PlayerNumber int

type LeafEvalY float64
type BackwardEvalY float64
type LeafEval[S any] func(*S) LeafEvalY
type BackwardEval func(LeafEvalY, NodeID, PlayerNumber) BackwardEvalY

type Policy[A comparable] map[A]float64
type Policies[A comparable] []Policy[A]
type GetPolicies[S any, A comparable] func(*S) Policies[A]

type GetC[S any] func(*S) float64

type Eval[S any]  struct {
	Leaf LeafEval[S]
	Backward BackwardEval
}

type StatePush[S any, A comparable] func(S, ...A) (S, error)
type EqualState[S any] func(*S, *S) bool
type IsEndState[S any] func(*S) bool

type StateFunc[S any, A comparable] struct {
	Push StatePush[S, A]
	Equal EqualState[S]
	IsEnd IsEndState[S]
}

type GetNodeID[S any] func(*S) NodeID

type Func[S any, A comparable] struct {
	Eval Eval[S]
	GetPolicies GetPolicies[S, A]
	State StateFunc[S, A]
	GetNodeID GetNodeID[S]
}

func (f *Func[S, A]) NewNode(state *S, c float64) *Node[S, A] {
	policies := f.GetPolicies(state)
	id := f.GetNodeID(state)
	return &Node[S, A]{State:*state, KUCBs:NewKUCBs(policies, c), ID:id}
}

type Node[S any, A comparable] struct {
	State S
	KUCBs KUCBs[A]
	NextNodes Nodes[S, A]
	SelectCount     int
	ID NodeID
}

func (node *Node[S, A])SelectAndExpansion(allNodes Nodes[S, A], f *Func[S, A], c float64, r *rand.Rand, simultaneous, cap int) (S, Nodes[S, A], Selectss[S, A], error) {
	selectss := make(Selectss[S, A], 0, cap)
	state := node.State
	var err error

	for {
		actions := make([]A, simultaneous)
		for i, kucb := range node.KUCBs {
			actions[i] = omw.RandomChoice(kucb.MaxKeys(), r)
		}

		selects := make(Selects[S, A], simultaneous)
		for i, action := range actions {
			selects[i] = Select[S, A]{Node:node, Action:action}
		}
		selectss = append(selectss, selects)

		node.SelectCount += 1

		state, err = f.State.Push(state, actions...)
		if err != nil {
			var zero S
			return zero, Nodes[S, A]{}, Selectss[S, A]{}, err
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

func (nodes Nodes[S, A]) Find(state *S, eq EqualState[S]) (*Node[S, A], error) {
	for _, v := range nodes {
		if eq(&v.State, state) {
			return v, nil
		}
	}
	return &Node[S, A]{}, fmt.Errorf("一致するNodeが見つからなかった")
}

type Select[S any, A comparable] struct {
	Node *Node[S, A]
	Action A
}

type Selects[S any, A comparable] []Select[S, A]

func (ss Selects[S, A]) Backward(y LeafEvalY, eval BackwardEval) {
	for i, s := range ss {
		node := s.Node
		action := s.Action
		node.KUCBs[i][action].AccumReward += float64(eval(y, node.ID, PlayerNumber(i)))
		node.KUCBs[i][action].Trial += 1
		node.SelectCount = 0
	}
}

type Selectss[S any, A comparable] []Selects[S, A]

func (sss Selectss[S, A]) Backward(y LeafEvalY, eval BackwardEval) {
	for _, ss := range sss {
		ss.Backward(y, eval)
	}
}

func Run[S any, A comparable](simulation int, rootState S, f *Func[S, A], c float64, r *rand.Rand) (Nodes[S, A], error) {
	rootNode := f.NewNode(&rootState, c)
	allNodes := Nodes[S, A]{rootNode}
	simultaneous := len(rootNode.KUCBs)

	var leafState S
	var selectss Selectss[S, A]
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

		selectss.Backward(y, f.Eval.Backward)
	}
	return allNodes, nil
}