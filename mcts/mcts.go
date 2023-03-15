package mcts

import (
	"fmt"
	"math"
	"math/rand"
	"github.com/sw965/omw"
)

func GetPUCB(p, v float64, n, a int, X float64) float64 {
	return v + (p * X * math.Sqrt(float64(n)) / float64(a+1))
}

type PUCB struct {
	P      float64
	AccumReward float64
	Trial       int
}

func (p *PUCB) AverageReward() float64 {
	return float64(p.AccumReward) / float64(p.Trial+1)
}

func (p *PUCB) Get(totalTrial int, X float64) float64 {
	v := p.AverageReward()
	return GetPUCB(p.P, v, totalTrial, p.Trial, X)
}

type MaPUCB[K comparable] map[K]*PUCB

func (mp MaPUCB[K]) Trials() []int {
	y := make([]int, 0, len(mp))
	for _, v := range mp {
		y = append(y, v.Trial)
	}
	return y
}
func (mp MaPUCB[K]) Max(X float64) float64 {
	total := omw.Sum(mp.Trials()...)
	y := make([]float64, 0, len(mp))
	for _, v := range mp {
		y = append(y, v.Get(total, X))
	}
	return omw.Max(y...)
}

func (mp MaPUCB[K]) MaxKeys(X float64) []K {
	max := mp.Max(X)
	total := omw.Sum(mp.Trials()...)
	ks := make([]K, 0, len(mp))
	for k, v := range mp {
		a := v.Get(total, X)
		if a == max {
			ks = append(ks, k)
		}
	}
	return ks
}

func (mp MaPUCB[K]) MaxTrialKeys(X float64) []K {
	max := omw.Max(mp.Trials()...)
	ks := make([]K, 0, len(mp))
	for k, v := range mp {
		if v.Trial == max {
			ks = append(ks, k)
		}
	}
	return ks 
}

type MaPUCBs[K comparable] []MaPUCB[K]

func NewMaPUCBs[S any, A comparable](state *S, policies Policies[S, A]) MaPUCBs[A] {
	psy := policies(state)
	y := make(MaPUCBs[A], len(psy))
	for i, py := range psy {
		y[i] = MaPUCB[A]{}
		for a, p := range py {
			y[i][a] = &PUCB{P:p}
		}
	}
	return y
}

type NodeID int
type PlayerNumber int

type Policies[S any, A comparable] func(*S) PoliciesY[A]
type PolicyY[A comparable] map[A]float64
type PoliciesY[A comparable] []PolicyY[A]

type LeafEval[S any] func(*S) float64
type BackwardEval func(float64, NodeID, PlayerNumber) float64

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
	Policies Policies[S, A]
	State StateFunc[S, A]
	GetNodeID GetNodeID[S]
}

func (f *Func[S, A]) NewNode(state *S) *Node[S, A] {
	id := f.GetNodeID(state)
	return &Node[S, A]{State:*state, MaPUCBs:NewMaPUCBs(state, f.Policies), ID:id}
}

type Node[S any, A comparable] struct {
	State S
	MaPUCBs MaPUCBs[A]
	NextNodes Nodes[S, A]
	SelectCount     int
	ID NodeID
}

func (node *Node[S, A])SelectAndExpansion(allNodes Nodes[S, A], f *Func[S, A], X float64, r *rand.Rand, simultaneousMove, capSize int) (S, Nodes[S, A], Selectss[S, A], error) {
	selectss := make(Selectss[S, A], 0, capSize)
	state := node.State
	var err error

	for {
		actions := make([]A, simultaneousMove)
		for i, mp := range node.MaPUCBs {
			actions[i] = omw.RandomChoice(mp.MaxKeys(X), r)
		}

		selects := make(Selects[S, A], simultaneousMove)
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
				nextNode = f.NewNode(&state)
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

func (nodes Nodes[S, A]) Find(state *S, equal EqualState[S]) (*Node[S, A], error) {
	for _, v := range nodes {
		if equal(&v.State, state) {
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

func (ss Selects[S, A]) Backward(leafEvalY float64, eval BackwardEval) {
	for i, s := range ss {
		node := s.Node
		action := s.Action
		node.MaPUCBs[i][action].AccumReward += eval(leafEvalY, node.ID, PlayerNumber(i))
		node.MaPUCBs[i][action].Trial += 1
		node.SelectCount = 0
	}
}

type Selectss[S any, A comparable] []Selects[S, A]

func (sss Selectss[S, A]) Backward(leafEvalY float64, eval BackwardEval) {
	for _, ss := range sss {
		ss.Backward(leafEvalY, eval)
	}
}

func Run[S any, A comparable](simulation int, rootState S, f *Func[S, A], X float64, r *rand.Rand) (Nodes[S, A], error) {
	rootNode := f.NewNode(&rootState)
	simultaneousMove := len(rootNode.MaPUCBs)
	allNodes := Nodes[S, A]{rootNode}

	node := rootNode
	var leafState S
	var selectss Selectss[S, A]
	capSize := 0
	var err error

	for i := 0; i < simulation; i++ {
		leafState, allNodes, selectss, err = node.SelectAndExpansion(allNodes, f, X, r, simultaneousMove, capSize + 1)
		if err != nil {
			return Nodes[S, A]{}, err
		}
		capSize = len(selectss)

		leafEvalY := f.Eval.Leaf(&leafState)
		if err != nil {
			return Nodes[S, A]{}, err
		}

		selectss.Backward(leafEvalY, f.Eval.Backward)
		node = rootNode
	}
	return allNodes, nil
}