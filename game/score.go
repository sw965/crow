package game

import (
	"fmt"
	"sort"
)

// ゲームが終了していない場合は、空あるいはnilにする
type RankByAgent[Ag comparable] map[Ag]int

func NewRankByAgent[Ag comparable](agentsPerRank [][]Ag) (RankByAgent[Ag], error) {
	ranks := RankByAgent[Ag]{}
	rank := 1
	for i, agents := range agentsPerRank {
		if len(agents) == 0 {
			return nil, fmt.Errorf("各順位に1体以上のエージェントが必要: len(agentsPerRank[%d]) == 0", i)
		}

		for _, agent := range agents {
			if _, ok := ranks[agent]; ok {
				return nil, fmt.Errorf("エージェントが重複している: agent = %v", agent)
			}
			ranks[agent] = rank
		}
		rank += len(agents)
	}
	return ranks, nil
}

func (r RankByAgent[Ag]) Validate() error {
	n := len(r)
	if n == 0 {
		return nil
	}

	ranks := make([]int, 0, n)
	for _, rank := range r {
		if rank < 1 {
			return fmt.Errorf("rank >= 1 であるべき: rank = %d", rank)
		}
		ranks = append(ranks, rank)
	}
	sort.Ints(ranks)

	current := ranks[0]
	if current != 1 {
		return fmt.Errorf("最小のrankは1であるべき: rank = %d", current)
	}
	expected := current + 1

	for i := 1; i < len(ranks); i++ {
		rank := ranks[i]
		switch rank {
		case current:
			// 同順の場合
			expected++
		case expected:
			// 順位が切り替わった場合
			current = rank
			expected = rank + 1
		default:
			return fmt.Errorf("順位の並びが不正: ranks[%d] = %d(同順) または %d であるべき: ranks = %v", i, current, expected, ranks)
		}
	}
	return nil
}

type RankByAgentFunc[S any, Ag comparable] func(S) (RankByAgent[Ag], error)
type ResultScoreByAgent[Ag comparable] map[Ag]float32
type ResultScoreByAgentFunc[Ag comparable] func(RankByAgent[Ag]) (ResultScoreByAgent[Ag], error)

func StandardResultScoreByAgentFunc[Ag comparable](ranks RankByAgent[Ag]) (ResultScoreByAgent[Ag], error) {
	if err := ranks.Validate(); err != nil {
		return nil, err
	}

	n := len(ranks)
	scores := map[Ag]float32{}

	if n == 1 {
		for agent := range ranks {
			scores[agent] = 1.0
		}
		return scores, nil
	}

	counts := map[int]int{}
	for _, rank := range ranks {
		counts[rank]++
	}

	den := float32(n - 1)
	tieScore := func(r, k int) float32 {
		return 1.0 - float32(2*r+k-3)/(2.0*den)
	}

	for agent, r := range ranks {
		k := counts[r]
		scores[agent] = tieScore(r, k)
	}
	return scores, nil
}
