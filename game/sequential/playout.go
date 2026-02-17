package sequential

import (
	"fmt"
	"github.com/sw965/omw/mathx/randx"
	"github.com/sw965/omw/parallel"
	"github.com/sw965/omw/slicesx"
	"math/rand/v2"
	"slices"
)

func (e *Engine[S, M, A]) Playouts(inits []S, ac ActorCritic[S, M, A], rngs []*rand.Rand) ([]S, error) {
	if err := e.Validate(); err != nil {
		return nil, err
	}

	if err := ac.Validate(); err != nil {
		return nil, err
	}

	n := len(inits)
	p := len(rngs)
	finals := make([]S, n)

	err := parallel.For(n, p, func(workerId, idx int) error {
		rng := rngs[workerId]
		state := inits[idx]
		for {
			isEnd, err := e.IsEnd(state)
			if err != nil {
				return err
			}

			if isEnd {
				break
			}

			legalMoves := e.Logic.LegalMovesFunc(state)
			// policy.ValidateForLegalMovesでもlegalMovesの空チェックをするが、PolicyFuncを安全に呼ぶ為に、ここでもチェックする
			if len(legalMoves) == 0 {
				return fmt.Errorf("game is not ended but no legal moves are available")
			}

			policy, _, err := ac.PolicyValueFunc(state, legalMoves)
			if err != nil {
				return err
			}

			// legalMovesがユニークならば、policyは合法手のみを持つ事が保障される
			// 第2引数がtrueならば、legalMovesがユニーク性をチェックするが、一手毎にチェックするのは、計算コストの観点から見送る
			err = policy.ValidateForLegalMoves(legalMoves, false)
			if err != nil {
				return err
			}

			agent := e.Logic.CurrentAgentFunc(state)
			move, err := ac.SelectFunc(policy, agent, rng)
			if err != nil {
				return err
			}

			state, err = e.Logic.MoveFunc(state, move)
			if err != nil {
				return err
			}
		}
		finals[idx] = state
		return nil
	})
	return finals, err
}

func (e *Engine[S, M, A]) RecordPlayouts(inits []S, ac ActorCritic[S, M, A], rngs []*rand.Rand, stepCap int) ([]Record[S, M, A], error) {
	if err := e.Validate(); err != nil {
		return nil, err
	}

	n := len(inits)
	p := len(rngs)
	records := make([]Record[S, M, A], n)

	err := parallel.For(n, p, func(workerId, idx int) error {
		rng := rngs[workerId]
		state := inits[idx]
		steps := make([]Step[S, M, A], 0, stepCap)

		for {
			isEnd, err := e.IsEnd(state)
			if err != nil {
				return err
			}
			if isEnd {
				break
			}

			legalMoves := e.Logic.LegalMovesFunc(state)
			if len(legalMoves) == 0 {
				return fmt.Errorf("game is not ended but no legal moves are available")
			}

			policy, value, err := ac.PolicyValueFunc(state, legalMoves)
			if err != nil {
				return err
			}

			if err := policy.ValidateForLegalMoves(legalMoves, false); err != nil {
				return err
			}

			agent := e.Logic.CurrentAgentFunc(state)
			move, err := ac.SelectFunc(policy, agent, rng)
			if err != nil {
				return err
			}

			steps = append(steps, Step[S, M, A]{
				State:  state,
				Agent:  agent,
				Move:   move,
				Policy: policy,
				Value:  value,
			})

			state, err = e.Logic.MoveFunc(state, move)
			if err != nil {
				return err
			}
		}

		scores, err := e.EvaluateResultScoreByAgent(state)
		if err != nil {
			return err
		}

		records[idx] = Record[S, M, A]{
			Steps:              steps,
			FinalState:         state,
			ResultScoreByAgent: scores,
		}
		return nil
	})

	return records, err
}

type CrossPlayoutRecorder[S any, M, A comparable] struct {
	engine  *Engine[S, M, A]
	inits   []S
	acPerms [][]ActorCritic[S, M, A]
	rands   []*rand.Rand
	stepCap int

	currentIdx         int
	numGames           int
	totalScoreByAcName map[ActorCriticName]float32
}

func (e *Engine[S, M, A]) NewCrossPlayoutRecorder(inits []S, acs []ActorCritic[S, M, A], p int) (*CrossPlayoutRecorder[S, M, A], error) {
	agentsN := len(e.Agents)
	if len(acs) < agentsN {
		return nil, fmt.Errorf("insufficient actors: expected at least %d, got %d", agentsN, len(acs))
	}

	acPerms := slices.Collect(slicesx.Permutations(acs, agentsN))
	rands := randx.NewPCGs(p)
	return &CrossPlayoutRecorder[S, M, A]{
		engine:             e,
		inits:              inits,
		acPerms:            acPerms,
		rands:              rands,
		stepCap:            256,
		totalScoreByAcName: make(map[ActorCriticName]float32),
	}, nil
}

func (cp *CrossPlayoutRecorder[S, M, A]) NumGames() int {
	return cp.numGames
}

func (cp *CrossPlayoutRecorder[S, M, A]) SetStepCap(c int) {
	cp.stepCap = c
}

func (cp *CrossPlayoutRecorder[S, M, A]) TotalScoreByActorCriticName() map[ActorCriticName]float32 {
	return cp.totalScoreByAcName
}

func (cp *CrossPlayoutRecorder[S, M, A]) Next() ([]Record[S, M, A], map[A]ActorCriticName, bool, error) {
	if cp.currentIdx >= len(cp.acPerms) {
		return nil, nil, false, nil
	}

	acPerm := cp.acPerms[cp.currentIdx]
	cp.currentIdx++

	acNameByAgent := map[A]ActorCriticName{}
	pvFuncByAgent := map[A]PolicyValueFunc[S, M]{}
	selectFuncByAgent := map[A]SelectFunc[M, A]{}

	for i, agent := range cp.engine.Agents {
		ac := acPerm[i]
		acNameByAgent[agent] = ac.Name
		pvFuncByAgent[agent] = ac.PolicyValueFunc
		selectFuncByAgent[agent] = ac.SelectFunc
	}

	// 現在の手番の Agent に応じて ActorCritic の振る舞いを切り替えるラッパー
	pvFunc := func(state S, legalMoves []M) (Policy[M], float32, error) {
		agent := cp.engine.Logic.CurrentAgentFunc(state)
		return pvFuncByAgent[agent](state, legalMoves)
	}

	selectFunc := func(p Policy[M], agent A, rng *rand.Rand) (M, error) {
		return selectFuncByAgent[agent](p, agent, rng)
	}

	wrapperActor := ActorCritic[S, M, A]{
		PolicyValueFunc: pvFunc,
		SelectFunc:      selectFunc,
	}

	// Playouts ではなく RecordPlayouts を実行
	records, err := cp.engine.RecordPlayouts(cp.inits, wrapperActor, cp.rands, cp.stepCap)
	if err != nil {
		return nil, nil, false, err
	}

	// スコアの集計
	for _, record := range records {
		for agent, score := range record.ResultScoreByAgent {
			acName := acNameByAgent[agent]
			cp.totalScoreByAcName[acName] += score
		}
	}

	cp.numGames += len(records)
	return records, acNameByAgent, true, nil
}

func (cp *CrossPlayoutRecorder[S, M, A]) AverageScoreByActorCriticName() (map[ActorCriticName]float32, error) {
	if cp.numGames <= 0 {
		return nil, fmt.Errorf("ゲームがまだ行われていないので、平均スコアを計算出来ません。")
	}
	avg := map[ActorCriticName]float32{}
	for k, v := range cp.totalScoreByAcName {
		avg[k] = v / float32(cp.numGames)
	}
	return avg, nil
}

type Step[S any, M, A comparable] struct {
	State  S
	Agent  A
	Move   M
	Policy Policy[M]
	Value  float32
}

type Record[S any, M, A comparable] struct {
	Steps              []Step[S, M, A]
	FinalState         S
	ResultScoreByAgent ResultScoreByAgent[A]
}
