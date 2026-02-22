package simultaneous

import (
	"fmt"
	"github.com/sw965/omw/mathx/randx"
	"github.com/sw965/omw/parallel"
	"github.com/sw965/omw/slicesx"
	"maps"
	"math/rand/v2"
	"slices"
	"github.com/sw965/crow/game"
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

			legalMovesByAgent := e.Logic.LegalMovesByAgentFunc(state)
			if len(legalMovesByAgent) == 0 {
				return fmt.Errorf("game is not ended but no legal moves are available")
			}

			jointPolicy, _, err := ac.PolicyValueFunc(state, legalMovesByAgent)
			if err != nil {
				return err
			}

			jointAction := make(map[A]M, len(e.Agents))
			for _, agent := range e.Agents {
				legalMoves := legalMovesByAgent[agent]
				policy := jointPolicy[agent]

				err = policy.ValidateForLegalMoves(legalMoves, false)
				if err != nil {
					return err
				}

				move, err := ac.SelectFunc(policy, agent, rng)
				if err != nil {
					return err
				}
				jointAction[agent] = move
			}

			state, err = e.Logic.MoveFunc(state, jointAction)
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

			legalMovesByAgent := e.Logic.LegalMovesByAgentFunc(state)
			if len(legalMovesByAgent) == 0 {
				return fmt.Errorf("game is not ended but no legal moves are available")
			}

			jointPolicy, jointValue, err := ac.PolicyValueFunc(state, legalMovesByAgent)
			if err != nil {
				return err
			}

			jointAction := make(map[A]M, len(e.Agents))
			for _, agent := range e.Agents {
				legalMoves := legalMovesByAgent[agent]
				policy := jointPolicy[agent]

				if err := policy.ValidateForLegalMoves(legalMoves, false); err != nil {
					return err
				}

				move, err := ac.SelectFunc(policy, agent, rng)
				if err != nil {
					return err
				}
				jointAction[agent] = move
			}

			steps = append(steps, Step[S, M, A]{
				State:       state,
				JointMove:   jointAction,
				PolicyByAgent: jointPolicy,
				ValueByAgent:  jointValue,
			})

			state, err = e.Logic.MoveFunc(state, jointAction)
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
	totalScoreByAcName map[game.ActorCriticName]float32
	numGamesByAcName   map[game.ActorCriticName]int
}

func (e *Engine[S, M, A]) NewCrossPlayoutRecorder(inits []S, acs []ActorCritic[S, M, A], p int) (*CrossPlayoutRecorder[S, M, A], error) {
	agentsN := len(e.Agents)
	if len(acs) < agentsN {
		return nil, fmt.Errorf("insufficient actors: expected at least %d, got %d", agentsN, len(acs))
	}

	acPerms := slices.Collect(slicesx.Permutations(acs, agentsN))
	rands := randx.NewPCGs(p)

	totalScoreByAcName := make(map[game.ActorCriticName]float32)
	numGamesByAcName := make(map[game.ActorCriticName]int)
	for _, ac := range acs {
		totalScoreByAcName[ac.Name] = 0
		numGamesByAcName[ac.Name] = 0
	}

	return &CrossPlayoutRecorder[S, M, A]{
		engine:             e,
		inits:              inits,
		acPerms:            acPerms,
		rands:              rands,
		stepCap:            256,
		totalScoreByAcName: totalScoreByAcName,
		numGamesByAcName:   numGamesByAcName,
	}, nil
}

func (cp *CrossPlayoutRecorder[S, M, A]) NumGames() int {
	return cp.numGames
}

func (cp *CrossPlayoutRecorder[S, M, A]) SetStepCap(c int) {
	cp.stepCap = c
}

func (cp *CrossPlayoutRecorder[S, M, A]) TotalScoreByActorCriticName() map[game.ActorCriticName]float32 {
	return maps.Clone(cp.totalScoreByAcName)
}

func (cp *CrossPlayoutRecorder[S, M, A]) AverageScoreByActorCriticName() (map[game.ActorCriticName]float32, error) {
	if cp.numGames <= 0 {
		return nil, fmt.Errorf("ゲームがまだ行われていないので、平均スコアを計算出来ません。")
	}
	avg := make(map[game.ActorCriticName]float32, len(cp.totalScoreByAcName))
	for k, v := range cp.totalScoreByAcName {
		numGames := cp.numGamesByAcName[k]
		if numGames > 0 {
			avg[k] = v / float32(numGames)
		} else {
			avg[k] = 0
		}
	}
	return avg, nil
}

func (cp *CrossPlayoutRecorder[S, M, A]) NumGamesByActorCriticName() map[game.ActorCriticName]int {
	return maps.Clone(cp.numGamesByAcName)
}

func (cp *CrossPlayoutRecorder[S, M, A]) Next() ([]Record[S, M, A], bool, error) {
	if cp.currentIdx >= len(cp.acPerms) {
		return nil, false, nil
	}

	agentsN := len(cp.engine.Agents)
	acNameByAgent := make(map[A]game.ActorCriticName, agentsN)
	pvFuncByAgent := make(map[A]PolicyValueFunc[S, M, A], agentsN)
	selectFuncByAgent := make(map[A]game.SelectFunc[M, A], agentsN)
	acPerm := cp.acPerms[cp.currentIdx]

	for i, agent := range cp.engine.Agents {
		ac := acPerm[i]
		acNameByAgent[agent] = ac.Name
		pvFuncByAgent[agent] = ac.PolicyValueFunc
		selectFuncByAgent[agent] = ac.SelectFunc
	}

	pvFunc := func(state S, legalMovesByAgent LegalMovesByAgent[M, A]) (PolicyByAgent[M, A], ValueByAgent[A], error) {
		jp := make(PolicyByAgent[M, A], agentsN)
		jv := make(ValueByAgent[A], agentsN)

		for _, agent := range cp.engine.Agents {
			acJP, acJV, err := pvFuncByAgent[agent](state, legalMovesByAgent)
			if err != nil {
				return nil, nil, err
			}
			jp[agent] = acJP[agent]
			jv[agent] = acJV[agent]
		}
		return jp, jv, nil
	}

	selectFunc := func(p game.Policy[M], agent A, rng *rand.Rand) (M, error) {
		return selectFuncByAgent[agent](p, agent, rng)
	}

	wrapperActor := ActorCritic[S, M, A]{
		PolicyValueFunc: pvFunc,
		SelectFunc:      selectFunc,
	}

	records, err := cp.engine.RecordPlayouts(cp.inits, wrapperActor, cp.rands, cp.stepCap)
	if err != nil {
		return nil, false, err
	}

	for i := range records {
		records[i].ActorCriticNameByAgent = acNameByAgent
	}

	// スコアの集計
	for _, record := range records {
		for agent, score := range record.ResultScoreByAgent {
			acName := acNameByAgent[agent]
			cp.totalScoreByAcName[acName] += score
			cp.numGamesByAcName[acName]++
		}
	}

	cp.currentIdx++
	cp.numGames += len(records)
	return records, true, nil
}

func (cp *CrossPlayoutRecorder[S, M, A]) Collect() ([]Record[S, M, A], error) {
	remainingPerms := len(cp.acPerms) - cp.currentIdx
	if remainingPerms <= 0 {
		return nil, nil
	}

	c := remainingPerms * len(cp.inits)
	collected := make([]Record[S, M, A], 0, c)
	for {
		records, hasNext, err := cp.Next()
		if err != nil {
			return nil, err
		}
		if !hasNext {
			break
		}
		collected = append(collected, records...)
	}
	return collected, nil
}

type Step[S any, M, A comparable] struct {
	State       S
	JointMove   map[A]M
	PolicyByAgent PolicyByAgent[M, A]
	ValueByAgent  ValueByAgent[A]
}

type Record[S any, M, A comparable] struct {
	Steps                  []Step[S, M, A]
	FinalState             S
	ResultScoreByAgent     game.ResultScoreByAgent[A]
	ActorCriticNameByAgent map[A]game.ActorCriticName
}

func (r Record[S, M, A]) ElmoSteps(alpha float32) []Step[S, M, A] {
	if alpha < 0.0 {
		alpha = 0.0
	} else if alpha > 1.0 {
		alpha = 1.0
	}

	elmoSteps := make([]Step[S, M, A], len(r.Steps))

	for i, step := range r.Steps {
		newValueByAgent := make(ValueByAgent[A], len(step.ValueByAgent))

		for agent, v := range step.ValueByAgent {
			z := r.ResultScoreByAgent[agent]
			newValueByAgent[agent] = alpha*z + (1.0-alpha)*v
		}

		elmoSteps[i] = Step[S, M, A]{
			State:       step.State,
			JointMove:   step.JointMove,
			PolicyByAgent: step.PolicyByAgent,
			ValueByAgent:  newValueByAgent,
		}
	}
	return elmoSteps
}
