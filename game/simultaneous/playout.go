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

func (e *Engine[S, Ac, Ag]) Playouts(inits []S, accr ActorCritic[S, Ac, Ag], rngs []*rand.Rand) ([]S, error) {
	if err := e.Validate(); err != nil {
		return nil, err
	}

	if err := accr.Validate(); err != nil {
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

			legalActionsByAgent := e.Logic.LegalActionsByAgentFunc(state)
			if len(legalActionsByAgent) == 0 {
				return fmt.Errorf("game is not ended but no legal actions are available")
			}

			jointPolicy, _, err := accr.PolicyValueFunc(state, legalActionsByAgent)
			if err != nil {
				return err
			}

			jointAction := make(map[Ag]Ac, len(e.Agents))
			for _, agent := range e.Agents {
				legalActions := legalActionsByAgent[agent]
				policy := jointPolicy[agent]

				err = policy.ValidateForLegalActions(legalActions, false)
				if err != nil {
					return err
				}

				action, err := accr.SelectFunc(policy, agent, rng)
				if err != nil {
					return err
				}
				jointAction[agent] = action
			}

			state, err = e.Logic.ActionFunc(state, jointAction)
			if err != nil {
				return err
			}
		}
		finals[idx] = state
		return nil
	})
	return finals, err
}

func (e *Engine[S, Ac, Ag]) RecordPlayouts(inits []S, accr ActorCritic[S, Ac, Ag], rngs []*rand.Rand, stepCap int) ([]Record[S, Ac, Ag], error) {
	if err := e.Validate(); err != nil {
		return nil, err
	}

	n := len(inits)
	p := len(rngs)
	records := make([]Record[S, Ac, Ag], n)

	err := parallel.For(n, p, func(workerId, idx int) error {
		rng := rngs[workerId]
		state := inits[idx]
		steps := make([]Step[S, Ac, Ag], 0, stepCap)

		for {
			isEnd, err := e.IsEnd(state)
			if err != nil {
				return err
			}
			if isEnd {
				break
			}

			legalActionsByAgent := e.Logic.LegalActionsByAgentFunc(state)
			if len(legalActionsByAgent) == 0 {
				return fmt.Errorf("game is not ended but no legal actions are available")
			}

			jointPolicy, jointValue, err := accr.PolicyValueFunc(state, legalActionsByAgent)
			if err != nil {
				return err
			}

			jointAction := make(map[Ag]Ac, len(e.Agents))
			for _, agent := range e.Agents {
				legalActions := legalActionsByAgent[agent]
				policy := jointPolicy[agent]

				if err := policy.ValidateForLegalActions(legalActions, false); err != nil {
					return err
				}

				action, err := accr.SelectFunc(policy, agent, rng)
				if err != nil {
					return err
				}
				jointAction[agent] = action
			}

			steps = append(steps, Step[S, Ac, Ag]{
				State:       state,
				JointAction:   jointAction,
				PolicyByAgent: jointPolicy,
				ValueByAgent:  jointValue,
			})

			state, err = e.Logic.ActionFunc(state, jointAction)
			if err != nil {
				return err
			}
		}

		scores, err := e.EvaluateResultScoreByAgent(state)
		if err != nil {
			return err
		}

		records[idx] = Record[S, Ac, Ag]{
			Steps:              steps,
			FinalState:         state,
			ResultScoreByAgent: scores,
		}
		return nil
	})

	return records, err
}

type CrossPlayoutRecorder[S any, Ac, Ag comparable] struct {
	engine  *Engine[S, Ac, Ag]
	inits   []S
	accrPerms [][]ActorCritic[S, Ac, Ag]
	rands   []*rand.Rand
	stepCap int

	currentIdx         int
	numGames           int
	totalScoreByAccrName map[game.ActorCriticName]float32
	numGamesByAccrName   map[game.ActorCriticName]int
}

func (e *Engine[S, Ac, Ag]) NewCrossPlayoutRecorder(inits []S, accrs []ActorCritic[S, Ac, Ag], p int) (*CrossPlayoutRecorder[S, Ac, Ag], error) {
	agentsN := len(e.Agents)
	if len(accrs) < agentsN {
		return nil, fmt.Errorf("insufficient actors: expected at least %d, got %d", agentsN, len(accrs))
	}

	accrPerms := slices.Collect(slicesx.Permutations(accrs, agentsN))
	rands := randx.NewPCGs(p)

	totalScoreByAccrName := make(map[game.ActorCriticName]float32)
	numGamesByAccrName := make(map[game.ActorCriticName]int)
	for _, accr := range accrs {
		totalScoreByAccrName[accr.Name] = 0
		numGamesByAccrName[accr.Name] = 0
	}

	return &CrossPlayoutRecorder[S, Ac, Ag]{
		engine:             e,
		inits:              inits,
		accrPerms:          accrPerms,
		rands:              rands,
		stepCap:            256,
		totalScoreByAccrName: totalScoreByAccrName,
		numGamesByAccrName:   numGamesByAccrName,
	}, nil
}

func (cp *CrossPlayoutRecorder[S, Ac, Ag]) NumGames() int {
	return cp.numGames
}

func (cp *CrossPlayoutRecorder[S, Ac, Ag]) SetStepCap(c int) {
	cp.stepCap = c
}

func (cp *CrossPlayoutRecorder[S, Ac, Ag]) TotalScoreByActorCriticName() map[game.ActorCriticName]float32 {
	return maps.Clone(cp.totalScoreByAccrName)
}

func (cp *CrossPlayoutRecorder[S, Ac, Ag]) AverageScoreByActorCriticName() (map[game.ActorCriticName]float32, error) {
	if cp.numGames <= 0 {
		return nil, fmt.Errorf("ゲームがまだ行われていないので、平均スコアを計算出来ません。")
	}
	avg := make(map[game.ActorCriticName]float32, len(cp.totalScoreByAccrName))
	for k, v := range cp.totalScoreByAccrName {
		numGames := cp.numGamesByAccrName[k]
		if numGames > 0 {
			avg[k] = v / float32(numGames)
		} else {
			avg[k] = 0
		}
	}
	return avg, nil
}

func (cp *CrossPlayoutRecorder[S, Ac, Ag]) NumGamesByActorCriticName() map[game.ActorCriticName]int {
	return maps.Clone(cp.numGamesByAccrName)
}

func (cp *CrossPlayoutRecorder[S, Ac, Ag]) Next() ([]Record[S, Ac, Ag], bool, error) {
	if cp.currentIdx >= len(cp.accrPerms) {
		return nil, false, nil
	}

	agentsN := len(cp.engine.Agents)
	accrNameByAgent := make(map[Ag]game.ActorCriticName, agentsN)
	pvFuncByAgent := make(map[Ag]PolicyValueFunc[S, Ac, Ag], agentsN)
	selectFuncByAgent := make(map[Ag]game.SelectFunc[Ac, Ag], agentsN)
	accrPerm := cp.accrPerms[cp.currentIdx]

	for i, agent := range cp.engine.Agents {
		accr := accrPerm[i]
		accrNameByAgent[agent] = accr.Name
		pvFuncByAgent[agent] = accr.PolicyValueFunc
		selectFuncByAgent[agent] = accr.SelectFunc
	}

	pvFunc := func(state S, legalActionsByAgent LegalActionsByAgent[Ac, Ag]) (PolicyByAgent[Ac, Ag], ValueByAgent[Ag], error) {
		jp := make(PolicyByAgent[Ac, Ag], agentsN)
		jv := make(ValueByAgent[Ag], agentsN)

		for _, agent := range cp.engine.Agents {
			accrJP, accrJV, err := pvFuncByAgent[agent](state, legalActionsByAgent)
			if err != nil {
				return nil, nil, err
			}
			jp[agent] = accrJP[agent]
			jv[agent] = accrJV[agent]
		}
		return jp, jv, nil
	}

	selectFunc := func(p game.Policy[Ac], agent Ag, rng *rand.Rand) (Ac, error) {
		return selectFuncByAgent[agent](p, agent, rng)
	}

	wrapperActor := ActorCritic[S, Ac, Ag]{
		PolicyValueFunc: pvFunc,
		SelectFunc:      selectFunc,
	}

	records, err := cp.engine.RecordPlayouts(cp.inits, wrapperActor, cp.rands, cp.stepCap)
	if err != nil {
		return nil, false, err
	}

	for i := range records {
		records[i].ActorCriticNameByAgent = accrNameByAgent
	}

	// スコアの集計
	for _, record := range records {
		for agent, score := range record.ResultScoreByAgent {
			accrName := accrNameByAgent[agent]
			cp.totalScoreByAccrName[accrName] += score
			cp.numGamesByAccrName[accrName]++
		}
	}

	cp.currentIdx++
	cp.numGames += len(records)
	return records, true, nil
}

func (cp *CrossPlayoutRecorder[S, Ac, Ag]) Collect() ([]Record[S, Ac, Ag], error) {
	remainingPerms := len(cp.accrPerms) - cp.currentIdx
	if remainingPerms <= 0 {
		return nil, nil
	}

	c := remainingPerms * len(cp.inits)
	collected := make([]Record[S, Ac, Ag], 0, c)
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

type Step[S any, Ac, Ag comparable] struct {
	State       S
	// TODO 型定義しておく？
	JointAction   map[Ag]Ac
	PolicyByAgent PolicyByAgent[Ac, Ag]
	ValueByAgent  ValueByAgent[Ag]
}

type Record[S any, Ac, Ag comparable] struct {
	Steps                  []Step[S, Ac, Ag]
	FinalState             S
	ResultScoreByAgent     game.ResultScoreByAgent[Ag]
	ActorCriticNameByAgent map[Ag]game.ActorCriticName
}

func (r Record[S, Ac, Ag]) ElmoSteps(alpha float32) []Step[S, Ac, Ag] {
	if alpha < 0.0 {
		alpha = 0.0
	} else if alpha > 1.0 {
		alpha = 1.0
	}

	elmoSteps := make([]Step[S, Ac, Ag], len(r.Steps))

	for i, step := range r.Steps {
		newValueByAgent := make(ValueByAgent[Ag], len(step.ValueByAgent))

		for agent, v := range step.ValueByAgent {
			z := r.ResultScoreByAgent[agent]
			newValueByAgent[agent] = alpha*z + (1.0-alpha)*v
		}

		elmoSteps[i] = Step[S, Ac, Ag]{
			State:       step.State,
			JointAction:   step.JointAction,
			PolicyByAgent: step.PolicyByAgent,
			ValueByAgent:  newValueByAgent,
		}
	}
	return elmoSteps
}
