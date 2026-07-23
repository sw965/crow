package simultaneous

import (
	"fmt"
	"math/rand/v2"
	"slices"

	"github.com/sw965/crow/game"
	"github.com/sw965/omw/mathx/randx"
	"github.com/sw965/omw/parallel"
	"github.com/sw965/omw/slicesx"
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

	err := parallel.For(n, p, func(workerID, idx int) error {
		rng := rngs[workerID]
		state := inits[idx]
		numSteps := 0
		for {
			isEnd, err := e.IsTerminal(state)
			if err != nil {
				return err
			}

			if isEnd {
				break
			}

			if e.MaxSteps > 0 && numSteps >= e.MaxSteps {
				return fmt.Errorf("手数がMaxSteps(%d)に達してもゲームが終了しませんでした", e.MaxSteps)
			}

			legalActionsByAgent := e.Logic.LegalActionsByAgentFunc(state)
			if len(legalActionsByAgent) == 0 {
				return fmt.Errorf("ゲームが終了していないのに合法手がありません")
			}

			policyByAgent, _, err := accr.PolicyValueFunc(state, legalActionsByAgent)
			if err != nil {
				return err
			}

			jointAction := make(JointAction[Ac, Ag], len(e.Agents))
			for _, agent := range e.Agents {
				legalActions := legalActionsByAgent[agent]
				policy := policyByAgent[agent]

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

			state, err = e.Logic.TransitionFunc(state, jointAction)
			if err != nil {
				return err
			}
			numSteps++
		}
		finals[idx] = state
		return nil
	})
	return finals, err
}

func (e *Engine[S, Ac, Ag]) RecordPlayouts(inits []S, accr ActorCritic[S, Ac, Ag], rngs []*rand.Rand, initStepsCap int) ([]Record[S, Ac, Ag], error) {
	if err := e.Validate(); err != nil {
		return nil, err
	}

	n := len(inits)
	p := len(rngs)
	records := make([]Record[S, Ac, Ag], n)

	err := parallel.For(n, p, func(workerID, idx int) error {
		rng := rngs[workerID]
		state := inits[idx]
		steps := make([]Step[S, Ac, Ag], 0, initStepsCap)

		for {
			isEnd, err := e.IsTerminal(state)
			if err != nil {
				return err
			}
			if isEnd {
				break
			}

			if e.MaxSteps > 0 && len(steps) >= e.MaxSteps {
				return fmt.Errorf("手数がMaxSteps(%d)に達してもゲームが終了しませんでした", e.MaxSteps)
			}

			legalActionsByAgent := e.Logic.LegalActionsByAgentFunc(state)
			if len(legalActionsByAgent) == 0 {
				return fmt.Errorf("ゲームが終了していないのに合法手がありません")
			}

			policyByAgent, valueByAgent, err := accr.PolicyValueFunc(state, legalActionsByAgent)
			if err != nil {
				return err
			}

			jointAction := make(JointAction[Ac, Ag], len(e.Agents))
			for _, agent := range e.Agents {
				legalActions := legalActionsByAgent[agent]
				policy := policyByAgent[agent]

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
				State:         state,
				JointAction:   jointAction,
				PolicyByAgent: policyByAgent,
				ValueByAgent:  valueByAgent,
			})

			state, err = e.Logic.TransitionFunc(state, jointAction)
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

// NewCrossPlayoutRecorderは、複数のActorCriticを総当たりで対戦させる game.CrossPlayoutRecorder を返す。
// 総当たりの進行とスコアの集計は共通実装(game側)が担い、ここでは並び1組分の対戦の実行方法だけを定義する。
func (e *Engine[S, Ac, Ag]) NewCrossPlayoutRecorder(inits []S, accrs []ActorCritic[S, Ac, Ag], p int) (*game.CrossPlayoutRecorder[Record[S, Ac, Ag], Ag], error) {
	agentsN := len(e.Agents)
	if len(accrs) < agentsN {
		return nil, fmt.Errorf("ActorCriticが不足しています: len(accrs) = %d: %d 以上であるべき", len(accrs), agentsN)
	}

	accrPerms := slices.Collect(slicesx.Permutations(accrs, agentsN))
	rngs := randx.NewPCGs(p)

	accrNames := make([]game.ActorCriticName, len(accrs))
	for i, accr := range accrs {
		accrNames[i] = accr.Name
	}

	playPermutationFunc := func(permIdx, initStepsCap int) ([]Record[S, Ac, Ag], map[Ag]game.ActorCriticName, error) {
		accrPerm := accrPerms[permIdx]
		accrNameByAgent := make(map[Ag]game.ActorCriticName, agentsN)
		pvFuncByAgent := make(map[Ag]PolicyValueFunc[S, Ac, Ag], agentsN)
		selectFuncByAgent := make(map[Ag]game.SelectFunc[Ac, Ag], agentsN)

		for i, agent := range e.Agents {
			accr := accrPerm[i]
			accrNameByAgent[agent] = accr.Name
			pvFuncByAgent[agent] = accr.PolicyValueFunc
			selectFuncByAgent[agent] = accr.SelectFunc
		}

		pvFunc := func(state S, legalActionsByAgent LegalActionsByAgent[Ac, Ag]) (PolicyByAgent[Ac, Ag], ValueByAgent[Ag], error) {
			policyByAgent := make(PolicyByAgent[Ac, Ag], agentsN)
			valueByAgent := make(ValueByAgent[Ag], agentsN)

			for _, agent := range e.Agents {
				actorPolicyByAgent, actorValueByAgent, err := pvFuncByAgent[agent](state, legalActionsByAgent)
				if err != nil {
					return nil, nil, err
				}
				policyByAgent[agent] = actorPolicyByAgent[agent]
				valueByAgent[agent] = actorValueByAgent[agent]
			}
			return policyByAgent, valueByAgent, nil
		}

		selectFunc := func(p game.Policy[Ac], agent Ag, rng *rand.Rand) (Ac, error) {
			return selectFuncByAgent[agent](p, agent, rng)
		}

		wrapperActor := ActorCritic[S, Ac, Ag]{
			PolicyValueFunc: pvFunc,
			SelectFunc:      selectFunc,
		}

		records, err := e.RecordPlayouts(inits, wrapperActor, rngs, initStepsCap)
		if err != nil {
			return nil, nil, err
		}

		for i := range records {
			records[i].ActorCriticNameByAgent = accrNameByAgent
		}
		return records, accrNameByAgent, nil
	}

	resultScoreFromRecordFunc := func(r Record[S, Ac, Ag]) game.ResultScoreByAgent[Ag] {
		return r.ResultScoreByAgent
	}

	return game.NewCrossPlayoutRecorder(accrNames, len(accrPerms), len(inits), playPermutationFunc, resultScoreFromRecordFunc), nil
}

type Step[S any, Ac, Ag comparable] struct {
	State         S
	JointAction   JointAction[Ac, Ag]
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
			State:         step.State,
			JointAction:   step.JointAction,
			PolicyByAgent: step.PolicyByAgent,
			ValueByAgent:  newValueByAgent,
		}
	}
	return elmoSteps
}
