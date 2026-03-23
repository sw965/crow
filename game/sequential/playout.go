package sequential

import (
	"fmt"
	"github.com/sw965/crow/game"
	"github.com/sw965/omw/mathx/randx"
	"github.com/sw965/omw/parallel"
	"github.com/sw965/omw/slicesx"
	"maps"
	"math/rand/v2"
	"slices"
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

			legalActions := e.Logic.LegalActionsFunc(state)
			// policy.ValidateForLegalActionsでもlegalActionsの空チェックをするが、PolicyFuncを安全に呼ぶ為に、ここでもチェックする
			if len(legalActions) == 0 {
				return fmt.Errorf("game is not ended but no legal actions are available")
			}

			policy, _, err := accr.PolicyValueFunc(state, legalActions)
			if err != nil {
				return err
			}

			// legalActionsがユニークならば、policyは合法手のみを持つ事が保障される
			// 第2引数がtrueならば、legalActionsがユニーク性をチェックするが、一手毎にチェックするのは、計算コストの観点から見送る
			err = policy.ValidateForLegalActions(legalActions, false)
			if err != nil {
				return err
			}

			agent := e.Logic.CurrentAgentFunc(state)
			action, err := accr.SelectFunc(policy, agent, rng)
			if err != nil {
				return err
			}

			state, err = e.Logic.TransitionFunc(state, action)
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

			legalActions := e.Logic.LegalActionsFunc(state)
			if len(legalActions) == 0 {
				return fmt.Errorf("game is not ended but no legal actions are available")
			}

			policy, value, err := accr.PolicyValueFunc(state, legalActions)
			if err != nil {
				return err
			}

			if err := policy.ValidateForLegalActions(legalActions, false); err != nil {
				return err
			}

			agent := e.Logic.CurrentAgentFunc(state)
			action, err := accr.SelectFunc(policy, agent, rng)
			if err != nil {
				return err
			}

			steps = append(steps, Step[S, Ac, Ag]{
				State:  state,
				Agent:  agent,
				Action: action,
				Policy: policy,
				Value:  value,
			})

			state, err = e.Logic.TransitionFunc(state, action)
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
	pvFuncByAgent := make(map[Ag]PolicyValueFunc[S, Ac], agentsN)
	selectFuncByAgent := make(map[Ag]game.SelectFunc[Ac, Ag], agentsN)
	accrPerm := cp.accrPerms[cp.currentIdx]

	for i, agent := range cp.engine.Agents {
		accr := accrPerm[i]
		accrNameByAgent[agent] = accr.Name
		pvFuncByAgent[agent] = accr.PolicyValueFunc
		selectFuncByAgent[agent] = accr.SelectFunc
	}

	pvFunc := func(state S, legalActions []Ac) (game.Policy[Ac], float32, error) {
		agent := cp.engine.Logic.CurrentAgentFunc(state)
		return pvFuncByAgent[agent](state, legalActions)
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
	State  S
	Agent  Ag
	Action Ac
	Policy game.Policy[Ac]
	Value  float32
}

type Record[S any, Ac, Ag comparable] struct {
	Steps                  []Step[S, Ac, Ag]
	FinalState             S
	ResultScoreByAgent     game.ResultScoreByAgent[Ag]
	ActorCriticNameByAgent map[Ag]game.ActorCriticName
}

func (r Record[S, Ac, Ag]) ElmoSteps(alpha float32) []Step[S, Ac, Ag] {
	// alpha が範囲外の場合はクリッピングするか、呼び出し側の責任とする
	if alpha < 0.0 {
		alpha = 0.0
	} else if alpha > 1.0 {
		alpha = 1.0
	}

	elmoSteps := make([]Step[S, Ac, Ag], len(r.Steps))

	for i, step := range r.Steps {
		// 1. そのステップの手番エージェントの、実際のゲーム結果(Z)を取得
		z := r.ResultScoreByAgent[step.Agent]

		// 2. そのステップでの探索評価値(V_search)を取得
		v := step.Value

		// 3. ブレンドして新しいターゲット価値を計算
		newValue := alpha*z + (1.0-alpha)*v

		// 4. 新しい Step を作成 (Policy は元のマップの参照をそのまま使い、メモリを節約)
		elmoSteps[i] = Step[S, Ac, Ag]{
			State:  step.State,
			Agent:  step.Agent,
			Action: step.Action,
			Policy: step.Policy,
			Value:  newValue,
		}
	}
	return elmoSteps
}
