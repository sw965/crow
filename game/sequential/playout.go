package sequential

import (
	"fmt"
	"github.com/sw965/omw/parallel"
	"github.com/sw965/omw/slicesx"
	"math/rand/v2"
	"slices"
)

func (e *Engine[S, M, A]) Playouts(inits []S, actor Actor[S, M, A], rngs []*rand.Rand) ([]S, error) {
	if err := e.Validate(); err != nil {
		return nil, err
	}

	if err := actor.Validate(); err != nil {
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

			policy, err := actor.PolicyFunc(state, legalMoves)
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
			move, err := actor.SelectFunc(policy, agent, rng)
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

func (e Engine[S, M, A]) RecordPlayouts(inits []S, actor ActorCritic[S, M, A], oneGameCap int, rngs []*rand.Rand) ([]Record[S, M, A], error) {
	if err := e.Validate(); err != nil {
		return nil, err
	}

	n := len(inits)
	p := len(rngs)
	records := make([]Record[S, M, A], n)

	err := parallel.For(n, p, func(workerId, idx int) error {
		rng := rngs[workerId]
		state := inits[idx]
		steps := make([]Step[S, M, A], 0, oneGameCap)

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

			policy, value, err := actor.PolicyValueFunc(state, legalMoves)
			if err != nil {
				return err
			}

			if err := policy.ValidateForLegalMoves(legalMoves, false); err != nil {
				return err
			}

			agent := e.Logic.CurrentAgentFunc(state)
			move, err := actor.SelectFunc(policy, agent, rng)
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

type CrossPlayouter[S any, M, A comparable] struct {
	engine     *Engine[S, M, A]
	inits      []S
	actors     []Actor[S, M, A]
	actorPerms [][]Actor[S, M, A]

	currentIdx       int
	ScoreByActorName map[ActorName]float32
	rngs             []*rand.Rand
}

func (e *Engine[S, M, A]) NewCrossPlayouter(inits []S, actors []Actor[S, M, A], rngs []*rand.Rand) (*CrossPlayouter[S, M, A], error) {
	agentsN := len(e.Agents)
	if len(actors) < agentsN {
		return nil, fmt.Errorf("insufficient actors: expected at least %d, got %d", agentsN, len(actors))
	}

	perms := slices.Collect(slicesx.Permutations(actors, agentsN))
	return &CrossPlayouter[S, M, A]{
		engine:           e,
		inits:            inits,
		actors:           actors,
		actorPerms:       perms,
		ScoreByActorName: make(map[ActorName]float32),
		rngs:             rngs,
	}, nil
}

func (cp *CrossPlayouter[S, M, A]) Next() ([]S, map[A]ActorName, bool, error) {
	if cp.currentIdx >= len(cp.actorPerms) {
		return nil, nil, false, nil
	}

	actorPerm := cp.actorPerms[cp.currentIdx]
	cp.currentIdx++

	actorByAgent := map[A]Actor[S, M, A]{}
	actorNameByAgent := map[A]ActorName{}
	policyFuncByAgent := map[A]PolicyFunc[S, M]{}
	selectFuncByAgent := map[A]SelectFunc[M, A]{}

	for i, agent := range cp.engine.Agents {
		actor := actorPerm[i]
		actorByAgent[agent] = actor
		actorNameByAgent[agent] = actor.Name
		policyFuncByAgent[agent] = actor.PolicyFunc
		selectFuncByAgent[agent] = actor.SelectFunc
	}

	policyFunc := func(state S, legalMoves []M) (Policy[M], error) {
		agent := cp.engine.Logic.CurrentAgentFunc(state)
		return policyFuncByAgent[agent](state, legalMoves)
	}

	selectFunc := func(p Policy[M], agent A, rng *rand.Rand) (M, error) {
		return selectFuncByAgent[agent](p, agent, rng)
	}

	newActor := Actor[S, M, A]{
		PolicyFunc: policyFunc,
		SelectFunc: selectFunc,
	}

	finals, err := cp.engine.Playouts(cp.inits, newActor, cp.rngs)
	if err != nil {
		return nil, nil, false, err
	}

	for _, final := range finals {
		scores, err := cp.engine.EvaluateResultScoreByAgent(final)
		if err != nil {
			return nil, nil, false, err
		}
		for agent, score := range scores {
			actorName := actorNameByAgent[agent]
			cp.ScoreByActorName[actorName] += score
		}
	}
	return finals, actorNameByAgent, true, nil
}

type CrossRecordPlayoutResult[S any, M, A comparable] struct {
	ActorByAgent map[A]ActorCritic[S, M, A]
	Records      []Record[S, M, A]
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
