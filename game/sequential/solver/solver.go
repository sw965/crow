package solver

import (
	"github.com/sw965/crow/game/sequential"
)

type Policy[A comparable] map[A]float64
type PolicyProvider[S any, As ~[]A, A comparable] func(*S, As) Policy[A]

func UniformPolicyProvider[S any, As ~[]A, A comparable](state *S, legalActions As) Policy[A] {
	n := len(legalActions)
	p := 1.0 / float64(n)
	policy := Policy[A]{}
	for _, a := range legalActions {
		policy[a] = p
	}
	return policy
}

type Eval float64

type ActorCritic[S any, As ~[]A, A comparable] func(*S, As) (A, Policy[A], Eval, error)

func PlayerToSolver[S any, As ~[]A, A comparable](player sequential.Player[S, As, A]) ActorCritic[S, As, A] {
	return func(state *S, legalActions As) (A, Policy[A], Eval, error) {
		action, err := player(state, legalActions)
		//Policy, Evalはゼロ値にする。
		return action, Policy[A]{}, 0.0, err
	}
}

func (ac ActorCritic[S, As, A]) ToPlayer() sequential.Player[S, As, A] {
	return func(state *S, legalActions As) (A, error) {
		//Policy, Evalは破棄する。
		action, _, _, err := ac(state, legalActions)
		return action, err
	}
}

type PerAgent[S any, As ~[]A, A, G comparable] map[G]ActorCritic[S, As, A]

func PlayerPerAgentToSolver[S any, As ~[]A, A, G comparable](players sequential.PlayerPerAgent[S, As, A, G]) PerAgent[S, As, A, G] {
	solvers := PerAgent[S, As, A, G]{}
	for k, player := range players {
		solvers[k] = PlayerToSolver(player)
	}
	return solvers
}

func (p PerAgent[S, As, A, G]) ToPlayer() sequential.PlayerPerAgent[S, As, A, G] {
	players := sequential.PlayerPerAgent[S, As, A, G]{}
	for k, solver := range p {
		players[k] = solver.ToPlayer()
	}
	return players
}

type Record[S any, A comparable] struct {
	State  S
	Action A
	Policy Policy[A]
	Eval   Eval
}

type Episode[S any, A comparable] []Record[S, A]

type EpisodeGenerator[S any, As ~[]A, A, G comparable] struct {
	GameLogic      sequential.Logic[S, As, A, G]
	Cap            int
	SolverPerAgent PerAgent[S, As, A, G] 
}

func (g EpisodeGenerator[S, As, A, G]) Generate(state S, f func(*S) bool) (S, Episode[S, A], error) {
	episode := make(Episode[S, A], 0, g.Cap)
	for {
		isEnd := g.GameLogic.IsEnd(&state)
		if isEnd {
			break
		}

		agent := g.GameLogic.CurrentTurnAgentGetter(&state)
		solver := g.SolverPerAgent[agent]
		legalActions := g.GameLogic.LegalActionsProvider(&state)

		action, policy, eval, err := solver(&state, legalActions)
		if err != nil {
			var s S
			return s, Episode[S, A]{}, err
		}

		record := Record[S, A]{State:state, Action:action, Policy:policy, Eval:eval}
		episode = append(episode, record)

		state, err = g.GameLogic.Transitioner(state, &action)
		if err != nil {
			var s S
			return s, Episode[S, A]{}, err
		}
	}
	return state, episode, nil
}