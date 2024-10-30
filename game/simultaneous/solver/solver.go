package solver

import (
	"github.com/sw965/crow/game/simultaneous"
)

type Policy[A comparable] map[A]float64
type Policies[A comparable] []Policy[A]
type PoliciesProvider[S any, Ass ~[]As, As ~[]A, A comparable] func(*S, Ass) Policies[A]

func UniformPoliciesProvider[S any, Ass ~[]As, As ~[]A, A comparable](state *S, legalActionTable Ass) Policies[A] {
	policies := make(Policies[A], len(legalActionTable))
	for i, actions := range legalActionTable {
		n := len(actions)
		p := 1.0 / float64(n)
		policy := Policy[A]{}
		for _, a := range actions {
			policy[a] = p
		}
		policies[i] = policy
	}
	return policies
}

type Eval float64
type Evals []Eval

func ResultScoresToEvals(scores simultaneous.ResultScores) Evals {
	evals := make(Evals, len(scores))
	for i, score := range scores {
		evals[i] = Eval(score)
	}
	return evals
}

type ActorCritic[S any, Ass ~[]As, As ~[]A, A comparable] func(*S, Ass) (As, Policies[A], Evals, error)

func PlayerToSolver[S any, Ass ~[]As, As ~[]A, A comparable](player simultaneous.Player[S, Ass, As, A]) ActorCritic[S, Ass, As, A] {
	return func(state *S, legalActionTable Ass) (As, Policies[A], Evals, error) {
		jointAction, err := player(state, legalActionTable)
		// PoliciesとEvalsはnilで返す
		return jointAction, nil, nil, err
	}
}

func (ac ActorCritic[S, Ass, As, A]) ToPlayer() simultaneous.Player[S, Ass, As, A] {
	return func(state *S, legalActionTable Ass) (As, error) {
		// PoliciesとEvalsは破棄する
		jointAction, _, _, err := ac(state, legalActionTable)
		return jointAction, err
	}
}

type Plural[S any, Ass ~[]As, As ~[]A, A comparable] []ActorCritic[S, Ass, As, A]

func PlayersToSolvers[S any, Ass ~[]As, As ~[]A, A comparable](players simultaneous.Players[S, Ass, As, A]) Plural[S, Ass, As, A] {
	solvers := make(Plural[S, Ass, As, A], len(players))
	for i, player := range players {
		solvers[i] = PlayerToSolver(player)
	}
	return solvers
}

func (p Plural[S, Ass, As, A]) ToPlayers() simultaneous.Players[S, Ass, As, A] {
	players := make(simultaneous.Players[S, Ass, As, A], len(p))
	for i, solver := range p {
		players[i] = solver.ToPlayer()
	}
	return players
}

type Record[S any, As ~[]A, A comparable] struct {
	State        S
	JointAction  As
	Policies     Policies[A]
	Evals        Evals
}

type Episode[S any, As ~[]A, A comparable] []Record[S, As, A]

type EpisodeGenerator[S any, Ass ~[]As, As ~[]A, A comparable] struct {
	GameLogic       simultaneous.Logic[S, Ass, As, A]
	Cap             int
	Solvers         Plural[S, Ass, As, A]
}

func (g EpisodeGenerator[S, Ass, As, A]) Generate(state S, f func(*S) bool) (S, Episode[S, As, A], error) {
	episode := make(Episode[S, As, A], 0, g.Cap)
	for {
		isEnd, err := g.GameLogic.IsEnd(&state)
		if err != nil {
			var zeroState S
			return zeroState, nil, err
		}
		if isEnd || f(&state) {
			break
		}

		legalActionTable := g.GameLogic.LegalActionTableProvider(&state)
		jointAction := make(As, len(g.Solvers))
		policies := make(Policies[A], len(g.Solvers))
		evals := make(Evals, len(g.Solvers))

		for i, solver := range g.Solvers {
			ja, ps, es, err := solver(&state, legalActionTable)
			if err != nil {
				var zero S
				return zero, nil, err
			}
			jointAction[i] = ja[i]
			policies[i] = ps[i]
			evals[i] = es[i]
		}

		record := Record[S, As, A]{
			State:    state,
			JointAction:  jointAction,
			Policies: policies,
			Evals:    evals,
		}
		episode = append(episode, record)

		state, err = g.GameLogic.Transitioner(state, jointAction)
		if err != nil {
			var zero S
			return zero, nil, err
		}
	}
	return state, episode, nil
}