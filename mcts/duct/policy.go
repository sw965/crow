package duct

import (
	"github.com/sw965/crow/mcts"
)

type PolicYs[A comparable] []mcts.PolicY[A]
type PoliciesFn[S any, A comparable] func(*S) PolicYs[A]