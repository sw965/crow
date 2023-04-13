package duct

import (
	"github.com/sw965/crow/mcts"
)

type LeafEvalYs []mcts.LeafEvalY
type LeafEvalsFn[S any] func(*S) LeafEvalYs