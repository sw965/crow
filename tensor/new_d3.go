package tensor

import (
	"github.com/sw965/omw"
)

func NewD3ZerosLike(d3 D3) D3 {
	return omw.MapFunc[D3](d3, NewD2ZerosLike)
}

func NewD3OnesLike(d3 D3) D3 {
	return omw.MapFunc[D3](d3, NewD2OnesLike)
}