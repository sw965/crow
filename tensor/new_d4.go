package tensor

func NewD4ZerosLike(d4 D4) D4 {
	zeros := make(D4, len(d4))
	for i, d3 := range d4 {
		zeros[i] = NewD3ZerosLike(d3)
	}
	return zeros
}