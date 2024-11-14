package ml

type ML[M any] struct {
	Optimizer Optimizer[M]
	
}

func (ml ML[M]) Run(model M) {
	Set
}