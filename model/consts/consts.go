package consts

type LayerType int

const (
	DotLayer LayerType = iota
	LeakyReLULayer
	InstanceNormalizationLayer
	SoftmaxLayer
)