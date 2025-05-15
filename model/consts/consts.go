package consts

type LayerType int

const (
	ConvLayer LayerType = iota
	DotLayer
	LeakyReLULayer
	InstanceNormalizationLayer
	SoftmaxLayer
)