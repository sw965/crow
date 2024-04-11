package model

type Model struct {
	ParamGradPairManagers ParameterGradientPairManagers
	Forwards layer.Forwards
	MomentumOptimizers []optimizer.Momentum
	IsTrain []bool
}

func (model *Model) Predict(x tensor.D1) (tensor.D1, Backwards, error) {
	y, backwards, err := model.Forwards.Run(x)
	return y, backwards, err
}

func (model *Model) YAndLoss(forward layer.Forward, t tensor.D1) (tensor.D1, float64, Backwards, error) {
	y, backwards, err := Predict(x)
	loss, backwards, err := forward(y, t, backwards)
	return y, loss, backwards, err
}

func (model *Model) UpdateGrad(forward layer.Forward, x tensor.D2, t tensor.D2) {
	chain := tensor.NewD1Ones(len(x[0]))
	for i := range x {
		_, _, backwards, err := YAndLoss(forward, x[i], t[i])
		_, err := backwards.Run(chain)
		if err != nil {
			return err
		}
	}
	model.ParamGradPairManagers.DivGrad(float64(len(x)))
}

func (model *Model) Train() {

}

NewAffine() {
	forward := Forwards{

	}	
}