package layer

type D1VarMap map[VarKey]tensor.D1

type D1TrainVarManager struct {
	Param D1VarMap
	Grad D1VarMap
}

func (v *D1TrainVarManager) AddGrad(dParam tensor.D1) {
	for key, _ := range v.Grad {
		grad := v.Grad[key]
		for i := range grad {
			grad[i] = dParam[i]
		}
	}
}

func (v *D1TrainVarManager) DivGrad(n float64) {
	for key, _ := range v.Grad {
		grad := v.Grad[key]
		for i := range g {
			grad[i] /= n
		}
	}
}

type D1Forward func(tensor.D1, D1Backwards) (tensor.D1, D1Backwards, error)
type D1Forwards []D1Forward

func (fs D1Forwards) Run(x tensor.D1) (tensor.D1, D1Backwards, error) {
	bs := make(D1Backwards, 0, len(fs))
	var err error
	for _, f := range fs {
		x, bs, err = f(x, bs)
		if err != nil {
			return tensor.D1{}, D1Backwards{}, err
		}
	}
	return x, bs, err
}

type D1Backward func(tensor.D1) (tensor.D1, error)
type D1Backwards []D1Backward

func (bs D1Backwards) Run(chain tensor.D1) (tensor.D1, error) {
	var err error
	bs = omw.Reverse(bs)
	for _, b := range bs {
		chain, err = b(chain)
		if err != nil {
			return tensor.D1{}, err
		}
	}
	return chain, nil
}

func NewD1AffineForward(d2v *D2TrainVarManager, d1v D1TrainVarManager) Forward {
	return func(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
		w := d2v.Param[KEY_W]
		b := d1v.Param[KEY_B]

		dot, err := tensor.D2{x}.DotProduct(w)
		if err != nil {
			return tensor.D1{}, Backwards{}, err
		}
		y, err := dot[0].Add(b[0])

		var backward Backward
		backward = func(chain tensor.D1) (tensor.D1, error) {
			// ∂L/∂x
			dx, err := tensor.D2{chain}.DotProduct(w.Transpose())
			if err != nil {
				return tensor.D1{}, err
			}
		
			// ∂L/∂w
			dw, err := tensor.D2{x}.Transpose().DotProduct(tensor.D2{chain})
			if err != nil {
				return tensor.D1{}, err
			}
			d2v.AddGrad(dw)

			// ∂L/∂b
			db := chain
			d1v.AddGrad(db)
			return dx[0], err
		}
		backwards = append(backwards, backward)
		return y, backwards, err
	}
}

func D1ReLUForward(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
	y := mlfuncs.ReLU(x)
	var backward Backward
	backward = func(chain tensor.D1) (tensor.D1, error) {
		dydx := mlfuncs.ReLUDerivative(x)
		// ∂L/∂x
		dx, err := dydx.Mul(chain)
		return dx, err
	}
	backwards = append(backwards, backward)
	return y, backwards, nil
}

func NewD1PReLUForward(sv *ScalarTrainVarManager) D1Forward {
	return func(x tensor.D1, backwards D1Backwards) (tensor.D1, D1Backwards, error) {
		alpha := sv.Param[KEY_ALPHA]
		y := mlfuncs.PReLU(x, alpha)
		var backward Backward
		backward = func(chain tensor.D1) (tensor.D1, error) {
			dydx, dydAlpha := mlfuncs.PReLUDerivative(x, alpha)

			// ∂L/∂alpha
			dAlpha := omw.Sum(chain.MulScalar(dydAlpha)...)
			sv.AddGrad(dAlpha)

			// ∂L/∂x
			dx, err := dydx.Mul(chain)
			return dx, err
		}
		backwards = append(backwards, backward)
		return y, backwards, nil
	}
}

func D1SigmoidForward(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
	y := mlfuncs.Sigmoid(x)
	var backward Backward
	backward = func(chain tensor.D1) (tensor.D1, error) {
		dydx := mlfuncs.SigmoidGrad(y)
		// ∂L/∂x
		dx, err := dydx.Mul(chain)
		return dx, err
	}
	backwards = append(backwards, backward)
	return y, backwards, nil
}

func D1TanhForward(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
	y := mlfuncs.Tanh(x)
	var backward Backward
	backward = func(chain tensor.D1) (tensor.D1, error) {
		dydx := mlfuncs.TanhGrad(y)
		// ∂L/∂x
		dx, err := dydx.Mul(chain)
		return dx, err
	}
	backwards = append(backwards, backward)
	return y, backwards, nil
}

func D1MeanSquaredErrorForward(y, t tensor.D1, backwards Backwards) (float64, Backwards, error) {
	z, err := mlfuncs.MeanSquaredError(y, t)
	var backward Backward
	backward = func(chain tensor.D1) (tensor.D1, error) {
		dy, err := mlfuncs.MeanSquaredErrorDerivative(y, t)
		if err != nil {
			return tensor.D1{}, err
		}
		return dy.Mul(chain)
	}
	backwards = append(backwards, backward)
	return z, backwards, err
}