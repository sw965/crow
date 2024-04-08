package layer

import (
	"math/rand"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/crow/optimizer"
	"github.com/sw965/crow/mlfuncs"
	"github.com/sw965/omw"
)

type Forward func(tensor.D1, Backwards) (tensor.D1, Backwards, error)
type Forwards []Forward

func (fs Forwards) Run(x tensor.D1) (tensor.D1, Backwards, error) {
	bs := make(Backwards, 0, len(fs))
	var err error
	for _, f := range fs {
		x, bs, err = f(x, bs)
		if err != nil {
			return tensor.D1{}, Backwards{}, err
		}
	}
	return x, bs, err
}

type Backward func(tensor.D1) (tensor.D1, error)
type Backwards []Backward

func (bs Backwards) Run(chain tensor.D1) (tensor.D1, error) {
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

type D1Affine struct {
	W tensor.D2
	B tensor.D1
	Dw tensor.D2
	Db tensor.D1

	WRow int
	WCol int

	Lambda float64
	WOptimizer optimizer.D2Momentum
	BOptimizer optimizer.D1Momentum
}

func NewD1Affine(r, c int, random *rand.Rand) D1Affine {
	return D1Affine{
		W:tensor.NewD2He(r, c, random),
		B:make(tensor.D1, c),
		WRow:r,
		WCol:c,
		Lambda:0.001,
		WOptimizer:optimizer.NewD2Momentum(r, c),
		BOptimizer:optimizer.NewD1Momentum(c),
	}
}

func (affine *D1Affine) Forward(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
	dot, err := tensor.D2{x}.DotProduct(affine.W)
	if err != nil {
		return tensor.D1{}, Backwards{}, err
	}
	y, err := dot[0].Add(affine.B)

	var backward Backward
	backward = func(chain tensor.D1) (tensor.D1, error) {
		// ∂L/∂x
		dx, err := tensor.D2{chain}.DotProduct(affine.W.Transpose())
		if err != nil {
			return tensor.D1{}, err
		}

		// ∂L/∂w
		affine.Dw, err = tensor.D2{x}.Transpose().DotProduct(tensor.D2{chain})
		if err != nil {
			return tensor.D1{}, err
		}

		affine.Dw, err = affine.Dw.Add(mlfuncs.D2L2RegularizationDerivative(affine.W, affine.Lambda))
		if err != nil {
			return tensor.D1{}, err
		}

		// ∂L/∂b
		affine.Db = chain
		return dx[0], nil
	}
	backwards = append(backwards, backward)
	return y, backwards, err
}

func (affine *D1Affine) Train(lr float64) {
	affine.WOptimizer.Update(affine.W, affine.Dw, lr)
	affine.BOptimizer.Update(affine.B, affine.Db, lr)
}

func (affine *D1Affine) SWA(old *D1Affine, wScale float64) (D1Affine , error) {
	scaledNewW := affine.W.MulScalar(wScale)
	scaledOldW := old.W.MulScalar(1.0 - wScale)
	avgW, err := scaledNewW.Add(scaledOldW)
	if err != nil {
		return D1Affine{}, err
	}

	scaledNewB := affine.B.MulScalar(wScale)
	scaledOldB := old.B.MulScalar(1.0 - wScale)
	avgB, err := scaledNewB.Add(scaledOldB)
	if err != nil {
		return D1Affine{}, err
	}

	return D1Affine{
		W:avgW,
		B:avgB,
		WRow:affine.WRow,
		WCol:affine.WCol,
		WOptimizer:optimizer.NewD2Momentum(affine.WRow, affine.WCol),
		BOptimizer:optimizer.NewD1Momentum(affine.WCol),
	}, nil
}

func D1SigmoidForward(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
	y := mlfuncs.D1Sigmoid(x)
	var backward Backward
	backward = func(chain tensor.D1) (tensor.D1, error) {
		dydx := mlfuncs.D1SigmoidGrad(y)
		// ∂L/∂x
		dx, err := dydx.Mul(chain)
		return dx, err
	}
	backwards = append(backwards, backward)
	return y, backwards, nil
}

func D1TanhForward(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
	y := mlfuncs.D1Tanh(x)
	var backward Backward
	backward = func(chain tensor.D1) (tensor.D1, error) {
		dydx := mlfuncs.D1TanhGrad(y)
		// ∂L/∂x
		dx, err := dydx.Mul(chain)
		return dx, err
	}
	backwards = append(backwards, backward)
	return y, backwards, nil
}

func D1ReLUForward(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
	y := mlfuncs.D1ReLU(x)
	var backward Backward
	backward = func(chain tensor.D1) (tensor.D1, error) {
		dydx := mlfuncs.D1ReLUDerivative(x)
		// ∂L/∂x
		dx, err := dydx.Mul(chain)
		return dx, err
	}
	backwards = append(backwards, backward)
	return y, backwards, nil
}

type D1Dropout struct {
	P float64
	Rand *rand.Rand
	IsTrain bool
}

func (drop *D1Dropout) Forward(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
	var p float64
	if drop.IsTrain {
		p = drop.P
	} else {
		p = 0.0
	}
	y, mask := mlfuncs.D1Dropout(x, p, drop.Rand)

	var backward Backward
	backward = func(chain tensor.D1) (tensor.D1, error) {
		dydx := mlfuncs.D1DropoutDerivative(mask)
		// ∂L/∂x
		dx, err := dydx.Mul(chain)
		return dx, err
	}
	backwards = append(backwards, backward)
	return y, backwards, nil
}

type D1PReLU struct {
	Alpha tensor.D1
	Dalpha tensor.D1
	AlphaOptimizer optimizer.D1Momentum
}

func NewD1PReLU(n int, min, max float64, r *rand.Rand) D1PReLU {
	return D1PReLU{
		Alpha:tensor.NewD1Random(n, min, max, r),
		AlphaOptimizer:optimizer.NewD1Momentum(n),
	}
}

func (relu *D1PReLU) Forward(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
	y, err := mlfuncs.D1PReLU(x, relu.Alpha)
	var backward Backward
	backward = func(chain tensor.D1) (tensor.D1, error) {
		dydx, dAlpha, err := mlfuncs.D1PReLUDerivative(x, relu.Alpha)
		if err != nil {
			return tensor.D1{}, err
		}

		relu.Dalpha, err = dAlpha.Mul(chain)
		if err != nil {
			return tensor.D1{}, err
		}

		dx, err := dydx.Mul(chain)
		return dx, err
	}
	backwards = append(backwards, backward)
	return y, backwards, err
}

func (relu *D1PReLU) Train(lr float64) {
	relu.AlphaOptimizer.Update(relu.Alpha, relu.Dalpha, lr)
}

func (relu *D1PReLU) SWA(old *D1PReLU, alphaScale float64) (D1PReLU, error) {
	scaledNewAlpha := relu.Alpha.MulScalar(alphaScale)
	scaledOldAlpha := old.Alpha.MulScalar(1.0 - alphaScale)
	avgAlpha, err := scaledNewAlpha.Add(scaledOldAlpha)
	return D1PReLU{
		Alpha:avgAlpha,
		AlphaOptimizer:optimizer.NewD1Momentum(len(relu.Alpha)),
	}, err
}

func NormalizeToProbDistForward(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
	y := mlfuncs.NormalizeToProbDist(x)
	var backward Backward
	backward = func(chain tensor.D1) (tensor.D1, error) {
		dydx := mlfuncs.NormalizeToProbDistDerivative(x)
		// ∂L/∂x
		return dydx.Mul(chain)
	}
	backwards = append(backwards, backward)
	return y, backwards, nil
}

func D1MeanSquaredErrorForward(y, t tensor.D1, backwards Backwards) (float64, Backwards, error) {
	z, err := mlfuncs.D1MeanSquaredError(y, t)
	var backward Backward
	backward = func(chain tensor.D1) (tensor.D1, error) {
		dy, err := mlfuncs.D1MeanSquaredErrorDerivative(y, t)
		if err != nil {
			return tensor.D1{}, err
		}
		return dy.Mul(chain)
	}
	backwards = append(backwards, backward)
	return z, backwards, err
}