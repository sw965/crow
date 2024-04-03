package layer

import (
	"math/rand"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/crow/mlfuncs"
)

type Backward func(tensor.D1) (tensor.D1, error)
type Backwards []Backward

type D1Affine struct {
	W tensor.D2
	B tensor.D1
	Dw tensor.D2
	Db tensor.D1
	WRow int
	WCol int
	Sample int
}

func NewD1Affine(r, c int, min, max float64, random *rand.Rand) D1Affine {
	return D1Affine{
		W:tensor.NewD2Random(r, c, min, max, random),
		B:make(tensor.D1, c),
		WRow:r,
		WCol:c,
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
		//∂L/∂x
		dx, err := tensor.D2{chain}.DotProduct(affine.W.Transpose())
		if err != nil {
			return tensor.D1{}, err
		}

		// ∂L/∂w
		dw, err := tensor.D2{x}.Transpose().DotProduct(tensor.D2{chain}) 
		if err != nil {
			return tensor.D1{}, err
		}

		affine.Dw, err = affine.Dw.Add(dw)
		if err != nil {
			return tensor.D1{}, err
		}

		// ∂L/∂b
		affine.Db, err = affine.Db.Add(chain)
		if err != nil {
			return tensor.D1{}, err
		}

		affine.Sample += 1
		return dx[0], nil
	}
	backwards = append(backwards, backward)
	return y, backwards, err
}

func (affine *D1Affine) SGD(lr float64) error {
	sample := float64(affine.Sample)
	dw := affine.Dw.DivScalar(sample)
	db := affine.Db.DivScalar(sample)

	var err error
	affine.W, err = affine.W.Sub(dw.MulScalar(lr))
	if err != nil {
		return err
	}

	affine.B, err = affine.B.Sub(db.MulScalar(lr))
	if err != nil {
		return err
	}

	affine.Dw = tensor.NewD2Zeros(affine.WRow, affine.WCol)
	affine.Db = make(tensor.D1, affine.WCol)
	affine.Sample = 0
	return nil
}

type D1Droput struct {
	P float64
	R tensor.D1
	IsTrain bool
	Len int
	Rand *rand.Rand
}

func NewD1Droput(p float64, n int, random *rand.Rand) D1Droput {
	return D1Droput{
		P:p,
		R:make(tensor.D1, n),
		IsTrain:true,
		Len:n,
		Rand:random,
	}
}

func (drop *D1Droput) Forward(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
	y := make(tensor.D1, drop.Len)
	r := make(tensor.D1, drop.Len)

	if drop.IsTrain {
		for i, ele := range x {
			if drop.Rand.Float64() < drop.P {
				y[i] = ele
				r[i] = 1
			} else {
				y[i] = 0
				r[i] = 0
			}
		}
	} else {
		for i, ele := range x {
			y[i] = ele * drop.P
		}
	}

	backward := func(chain tensor.D1) (tensor.D1, error) {
		return r.Mul(chain)
	}
	backwards = append(backwards, backward)
	return y, backwards, nil
}

type D1PRReLU struct {
	Alpha  tensor.D1
	Beta   tensor.D1
	Gamma  float64
	BetaMin float64
	BetaMax float64
	Dalpha tensor.D1
	IsTrain bool
	Len    int
	Sample int
	Rand *rand.Rand
}

func NewD1PRReLU(n int, aMin, aMax, bMin, bMax float64, r *rand.Rand) D1PRReLU {
	alpha := tensor.NewD1Random(n, aMax, aMin, r)
	gamma := (bMin +  bMax) / 2.0
	return D1PRReLU{
		Alpha:  alpha,
		Beta:   tensor.NewD1Random(n, bMin, bMax, r) ,
		Gamma:  gamma,
		BetaMin:bMin,
		BetaMax:bMax,
		Dalpha: make(tensor.D1, n),
		IsTrain:true,
		Len:    n,
		Rand:r,
	}
}

func (prrelu *D1PRReLU) Forward(x tensor.D1, backwards Backwards) (tensor.D1, Backwards, error) {
	var err error
	var beta tensor.D1
	var y tensor.D1

	if prrelu.IsTrain {
		beta = tensor.NewD1Random(prrelu.Len, prrelu.BetaMin, prrelu.BetaMax, prrelu.Rand) 
		y, err = mlfuncs.D1PRReLU(x, prrelu.Alpha, prrelu.Beta, prrelu.Gamma, prrelu.IsTrain)
	} else {
		beta = make(tensor.D1, prrelu.Len)
		y, err = mlfuncs.D1PRReLU(x, prrelu.Alpha, beta, prrelu.Gamma, prrelu.IsTrain)
	}

	var backward Backward
	backward = func(chain tensor.D1) (tensor.D1, error) {
		dx := make(tensor.D1, prrelu.Len)
		for i := range dx {
			xi := x[i]
			if xi >= 0 {
				dx[i] = chain[i]
				prrelu.Dalpha[i] += 0
			} else {
				betai := beta[i]
				dx[i] = prrelu.Alpha[i] * betai * chain[i]
				prrelu.Dalpha[i] += xi * betai * chain[i]
			}
		}
		prrelu.Sample += 1
		return dx, nil
	}
	backwards = append(backwards, backward)
	return y, backwards, err
}

func (prrelu *D1PRReLU) SGD(lr float64) error {
	var err error
	dAlpha := prrelu.Dalpha.DivScalar(float64(prrelu.Sample))
	prrelu.Alpha, err = prrelu.Alpha.Sub(dAlpha.MulScalar(lr))
	if err != nil {
		return err
	}
	prrelu.Dalpha = make(tensor.D1, prrelu.Len)
	prrelu.Sample = 0
	return nil
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

type D1MeanSquaredError struct {
	Y tensor.D1
	T tensor.D1
}

func (mse *D1MeanSquaredError) Forward(y, t tensor.D1) float64 {
	mse.Y = y
	mse.T = t
	return mlfuncs.D1MeanSquaredError(y, t)
}

func (mse *D1MeanSquaredError) Backward() tensor.D1 {
	return mlfuncs.D1MeanSquaredErrorDerivative(mse.Y, mse.T)
}
