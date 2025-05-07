package layer1d

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
	"slices"
	"github.com/sw965/crow/model/layer"
	omath "github.com/sw965/omw/math"
	"github.com/chewxy/math32"
)

type Forward func(blas32.Vector) (blas32.Vector, Backward, error)
type Forwards []Forward

func (fs Forwards) Propagate(x blas32.Vector) (blas32.Vector, Backwards, error) {
	var err error
	var backward Backward
	backwards := make(Backwards, len(fs))
	for i, f := range fs {
		x, backward, err = f(x)
		if err != nil {
			return blas32.Vector{}, nil, err
		}
		backwards[i] = backward
	}
	y := x
	slices.Reverse(backwards)
	return y, backwards, nil
}

type Backward func(blas32.Vector, *layer.GradBuffer) (blas32.Vector, error)
type Backwards []Backward

func (bs Backwards) Propagate(chain blas32.Vector) (blas32.Vector, layer.GradBuffer, error) {
	n := len(bs)
	grad := layer.GradBuffer{
		Weights:make([]blas32.General, 0, n),
		Biases: make([]blas32.Vector, 0, n),
	}
	var err error

	for _, b := range bs {
		chain, err = b(chain, &grad)
		if err != nil {
			return blas32.Vector{}, layer.GradBuffer{}, err
		}
	}

	slices.Reverse(grad.Weights)
	slices.Reverse(grad.Biases)
	dx := chain
	return dx, grad, nil
}

func NewAffineForward(w blas32.General, b blas32.Vector) Forward {
	return func(x blas32.Vector) (blas32.Vector, Backward, error) {
		yn := w.Cols
		y := blas32.Vector{N: yn, Inc: 1, Data: make([]float32, yn)}
		blas32.Copy(b, y)
		blas32.Gemv(blas.Trans, 1.0, w, x, 1.0, y)
	
		var backward Backward
		backward = func(chain blas32.Vector, grad *layer.GradBuffer) (blas32.Vector, error) {
			wRows := w.Rows
			wCols := w.Cols
	
			dx := blas32.Vector{
				N:    wRows,
				Inc:  1,
				Data: make([]float32, wRows),
			}
			blas32.Gemv(blas.NoTrans, 1.0, w, chain, 1.0, dx)
	
			dw := blas32.General{
				Rows:   wRows,
				Cols:   wCols,
				Stride: wCols,
				Data:   make([]float32, wRows*wCols),
			}
			blas32.Ger(1.0, x, chain, dw)
	
			db := blas32.Vector{
				N:    chain.N,
				Inc:  1,
				Data: make([]float32, chain.N),
			}
			blas32.Copy(chain, db)

			grad.Weights = append(grad.Weights, dw)
			grad.Biases  = append(grad.Biases, db)
			return dx, nil
		}
		return y, backward, nil
	}
}

func NewLeakyReLUForward(alpha float32) Forward {
	return func(x blas32.Vector) (blas32.Vector, Backward, error) {
		xData := x.Data
		yData := make([]float32, x.N)
		for i := range yData {
			e := xData[i]
			if e > 0 {
				yData[i] = e
			} else {
				yData[i] = alpha * e
			}
		}

		y := blas32.Vector{
			N:    x.N,
			Inc:  x.Inc,
			Data: yData,
		}

		var backward Backward
		backward = func(chain blas32.Vector, _ *layer.GradBuffer) (blas32.Vector, error) {
			chainData := chain.Data
			dxData := make([]float32, chain.N)
			for i, e := range xData {
				if e > 0 {
					dxData[i] = chainData[i]
				} else {
					dxData[i] = alpha * chainData[i]
				}
			}
			dx := blas32.Vector{
				N:    chain.N,
				Inc:  chain.Inc,
				Data: dxData,
			}
			return dx, nil
		}

		return y, backward, nil
	}
}

func SoftmaxForOutputForward(x blas32.Vector) (blas32.Vector, Backward, error) {
	xData := x.Data
	maxX := omath.Max(xData...) // オーバーフロー対策
	expX := make([]float32, x.N)
	sumExpX := float32(0.0)
	for i, e := range xData {
		expX[i] = math32.Exp(e - maxX)
		sumExpX += expX[i]
	}

	yData := make([]float32, x.N)
	for i := range expX {
		yData[i] = expX[i] / sumExpX
	}

	y := blas32.Vector{
		N:    x.N,
		Inc:  x.Inc,
		Data: yData,
	}

	var backward Backward
	backward = func(chain blas32.Vector, _ *layer.GradBuffer) (blas32.Vector, error) {
		//クロスエントロピーが損失関数である事を前提
		dx := chain
		return dx, nil
	}
	return y, backward, nil
}