package crow

import (
  "math/rand"
)

type Affine2DLayer struct {
  X Array2D
  W Array2D
  B Array1D
  WGrad Array2D
  BGrad Array1D
}

func NewAffine2DLayer(h, w int, std float64, random *rand.Rand) Affine2DLayer {
  result := Affine2DLayer{}
  result.W = NewRandomArray2D(h, w, std, random)
  result.B = NewRandomArray1D(w, std, random)
  return result
}

func (affine2DLayer *Affine2DLayer) Forward(x Array2D) Array2D {
  affine2DLayer.X = x
  return affine2DLayer.Output(x)
}

func (affine2DLayer *Affine2DLayer) Backward(dout Array2D) Array2D {
  dx := dout.Matmul(affine2DLayer.W.Transpose())
  affine2DLayer.WGrad = affine2DLayer.X.Transpose().Matmul(dout)
  affine2DLayer.BGrad = dout.Transpose().Sum()
  return dx
}

func (affine2DLayer *Affine2DLayer) SGDTraining(learingRate float64) {
  for i, grads := range affine2DLayer.WGrad {
    for j, grad := range grads {
      affine2DLayer.W[i][j] -= (grad * learingRate)
    }
  }

  for i, grad := range affine2DLayer.BGrad {
    affine2DLayer.B[i] -= (grad * learingRate)
  }
}

func (affine2DLayer *Affine2DLayer) Output(x Array2D) Array2D {
  return x.Matmul(affine2DLayer.W).Add1D(affine2DLayer.B)
}

type Relu2DLayer struct {
  Y Array2D
}

func (relu2DLayer *Relu2DLayer) Forward(x Array2D) Array2D {
  y := x.Relu()
  relu2DLayer.Y = y
  return y
}

func (relu2DLayer *Relu2DLayer) Backward(dout Array2D) Array2D {
  dx := make(Array2D, len(dout))
  for i, array1D := range relu2DLayer.Y {
    dxElement := make(Array1D, len(array1D))
    for j, v := range array1D {
      if v > 0 {
        dxElement[j] = dout[i][j]
      } else {
        dxElement[j] = 0
      }
    }
    dx[i] = dxElement
  }
  return dx
}

type SoftmaxWithCrossEntropy struct {
  Y Array1D
  Target Array1D
  Loss float64
}

func (swct *SoftmaxWithCrossEntropy) Forward(x, target Array1D) float64 {
  swct.Y = x.Softmax()
  swct.Target = target
  swct.Loss = swct.Y.CrossEntropy(target)
  return swct.Loss
}

func (swct *SoftmaxWithCrossEntropy) Backward() Array1D {
  dx := swct.Y.Sub(swct.Target)
  return dx
}
