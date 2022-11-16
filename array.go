package crow

import (
  "math"
  "math/rand"
)

type Array1D []float64

func NewRandomArray1D(size int, std float64, random *rand.Rand) Array1D {
  result := make(Array1D, size)
  for i := 0; i < size; i++ {
    result[i] = random.NormFloat64() * std
  }
  return result
}

func (array1D Array1D) Max() float64 {
  result := array1D[0]
  for _, v := range array1D[1:] {
    if v > result {
      result = v
    }
  }
  return result
}

func (array1D Array1D) Sum() float64 {
  result := 0.0
  for _, v := range array1D {
    result += v
  }
  return result
}

func (array1D Array1D) Log() Array1D {
  result := make(Array1D, len(array1D))
  for i, v := range array1D {
    result[i] = math.Log(v)
  }
  return result
}

func (array1D Array1D) Exp() Array1D {
  result := make(Array1D, len(array1D))
  for i, v := range array1D {
    result[i] = math.Exp(v)
  }
  return result
}

func (array1D Array1D) AddFloat64(x float64) Array1D {
  result := make(Array1D, len(array1D))
  for i, v := range array1D {
    result[i] = v + x
  }
  return result
}

func (array1D Array1D) SubFloat64(x float64) Array1D {
  result := make(Array1D, len(array1D))
  for i, v := range array1D {
    result[i] = v - x
  }
  return result
}

func (array1D Array1D) DivFloat64(x float64) Array1D {
  result := make(Array1D, len(array1D))
  for i, v := range array1D {
    result[i] = v / x
  }
  return result
}

func (array1D_1 Array1D) Add(array1D_2 Array1D) Array1D {
  result := make(Array1D, len(array1D_1))
  for i := 0; i < len(array1D_1); i++ {
    result[i] = array1D_1[i] + array1D_2[i]
  }
  return result
}

func (array1D_1 Array1D) Sub(array1D_2 Array1D) Array1D {
  result := make(Array1D, len(array1D_1))
  for i := 0; i < len(array1D_1); i++ {
    result[i] = array1D_1[i] - array1D_2[i]
  }
  return result
}

func (array1D_1 Array1D) Mul(array1D_2 Array1D) Array1D {
  result := make(Array1D, len(array1D_1))
  for i := 0; i < len(array1D_1); i++ {
    result[i] = array1D_1[i] * array1D_2[i]
  }
  return result
}

func (array1D_1 Array1D) InnerProduct(array1D_2 Array1D) float64 {
  result := 0.0
  for i := 0; i < len(array1D_1); i++ {
    result += array1D_1[i] * array1D_2[i]
  }
  return result
}

func (array1D Array1D) Relu() Array1D {
  result := make(Array1D, len(array1D))
  for i, v := range array1D {
    if v > 0 {
      result[i] = v
    } else {
      result[i] = 0
    }
  }
  return result
}

func (array1D Array1D) Softmax() Array1D {
  max := array1D.Max()
  a := array1D.SubFloat64(max).Exp()
  sumA := a.Sum()
  return a.DivFloat64(sumA)
}

func (y Array1D) CrossEntropy(target Array1D) float64 {
  y = y.AddFloat64(0.0001)
  result := target.Mul(y.Log()).Sum()
  return -result
}

type Array2D []Array1D

func NewRandomArray2D(h, w int, std float64, random *rand.Rand) Array2D {
  result := make(Array2D, h)
  for i := 0; i < h; i++ {
    result[i] = NewRandomArray1D(w, std, random)
  }
  return result
}

func (array2D Array2D) Sum() Array1D {
  result := make(Array1D, len(array2D))
  for i, array1D := range array2D {
    result[i] = array1D.Sum()
  }
  return result
}

func (array2D Array2D) Add1D(array1D Array1D) Array2D {
  result := make(Array2D, len(array2D))
  for i, iArray1D := range array2D {
    result[i] = iArray1D.Add(array1D)
  }
  return result
}

func (array2D Array2D) Transform() Array2D {
  array2DColumn := len(array2D)
  array2DRow := len(array2D[0])

  result := make(Array2D, array2DRow)
  for i := 0; i < array2DRow; i++ {
    resultElement := make(Array1D, array2DColumn)
    for j, array1D := range array2D {
      resultElement[j] = array1D[i]
    }
    result[i] = resultElement
  }
  return result
}

func (array2D_1 Array2D) Matmul(array2D_2 Array2D) Array2D {
  array2D_2 = array2D_2.Transform()
  result := make(Array2D, len(array2D_1))
  array2D_2Row := len(array2D_2)
  for i, array1D_1 := range array2D_1 {
    resultElement := make(Array1D, array2D_2Row)
    for j, array1D_2 := range array2D_2 {
      innerProduct := array1D_1.InnerProduct(array1D_2)
      resultElement[j] = innerProduct
    }
    result[i] = resultElement
  }
  return result
}

func (array2D Array2D) Relu() Array2D {
  result := make(Array2D, len(array2D))
  for i, array1D := range array2D {
    result[i] = array1D.Relu()
  }
  return result
}
