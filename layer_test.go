package crow

import (
  "testing"
  "fmt"
  "time"
  "math/rand"
  "github.com/seehuhn/mt19937"
)

func TestLayer(t *testing.T) {
	mtRandom := rand.New(mt19937.New())
	mtRandom.Seed(time.Now().UnixNano())
  x := Array2D{{0.1, 0.2, 0.3, 0.4, 0.5}}
  affine2DLayer := NewAffine2DLayer(len(x[0]), 5, 1.42, mtRandom)
  output := affine2DLayer.Output(x)[0]
  for i, v := range affine2DLayer.Forward(x)[0] {
    fmt.Println(v, "≒", output[i])
  }
}
