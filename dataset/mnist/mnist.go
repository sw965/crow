package mnist

import (
	"os"
)

var PATH = os.Getenv("GOPATH") + "/mnist/gob/"

type Flat struct {
	Xs []blas32.Vector
	Labels []blas32.Vector
}

type Image struct {
	Xs []blas32.General
	Labels []blas32.Vector
}


