package model_test

import (
	"testing"
	"fmt"
	"github.com/sw965/crow/dataset"
	"github.com/sw965/crow/model"
	"github.com/sw965/crow/layer/1d"
	"github.com/sw965/omw"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/crow/mlfuncs/1d"
	"github.com/sw965/crow/mlfuncs/2d"
	"github.com/sw965/crow/mlfuncs/3d"
)

func TestModel(t *testing.T) {
	r := omw.NewMt19937()
	xn := 784
	h1 := 128
	h2 := 32
	yn := 10

	isTrain := []bool{true}
	variable := model.Variable{
		Param1D:tensor.D1{0.1, 0.1, 0.1},
		Param2D:tensor.D2{
			tensor.NewD1Zeros(h1),
			tensor.NewD1Zeros(h2),
			tensor.NewD1Zeros(yn),
		},
		Param3D:tensor.D3{
			tensor.NewD2He(xn, h1, r),
			tensor.NewD2He(h1, h2, r),
			tensor.NewD2He(h2, yn, r),
		},
	}
	variable.Init()

	affine := model.NewSequentialInputOutput1D(variable)
	affine.L2NormGradClipThreshold = 64.0

	forwards := layer1d.Forwards{
		layer1d.NewAffineForward(variable.Param3D[0], variable.Param2D[0], variable.GetGrad3D()[0], variable.GetGrad2D()[0]),
		//layer.NewD1ReLUForward(),
		//layer1d.NewLeakyReLUForward(0.1),
		//layer1d.NewParamReLUForward(&variable.Param1D[0], &variable.GetGrad1D()[0]),
		//layer.NewD1RandReLUForward(0.05, 0.15, &isTrain[0], r),
		layer1d.NewParamRandReLUForward(&variable.Param1D[0], 0.25, 1.75, &variable.GetGrad1D()[0], &isTrain[0], r),

		layer1d.NewAffineForward(variable.Param3D[1], variable.Param2D[1], variable.GetGrad3D()[1], variable.GetGrad2D()[1]),
		//layer.NewD1ReLUForward(),
		//layer1d.NewLeakyReLUForward(0.1),
		//layer1d.NewParamReLUForward(&variable.Param1D[1], &variable.GetGrad1D()[1]),
		//layer.NewD1RandReLUForward(0.05, 0.15, &isTrain[0], r),
		layer1d.NewParamRandReLUForward(&variable.Param1D[1], 0.25, 1.75, &variable.GetGrad1D()[1], &isTrain[0], r),

		layer1d.NewAffineForward(variable.Param3D[2], variable.Param2D[2], variable.GetGrad3D()[2], variable.GetGrad2D()[2]),
		//layer.NewD1ReLUForward(),
		//layer1d.NewLeakyReLUForward(0.1),
		//layer1d.NewParamReLUForward(&variable.Param1D[2], &variable.GetGrad1D()[2]),
		//layer.NewD1RandReLUForward(0.05, 0.15, &isTrain[0], r),
		layer1d.NewParamRandReLUForward(&variable.Param1D[2], 0.25, 1.75, &variable.GetGrad1D()[2], &isTrain[0], r),

		//layer1d.NewSigmoidForward(),
		layer1d.NewTanhForward(),
	}
	//Tanh Prelu × D1
	//Sigmoid Prelu × D3
	//Sigmoid LRelu × D2 D3

	affine.Forwards = forwards
	affine.YLossFunc = mlfuncs1d.SumSquaredError
	affine.YLossDerivative = mlfuncs1d.SumSquaredErrorDerivative

	c := 0.0001
	affine.Param3DLossFunc = mlfuncs3d.L2Regularization(c)
	affine.Param3DLossDerivative = mlfuncs3d.L2RegularizationDerivative(c)

	mnist, err := dataset.LoadFlatMnist()
	if err != nil {
		panic(err)
	}

	mnist.TrainLabel = mlfuncs2d.SigmoidToTanh(mnist.TrainLabel)
	mnist.TestLabel = mlfuncs2d.SigmoidToTanh(mnist.TestLabel)

	trainImgNum := len(mnist.TrainImg)
	testImgNum := len(mnist.TestImg)
	trainNum := 256000
	testSize := 196

	for i := 0; i < trainNum; i++ {
		idx := r.Intn(trainImgNum)
		affine.SGD(mnist.TrainImg[idx], mnist.TrainLabel[idx], 0.01)
		if i%196 == 0 {
			//affine.ValidateBackwardAndNumericalGradientDifference(mnist.TrainImg[idx], mnist.TrainLabel[idx])
			idxs := omw.RandIntsUniform(testSize, 0, testImgNum, r)
			miniBatchTestImg := omw.ElementsAtIndices(mnist.TestImg, idxs...)
			miniBatchTestLabel := omw.ElementsAtIndices(mnist.TestLabel, idxs...)
			isTrain[0] = false
			loss, err := affine.MeanLoss(miniBatchTestImg, miniBatchTestLabel)
			if err != nil {
				panic(err)
			}
			accuracy, err := affine.Accuracy(miniBatchTestImg, miniBatchTestLabel)
			if err != nil {
				panic(err)
			}

			fmt.Println("i =", i, "loss =", loss, "accuracy =", accuracy)
			isTrain[0] = true
		}
	}
}

func TestLinearSumGrad(test *testing.T) {
	rng := omw.NewMt19937()
	linear := model.NewLinearSumIdentityMSE(0.001)
	//linear := model.NewLinearSumSigmoidMSE(0.001)
	r, c := 10, 5
	min, max := -5.0, 5.0
	linear.W = tensor.NewD2RandUniform(r, c, min, max, rng)
	linear.B = tensor.NewD1RandUniform(r, min, max, rng)
	x := tensor.NewD2RandUniform(r, c, min, max, rng)
	t := tensor.NewD1RandUniform(r, min, max, rng)
	err := linear.ValidateBackwardAndNumericalGradientDifference(x, t)
	if err != nil {
		panic(err)
	}
}