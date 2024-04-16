package model

import (
	"fmt"
	"math"
	"github.com/sw965/crow/layer"
	"github.com/sw965/crow/tensor"
)

type D1 struct {
	PerLayerScalarVar PerLayerScalarVar
	PerLayerD1Var PerLayerD1Var
	PerLayerD2Var PerLayerD2Var

	Forwards layer.D1Forwards
	LossForward layer.D1LossForward
}

func (model *D1) Init(lossForward layer.D1LossForward) {
	model.PerLayerScalarVar.Init()
	model.PerLayerD1Var.Init()
	model.PerLayerD2Var.Init()
	model.LossForward = lossForward
}

func (model *D1) Predict(x tensor.D1) (tensor.D1, layer.D1Backwards, error) {
	y, backwards, err := model.Forwards.Run(x)
	return y, backwards, err
}

func (model *D1) YAndLoss(x, t tensor.D1) (tensor.D1, float64, layer.D1BackwardPropagator, error) {
	y, backwards, err := model.Predict(x)
	loss, lastBackward, err := model.LossForward(y, t)
	loss += model.PerLayerScalarVar.L2Regularization()
	loss += model.PerLayerD1Var.L2Regularization()
	loss += model.PerLayerD2Var.L2Regularization()
	propagator := layer.NewD1BackwardPropagator(lastBackward, backwards)
	return y, loss, propagator, err
}

func (model *D1) Accuracy(x, t tensor.D1) (float64, error) {
	y, _, err := model.Predict(x)
	if err != nil {
		return 0.0, err
	}

	maxY := x[0]
	maxYIdx := 0
	for i, yi := range y[1:] {
		if yi > maxY {
			maxYIdx = i
		}
	}

	maxT := t[0]
	maxTIdx := 0

	for i, ti := range t[:] {
		if ti > maxT {
			maxTIdx = i
		}
	}

	if maxYIdx == maxTIdx {
		return 1.0, nil
	} else {
		return 0.0, nil
	}
}

func (model *D1) UpdateGrad(x, t tensor.D1) error {
	_, _, backPropagator, err := model.YAndLoss(x, t)
	if err != nil {
		return err
	}
	_, err = backPropagator()
	model.PerLayerScalarVar.AddL2RegularizationGrad()
	model.PerLayerD1Var.AddL2RegularizationGrad()
	model.PerLayerD2Var.AddL2RegularizationGrad()
	return err
}

func (model *D1) Train(lr float64) {
	model.PerLayerScalarVar.Train(lr)
	model.PerLayerD1Var.Train(lr)
	model.PerLayerD2Var.Train(lr)
}

func (model *D1) UpdateGradAndTrain(x, t tensor.D1, lr float64) {
	model.UpdateGrad(x, t)
	model.Train(lr)
}

func (model *D1) ValidateBackwardGrad(x, t tensor.D1, tolerance float64) error {
	err := model.UpdateGrad(x, t)
	if err != nil {
		return err
	}
	h := 0.001

	numericalDifferentiation := func(param, grad tensor.D1, name string) error {
		for i := range param {
			tmp := param[i]
	
			param[i] = tmp + h
			_, loss1, _, err := model.YAndLoss(x, t)
			if err != nil {
				return err
			}
	
			param[i] = tmp - h
			_, loss2, _, err := model.YAndLoss(x, t)
			if err != nil {
				return err
			}
	
			param[i] = tmp
	
			numericalGrad := (loss1 - loss2) / (2*h)
			diff := grad[i] - numericalGrad
			if math.Abs(diff) > tolerance {
				fmt.Println("許容誤差を超えた", name, "grad=", grad[i], "numGrad=", numericalGrad, "diff=", diff, loss1, loss2)
				return nil
			}
		}
		return nil
	}

	err = numericalDifferentiation(model.PerLayerScalarVar.Param, model.PerLayerScalarVar.Grad, "alpha")
	if err != nil {
		return err
	}

	for i := range model.PerLayerD1Var.Param {
		err = numericalDifferentiation(model.PerLayerD1Var.Param[i], model.PerLayerD1Var.Grad[i], "b")
		if err != nil {
			return err
		}
	}

	for i := range model.PerLayerD2Var.Param {
		for j := range model.PerLayerD2Var.Param[i] {
			err = numericalDifferentiation(model.PerLayerD2Var.Param[i][j], model.PerLayerD2Var.Grad[i][j], "w")
			if err != nil {
				return err
			}
		}
	}
	return nil
}