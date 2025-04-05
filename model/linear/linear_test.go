package linear_test

import (
	"fmt"
	"testing"
	"github.com/sw965/crow/model/linear"
)

func Test(*testing.T) {
	wf := func(wc linear.WeightCoordinate) linear.WeightCoordinate {
		wc.Row = 2 - wc.Row
		wc.Column = 2 - wc.Column
		return wc
	}

	bf := func(bi int) int {
		return 2 - bi
	}

	param := linear.NewParameter([]int{3, 3, 3}, wf, bf)
	for _, wi := range param.Weight {
		fmt.Println(wi)
	}
	fmt.Println(param.Bias)
	fmt.Println("")

	clone := param.Clone()
	
	for _, wi := range clone.Weight {
		fmt.Println(wi)
	}
	fmt.Println(clone.Bias)
}