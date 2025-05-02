package main

import (
	"fmt"
	"slices"
	"gonum.org/v1/gonum/blas/blas32"
	"gonum.org/v1/gonum/blas"
	"github.com/sw965/crow/tensor"
)

func OutputRows(img tensor.D3, filterRows int) int {
	return img.Rows - filterRows + 1
}

func OutputCols(img tensor.D3, filterCols int) int {
	return img.Cols - filterCols + 1
}

func ZeroPadding2D(img tensor.D3, top, bot, left, right int) tensor.D3 {
    padded := tensor.NewD3Zeros(img.Channels, img.Rows+top+bot, img.Cols+left+right)
    for ch := 0; ch < img.Channels; ch++ {
        for row := 0; row < img.Rows; row++ {
            for col := 0; col < img.Cols; col++ {
                oldIdx := img.FlatIndex(ch, row, col)
                newIdx := padded.FlatIndex(ch, row+top, col+left)
                padded.Data[newIdx] = img.Data[oldIdx]
            }
        }
    }
    return padded
}

func SameZeroPadding2D(img tensor.D3, filterRows, filterCols int) tensor.D3 {
	top := (filterRows - 1) / 2
    bot := filterRows - 1 - top
    left := (filterCols - 1) / 2
    right := filterCols - 1 - left
    return ZeroPadding2D(img, top, bot, left, right)
}

func Im2Col(img tensor.D3, filterRows, filterCols int) blas32.General {
	chs := img.Channels
	outRows := OutputRows(img, filterRows)
	outCols := OutputCols(img, filterCols)
	imgData := img.Data
	newData := make([]float32, outRows*outCols*chs*filterRows*filterCols)
	newIdx := 0

	for or := 0; or < outRows; or++ {
		for oc := 0; oc < outCols; oc++ {
			for ch := 0; ch < chs; ch++ {
				for fr := 0; fr < filterRows; fr++ {
					for fc := 0; fc < filterCols; fc++ {
						row := fr + or
						col := fc + oc
						imgIdx := img.FlatIndex(ch, row, col)
						newData[newIdx] = imgData[imgIdx]
						newIdx++
					}
				}
			}
		}
	}

	newCols := filterRows*filterCols*chs
	return blas32.General{
		Rows:outRows*outCols,
		Cols:newCols,
		Stride:newCols,
		Data:newData,
	}
}

func FilterToGeneral(f tensor.D4) blas32.General {
	return blas32.General{
		Rows:f.Batches,
		Cols:f.BatchStride,
		Stride:f.BatchStride,
		Data:slices.Clone(f.Data),
	}
}

func DotResultToImage(result blas32.General, outRows, outCols int) (tensor.D3, error) {
	if result.Rows != outRows*outCols {
		err := fmt.Errorf("dimension mismatch: result.Rows=%d, want %d (=outRows*outCols)", result.Rows, outRows*outCols)
		return tensor.D3{}, err
	}

	img := tensor.NewD3Zeros(result.Cols, outRows, outCols)

	for row := 0; row < outRows; row++ {
		for col := 0; col < outCols; col++ {
			base := (row*outCols + col) * result.Stride
			for ch := 0; ch < result.Cols; ch++ {
				img.Data[img.FlatIndex(ch, row, col)] = result.Data[base+ch]
			}
		}
	}
	return img, nil
}

func Conv2DAndAddChannel(img tensor.D3, filter tensor.D4, b blas32.Vector, isSamePad bool) (tensor.D3, error) {
	if len(b.Data) != b.N {
		return tensor.D3{}, fmt.Errorf("len(b.Data) != b.N")
	}

	if filter.Batches != b.N {
		return tensor.D3{}, fmt.Errorf("filter.Batches != b.N")
	}

	if isSamePad {
		img = SameZeroPadding2D(img, filter.Rows, filter.Cols)
	}

	imgCol := Im2Col(img, filter.Rows, filter.Cols)
	filterGen := FilterToGeneral(filter)

	dotRows := imgCol.Rows
	dotCols := filter.Batches

	dotResult := blas32.General{
		Rows:dotRows,
		Cols:dotCols,
		Stride:dotCols,
		Data:make([]float32, dotRows*dotCols),
	}

	//チャネル毎に加算する
	for row := 0; row < dotRows; row++ {
        base := row * dotCols
        for col := 0; col < dotCols; col++ {
            dotResult.Data[base+col] += b.Data[col]
        }
    }

	blas32.Gemm(blas.NoTrans, blas.Trans, 1.0, imgCol, filterGen, 1.0, dotResult)
	return DotResultToImage(dotResult, OutputRows(img, filter.Rows), OutputCols(img, filter.Cols))
}

// type Parameter struct {
// 	Filters []Filter
// 	Biases []blas32.Vector

// 	Weights []blas32.General


// }

// type Model struct {
// 	Parameter Parameter
// 	ConvForwards ConvForwards
// 	Flatten func(Image) blas32.Vector
// 	FullyForwards FullyForwards
// }

// func (m *Model) AppendConv2D(filter Filter, b blas32.Vector, isSamePad bool) {
// 	forward := func(img Image) (Image, error) {
// 		return Conv2D(filter, b, isSamePad)
// 	}
// 	m.Forwards = append(m.Forwards, forward)
// }

// func (m *Model) AppendLeakyReLU(alpha float32) {
// 	forward := func(img Image) (Image, error) {
// 		x := img.Data
// 		y := make([]float32, len(img.Data))
// 		for i := range y {
// 			xi := img.Data[i]
// 			if xi > 0 {
// 				y[i] = xi
// 			} else {
// 				y[i] = xi*alpha
// 			}
// 		}

// 		return Image{
// 			Channels:img.Channels,
// 			Rows:img.Rows,
// 			Cols:img.Cols,
// 			ChannelStride:img.Rows*img.Cols,
// 			Data:y,
// 		}
// 	}
// 	m.Forwards = append(m.Forwards, forward)
// }

// func (m *Model)