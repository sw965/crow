package nn

import (
	"github.com/sw965/crow/tensor"
)

func ConvOutputRows(img tensor.D3, filterRows, stride int) int {
    return (img.Rows-filterRows)/stride + 1
}

func ConvOutputCols(img tensor.D3, filterCols, stride int) int {
    return (img.Cols-filterCols)/stride + 1
}

func ZeroPadding2D(img tensor.D3, top, bot, left, right int) tensor.D3 {
    padded := tensor.NewD3Zeros(img.Channels, img.Rows+top+bot, img.Cols+left+right)
    for ch := 0; ch < img.Channels; ch++ {
        for row := 0; row < img.Rows; row++ {
            for col := 0; col < img.Cols; col++ {
                oldIdx := img.At(ch, row, col)
                newIdx := padded.At(ch, row+top, col+left)
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

func Im2Col(img tensor.D3, filterRows, filterCols, stride int) tensor.D2 {
    chs := img.Channels
    outRows := ConvOutputRows(img, filterRows, stride)
    outCols := ConvOutputCols(img, filterCols, stride)

    imgData := img.Data
    newData := make([]float32, outRows*outCols*chs*filterRows*filterCols)
    newIdx := 0

    for or := 0; or < outRows; or++ {
        baseRow := or * stride
        for oc := 0; oc < outCols; oc++ {
            baseCol := oc * stride
            for ch := 0; ch < chs; ch++ {
                for fr := 0; fr < filterRows; fr++ {
                    row := baseRow + fr
                    for fc := 0; fc < filterCols; fc++ {
                        col := baseCol + fc
                        imgIdx := img.At(ch, row, col)
                        newData[newIdx] = imgData[imgIdx]
                        newIdx++
                    }
                }
            }
        }
    }

    newCols := filterRows * filterCols * chs
    return tensor.D2{
        Rows:   outRows * outCols,
        Cols:   newCols,
        Stride: newCols,
        Data:   newData,
    }
}

func Col2Im(col tensor.D2, imgShape tensor.D3, filterRows, filterCols, stride int) tensor.D3 {
	chs := imgShape.Channels
	outRows := ConvOutputRows(imgShape, filterRows, stride)
	outCols := ConvOutputCols(imgShape, filterCols, stride)

	if col.Rows != outRows*outCols {
		panic("Col2Im: unexpected number of rows")
	}
	if col.Cols != chs*filterRows*filterCols {
		panic("Col2Im: unexpected number of cols")
	}

	im := make([]float32, imgShape.N())
	colIdx := 0

	for orow := 0; orow < outRows; orow++ {
		for ocol := 0; ocol < outCols; ocol++ {
			for ch := 0; ch < chs; ch++ {
				for fr := 0; fr < filterRows; fr++ {
					for fc := 0; fc < filterCols; fc++ {
						row := fr + orow*stride
						colPos := fc + ocol*stride
						imgIdx := imgShape.At(ch, row, colPos)
						im[imgIdx] += col.Data[colIdx]
						colIdx++
					}
				}
			}
		}
	}

	return tensor.D3{
		Channels:      chs,
		Rows:          imgShape.Rows,
		Cols:          imgShape.Cols,
		ChannelStride: imgShape.ChannelStride,
		RowStride:     imgShape.RowStride,
		Data:          im,
	}
}

func Conv2D(img tensor.D3, filter tensor.D4, stride int) tensor.D3 {
    col := Im2Col(img, filter.Rows, filter.Cols, stride)
    colFilter := filter.ToD1().Reshape2D(filter.Batches, -1).Transpose()

    outRows := ConvOutputRows(img, filter.Rows, stride)
    outCols := ConvOutputCols(img, filter.Cols, stride)

    colOut := col.NoTransDotNoTrans(colFilter)
    y := colOut.ToD1().Reshape3D(outRows, outCols, -1).Transpose201()
    return y
}

func Conv2DWithColVar(img tensor.D3, filter tensor.D4, stride int) (tensor.D3, tensor.D2, tensor.D2) {
    col := Im2Col(img, filter.Rows, filter.Cols, stride)
    colFilter := filter.ToD1().Reshape2D(filter.Batches, -1).Transpose()

    outRows := ConvOutputRows(img, filter.Rows, stride)
    outCols := ConvOutputCols(img, filter.Cols, stride)

    colOut := col.NoTransDotNoTrans(colFilter)
    y := colOut.ToD1().Reshape3D(outRows, outCols, -1).Transpose201()
    return y, col, colFilter
}