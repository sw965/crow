func LeakyReLU1D(x tensor.D1, alpha float32) D1 {
	y := make([]float32, x.N)
	for i, e := range x.Data {
		if e > 0 {
			y[i] = e
		} else {
			y[i] = alpha * e
		}
	}

	x.Data = y
	return x
}

func LeakyReLU1DDerivative(x tensor.D1, alpha float32) tensor.D1 {
	grad := make([]float32, x.N)
	for i, e := range x.Data {
		if e > 0 {
			grad[i] = 1.0
		} else {
			grad[i] = alpha
		}
	}
	x.Data = grad
	return x
}

func Col2Im(col blas32.General, imgShape tensor3d.General, filterRows, filterCols int) tensor3d.General {
	chs := imgShape.Channels
	outRows := imgShape.ConvOutputRows(filterRows)
	outCols := imgShape.ConvOutputCols(filterCols)

	// （念のための簡易チェック）
	if col.Rows != outRows*outCols {
		panic("Col2Im: unexpected number of rows")
	}
	if col.Cols != chs*filterRows*filterCols {
		panic("Col2Im: unexpected number of cols")
	}

	recon := make([]float32, len(imgShape.Data))
	colIdx := 0

	for or := 0; or < outRows; or++ {
		for oc := 0; oc < outCols; oc++ {
			for ch := 0; ch < chs; ch++ {
				for fr := 0; fr < filterRows; fr++ {
					for fc := 0; fc < filterCols; fc++ {
						row := fr + or
						colPos := fc + oc
						imgIdx := imgShape.At(ch, row, colPos)
						recon[imgIdx] += col.Data[colIdx]
						colIdx++
					}
				}
			}
		}
	}

	return tensor3d.General{
		Channels:      imgShape.Channels,
		Rows:          imgShape.Rows,
		Cols:          imgShape.Cols,
		ChannelStride: imgShape.ChannelStride,
		RowStride:     imgShape.RowStride,
		Data:          recon,
	}
}

func (imd3 D3) ZeroPadding2D(top, bot, left, right int) D3 {
    padded := NewD3Zeros(imd3.Channels, imd3.Rows+top+bot, imd3.Cols+left+right)
    for ch := 0; ch < imd3.Channels; ch++ {
        for row := 0; row < imd3.Rows; row++ {
            for col := 0; col < imd3.Cols; col++ {
                oldIdx := imd3.At(ch, row, col)
                newIdx := padded.At(ch, row+top, col+left)
                padded.Data[newIdx] = imd3.Data[oldIdx]
            }
        }
    }
    return padded
}

func (imd3 D3) SameZeroPadding2D(filterRows, filterCols int) D3 {
	top := (filterRows - 1) / 2
    bot := filterRows - 1 - top
    left := (filterCols - 1) / 2
    right := filterCols - 1 - left
    return imd3.ZeroPadding2D(top, bot, left, right)
}

func (imd3 D3) ConvOutputRows(filterRows int) int {
	return imd3.Rows - filterRows + 1
}

func (imd3 D3) ConvOutputCols(filterCols int) int {
	return imd3.Cols - filterCols + 1
}

func (imd3 D3) ToCol(filterRows, filterCols int) blas32.D3 {
	chs := imd3.Channels
	outRows := imd3.ConvOutputRows(filterRows)
	outCols := imd3.ConvOutputCols(filterCols)
	imgData := imd3.Data
	newData := make([]float32, outRows*outCols*chs*filterRows*filterCols)
	newIdx := 0

	for or := 0; or < outRows; or++ {
		for oc := 0; oc < outCols; oc++ {
			for ch := 0; ch < chs; ch++ {
				for fr := 0; fr < filterRows; fr++ {
					for fc := 0; fc < filterCols; fc++ {
						row := fr + or
						col := fc + oc
						imgIdx := imd3.At(ch, row, col)
						newData[newIdx] = imgData[imgIdx]
						newIdx++
					}
				}
			}
		}
	}

	newCols := filterRows*filterCols*chs
	return blas32.D3{
		Rows:outRows*outCols,
		Cols:newCols,
		Stride:newCols,
		Data:newData,
	}
}