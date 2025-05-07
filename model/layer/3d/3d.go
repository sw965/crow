package layer3d

func im2Col(img tensor3d.General, filterRows, filterCols int) blas32.General {
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