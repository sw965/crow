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

// func Col2Im(col tensor.D2 img tensor.D3, filterRows, filterCols int) tensor.D3 {
// 	chs := imgShape.Channels
// 	outRows := imgShape.ConvOutputRows(filterRows)
// 	outCols := imgShape.ConvOutputCols(filterCols)

// 	// （念のための簡易チェック）
// 	if col.Rows != outRows*outCols {
// 		panic("Col2Im: unexpected number of rows")
// 	}
// 	if col.Cols != chs*filterRows*filterCols {
// 		panic("Col2Im: unexpected number of cols")
// 	}

// 	recon := make([]float32, len(imgShape.Data))
// 	colIdx := 0

// 	for or := 0; or < outRows; or++ {
// 		for oc := 0; oc < outCols; oc++ {
// 			for ch := 0; ch < chs; ch++ {
// 				for fr := 0; fr < filterRows; fr++ {
// 					for fc := 0; fc < filterCols; fc++ {
// 						row := fr + or
// 						colPos := fc + oc
// 						imgIdx := imgShape.At(ch, row, colPos)
// 						recon[imgIdx] += col.Data[colIdx]
// 						colIdx++
// 					}
// 				}
// 			}
// 		}
// 	}

// 	return tensor3d.General{
// 		Channels:      imgShape.Channels,
// 		Rows:          imgShape.Rows,
// 		Cols:          imgShape.Cols,
// 		ChannelStride: imgShape.ChannelStride,
// 		RowStride:     imgShape.RowStride,
// 		Data:          recon,
// 	}
// }

// func (imd3 D3) ZeroPadding2D(top, bot, left, right int) D3 {
//     padded := NewD3Zeros(imd3.Channels, imd3.Rows+top+bot, imd3.Cols+left+right)
//     for ch := 0; ch < imd3.Channels; ch++ {
//         for row := 0; row < imd3.Rows; row++ {
//             for col := 0; col < imd3.Cols; col++ {
//                 oldIdx := imd3.At(ch, row, col)
//                 newIdx := padded.At(ch, row+top, col+left)
//                 padded.Data[newIdx] = imd3.Data[oldIdx]
//             }
//         }
//     }
//     return padded
// }

// func (imd3 D3) SameZeroPadding2D(filterRows, filterCols int) D3 {
// 	top := (filterRows - 1) / 2
//     bot := filterRows - 1 - top
//     left := (filterCols - 1) / 2
//     right := filterCols - 1 - left
//     return imd3.ZeroPadding2D(top, bot, left, right)
// }

// func (imd3 D3) ToCol(filterRows, filterCols int) blas32.D3 {
// 	chs := imd3.Channels
// 	outRows := imd3.ConvOutputRows(filterRows)
// 	outCols := imd3.ConvOutputCols(filterCols)
// 	imgData := imd3.Data
// 	newData := make([]float32, outRows*outCols*chs*filterRows*filterCols)
// 	newIdx := 0

// 	for or := 0; or < outRows; or++ {
// 		for oc := 0; oc < outCols; oc++ {
// 			for ch := 0; ch < chs; ch++ {
// 				for fr := 0; fr < filterRows; fr++ {
// 					for fc := 0; fc < filterCols; fc++ {
// 						row := fr + or
// 						col := fc + oc
// 						imgIdx := imd3.At(ch, row, col)
// 						newData[newIdx] = imgData[imgIdx]
// 						newIdx++
// 					}
// 				}
// 			}
// 		}
// 	}

// 	newCols := filterRows*filterCols*chs
// 	return blas32.D3{
// 		Rows:outRows*outCols,
// 		Cols:newCols,
// 		Stride:newCols,
// 		Data:newData,
// 	}
// }