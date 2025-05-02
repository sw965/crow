package main

import (
	"fmt"
	"strings"
	"slices"
	"math"
	"gonum.org/v1/gonum/blas/blas32"
	"gonum.org/v1/gonum/blas"
	"strconv"
	"math/rand"
	omwrand "github.com/sw965/omw/math/rand"

	"image"
	_ "image/gif"  // デコーダ登録
	_ "image/jpeg" //   〃
	"image/png"  //   〃
	"os"

	"image/color"
)

func clampUint8(v float32) uint8 {
	switch {
	case v < 0:
		return 0
	case v > 255:
		return 255
	default:
		return uint8(v + 0.5) // 四捨五入
	}
}

type Image struct {
	Channels      int
	Rows          int
	Cols          int
	ChannelStride int
	Data          []float32
}

func LoadImage(path string) (Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return Image{}, err
	}
	defer f.Close()

	src, _, err := image.Decode(f)
	if err != nil {
		return Image{}, err
	}
	bounds := src.Bounds()
	rows, cols := bounds.Dy(), bounds.Dx()

	chs := 3
	if _, ok := src.(*image.Gray); ok {
		chs = 1
	}

	img := NewImageZeros(chs, rows, cols)

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := src.At(x, y).RGBA() // 0-65535
			r8, g8, b8 := float32(r>>8), float32(g>>8), float32(b>>8)

			base := (y-bounds.Min.Y)*cols + (x - bounds.Min.X)
			switch chs {
			case 1:
				lum := (r8 + g8 + b8) / 3
				img.Data[base] = lum
			case 3:
				img.Data[base] = r8
				img.Data[rows*cols+base] = g8
				img.Data[2*rows*cols+base] = b8
			}
		}
	}
	return img, nil
}

func NewImageZeros(chs, rows, cols int) Image {
	data := make([]float32, chs*rows*cols)
	return Image{
		Channels:      chs,
		Rows:          rows,
		Cols:          cols,
		ChannelStride: rows * cols,
		Data:          data,
	}
}

func NewImageOnes(chs, rows, cols int) Image {
	img := NewImageZeros(chs, rows, cols)
	for i := range img.Data {
		img.Data[i] = 1.0
	}
	return img
}

func (img Image) FlatIndex(ch, row, col int) int {
	return ch*img.ChannelStride + row*img.Cols + col
}

func (img Image) OutputRows(filterRows int) int {
	return img.Rows - filterRows + 1
}

func (img Image) OutputCols(filterCols int) int {
	return img.Cols - filterCols + 1
}

func (img Image) ZeroPadding2D(top, bot, left, right int) Image {
    padded := NewImageZeros(img.Channels, img.Rows+top+bot, img.Cols+left+right)
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

func (img Image) SameZeroPadding2D(filterRows, filterCols int) Image {
	top := (filterRows - 1) / 2
    bot := filterRows - 1 - top
    left := (filterCols - 1) / 2
    right := filterCols - 1 - left
    return img.ZeroPadding2D(top, bot, left, right)
}

func(img Image) ToCol(filterRows, filterCols int) blas32.General {
	chs := img.Channels
	outRows := img.OutputRows(filterRows)
	outCols := img.OutputCols(filterCols)
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

func (img Image) ToString() string {
	const width = 7 // "123.45 " 程度
	est := img.Channels*(img.Rows*img.Cols*width + img.Rows + 16)
	var sb strings.Builder
	sb.Grow(est)

	for ch := 0; ch < img.Channels; ch++ {
		sb.WriteString("Channel:")
		sb.WriteString(strconv.Itoa(ch))
		sb.WriteByte('\n')

		for r := 0; r < img.Rows; r++ {
			rowBase := img.FlatIndex(ch, r, 0)
			for c := 0; c < img.Cols; c++ {
				v := img.Data[rowBase+c]
				// strconv.AppendFloat は dst を返すのでそのまま使う
				sb.Write(strconv.AppendFloat(make([]byte, 0, width),
					float64(v), 'f', 2, 32))
				sb.WriteByte(' ')
			}
			sb.WriteByte('\n')
		}
		sb.WriteByte('\n')
	}
	return sb.String()
}

func (img Image) SavePNG(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	rect := image.Rect(0, 0, img.Cols, img.Rows)

	switch img.Channels {
	case 1: // Grayscale
		dst := image.NewGray(rect)
		for y := 0; y < img.Rows; y++ {
			for x := 0; x < img.Cols; x++ {
				v := clampUint8(img.Data[y*img.Cols+x])
				dst.SetGray(x, y, color.Gray{Y: v})
			}
		}
		return png.Encode(f, dst)

	case 3, 4: // RGB or RGBA
		dst := image.NewRGBA(rect)
		for y := 0; y < img.Rows; y++ {
			for x := 0; x < img.Cols; x++ {
				base := y*img.Cols + x
				r := clampUint8(img.Data[base])
				g := clampUint8(img.Data[img.ChannelStride+base])
				b := clampUint8(img.Data[2*img.ChannelStride+base])
				a := uint8(255)
				if img.Channels == 4 {
					a = clampUint8(img.Data[3*img.ChannelStride+base])
				}
				dst.SetRGBA(x, y, color.RGBA{R: r, G: g, B: b, A: a})
			}
		}
		return png.Encode(f, dst)

	default:
		return fmt.Errorf("unsupported channel count: %d", img.Channels)
	}
}

type Filter struct {
    Batches       int
	Channels      int
	Rows          int
	Cols          int
    BatchStride   int
    Data          []float32
}

func NewFilterHe(batches, chs, rows, cols int, rng *rand.Rand) Filter {
	in := chs * rows * cols
	std := math.Sqrt(2.0 / float64(in))
	dataLen := batches * in

	data := make([]float32, dataLen)
	for i := 0; i < dataLen; i++ {
		data[i] = float32(rng.NormFloat64() * std)
	}

	return Filter{
		Batches:     batches,
		Channels:    chs,
		Rows:        rows,
		Cols:        cols,
		BatchStride: in,
		Data:        data,
	}
}

func (f Filter) ToGeneral() blas32.General {
	return blas32.General{
		Rows:f.Batches,
		Cols:f.BatchStride,
		Stride:f.BatchStride,
		Data:slices.Clone(f.Data),
	}
}

func DotResultToImage(result blas32.General, outRows, outCols int) (Image, error) {
	if result.Rows != outRows*outCols {
		err := fmt.Errorf("dimension mismatch: res.Rows=%d, want %d (=outRows*outCols)")
		return Image{}, err
	}

	img := NewImageZeros(result.Cols, outRows, outCols)

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

func Conv2D(img Image, filter Filter, b blas32.Vector, isSamePad bool) (Image, error) {
	if len(b.Data) != b.N {
		return Image{}, fmt.Errorf("len(b.Data) != b.N")
	}

	if filter.Batches != b.N {
		return Image{}, fmt.Errorf("filter.Batches != b.N")
	}

	if isSamePad {
		img = img.SameZeroPadding2D(filter.Rows, filter.Cols)
	}

	imgCol := img.ToCol(filter.Rows, filter.Cols)
	filterGen := filter.ToGeneral()

	dotRows := imgCol.Rows
	dotCols := filter.Batches

	dotResult := blas32.General{
		Rows:dotRows,
		Cols:dotCols,
		Stride:dotCols,
		Data:make([]float32, dotRows*dotCols),
	}

	for row := 0; row < dotRows; row++ {
        base := row * dotCols
        for col := 0; col < dotCols; col++ {
            dotResult.Data[base+col] += b.Data[col]
        }
    }

	blas32.Gemm(blas.NoTrans, blas.Trans, 1.0, imgCol, filterGen, 1.0, dotResult)
	return DotResultToImage(dotResult, img.OutputRows(filter.Rows), img.OutputCols(filter.Cols))
}