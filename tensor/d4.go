package tensor

import (
	"gonum.org/v1/gonum/blas/blas32"
	"slices"
	"math"
	"math/rand"
	crand "github.com/sw965/crow/math/rand"
)

type D4 struct {
	Batches       int
	Channels      int
	Rows          int
	Cols          int
	BatchStride   int
	ChannelStride int
	RowStride     int
	Data          []float32
}

func NewD4Zeros(batches, chs, rows, cols int) D4 {
	rowStride := cols
	chStride := rows * rowStride
	batchStride := chs * chStride
	n := batches * batchStride

	return D4{
		Batches:       batches,
		Channels:      chs,
		Rows:          rows,
		Cols:          cols,
		BatchStride:   batchStride,
		ChannelStride: chStride,
		RowStride:     rowStride,
		Data:          make([]float32, n),
	}
}

func NewD4Ones(batches, chs, rows, cols int) D4 {
	gen := NewD4Zeros(batches, chs, rows, cols)
	for i := range gen.Data {
		gen.Data[i] = 1.0
	}
	return gen
}

func NewD4Rademacher(batches, chs, rows, cols int, rng *rand.Rand) D4 {
	gen := NewD4Zeros(batches, chs, rows, cols)
	for i := range gen.Data {
		gen.Data[i] = crand.Rademacher(rng)
	}
	return gen
}

func NewD4He(batches, chs, rows, cols int, rng *rand.Rand) D4 {
    gen := NewD4Zeros(batches, chs, rows, cols)
    fanIn := float64(chs * rows * cols)
    std := float32(math.Sqrt(2.0 / fanIn))
    for i := range gen.Data {
        gen.Data[i] = float32(rng.NormFloat64()) * std
    }
    return gen
}

func (d4 D4) NewZerosLike() D4 {
	return NewD4Zeros(d4.Batches, d4.Channels, d4.Rows, d4.Cols)
}

func (d4 D4) NewOnesLike() D4 {
	return NewD4Ones(d4.Batches, d4.Channels, d4.Rows, d4.Cols)
}

func (d4 D4) NewRademacherLike(rng *rand.Rand) D4 {
	return NewD4Rademacher(d4.Batches, d4.Channels, d4.Rows, d4.Cols, rng)
}

func (d4 D4) N() int {
	return d4.Batches * d4.Channels * d4.Rows * d4.Cols
}

func (g D4) Clone() D4 {
	return D4{
		Batches:       g.Batches,
		Channels:      g.Channels,
		Rows:          g.Rows,
		Cols:          g.Cols,
		BatchStride:   g.BatchStride,
		ChannelStride: g.ChannelStride,
		RowStride:     g.RowStride,
		Data:          slices.Clone(g.Data),
	}
}

func (d4 D4) At(batch, ch, row, col int) int {
	return (batch * d4.BatchStride) + (ch * d4.ChannelStride) + (row * d4.RowStride) + col
}

func (d4 D4) ToD1() D1 {
	return D1{
		N:d4.N(),
		Inc:1,
		Data:slices.Clone(d4.Data),
	}
}

func (d4 D4) ToBlas32Vector() blas32.Vector {
	return blas32.Vector{
		N:    d4.N(),
		Inc:  1,
		Data: slices.Clone(d4.Data),
	}
}

func (d4 D4) Axpy(alpha float32, x D4) D4 {
	yv := d4.ToBlas32Vector()
	xv := x.ToBlas32Vector()
	blas32.Axpy(alpha, xv, yv)
	return D4{
		Batches:d4.Batches,
		Channels:d4.Channels,
		Rows:d4.Rows,
		Cols:d4.Cols,
		BatchStride:d4.BatchStride,
		ChannelStride:d4.ChannelStride,
		RowStride:d4.RowStride,
		Data:yv.Data,
	}
}

func (d4 D4) Scal(alpha float32) D4 {
	v := d4.ToBlas32Vector()
	blas32.Scal(alpha, v)
	return D4{
		Batches:d4.Batches,
		Channels:d4.Channels,
		Rows:d4.Rows,
		Cols:d4.Cols,
		BatchStride:d4.BatchStride,
		ChannelStride:d4.ChannelStride,
		RowStride:d4.RowStride,
		Data:v.Data,
	}
}

func (g *D4) Transpose0132() D4 {
    t := NewD4Zeros(g.Batches, g.Channels, g.Cols, g.Rows)
    idx := 0
    for b := 0; b < g.Batches; b++ {
        for c := 0; c < g.Channels; c++ {
            for col := 0; col < g.Cols; col++ {
                for r := 0; r < g.Rows; r++ {
                    t.Data[idx] = g.Data[g.At(b, c, r, col)]
                    idx++
                }
            }
        }
    }
    return t
}

func (g *D4) Transpose0213() D4 {
    t := NewD4Zeros(g.Batches, g.Rows, g.Channels, g.Cols)
    idx := 0
    for b := 0; b < g.Batches; b++ {
        for r := 0; r < g.Rows; r++ {
            for c := 0; c < g.Channels; c++ {
                srcBase := g.At(b, c, r, 0)
                copy(t.Data[idx:idx+g.Cols], g.Data[srcBase:srcBase+g.Cols])
                idx += g.Cols
            }
        }
    }
    return t
}

func (g *D4) Transpose0231() D4 {
    t := NewD4Zeros(g.Batches, g.Rows, g.Cols, g.Channels)
    idx := 0
    for b := 0; b < g.Batches; b++ {
        for r := 0; r < g.Rows; r++ {
            for col := 0; col < g.Cols; col++ {
                for c := 0; c < g.Channels; c++ {
                    t.Data[idx] = g.Data[g.At(b, c, r, col)]
                    idx++
                }
            }
        }
    }
    return t
}

func (g *D4) Transpose0312() D4 {
    t := NewD4Zeros(g.Batches, g.Cols, g.Channels, g.Rows)
    idx := 0
    for b := 0; b < g.Batches; b++ {
        for col := 0; col < g.Cols; col++ {
            for c := 0; c < g.Channels; c++ {
                srcBase := g.At(b, c, 0, col)
                for r := 0; r < g.Rows; r++ {
                    t.Data[idx] = g.Data[srcBase+r*g.RowStride]
                    idx++
                }
            }
        }
    }
    return t
}

func (g *D4) Transpose0321() D4 {
    t := NewD4Zeros(g.Batches, g.Cols, g.Rows, g.Channels)
    idx := 0
    for b := 0; b < g.Batches; b++ {
        for col := 0; col < g.Cols; col++ {
            for r := 0; r < g.Rows; r++ {
                for c := 0; c < g.Channels; c++ {
                    t.Data[idx] = g.Data[g.At(b, c, r, col)]
                    idx++
                }
            }
        }
    }
    return t
}

func (g *D4) Transpose1023() D4 {
    t := NewD4Zeros(g.Channels, g.Batches, g.Rows, g.Cols)
    idx := 0
    for c := 0; c < g.Channels; c++ {
        for b := 0; b < g.Batches; b++ {
            srcBase := g.At(b, c, 0, 0)
            copy(t.Data[idx:idx+g.Rows*g.Cols], g.Data[srcBase:srcBase+g.Rows*g.Cols])
            idx += g.Rows * g.Cols
        }
    }
    return t
}

func (g *D4) Transpose1032() D4 {
    t := NewD4Zeros(g.Channels, g.Batches, g.Cols, g.Rows)
    idx := 0
    for c := 0; c < g.Channels; c++ {
        for b := 0; b < g.Batches; b++ {
            for col := 0; col < g.Cols; col++ {
                for r := 0; r < g.Rows; r++ {
                    t.Data[idx] = g.Data[g.At(b, c, r, col)]
                    idx++
                }
            }
        }
    }
    return t
}

func (g *D4) Transpose1203() D4 {
    t := NewD4Zeros(g.Channels, g.Rows, g.Batches, g.Cols)
    idx := 0
    for c := 0; c < g.Channels; c++ {
        for r := 0; r < g.Rows; r++ {
            for b := 0; b < g.Batches; b++ {
                srcBase := g.At(b, c, r, 0)
                copy(t.Data[idx:idx+g.Cols], g.Data[srcBase:srcBase+g.Cols])
                idx += g.Cols
            }
        }
    }
    return t
}

func (g *D4) Transpose1230() D4 {
    t := NewD4Zeros(g.Channels, g.Rows, g.Cols, g.Batches)
    idx := 0
    for c := 0; c < g.Channels; c++ {
        for r := 0; r < g.Rows; r++ {
            for col := 0; col < g.Cols; col++ {
                for b := 0; b < g.Batches; b++ {
                    t.Data[idx] = g.Data[g.At(b, c, r, col)]
                    idx++
                }
            }
        }
    }
    return t
}

func (g *D4) Transpose1302() D4 {
    t := NewD4Zeros(g.Channels, g.Cols, g.Batches, g.Rows)
    idx := 0
    for c := 0; c < g.Channels; c++ {
        for col := 0; col < g.Cols; col++ {
            for b := 0; b < g.Batches; b++ {
                for r := 0; r < g.Rows; r++ {
                    t.Data[idx] = g.Data[g.At(b, c, r, col)]
                    idx++
                }
            }
        }
    }
    return t
}

func (g *D4) Transpose1320() D4 {
    t := NewD4Zeros(g.Channels, g.Cols, g.Rows, g.Batches)
    idx := 0
    for c := 0; c < g.Channels; c++ {
        for col := 0; col < g.Cols; col++ {
            for r := 0; r < g.Rows; r++ {
                for b := 0; b < g.Batches; b++ {
                    t.Data[idx] = g.Data[g.At(b, c, r, col)]
                    idx++
                }
            }
        }
    }
    return t
}

func (g *D4) Transpose2013() D4 {
    t := NewD4Zeros(g.Rows, g.Batches, g.Channels, g.Cols)
    idx := 0
    for r := 0; r < g.Rows; r++ {
        for b := 0; b < g.Batches; b++ {
            for c := 0; c < g.Channels; c++ {
                srcBase := g.At(b, c, r, 0)
                copy(t.Data[idx:idx+g.Cols], g.Data[srcBase:srcBase+g.Cols])
                idx += g.Cols
            }
        }
    }
    return t
}

func (g *D4) Transpose2031() D4 {
    t := NewD4Zeros(g.Rows, g.Batches, g.Cols, g.Channels)
    idx := 0
    for r := 0; r < g.Rows; r++ {
        for b := 0; b < g.Batches; b++ {
            for col := 0; col < g.Cols; col++ {
                for c := 0; c < g.Channels; c++ {
                    t.Data[idx] = g.Data[g.At(b, c, r, col)]
                    idx++
                }
            }
        }
    }
    return t
}

func (g *D4) Transpose2103() D4 {
    t := NewD4Zeros(g.Rows, g.Channels, g.Batches, g.Cols)
    idx := 0
    for r := 0; r < g.Rows; r++ {
        for c := 0; c < g.Channels; c++ {
            for b := 0; b < g.Batches; b++ {
                srcBase := g.At(b, c, r, 0)
                copy(t.Data[idx:idx+g.Cols], g.Data[srcBase:srcBase+g.Cols])
                idx += g.Cols
            }
        }
    }
    return t
}

func (g *D4) Transpose2130() D4 {
    t := NewD4Zeros(g.Rows, g.Channels, g.Cols, g.Batches)
    idx := 0
    for r := 0; r < g.Rows; r++ {
        for c := 0; c < g.Channels; c++ {
            for col := 0; col < g.Cols; col++ {
                for b := 0; b < g.Batches; b++ {
                    t.Data[idx] = g.Data[g.At(b, c, r, col)]
                    idx++
                }
            }
        }
    }
    return t
}

func (g *D4) Transpose2301() D4 {
    t := NewD4Zeros(g.Rows, g.Cols, g.Batches, g.Channels)
    idx := 0
    for r := 0; r < g.Rows; r++ {
        for col := 0; col < g.Cols; col++ {
            for b := 0; b < g.Batches; b++ {
                for c := 0; c < g.Channels; c++ {
                    t.Data[idx] = g.Data[g.At(b, c, r, col)]
                    idx++
                }
            }
        }
    }
    return t
}

func (g *D4) Transpose2310() D4 {
    t := NewD4Zeros(g.Rows, g.Cols, g.Channels, g.Batches)
    idx := 0
    for r := 0; r < g.Rows; r++ {
        for col := 0; col < g.Cols; col++ {
            for c := 0; c < g.Channels; c++ {
                for b := 0; b < g.Batches; b++ {
                    t.Data[idx] = g.Data[g.At(b, c, r, col)]
                    idx++
                }
            }
        }
    }
    return t
}

func (g *D4) Transpose3012() D4 {
    t := NewD4Zeros(g.Cols, g.Batches, g.Channels, g.Rows)
    idx := 0
    for col := 0; col < g.Cols; col++ {
        for b := 0; b < g.Batches; b++ {
            for c := 0; c < g.Channels; c++ {
                srcBase := g.At(b, c, 0, col)
                for r := 0; r < g.Rows; r++ {
                    t.Data[idx] = g.Data[srcBase+r*g.RowStride]
                    idx++
                }
            }
        }
    }
    return t
}

func (g *D4) Transpose3021() D4 {
    t := NewD4Zeros(g.Cols, g.Batches, g.Rows, g.Channels)
    idx := 0
    for col := 0; col < g.Cols; col++ {
        for b := 0; b < g.Batches; b++ {
            for r := 0; r < g.Rows; r++ {
                for c := 0; c < g.Channels; c++ {
                    t.Data[idx] = g.Data[g.At(b, c, r, col)]
                    idx++
                }
            }
        }
    }
    return t
}

func (g *D4) Transpose3102() D4 {
    t := NewD4Zeros(g.Cols, g.Channels, g.Batches, g.Rows)
    idx := 0
    for col := 0; col < g.Cols; col++ {
        for c := 0; c < g.Channels; c++ {
            for b := 0; b < g.Batches; b++ {
                srcBase := g.At(b, c, 0, col)
                for r := 0; r < g.Rows; r++ {
                    t.Data[idx] = g.Data[srcBase+r*g.RowStride]
                    idx++
                }
            }
        }
    }
    return t
}

func (g *D4) Transpose3120() D4 {
    t := NewD4Zeros(g.Cols, g.Channels, g.Rows, g.Batches)
    idx := 0
    for col := 0; col < g.Cols; col++ {
        for c := 0; c < g.Channels; c++ {
            for r := 0; r < g.Rows; r++ {
                for b := 0; b < g.Batches; b++ {
                    t.Data[idx] = g.Data[g.At(b, c, r, col)]
                    idx++
                }
            }
        }
    }
    return t
}

func (g *D4) Transpose3201() D4 {
    t := NewD4Zeros(g.Cols, g.Rows, g.Batches, g.Channels)
    idx := 0
    for col := 0; col < g.Cols; col++ {
        for r := 0; r < g.Rows; r++ {
            for b := 0; b < g.Batches; b++ {
                for c := 0; c < g.Channels; c++ {
                    t.Data[idx] = g.Data[g.At(b, c, r, col)]
                    idx++
                }
            }
        }
    }
    return t
}

func (g *D4) Transpose3210() D4 {
    t := NewD4Zeros(g.Cols, g.Rows, g.Channels, g.Batches)
    idx := 0
    for col := 0; col < g.Cols; col++ {
        for r := 0; r < g.Rows; r++ {
            for c := 0; c < g.Channels; c++ {
                for b := 0; b < g.Batches; b++ {
                    t.Data[idx] = g.Data[g.At(b, c, r, col)]
                    idx++
                }
            }
        }
    }
    return t
}