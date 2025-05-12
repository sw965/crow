package tensor4d

import (
	"gonum.org/v1/gonum/blas/blas32"
	"slices"
	"math"
	"math/rand"
	crand "github.com/sw965/crow/math/rand"
)

type General struct {
	Batches       int
	Channels      int
	Rows          int
	Cols          int
	BatchStride   int
	ChannelStride int
	RowStride     int
	Data          []float32
}

func NewZeros(batches, chs, rows, cols int) General {
	rowStride := cols
	chStride := rows * rowStride
	batchStride := chs * chStride
	n := batches * batchStride

	return General{
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

func NewZerosLike(gen General) General {
	return NewZeros(gen.Batches, gen.Channels, gen.Rows, gen.Cols)
}

func NewOnes(batches, chs, rows, cols int) General {
	gen := NewZeros(batches, chs, rows, cols)
	for i := range gen.Data {
		gen.Data[i] = 1.0
	}
	return gen
}

func NewOnesLike(gen General) General {
	return NewOnes(gen.Batches, gen.Channels, gen.Rows, gen.Cols)
}

func NewHe(batches, chs, rows, cols int, rng *rand.Rand) General {
    gen := NewZeros(batches, chs, rows, cols)
    fanIn := float64(chs * rows * cols)
    std := float32(math.Sqrt(2.0 / fanIn))
    for i := range gen.Data {
        gen.Data[i] = float32(rng.NormFloat64()) * std
    }
    return gen
}

func NewRademacher(batches, chs, rows, cols int, rng *rand.Rand) General {
	gen := NewZeros(batches, chs, rows, cols)
	for i := range gen.Data {
		gen.Data[i] = crand.Rademacher(rng)
	}
	return gen
}

func NewRademacherLike(gen General, rng *rand.Rand) General {
	return NewRademacher(gen.Batches, gen.Channels, gen.Rows, gen.Cols, rng)
}

func (g General) N() int {
	return g.Batches * g.Channels * g.Rows * g.Cols
}

func (g General) Clone() General {
	return General{
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

func (g General) At(batch, ch, row, col int) int {
	return (batch * g.BatchStride) + (ch * g.ChannelStride) + (row * g.RowStride) + col
}

func (g General) ToVector() blas32.Vector {
	return blas32.Vector{
		N:    g.N(),
		Inc:  1,
		Data: g.Data,
	}
}

func (g General) Axpy(alpha float32, x General) {
	xv := x.ToVector()
	yv := g.ToVector()
	blas32.Axpy(alpha, xv, yv)
}

func (g *General) Transpose0132() General {
    dst := NewZeros(g.Batches, g.Channels, g.Cols, g.Rows)
    idx := 0
    for b := 0; b < g.Batches; b++ {
        for c := 0; c < g.Channels; c++ {
            for col := 0; col < g.Cols; col++ {
                for r := 0; r < g.Rows; r++ {
                    dst.Data[idx] = g.Data[g.At(b, c, r, col)]
                    idx++
                }
            }
        }
    }
    return dst
}

func (g *General) Transpose0213() General {
    dst := NewZeros(g.Batches, g.Rows, g.Channels, g.Cols)
    idx := 0
    for b := 0; b < g.Batches; b++ {
        for r := 0; r < g.Rows; r++ {
            for c := 0; c < g.Channels; c++ {
                srcBase := g.At(b, c, r, 0)
                copy(dst.Data[idx:idx+g.Cols], g.Data[srcBase:srcBase+g.Cols])
                idx += g.Cols
            }
        }
    }
    return dst
}

func (g *General) Transpose0231() General {
    dst := NewZeros(g.Batches, g.Rows, g.Cols, g.Channels)
    idx := 0
    for b := 0; b < g.Batches; b++ {
        for r := 0; r < g.Rows; r++ {
            for col := 0; col < g.Cols; col++ {
                for c := 0; c < g.Channels; c++ {
                    dst.Data[idx] = g.Data[g.At(b, c, r, col)]
                    idx++
                }
            }
        }
    }
    return dst
}

func (g *General) Transpose0312() General {
    dst := NewZeros(g.Batches, g.Cols, g.Channels, g.Rows)
    idx := 0
    for b := 0; b < g.Batches; b++ {
        for col := 0; col < g.Cols; col++ {
            for c := 0; c < g.Channels; c++ {
                srcBase := g.At(b, c, 0, col)
                for r := 0; r < g.Rows; r++ {
                    dst.Data[idx] = g.Data[srcBase+r*g.RowStride]
                    idx++
                }
            }
        }
    }
    return dst
}

func (g *General) Transpose0321() General {
    dst := NewZeros(g.Batches, g.Cols, g.Rows, g.Channels)
    idx := 0
    for b := 0; b < g.Batches; b++ {
        for col := 0; col < g.Cols; col++ {
            for r := 0; r < g.Rows; r++ {
                for c := 0; c < g.Channels; c++ {
                    dst.Data[idx] = g.Data[g.At(b, c, r, col)]
                    idx++
                }
            }
        }
    }
    return dst
}

func (g *General) Transpose1023() General {
    dst := NewZeros(g.Channels, g.Batches, g.Rows, g.Cols)
    idx := 0
    for c := 0; c < g.Channels; c++ {
        for b := 0; b < g.Batches; b++ {
            srcBase := g.At(b, c, 0, 0)
            copy(dst.Data[idx:idx+g.Rows*g.Cols], g.Data[srcBase:srcBase+g.Rows*g.Cols])
            idx += g.Rows * g.Cols
        }
    }
    return dst
}

func (g *General) Transpose1032() General {
    dst := NewZeros(g.Channels, g.Batches, g.Cols, g.Rows)
    idx := 0
    for c := 0; c < g.Channels; c++ {
        for b := 0; b < g.Batches; b++ {
            for col := 0; col < g.Cols; col++ {
                for r := 0; r < g.Rows; r++ {
                    dst.Data[idx] = g.Data[g.At(b, c, r, col)]
                    idx++
                }
            }
        }
    }
    return dst
}

func (g *General) Transpose1203() General {
    dst := NewZeros(g.Channels, g.Rows, g.Batches, g.Cols)
    idx := 0
    for c := 0; c < g.Channels; c++ {
        for r := 0; r < g.Rows; r++ {
            for b := 0; b < g.Batches; b++ {
                srcBase := g.At(b, c, r, 0)
                copy(dst.Data[idx:idx+g.Cols], g.Data[srcBase:srcBase+g.Cols])
                idx += g.Cols
            }
        }
    }
    return dst
}

func (g *General) Transpose1230() General {
    dst := NewZeros(g.Channels, g.Rows, g.Cols, g.Batches)
    idx := 0
    for c := 0; c < g.Channels; c++ {
        for r := 0; r < g.Rows; r++ {
            for col := 0; col < g.Cols; col++ {
                for b := 0; b < g.Batches; b++ {
                    dst.Data[idx] = g.Data[g.At(b, c, r, col)]
                    idx++
                }
            }
        }
    }
    return dst
}

func (g *General) Transpose1302() General {
    dst := NewZeros(g.Channels, g.Cols, g.Batches, g.Rows)
    idx := 0
    for c := 0; c < g.Channels; c++ {
        for col := 0; col < g.Cols; col++ {
            for b := 0; b < g.Batches; b++ {
                for r := 0; r < g.Rows; r++ {
                    dst.Data[idx] = g.Data[g.At(b, c, r, col)]
                    idx++
                }
            }
        }
    }
    return dst
}

func (g *General) Transpose1320() General {
    dst := NewZeros(g.Channels, g.Cols, g.Rows, g.Batches)
    idx := 0
    for c := 0; c < g.Channels; c++ {
        for col := 0; col < g.Cols; col++ {
            for r := 0; r < g.Rows; r++ {
                for b := 0; b < g.Batches; b++ {
                    dst.Data[idx] = g.Data[g.At(b, c, r, col)]
                    idx++
                }
            }
        }
    }
    return dst
}

func (g *General) Transpose2013() General {
    dst := NewZeros(g.Rows, g.Batches, g.Channels, g.Cols)
    idx := 0
    for r := 0; r < g.Rows; r++ {
        for b := 0; b < g.Batches; b++ {
            for c := 0; c < g.Channels; c++ {
                srcBase := g.At(b, c, r, 0)
                copy(dst.Data[idx:idx+g.Cols], g.Data[srcBase:srcBase+g.Cols])
                idx += g.Cols
            }
        }
    }
    return dst
}

func (g *General) Transpose2031() General {
    dst := NewZeros(g.Rows, g.Batches, g.Cols, g.Channels)
    idx := 0
    for r := 0; r < g.Rows; r++ {
        for b := 0; b < g.Batches; b++ {
            for col := 0; col < g.Cols; col++ {
                for c := 0; c < g.Channels; c++ {
                    dst.Data[idx] = g.Data[g.At(b, c, r, col)]
                    idx++
                }
            }
        }
    }
    return dst
}

func (g *General) Transpose2103() General {
    dst := NewZeros(g.Rows, g.Channels, g.Batches, g.Cols)
    idx := 0
    for r := 0; r < g.Rows; r++ {
        for c := 0; c < g.Channels; c++ {
            for b := 0; b < g.Batches; b++ {
                srcBase := g.At(b, c, r, 0)
                copy(dst.Data[idx:idx+g.Cols], g.Data[srcBase:srcBase+g.Cols])
                idx += g.Cols
            }
        }
    }
    return dst
}

func (g *General) Transpose2130() General {
    dst := NewZeros(g.Rows, g.Channels, g.Cols, g.Batches)
    idx := 0
    for r := 0; r < g.Rows; r++ {
        for c := 0; c < g.Channels; c++ {
            for col := 0; col < g.Cols; col++ {
                for b := 0; b < g.Batches; b++ {
                    dst.Data[idx] = g.Data[g.At(b, c, r, col)]
                    idx++
                }
            }
        }
    }
    return dst
}

func (g *General) Transpose2301() General {
    dst := NewZeros(g.Rows, g.Cols, g.Batches, g.Channels)
    idx := 0
    for r := 0; r < g.Rows; r++ {
        for col := 0; col < g.Cols; col++ {
            for b := 0; b < g.Batches; b++ {
                for c := 0; c < g.Channels; c++ {
                    dst.Data[idx] = g.Data[g.At(b, c, r, col)]
                    idx++
                }
            }
        }
    }
    return dst
}

func (g *General) Transpose2310() General {
    dst := NewZeros(g.Rows, g.Cols, g.Channels, g.Batches)
    idx := 0
    for r := 0; r < g.Rows; r++ {
        for col := 0; col < g.Cols; col++ {
            for c := 0; c < g.Channels; c++ {
                for b := 0; b < g.Batches; b++ {
                    dst.Data[idx] = g.Data[g.At(b, c, r, col)]
                    idx++
                }
            }
        }
    }
    return dst
}

func (g *General) Transpose3012() General {
    dst := NewZeros(g.Cols, g.Batches, g.Channels, g.Rows)
    idx := 0
    for col := 0; col < g.Cols; col++ {
        for b := 0; b < g.Batches; b++ {
            for c := 0; c < g.Channels; c++ {
                srcBase := g.At(b, c, 0, col)
                for r := 0; r < g.Rows; r++ {
                    dst.Data[idx] = g.Data[srcBase+r*g.RowStride]
                    idx++
                }
            }
        }
    }
    return dst
}

func (g *General) Transpose3021() General {
    dst := NewZeros(g.Cols, g.Batches, g.Rows, g.Channels)
    idx := 0
    for col := 0; col < g.Cols; col++ {
        for b := 0; b < g.Batches; b++ {
            for r := 0; r < g.Rows; r++ {
                for c := 0; c < g.Channels; c++ {
                    dst.Data[idx] = g.Data[g.At(b, c, r, col)]
                    idx++
                }
            }
        }
    }
    return dst
}

func (g *General) Transpose3102() General {
    dst := NewZeros(g.Cols, g.Channels, g.Batches, g.Rows)
    idx := 0
    for col := 0; col < g.Cols; col++ {
        for c := 0; c < g.Channels; c++ {
            for b := 0; b < g.Batches; b++ {
                srcBase := g.At(b, c, 0, col)
                for r := 0; r < g.Rows; r++ {
                    dst.Data[idx] = g.Data[srcBase+r*g.RowStride]
                    idx++
                }
            }
        }
    }
    return dst
}

func (g *General) Transpose3120() General {
    dst := NewZeros(g.Cols, g.Channels, g.Rows, g.Batches)
    idx := 0
    for col := 0; col < g.Cols; col++ {
        for c := 0; c < g.Channels; c++ {
            for r := 0; r < g.Rows; r++ {
                for b := 0; b < g.Batches; b++ {
                    dst.Data[idx] = g.Data[g.At(b, c, r, col)]
                    idx++
                }
            }
        }
    }
    return dst
}

func (g *General) Transpose3201() General {
    dst := NewZeros(g.Cols, g.Rows, g.Batches, g.Channels)
    idx := 0
    for col := 0; col < g.Cols; col++ {
        for r := 0; r < g.Rows; r++ {
            for b := 0; b < g.Batches; b++ {
                for c := 0; c < g.Channels; c++ {
                    dst.Data[idx] = g.Data[g.At(b, c, r, col)]
                    idx++
                }
            }
        }
    }
    return dst
}

func (g *General) Transpose3210() General {
    dst := NewZeros(g.Cols, g.Rows, g.Channels, g.Batches)
    idx := 0
    for col := 0; col < g.Cols; col++ {
        for r := 0; r < g.Rows; r++ {
            for c := 0; c < g.Channels; c++ {
                for b := 0; b < g.Batches; b++ {
                    dst.Data[idx] = g.Data[g.At(b, c, r, col)]
                    idx++
                }
            }
        }
    }
    return dst
}