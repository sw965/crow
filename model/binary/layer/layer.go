package layer

import (
	"cmp"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/sw965/omw/mathx"
	"github.com/sw965/omw/mathx/bitsx"
	"github.com/sw965/omw/mathx/randx"
	"github.com/sw965/omw/parallel"
	"github.com/sw965/omw/slicesx"
)

type Delta []int16

type Interface interface {
	Forward(bitsx.Matrix) (bitsx.Matrix, error)
	Backward(bitsx.Matrix) (bitsx.Matrix, error)
	Predict(bitsx.Matrix) (bitsx.Matrix, error)
	Delta() Delta
	ClearDelta()
}

type SignDot struct {
	W             bitsx.Matrix
	WT            bitsx.Matrix
	GateThreshold int

	IsNoisy  bool
	NoiseStd float64
	Rng      *rand.Rand

	x     bitsx.Matrix
	zs    []int
	delta Delta
}

func NewSignDot(w, wt bitsx.Matrix) (*SignDot, error) {
	if w.Rows != wt.Cols {
		return nil, fmt.Errorf("後でエラーメッセージを書く")
	}

	if w.Cols != wt.Rows {
		return nil, fmt.Errorf("後でエラーメッセージを書く")
	}

	//√(zの最大値)
	std := math.Sqrt(float64(w.Cols))
	threshold := int(std)
	n := w.Rows * w.Cols
	return &SignDot{
		W:             w,
		WT:            wt,
		GateThreshold: threshold,
		NoiseStd:      std,
		zs:            make([]int, n),
		delta:         make([]int16, n),
	}, nil
}

func (sd *SignDot) Forward(x bitsx.Matrix) (bitsx.Matrix, error) {
	u, err := x.Dot(sd.W)
	if err != nil {
		return bitsx.Matrix{}, err
	}

	maxZ := sd.W.Cols
	minZ := -maxZ

	if sd.IsNoisy {
		for i, count := range u {
			z := 2*count - maxZ
			noise, err := randx.NormalInt(minZ, maxZ, 0, sd.NoiseStd, sd.Rng)
			if err != nil {
				return bitsx.Matrix{}, err
			}
			sd.zs[i] = z + noise
		}
	} else {
		for i, count := range u {
			sd.zs[i] = 2*count - maxZ
		}
	}

	yRows := x.Rows
	yCols := sd.W.Rows
	sign, err := bitsx.NewSignMatrix(yRows, yCols, sd.zs)
	if err != nil {
		return bitsx.Matrix{}, err
	}
	sd.x = x
	return sign, nil
}

// func (sd *SignDot) Backward(target bitsx.Matrix) (bitsx.Matrix, error) {
// 	yRows := target.Rows
// 	yCols := target.Cols
// 	wCols := sd.W.Cols
// 	xStride := sd.x.Stride
// 	tStride := target.Stride

// 	if target.Rows != sd.x.Rows || target.Cols != sd.W.Rows {
// 		return bitsx.Matrix{}, fmt.Errorf("target shape mismatch: expected %dx%d, got %dx%d", yRows, yCols, target.Rows, target.Cols)
// 	}

// 	keepGate, err := bitsx.NewZerosMatrix(yRows, yCols)
// 	if err != nil {
// 		return bitsx.Matrix{}, err
// 	}

// 	for r := 0; r < yRows; r++ {
// 		xRowBase := r * xStride
// 		tRowBase := r * tStride
// 		zsBase := r * yCols
// 		kgRowBase := r * keepGate.Stride

// 		for tw := 0; tw < tStride; tw++ {
// 			tWord := target.Data[tRowBase+tw]
// 			var kgWord uint64

// 			cStart := tw * 64
// 			cEnd := cStart + 64
// 			if cEnd > yCols {
// 				cEnd = yCols
// 			}

// 			for c := cStart; c < cEnd; c++ {
// 				z := sd.zs[zsBase+c]
// 				if mathx.Abs(z) > sd.GateThreshold {
// 					continue
// 				}

// 				bitPos := uint(c - cStart)
// 				kgWord |= (1 << bitPos)

// 				tBit := (tWord >> bitPos) & 1
// 				yBit := uint64(0)
// 				if z >= 0 {
// 					yBit = 1
// 				}

// 				// 正解なら更新をスキップ
// 				if tBit == yBit {
// 					continue
// 				}

// 				deltaBase := c * wCols
// 				for i := 0; i < xStride; i++ {
// 					xWord := sd.x.Data[xRowBase+i]
// 					dIdx := deltaBase + i*64

// 					limit := 64
// 					if i*64+64 > wCols {
// 						limit = wCols - i*64
// 					}

// 					subDelta := sd.delta[dIdx : dIdx+limit]

// 					if tBit == 1 {
// 						for b := 0; b < limit; b++ {
// 							if (xWord & 1) == 1 {
// 								subDelta[b]++
// 							} else {
// 								subDelta[b]--
// 							}
// 							xWord >>= 1
// 						}
// 					} else {
// 						for b := 0; b < limit; b++ {
// 							if (xWord & 1) == 1 {
// 								subDelta[b]--
// 							} else {
// 								subDelta[b]++
// 							}
// 							xWord >>= 1
// 						}
// 					}
// 				}
// 			}
// 			keepGate.Data[kgRowBase+tw] = kgWord
// 		}
// 	}

// 	rawNextTargetT, err := sd.WT.DotTernary(target, keepGate)
// 	if err != nil {
// 		return bitsx.Matrix{}, err
// 	}

// 	nextTarget, err := bitsx.NewZerosMatrix(yRows, wCols)
// 	if err != nil {
// 		return bitsx.Matrix{}, err
// 	}

// 	for r := 0; r < yRows; r++ {
// 		rowIdx := r * nextTarget.Stride
// 		for cw := 0; cw < nextTarget.Stride; cw++ {
// 			var word uint64
// 			cStart := cw * 64
// 			cEnd := cStart + 64
// 			if cEnd > wCols {
// 				cEnd = wCols
// 			}

// 			for c := cStart; c < cEnd; c++ {
// 				if rawNextTargetT[c*yRows+r] >= 0 {
// 					word |= (1 << uint(c-cStart))
// 				}
// 			}
// 			nextTarget.Data[rowIdx+cw] = word
// 		}
// 	}
// 	return nextTarget, nil
// }

func (sd *SignDot) Backward(target bitsx.Matrix) (bitsx.Matrix, error) {
	yRows, yCols := target.Rows, target.Cols
	wCols := sd.W.Cols

	if yRows != sd.x.Rows || yCols != sd.W.Rows {
		return bitsx.Matrix{}, fmt.Errorf("target shape mismatch")
	}

	keepGate, err := bitsx.NewZerosMatrix(yRows, yCols)
	if err != nil {
		return bitsx.Matrix{}, err
	}

	// wRows := sd.W.Rows
	err = target.ScanRowsWord(nil, func(tCtx bitsx.MatrixWordContext) error {
		// 64ビット毎に操作するための宣言
		zsWord := sd.zs[tCtx.GlobalStart:tCtx.GlobalEnd]
		tWord := target.Data[tCtx.WordIndex]
		var keepGateWord uint64

		// ScanBitsで上記の64ビット(Word)に対して操作する
		// 引数iに代入される値は、100ビットの場合、一週目は0～63、二週目は64～99
		tCtx.ScanBits(func(i, col, colT int) {
			z := zsWord[i]
			if mathx.Abs(z) > sd.GateThreshold {
				return
			}
			keepGateWord |= (1 << uint64(i))

			tBit := (tWord >> uint64(i)) & 1
			yBit := uint64(0)
			if z >= 0 {
				yBit = 1
			}

			// 正解なら更新しない
			if tBit == yBit {
				return
			}

			outputCol := tCtx.ColStart + i
			deltaRow := sd.delta[outputCol*wCols : (outputCol+1)*wCols]

			// 64ビット毎にxを見て重みの勾配を求める
			sd.x.ScanRowsWord([]int{tCtx.Row}, func(xCtx bitsx.MatrixWordContext) error {
				xWord := sd.x.Data[xCtx.WordIndex]
				deltaWord := deltaRow[xCtx.ColStart:xCtx.ColEnd]
				for b := range deltaWord {
					xBit := (xWord >> uint(b)) & 1
					deltaWord[b] += int16(1 - 2*int(xBit^tBit))
				}
				return nil
			})
		})

		keepGate.Data[tCtx.WordIndex] = keepGateWord
		return nil
	})

	if err != nil {
		return bitsx.Matrix{}, err
	}

	rawNextTargetT, err := sd.WT.DotTernary(target, keepGate)
	if err != nil {
		return bitsx.Matrix{}, err
	}

	nextTarget, err := bitsx.NewZerosMatrix(yRows, wCols)
	if err != nil {
		return bitsx.Matrix{}, err
	}

	err = nextTarget.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
		var word uint64
		ctx.ScanBits(func(i, col, colT int) {
			if rawNextTargetT[colT] >= 0 {
				word |= (1 << uint(i))
			}
		})
		nextTarget.Data[ctx.WordIndex] = word
		return nil
	})

	if err != nil {
		return bitsx.Matrix{}, err
	}
	return nextTarget, nil
}

func (sd *SignDot) Predict(x bitsx.Matrix) (bitsx.Matrix, error) {
	u, err := x.Dot(sd.W)
	if err != nil {
		return bitsx.Matrix{}, err
	}

	maxZ := sd.W.Cols
	zs := make([]int, len(u))
	for i, count := range u {
		zs[i] = 2*count - maxZ
	}

	yRows := x.Rows
	yCols := sd.W.Rows
	sign, err := bitsx.NewSignMatrix(yRows, yCols, zs)
	if err != nil {
		return bitsx.Matrix{}, err
	}
	return sign, nil
}

func (sd *SignDot) Delta() Delta {
	return sd.delta
}

func (sd *SignDot) ClearDelta() {
	clear(sd.delta)
}

type Sequence []Interface

func (s Sequence) Forwards(x bitsx.Matrix) (bitsx.Matrix, error) {
	var err error
	for _, layer := range s {
		x, err = layer.Forward(x)
		if err != nil {
			return bitsx.Matrix{}, err
		}
	}
	y := x
	return y, nil
}

func (s Sequence) Backwards(target bitsx.Matrix) (bitsx.Matrix, error) {
	var err error
	for layerI := len(s) - 1; layerI >= 0; layerI-- {
		target, err = s[layerI].Backward(target)
		if err != nil {
			return bitsx.Matrix{}, err
		}
	}
	return target, nil
}

func (s Sequence) Predict(x bitsx.Matrix) (bitsx.Matrix, error) {
	var err error
	for _, layer := range s {
		x, err = layer.Predict(x)
		if err != nil {
			return bitsx.Matrix{}, err
		}
	}
	return x, nil
}

func (s Sequence) PredictLogits(x bitsx.Matrix, prototypes bitsx.Matrices) ([]int, error) {
	y, err := s.Predict(x)
	if err != nil {
		return nil, err
	}

	n := len(prototypes)
	logits := make([]int, n)
	maxMatch := y.Rows * y.Cols

	for i, proto := range prototypes {
		if y.Rows != proto.Rows || y.Cols != proto.Cols {
			return nil, fmt.Errorf("後でエラーメッセージを書く")
		}
		mismatch, err := y.HammingDistance(proto)
		if err != nil {
			return nil, err
		}
		logits[i] = maxMatch - mismatch
	}
	return logits, nil
}

func (s Sequence) Accuracy(xs []bitsx.Matrix, labels []int, prototypes bitsx.Matrices, p int) (float32, error) {
	n := len(xs)
	if n != len(labels) {
		return 0.0, fmt.Errorf("length mismatch: xs %d != labels %d", n, len(labels))
	}
	correctCounts := make([]int, p)

	err := parallel.For(n, p, func(workerId, idx int) error {
		x := xs[idx]
		label := labels[idx]

		logits, err := s.PredictLogits(x, prototypes)
		if err != nil {
			return err
		}

		if len(logits) == 0 {
			return fmt.Errorf("logits is empty")
		}

		yMaxIdx := slicesx.Argsort(logits)[len(logits)-1]
		if yMaxIdx == label {
			correctCounts[workerId]++
		}
		return nil
	})

	if err != nil {
		return 0.0, err
	}

	totalCorrect := 0
	for _, c := range correctCounts {
		totalCorrect += c
	}
	return float32(totalCorrect) / float32(n), nil
}

func (s Sequence) ClearDelta() {
	for _, layer := range s {
		layer.ClearDelta()
	}
}

type Sequences []Sequence

func (ss Sequences) ComputeSignDeltas(xs, ts bitsx.Matrices) ([]Delta, error) {
	n := len(xs)
	if n != len(ts) {
		return nil, fmt.Errorf("後でエラーメッセージを書く")
	}

	// バッチサイズがmath.MaxInt16より大きいと、オーバーフローが起きる可能性があるため、ガードする
	// ※ ワーカー毎にDeltaを切り分けるが、合計して集約するステップがある事に注意
	if n > math.MaxInt16 {
		return nil, fmt.Errorf("後でエラーメッセージを書く")
	}
	p := len(ss)

	for _, seq := range ss {
		seq.ClearDelta()
	}

	err := parallel.For(n, p, func(workerId, idx int) error {
		x, t := xs[idx], ts[idx]
		seq := ss[workerId]

		_, err := seq.Forwards(x)
		if err != nil {
			return err
		}

		_, err = seq.Backwards(t)
		if err != nil {
			return err
		}
		return nil
	})

	if err != nil {
		return nil, err
	}

	numLayers := len(ss[0])
	signDeltas := make([]Delta, numLayers)

	for layerI := range numLayers {
		worker0Delta := ss[0][layerI].Delta()
		worker0DeltaN := len(worker0Delta)
		sum := make([]int, worker0DeltaN)

		for _, seq := range ss {
			delta := seq[layerI].Delta()
			if len(delta) != worker0DeltaN {
				return nil, fmt.Errorf("後でエラーメッセージを書く")
			}

			for i, v := range delta {
				sum[i] += int(v)
			}
		}

		sign := make(Delta, worker0DeltaN)
		for i, v := range sum {
			sign[i] = int16(cmp.Compare(v, 0))
		}
		signDeltas[layerI] = sign
	}
	return signDeltas, err
}