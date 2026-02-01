// package mixer

// import (
// 	"cmp"
// 	"fmt"
// 	"math"
// 	"math/bits"
// 	"math/rand/v2"

// 	"github.com/sw965/omw/mathx"
// 	"github.com/sw965/omw/mathx/bitsx"
// 	"github.com/sw965/omw/mathx/randx"
// 	"github.com/sw965/omw/parallel"
// 	"github.com/sw965/omw/slicesx"
// )

// // H holds the integer hidden weights for one Mixer Layer.
// type H struct {
// 	DataT1 []int8
// 	DataT2 []int8
// 	DataC1 []int8
// 	DataC2 []int8
// }

// // Delta holds the accumulated gradients for one Mixer Layer.
// type Delta struct {
// 	DataT1 []int16
// 	DataT2 []int16
// 	DataC1 []int16
// 	DataC2 []int16
// }

// // ----------------------------------------------------------------------------
// // Helper Functions
// // ----------------------------------------------------------------------------

// // linearBlock computes X . W, applies noise, and Sign activation.
// // Returns the binary activation matrix and the pre-activation Zs.
// func linearBlock(x, w bitsx.Matrix, std float64, rng *rand.Rand, isNoisy bool) (bitsx.Matrix, []int, error) {
// 	counts, err := x.Dot(w)
// 	if err != nil {
// 		return bitsx.Matrix{}, nil, err
// 	}

// 	rows := x.Rows
// 	cols := w.Rows // Transposed in bitsx Dot logic (Result Cols = W.Rows)

// 	y, err := bitsx.NewZerosMatrix(rows, cols)
// 	if err != nil {
// 		return bitsx.Matrix{}, nil, err
// 	}

// 	totalBits := w.Cols
// 	zs := make([]int, rows*cols)

// 	for r := 0; r < rows; r++ {
// 		offset := r * cols
// 		for c := 0; c < cols; c++ {
// 			idx := offset + c
// 			count := counts[idx]

// 			// Convert bit count (0..N) to bipolar sum (-N..N)
// 			z := 2*count - totalBits

// 			if isNoisy {
// 				noise, err := randx.NormalInt(int(math.MinInt8), int(math.MaxInt8), 0.0, std, rng)
// 				if err != nil {
// 					return bitsx.Matrix{}, nil, err
// 				}
// 				z += noise
// 			}
// 			zs[idx] = z

// 			// Sign Activation: z >= 0 -> 1, else 0
// 			if z >= 0 {
// 				if err := y.Set(r, c); err != nil {
// 					return bitsx.Matrix{}, nil, err
// 				}
// 			}
// 		}
// 	}
// 	return y, zs, nil
// }

// // projectBlock backpropagates the target activation to the previous layer.
// // It uses dynamic thresholding based on the number of active gates (BEP logic).
// func projectBlock(gatedTarget, projector bitsx.Matrix, gate bitsx.Matrix) (bitsx.Matrix, error) {
// 	// Projector is usually WT. gatedTarget . WT
// 	counts, err := gatedTarget.Dot(projector)
// 	if err != nil {
// 		return bitsx.Matrix{}, err
// 	}

// 	rows := gatedTarget.Rows
// 	cols := projector.Rows // Result cols

// 	nextTarget, err := bitsx.NewZerosMatrix(rows, cols)
// 	if err != nil {
// 		return bitsx.Matrix{}, err
// 	}

// 	// BEP logic: Threshold depends on how many neurons passed the gate
// 	for r := 0; r < rows; r++ {
// 		// Calculate threshold for this row
// 		rowActiveCount := 0
// 		gateRowOffset := r * gate.Stride
// 		// This is a manual PopCount for the row in bitsx
// 		for k := 0; k < gate.Stride; k++ {
// 			word := gate.Data[gateRowOffset+k]
// 			if k == gate.Stride-1 {
// 				word &= gate.RowMask
// 			}
// 			rowActiveCount += bits.OnesCount64(word)
// 		}
		
// 		if rowActiveCount == 0 {
//             continue
//         }

// 		threshold := (rowActiveCount + 1) / 2
		
// 		resOffset := r * cols
// 		for c := 0; c < cols; c++ {
// 			if counts[resOffset+c] >= threshold {
// 				if err := nextTarget.Set(r, c); err != nil {
// 					return bitsx.Matrix{}, err
// 				}
// 			}
// 		}
// 	}
// 	return nextTarget, nil
// }

// // accumulateSparseDelta computes the gradient for weights.
// // `target` must be the desired state (0/1), NOT masked by the gate.
// func accumulateSparseDelta(x, target bitsx.Matrix, zs []int, deltaData []int16, groupSize int) error {
// 	l := x.Rows
// 	dIn := x.Cols
// 	dOut := target.Cols

// 	if len(deltaData) != dOut*dIn {
// 		return fmt.Errorf("delta size mismatch: expected %d, got %d", dOut*dIn, len(deltaData))
// 	}

// 	if groupSize <= 0 {
// 		groupSize = 1
// 	}

// 	for i := 0; i < l; i++ {
// 		zsStart := i * dOut
// 		zsEnd := zsStart + dOut
// 		rowZs := zs[zsStart:zsEnd]

// 		absZs := make([]int, dOut)
// 		for j, z := range rowZs {
// 			absZs[j] = mathx.Abs(z)
// 		}

// 		ascAbsZIdxs := slicesx.Argsort(absZs)

// 		k := dOut / groupSize
// 		if k == 0 && dOut > 0 {
// 			k = 1
// 		}

// 		for _, neuronIdx := range ascAbsZIdxs[:k] {
			
// 			targetBit, _ := target.Bit(i, neuronIdx)

// 			baseIdx := neuronIdx * dIn
// 			colIdx := 0
// 			xRowOffset := i * x.Stride

// 			for wIdx := 0; wIdx < x.Stride; wIdx++ {
// 				xWord := x.Data[xRowOffset+wIdx]
				
// 				validBits := 64
// 				if colIdx+validBits > dIn {
// 					validBits = dIn - colIdx
// 				}

// 				for b := 0; b < validBits; b++ {
// 					xBit := (xWord >> b) & 1
// 					val := int16(1 - 2*int(xBit^targetBit))
// 					deltaData[baseIdx+colIdx+b] += val
// 				}
// 				colIdx += validBits
// 			}
// 		}
// 	}
// 	return nil
// }

// // ----------------------------------------------------------------------------
// // Core Structures
// // ----------------------------------------------------------------------------

// type Forward func(bitsx.Matrix, *rand.Rand) (bitsx.Matrix, Backward, error)
// type Backward func(bitsx.Matrix, Delta) (bitsx.Matrix, error)
// type Forwards []Forward
// type Backwards []Backward

// type MixerLayer struct {
// 	// Token Mixing
// 	WT1  bitsx.Matrix
// 	WT1T bitsx.Matrix
// 	WT2  bitsx.Matrix
// 	WT2T bitsx.Matrix

// 	// Channel Mixing
// 	WC1  bitsx.Matrix
// 	WC1T bitsx.Matrix
// 	WC2  bitsx.Matrix
// 	WC2T bitsx.Matrix

// 	H         H
// 	IsNoisy   bool
// 	GroupSize int

// 	DimTok  int
// 	DimChan int
// }

// type Model struct {
// 	forwards        Forwards
// 	Layers          []MixerLayer
// 	Prototypes      []bitsx.Matrix
// 	deltasPerWorker [][]Delta
// }

// func NewForward(layer *MixerLayer) (Forward, error) {
// 	// Standard Deviations for Noise
// 	stdT1 := math.Sqrt(float64(layer.WT1.Cols))
// 	stdT2 := math.Sqrt(float64(layer.WT2.Cols))
// 	stdC1 := math.Sqrt(float64(layer.WC1.Cols))
// 	stdC2 := math.Sqrt(float64(layer.WC2.Cols))

// 	// Thresholds for Gate (sqrt(N))
// 	gateThreshT1 := int(stdT1)
// 	gateThreshT2 := int(stdT2)
// 	gateThreshC1 := int(stdC1)
// 	gateThreshC2 := int(stdC2)

// 	return func(x bitsx.Matrix, rng *rand.Rand) (bitsx.Matrix, Backward, error) {
// 		// --- 1. Token Mixing (Spatial) ---
// 		// x is (L x C)
// 		xT, err := x.Transpose() // (C x L)
// 		if err != nil { return bitsx.Matrix{}, nil, err }

// 		// Linear 1: xT . WT1 -> t1
// 		t1, t1Zs, err := linearBlock(xT, layer.WT1, stdT1, rng, layer.IsNoisy)
// 		if err != nil { return bitsx.Matrix{}, nil, err }

// 		// Linear 2: t1 . WT2 -> t2
// 		t2, t2Zs, err := linearBlock(t1, layer.WT2, stdT2, rng, layer.IsNoisy)
// 		if err != nil { return bitsx.Matrix{}, nil, err }

// 		yTok, err := t2.Transpose() // (L x C)
// 		if err != nil { return bitsx.Matrix{}, nil, err }

// 		// --- 2. Channel Mixing (Feature) ---
// 		// Linear 3: yTok . WC1 -> c1
// 		c1, c1Zs, err := linearBlock(yTok, layer.WC1, stdC1, rng, layer.IsNoisy)
// 		if err != nil { return bitsx.Matrix{}, nil, err }

// 		// Linear 4: c1 . WC2 -> yOut
// 		yOut, c2Zs, err := linearBlock(c1, layer.WC2, stdC2, rng, layer.IsNoisy)
// 		if err != nil { return bitsx.Matrix{}, nil, err }

// 		// --- Backward ---
// 		var backward Backward
// 		backward = func(target bitsx.Matrix, delta Delta) (bitsx.Matrix, error) {
			
// 			// Helper to create Gate matrix based on Zs
// 			makeGate := func(zs []int, rows, cols, threshold int) (bitsx.Matrix, error) {
// 				g, err := bitsx.NewZerosMatrix(rows, cols)
// 				if err != nil { return bitsx.Matrix{}, err }
// 				for r := 0; r < rows; r++ {
// 					offset := r * cols
// 					for c := 0; c < cols; c++ {
// 						z := zs[offset+c]
// 						if mathx.Abs(z) <= threshold {
// 							g.Set(r, c)
// 						}
// 					}
// 				}
// 				return g, nil
// 			}

// 			// --- Channel Mixing Backward ---
// 			gateC2, err := makeGate(c2Zs, yOut.Rows, yOut.Cols, gateThreshC2)
// 			if err != nil { return bitsx.Matrix{}, err }
			
// 			gatedTargetC2, err := target.And(gateC2)
// 			if err != nil { return bitsx.Matrix{}, err }

// 			// Project C2 -> C1
// 			c1Target, err := projectBlock(gatedTargetC2, layer.WC2T, gateC2)
// 			if err != nil { return bitsx.Matrix{}, err }

// 			gateC1, err := makeGate(c1Zs, c1.Rows, c1.Cols, gateThreshC1)
// 			if err != nil { return bitsx.Matrix{}, err }
// 			gatedTargetC1, err := c1Target.And(gateC1)
// 			if err != nil { return bitsx.Matrix{}, err }

// 			// Project C1 -> YTok
// 			yTokTarget, err := projectBlock(gatedTargetC1, layer.WC1T, gateC1)
// 			if err != nil { return bitsx.Matrix{}, err }

// 			// Transpose for Token Mixing
// 			t2Target, err := yTokTarget.Transpose()
// 			if err != nil { return bitsx.Matrix{}, err }

// 			// --- Token Mixing Backward ---
// 			gateT2, err := makeGate(t2Zs, t2.Rows, t2.Cols, gateThreshT2)
// 			if err != nil { return bitsx.Matrix{}, err }
// 			gatedTargetT2, err := t2Target.And(gateT2)
// 			if err != nil { return bitsx.Matrix{}, err }

// 			// Project T2 -> T1
// 			t1Target, err := projectBlock(gatedTargetT2, layer.WT2T, gateT2)
// 			if err != nil { return bitsx.Matrix{}, err }

// 			gateT1, err := makeGate(t1Zs, t1.Rows, t1.Cols, gateThreshT1)
// 			if err != nil { return bitsx.Matrix{}, err }
// 			gatedTargetT1, err := t1Target.And(gateT1)
// 			if err != nil { return bitsx.Matrix{}, err }

// 			// Project T1 -> XT
// 			xTTarget, err := projectBlock(gatedTargetT1, layer.WT1T, gateT1)
// 			if err != nil { return bitsx.Matrix{}, err }

// 			xTarget, err := xTTarget.Transpose()
// 			if err != nil { return bitsx.Matrix{}, err }

// 			// --- Accumulate Gradients ---
// 			if err := accumulateSparseDelta(c1, target, c2Zs, delta.DataC2, layer.GroupSize); err != nil { return bitsx.Matrix{}, err }
// 			if err := accumulateSparseDelta(yTok, c1Target, c1Zs, delta.DataC1, layer.GroupSize); err != nil { return bitsx.Matrix{}, err }
// 			if err := accumulateSparseDelta(t1, t2Target, t2Zs, delta.DataT2, layer.GroupSize); err != nil { return bitsx.Matrix{}, err }
// 			// FIX: Use t1Target here, not xTTarget
// 			if err := accumulateSparseDelta(xT, t1Target, t1Zs, delta.DataT1, layer.GroupSize); err != nil { return bitsx.Matrix{}, err }

// 			return xTarget, nil
// 		}
// 		return yOut, backward, nil
// 	}, nil
// }

// func NewModel(L, C, dimTok, dimChan, numLayers int, p int, rng *rand.Rand) (Model, error) {
// 	m := Model{
// 		forwards:        make(Forwards, numLayers),
// 		Layers:          make([]MixerLayer, numLayers),
// 		deltasPerWorker: make([][]Delta, p),
// 	}

// 	for i := 0; i < numLayers; i++ {
// 		layer := MixerLayer{
// 			DimTok:    dimTok,
// 			DimChan:   dimChan,
// 			IsNoisy:   false,
// 			GroupSize: 2,
// 		}

// 		// Helper to init weights and hidden weights
// 		initW := func(r, c int) (bitsx.Matrix, bitsx.Matrix, []int8, error) {
// 			w, err := bitsx.NewRandMatrix(r, c, 0, rng)
// 			if err != nil { return bitsx.Matrix{}, bitsx.Matrix{}, nil, err }
// 			wt, err := w.Transpose()
// 			if err != nil { return bitsx.Matrix{}, bitsx.Matrix{}, nil, err }
			
// 			hData := make([]int8, r*c)
// 			for rr := 0; rr < r; rr++ {
// 				for cc := 0; cc < c; cc++ {
// 					idx := rr*c + cc // Row-major index for H
					
// 					// w.Bit(rr, cc) uses bit indexing
// 					bit, _ := w.Bit(rr, cc)
					
// 					// Init with synaptic inertia (e.g., +/- 31)
// 					if bit == 1 {
// 						hData[idx] = 31
// 					} else {
// 						hData[idx] = -31
// 					}
// 				}
// 			}
// 			return w, wt, hData, nil
// 		}

// 		var err error
// 		// WT1: (L -> DimTok) -> cols=L, rows=DimTok
// 		layer.WT1, layer.WT1T, layer.H.DataT1, err = initW(dimTok, L)
// 		if err != nil { return Model{}, err }
		
// 		// WT2: (DimTok -> L) -> cols=DimTok, rows=L
// 		layer.WT2, layer.WT2T, layer.H.DataT2, err = initW(L, dimTok)
// 		if err != nil { return Model{}, err }
		
// 		// WC1: (C -> DimChan) -> cols=C, rows=DimChan
// 		layer.WC1, layer.WC1T, layer.H.DataC1, err = initW(dimChan, C)
// 		if err != nil { return Model{}, err }
		
// 		// WC2: (DimChan -> C) -> cols=DimChan, rows=C
// 		layer.WC2, layer.WC2T, layer.H.DataC2, err = initW(C, dimChan)
// 		if err != nil { return Model{}, err }

// 		m.Layers[i] = layer
// 		m.forwards[i], err = NewForward(&m.Layers[i])
// 		if err != nil { return Model{}, err }
// 	}

// 	for workerId := 0; workerId < p; workerId++ {
// 		m.deltasPerWorker[workerId] = make([]Delta, numLayers)
// 		for i := 0; i < numLayers; i++ {
// 			// Sizes match the H Data sizes
// 			m.deltasPerWorker[workerId][i] = Delta{
// 				DataT1: make([]int16, len(m.Layers[i].H.DataT1)),
// 				DataT2: make([]int16, len(m.Layers[i].H.DataT2)),
// 				DataC1: make([]int16, len(m.Layers[i].H.DataC1)),
// 				DataC2: make([]int16, len(m.Layers[i].H.DataC2)),
// 			}
// 		}
// 	}
// 	return m, nil
// }

// func (m *Model) SetIsTraining(isTrain bool) {
// 	for i := range m.Layers {
// 		m.Layers[i].IsNoisy = isTrain
// 	}
// }

// func (fs Forwards) Propagate(x bitsx.Matrix, rng *rand.Rand) (bitsx.Matrix, Backwards, error) {
// 	backwards := make(Backwards, len(fs))
// 	var backward Backward
// 	var err error
// 	for i, f := range fs {
// 		x, backward, err = f(x, rng)
// 		if err != nil { return bitsx.Matrix{}, nil, err }
// 		backwards[i] = backward
// 	}
// 	return x, backwards, err
// }

// func (bs Backwards) Propagate(target bitsx.Matrix, deltas []Delta) (bitsx.Matrix, error) {
// 	var err error
// 	for layerI := len(bs) - 1; layerI >= 0; layerI-- {
// 		target, err = bs[layerI](target, deltas[layerI])
// 		if err != nil { return bitsx.Matrix{}, err }
// 	}
// 	return target, nil
// }

// func (m Model) Predict(x bitsx.Matrix, rng *rand.Rand) (bitsx.Matrix, error) {
// 	y, _, err := m.forwards.Propagate(x, rng)
// 	return y, err
// }

// func (m Model) PredictLogits(x bitsx.Matrix, rng *rand.Rand) ([]int, error) {
// 	y, err := m.Predict(x, rng)
// 	if err != nil { return nil, err }

// 	// y: (L x C)
// 	// Prototypes: (1 x C)

// 	n := len(m.Prototypes)
// 	logits := make([]int, n)

// 	for i, prototype := range m.Prototypes {
// 		counts, err := y.Dot(prototype) // counts has length L
// 		if err != nil { return nil, err }
		
// 		sumScore := 0
// 		for _, c := range counts {
// 			// Convert binary count to bipolar score: 2*c - width
// 			width := y.Cols
// 			z := 2*c - width
// 			sumScore += z
// 		}
// 		logits[i] = sumScore
// 	}
// 	return logits, nil
// }

// func (m *Model) ComputeSignDeltas(xs, ts []bitsx.Matrix, margin float32, rngs []*rand.Rand) ([]Delta, error) {
// 	n := len(xs)
// 	p := len(rngs)

// 	err := parallel.For(n, p, func(workerId, idx int) error {
// 		rng := rngs[workerId]
// 		deltas := m.deltasPerWorker[workerId]
// 		x, t := xs[idx], ts[idx]

// 		y, backwards, err := m.forwards.Propagate(x, rng)
// 		if err != nil { return err }

// 		// --- BEP/Margin Logic ---
// 		if margin > 0 && len(m.Prototypes) > 0 {
// 			// Calculate Score for Correct Class (t)
// 			correctCounts, err := y.Dot(t) // t is the prototype for correct class
// 			if err != nil { return err }
// 			correctScore := 0
// 			for _, c := range correctCounts { correctScore += (2*c - y.Cols) }

// 			maxWrongScore := -999999999
// 			for _, proto := range m.Prototypes {
// 				counts, err := y.Dot(proto)
// 				if err != nil { return err }
// 				score := 0
// 				for _, c := range counts { score += (2*c - y.Cols) }

// 				if score == correctScore {
// 					// Check if proto is t
// 					checkC, _ := t.Dot(proto)
// 					if checkC[0] == t.Cols { continue }
// 				}
// 				if score > maxWrongScore { maxWrongScore = score }
// 			}

// 			// Global max possible score = L * C
// 			maxPossible := float32(y.Rows * y.Cols)
// 			limit := int(maxPossible * margin)
// 			if (correctScore - maxWrongScore) >= limit { return nil }
// 		}

// 		// Backward Propagation
// 		// 'y' has shape (L, C)
// 		// 't' (prototype) has shape (1, C)
// 		// We must broadcast 't' to shape (L, C) to match 'y' for the backward pass.
		
// 		rows, cols := y.Rows, y.Cols
// 		tSeq, err := bitsx.NewZerosMatrix(rows, cols)
// 		if err != nil { return err }

// 		// 't' is (1, C). We want to copy the active bits of t[0] to all rows of tSeq.
// 		// bitsx.Set operates bit by bit, which is slow for copying.
// 		// A faster way is to copy the underlying uint64 words if stride matches.
// 		// Since t and tSeq have the same Cols, they have the same Stride.
// 		// t has 1 row, tSeq has 'rows' rows.
		
// 		tData := t.Data // length = stride
// 		stride := t.Stride
// 		tSeqData := tSeq.Data // length = rows * stride

// 		for r := 0; r < rows; r++ {
// 			// Copy stride uint64s
// 			copy(tSeqData[r*stride : (r+1)*stride], tData)
// 		}

// 		_, err = backwards.Propagate(tSeq, deltas)
// 		return err
// 	})

// 	if err != nil { return nil, err }

// 	layerNum := len(m.Layers)
// 	signDeltas := make([]Delta, layerNum)

// 	for i := 0; i < layerNum; i++ {
// 		l := &m.Layers[i]
// 		signDeltas[i] = Delta{
// 			DataT1: make([]int16, len(l.H.DataT1)),
// 			DataT2: make([]int16, len(l.H.DataT2)),
// 			DataC1: make([]int16, len(l.H.DataC1)),
// 			DataC2: make([]int16, len(l.H.DataC2)),
// 		}

// 		// Accumulate
// 		for wId := 0; wId < p; wId++ {
// 			wd := m.deltasPerWorker[wId][i]
// 			for k, v := range wd.DataT1 { signDeltas[i].DataT1[k] += v }
// 			for k, v := range wd.DataT2 { signDeltas[i].DataT2[k] += v }
// 			for k, v := range wd.DataC1 { signDeltas[i].DataC1[k] += v }
// 			for k, v := range wd.DataC2 { signDeltas[i].DataC2[k] += v }
// 		}

// 		// Sign conversion
// 		convert := func(data []int16) {
// 			for k, v := range data { data[k] = int16(cmp.Compare(v, 0)) }
// 		}
// 		convert(signDeltas[i].DataT1)
// 		convert(signDeltas[i].DataT2)
// 		convert(signDeltas[i].DataC1)
// 		convert(signDeltas[i].DataC2)
// 	}

// 	// Clear buffers
// 	for i := range m.deltasPerWorker {
// 		for j := range m.deltasPerWorker[i] {
// 			clear(m.deltasPerWorker[i][j].DataT1)
// 			clear(m.deltasPerWorker[i][j].DataT2)
// 			clear(m.deltasPerWorker[i][j].DataC1)
// 			clear(m.deltasPerWorker[i][j].DataC2)
// 		}
// 	}
// 	return signDeltas, nil
// }

// func (m *Model) UpdateWeight(deltas []Delta, prob float32, rng *rand.Rand) error {
// 	updateMat := func(dataH []int8, deltaData []int16, matW, matWT *bitsx.Matrix) error {
// 		rows, cols := matW.Rows, matW.Cols // W dimensions
// 		for r := 0; r < rows; r++ {
// 			for c := 0; c < cols; c++ {
// 				idx := r*cols + c
// 				d := deltaData[idx]
// 				if d == 0 { continue }
// 				if rng.Float32() > prob { continue }

// 				old := dataH[idx]
// 				newVal := int(old) + int(2*d)
// 				clipped := int8(max(math.MinInt8, min(math.MaxInt8, newVal)))
// 				dataH[idx] = clipped

// 				// If sign changes, toggle the bit in W and WT
// 				isOldPlus := old >= 0
// 				isNewPlus := clipped >= 0
				
// 				if isOldPlus != isNewPlus {
// 					if err := matW.Toggle(r, c); err != nil { return err }
// 					if err := matWT.Toggle(c, r); err != nil { return err }
// 				}
// 			}
// 		}
// 		return nil
// 	}

// 	for i, delta := range deltas {
// 		l := &m.Layers[i]
// 		if err := updateMat(l.H.DataT1, delta.DataT1, &l.WT1, &l.WT1T); err != nil { return err }
// 		if err := updateMat(l.H.DataT2, delta.DataT2, &l.WT2, &l.WT2T); err != nil { return err }
// 		if err := updateMat(l.H.DataC1, delta.DataC1, &l.WC1, &l.WC1T); err != nil { return err }
// 		if err := updateMat(l.H.DataC2, delta.DataC2, &l.WC2, &l.WC2T); err != nil { return err }
// 	}
// 	return nil
// }

package mixer

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

// H holds the integer hidden weights for one Mixer Layer.
type H struct {
	DataT1 []int8
	DataT2 []int8
	DataC1 []int8
	DataC2 []int8
}

// Delta holds the accumulated gradients for one Mixer Layer.
type Delta struct {
	DataT1 []int16
	DataT2 []int16
	DataC1 []int16
	DataC2 []int16
}

// ----------------------------------------------------------------------------
// Helper Functions
// ----------------------------------------------------------------------------

// linearBlock computes X . W, applies noise, and Sign activation.
// Returns the binary activation matrix and the pre-activation Zs.
func linearBlock(x, w bitsx.Matrix, std float64, rng *rand.Rand, isNoisy bool) (bitsx.Matrix, []int, error) {
	counts, err := x.Dot(w)
	if err != nil {
		return bitsx.Matrix{}, nil, err
	}

	rows := x.Rows
	cols := w.Rows // Transposed in bitsx Dot logic (Result Cols = W.Rows)

	y, err := bitsx.NewZerosMatrix(rows, cols)
	if err != nil {
		return bitsx.Matrix{}, nil, err
	}

	totalBits := w.Cols
	zs := make([]int, rows*cols)

	for r := 0; r < rows; r++ {
		offset := r * cols
		for c := 0; c < cols; c++ {
			idx := offset + c
			count := counts[idx]

			// Convert bit count (0..N) to bipolar sum (-N..N)
			z := 2*count - totalBits

			if isNoisy {
				noise, err := randx.NormalInt(int(math.MinInt8), int(math.MaxInt8), 0.0, std, rng)
				if err != nil {
					return bitsx.Matrix{}, nil, err
				}
				z += noise
			}
			zs[idx] = z

			// Sign Activation: z >= 0 -> 1, else 0
			if z >= 0 {
				if err := y.Set(r, c); err != nil {
					return bitsx.Matrix{}, nil, err
				}
			}
		}
	}
	return y, zs, nil
}

// getRowAsMatrix creates a view of a single row as a Matrix (1 x Cols).
// This avoids copying data and allows using DotTernaryVec on rows.
func getRowAsMatrix(m bitsx.Matrix, r int) bitsx.Matrix {
	return bitsx.Matrix{
		Rows:    1,
		Cols:    m.Cols,
		Stride:  m.Stride,
		Data:    m.Data[r*m.Stride : (r+1)*m.Stride],
		RowMask: m.RowMask,
	}
}

// projectBlock backpropagates the target activation to the previous layer.
// It uses ternary logic: (Target AND Gate) . Projector.
// 'target' and 'gate' must be passed separately to distinguish 0 (masked) from -1.
func projectBlock(target, projector, gate bitsx.Matrix) (bitsx.Matrix, error) {
	rows := target.Rows
	cols := projector.Rows // projector is WT, so output dim is WT.Rows

	nextTarget, err := bitsx.NewZerosMatrix(rows, cols)
	if err != nil {
		return bitsx.Matrix{}, err
	}

	// We iterate row by row to perform ternary projection for each token/channel vector
	for r := 0; r < rows; r++ {
		targetRow := getRowAsMatrix(target, r)
		gateRow := getRowAsMatrix(gate, r)

		// Calculate threshold based on active voters in this row's gate
		activeCount := gateRow.PopCount()
		if activeCount == 0 {
			continue
		}
		threshold := (activeCount + 1) / 2

		// Compute Ternary Dot Product: WT . (Target \odot Gate)
		// DotTernaryVec correctly treats gated-off bits as 0, not -1.
		counts, err := projector.DotTernaryVec(targetRow, gateRow)
		if err != nil {
			return bitsx.Matrix{}, err
		}

		for c := 0; c < cols; c++ {
			if counts[c] >= threshold {
				if err := nextTarget.Set(r, c); err != nil {
					return bitsx.Matrix{}, err
				}
			}
		}
	}
	return nextTarget, nil
}

// accumulateSparseDelta computes the gradient for weights.
// `target` must be the desired state (0/1), NOT masked by the gate.
func accumulateSparseDelta(x, target bitsx.Matrix, zs []int, deltaData []int16, groupSize int) error {
	l := x.Rows
	dIn := x.Cols
	dOut := target.Cols

	if len(deltaData) != dOut*dIn {
		return fmt.Errorf("delta size mismatch: expected %d, got %d", dOut*dIn, len(deltaData))
	}

	if groupSize <= 0 {
		groupSize = 1
	}

	for i := 0; i < l; i++ {
		zsStart := i * dOut
		zsEnd := zsStart + dOut
		rowZs := zs[zsStart:zsEnd]

		absZs := make([]int, dOut)
		for j, z := range rowZs {
			absZs[j] = mathx.Abs(z)
		}

		ascAbsZIdxs := slicesx.Argsort(absZs)

		k := dOut / groupSize
		if k == 0 && dOut > 0 {
			k = 1
		}

		// Sparse update: only update weights for neurons close to the decision boundary
		for _, neuronIdx := range ascAbsZIdxs[:k] {

			targetBit, _ := target.Bit(i, neuronIdx)

			baseIdx := neuronIdx * dIn
			colIdx := 0
			xRowOffset := i * x.Stride

			for wIdx := 0; wIdx < x.Stride; wIdx++ {
				xWord := x.Data[xRowOffset+wIdx]

				validBits := 64
				if colIdx+validBits > dIn {
					validBits = dIn - colIdx
				}

				for b := 0; b < validBits; b++ {
					xBit := (xWord >> b) & 1
					// Gradient: (target - prediction) * input
					// In binary: 1 - 2 * (x ^ target)
					val := int16(1 - 2*int(xBit^targetBit))
					deltaData[baseIdx+colIdx+b] += val
				}
				colIdx += validBits
			}
		}
	}
	return nil
}

// ----------------------------------------------------------------------------
// Core Structures
// ----------------------------------------------------------------------------

type Forward func(bitsx.Matrix, *rand.Rand) (bitsx.Matrix, Backward, error)
type Backward func(bitsx.Matrix, Delta) (bitsx.Matrix, error)
type Forwards []Forward
type Backwards []Backward

type MixerLayer struct {
	// Token Mixing
	WT1  bitsx.Matrix
	WT1T bitsx.Matrix
	WT2  bitsx.Matrix
	WT2T bitsx.Matrix

	// Channel Mixing
	WC1  bitsx.Matrix
	WC1T bitsx.Matrix
	WC2  bitsx.Matrix
	WC2T bitsx.Matrix

	H         H
	IsNoisy   bool
	GroupSize int

	DimTok  int
	DimChan int
}

type Model struct {
	forwards        Forwards
	Layers          []MixerLayer
	Prototypes      []bitsx.Matrix
	deltasPerWorker [][]Delta
}

func NewForward(layer *MixerLayer) (Forward, error) {
	// Standard Deviations for Noise
	stdT1 := math.Sqrt(float64(layer.WT1.Cols))
	stdT2 := math.Sqrt(float64(layer.WT2.Cols))
	stdC1 := math.Sqrt(float64(layer.WC1.Cols))
	stdC2 := math.Sqrt(float64(layer.WC2.Cols))

	// Thresholds for Gate (sqrt(N))
	gateThreshT1 := int(stdT1)
	gateThreshT2 := int(stdT2)
	gateThreshC1 := int(stdC1)
	gateThreshC2 := int(stdC2)

	return func(x bitsx.Matrix, rng *rand.Rand) (bitsx.Matrix, Backward, error) {
		// --- 1. Token Mixing (Spatial) ---
		// x is (L x C)
		xT, err := x.Transpose() // (C x L)
		if err != nil {
			return bitsx.Matrix{}, nil, err
		}

		// Linear 1: xT . WT1 -> t1
		t1, t1Zs, err := linearBlock(xT, layer.WT1, stdT1, rng, layer.IsNoisy)
		if err != nil {
			return bitsx.Matrix{}, nil, err
		}

		// Linear 2: t1 . WT2 -> t2
		t2, t2Zs, err := linearBlock(t1, layer.WT2, stdT2, rng, layer.IsNoisy)
		if err != nil {
			return bitsx.Matrix{}, nil, err
		}

		yTok, err := t2.Transpose() // (L x C)
		if err != nil {
			return bitsx.Matrix{}, nil, err
		}

		// --- 2. Channel Mixing (Feature) ---
		// Linear 3: yTok . WC1 -> c1
		c1, c1Zs, err := linearBlock(yTok, layer.WC1, stdC1, rng, layer.IsNoisy)
		if err != nil {
			return bitsx.Matrix{}, nil, err
		}

		// Linear 4: c1 . WC2 -> yOut
		yOut, c2Zs, err := linearBlock(c1, layer.WC2, stdC2, rng, layer.IsNoisy)
		if err != nil {
			return bitsx.Matrix{}, nil, err
		}

		// --- Backward ---
		var backward Backward
		backward = func(target bitsx.Matrix, delta Delta) (bitsx.Matrix, error) {

			// Helper to create Gate matrix based on Zs
			makeGate := func(zs []int, rows, cols, threshold int) (bitsx.Matrix, error) {
				g, err := bitsx.NewZerosMatrix(rows, cols)
				if err != nil {
					return bitsx.Matrix{}, err
				}
				for r := 0; r < rows; r++ {
					offset := r * cols
					for c := 0; c < cols; c++ {
						z := zs[offset+c]
						if mathx.Abs(z) <= threshold {
							g.Set(r, c)
						}
					}
				}
				return g, nil
			}

			// --- Channel Mixing Backward ---
			gateC2, err := makeGate(c2Zs, yOut.Rows, yOut.Cols, gateThreshC2)
			if err != nil {
				return bitsx.Matrix{}, err
			}

			// Project C2 -> C1
			// Note: Pass target and gate separately to allow Ternary projection
			c1Target, err := projectBlock(target, layer.WC2T, gateC2)
			if err != nil {
				return bitsx.Matrix{}, err
			}

			gateC1, err := makeGate(c1Zs, c1.Rows, c1.Cols, gateThreshC1)
			if err != nil {
				return bitsx.Matrix{}, err
			}

			// Project C1 -> YTok
			yTokTarget, err := projectBlock(c1Target, layer.WC1T, gateC1)
			if err != nil {
				return bitsx.Matrix{}, err
			}

			// Transpose for Token Mixing
			t2Target, err := yTokTarget.Transpose()
			if err != nil {
				return bitsx.Matrix{}, err
			}

			// --- Token Mixing Backward ---
			gateT2, err := makeGate(t2Zs, t2.Rows, t2.Cols, gateThreshT2)
			if err != nil {
				return bitsx.Matrix{}, err
			}

			// Project T2 -> T1
			t1Target, err := projectBlock(t2Target, layer.WT2T, gateT2)
			if err != nil {
				return bitsx.Matrix{}, err
			}

			gateT1, err := makeGate(t1Zs, t1.Rows, t1.Cols, gateThreshT1)
			if err != nil {
				return bitsx.Matrix{}, err
			}

			// Project T1 -> XT
			xTTarget, err := projectBlock(t1Target, layer.WT1T, gateT1)
			if err != nil {
				return bitsx.Matrix{}, err
			}

			xTarget, err := xTTarget.Transpose()
			if err != nil {
				return bitsx.Matrix{}, err
			}

			// --- Accumulate Gradients ---
			if err := accumulateSparseDelta(c1, target, c2Zs, delta.DataC2, layer.GroupSize); err != nil {
				return bitsx.Matrix{}, err
			}
			if err := accumulateSparseDelta(yTok, c1Target, c1Zs, delta.DataC1, layer.GroupSize); err != nil {
				return bitsx.Matrix{}, err
			}
			if err := accumulateSparseDelta(t1, t2Target, t2Zs, delta.DataT2, layer.GroupSize); err != nil {
				return bitsx.Matrix{}, err
			}
			if err := accumulateSparseDelta(xT, t1Target, t1Zs, delta.DataT1, layer.GroupSize); err != nil {
				return bitsx.Matrix{}, err
			}

			return xTarget, nil
		}
		return yOut, backward, nil
	}, nil
}

func NewModel(L, C, dimTok, dimChan, numLayers int, p int, rng *rand.Rand) (Model, error) {
	m := Model{
		forwards:        make(Forwards, numLayers),
		Layers:          make([]MixerLayer, numLayers),
		deltasPerWorker: make([][]Delta, p),
	}

	for i := 0; i < numLayers; i++ {
		layer := MixerLayer{
			DimTok:    dimTok,
			DimChan:   dimChan,
			IsNoisy:   false,
			GroupSize: 2,
		}

		// Helper to init weights and hidden weights
		initW := func(r, c int) (bitsx.Matrix, bitsx.Matrix, []int8, error) {
			w, err := bitsx.NewRandMatrix(r, c, 0, rng)
			if err != nil {
				return bitsx.Matrix{}, bitsx.Matrix{}, nil, err
			}
			wt, err := w.Transpose()
			if err != nil {
				return bitsx.Matrix{}, bitsx.Matrix{}, nil, err
			}

			hData := make([]int8, r*c)
			for rr := 0; rr < r; rr++ {
				for cc := 0; cc < c; cc++ {
					idx := rr*c + cc // Row-major index for H

					// w.Bit(rr, cc) uses bit indexing
					bit, _ := w.Bit(rr, cc)

					// Init with synaptic inertia (e.g., +/- 31)
					if bit == 1 {
						hData[idx] = 31
					} else {
						hData[idx] = -31
					}
				}
			}
			return w, wt, hData, nil
		}

		var err error
		// WT1: (L -> DimTok) -> cols=L, rows=DimTok
		layer.WT1, layer.WT1T, layer.H.DataT1, err = initW(dimTok, L)
		if err != nil {
			return Model{}, err
		}

		// WT2: (DimTok -> L) -> cols=DimTok, rows=L
		layer.WT2, layer.WT2T, layer.H.DataT2, err = initW(L, dimTok)
		if err != nil {
			return Model{}, err
		}

		// WC1: (C -> DimChan) -> cols=C, rows=DimChan
		layer.WC1, layer.WC1T, layer.H.DataC1, err = initW(dimChan, C)
		if err != nil {
			return Model{}, err
		}

		// WC2: (DimChan -> C) -> cols=DimChan, rows=C
		layer.WC2, layer.WC2T, layer.H.DataC2, err = initW(C, dimChan)
		if err != nil {
			return Model{}, err
		}

		m.Layers[i] = layer
		m.forwards[i], err = NewForward(&m.Layers[i])
		if err != nil {
			return Model{}, err
		}
	}

	for workerId := 0; workerId < p; workerId++ {
		m.deltasPerWorker[workerId] = make([]Delta, numLayers)
		for i := 0; i < numLayers; i++ {
			// Sizes match the H Data sizes
			m.deltasPerWorker[workerId][i] = Delta{
				DataT1: make([]int16, len(m.Layers[i].H.DataT1)),
				DataT2: make([]int16, len(m.Layers[i].H.DataT2)),
				DataC1: make([]int16, len(m.Layers[i].H.DataC1)),
				DataC2: make([]int16, len(m.Layers[i].H.DataC2)),
			}
		}
	}
	return m, nil
}

func (m *Model) SetIsTraining(isTrain bool) {
	for i := range m.Layers {
		m.Layers[i].IsNoisy = isTrain
	}
}

func (fs Forwards) Propagate(x bitsx.Matrix, rng *rand.Rand) (bitsx.Matrix, Backwards, error) {
	backwards := make(Backwards, len(fs))
	var backward Backward
	var err error
	for i, f := range fs {
		x, backward, err = f(x, rng)
		if err != nil {
			return bitsx.Matrix{}, nil, err
		}
		backwards[i] = backward
	}
	return x, backwards, err
}

func (bs Backwards) Propagate(target bitsx.Matrix, deltas []Delta) (bitsx.Matrix, error) {
	var err error
	for layerI := len(bs) - 1; layerI >= 0; layerI-- {
		target, err = bs[layerI](target, deltas[layerI])
		if err != nil {
			return bitsx.Matrix{}, err
		}
	}
	return target, nil
}

func (m Model) Predict(x bitsx.Matrix, rng *rand.Rand) (bitsx.Matrix, error) {
	y, _, err := m.forwards.Propagate(x, rng)
	return y, err
}

func (m Model) PredictLogits(x bitsx.Matrix, rng *rand.Rand) ([]int, error) {
	y, err := m.Predict(x, rng)
	if err != nil {
		return nil, err
	}

	// y: (L x C)
	// Prototypes: (1 x C)

	n := len(m.Prototypes)
	logits := make([]int, n)

	for i, prototype := range m.Prototypes {
		counts, err := y.Dot(prototype) // counts has length L
		if err != nil {
			return nil, err
		}

		sumScore := 0
		for _, c := range counts {
			// Convert binary count to bipolar score: 2*c - width
			width := y.Cols
			z := 2*c - width
			sumScore += z
		}
		logits[i] = sumScore
	}
	return logits, nil
}

func (m *Model) ComputeSignDeltas(xs, ts []bitsx.Matrix, margin float32, rngs []*rand.Rand) ([]Delta, error) {
	n := len(xs)
	p := len(rngs)

	err := parallel.For(n, p, func(workerId, idx int) error {
		rng := rngs[workerId]
		deltas := m.deltasPerWorker[workerId]
		x, t := xs[idx], ts[idx]

		y, backwards, err := m.forwards.Propagate(x, rng)
		if err != nil {
			return err
		}

		// --- BEP/Margin Logic ---
		if margin > 0 && len(m.Prototypes) > 0 {
			// Calculate Score for Correct Class (t)
			correctCounts, err := y.Dot(t) // t is the prototype for correct class
			if err != nil {
				return err
			}
			correctScore := 0
			for _, c := range correctCounts {
				correctScore += (2*c - y.Cols)
			}

			maxWrongScore := -999999999
			for _, proto := range m.Prototypes {
				counts, err := y.Dot(proto)
				if err != nil {
					return err
				}
				score := 0
				for _, c := range counts {
					score += (2*c - y.Cols)
				}

				if score == correctScore {
					// Check if proto is t
					checkC, _ := t.Dot(proto)
					if checkC[0] == t.Cols {
						continue
					}
				}
				if score > maxWrongScore {
					maxWrongScore = score
				}
			}

			// Global max possible score = L * C
			maxPossible := float32(y.Rows * y.Cols)
			limit := int(maxPossible * margin)
			if (correctScore - maxWrongScore) >= limit {
				return nil
			}
		}

		// Backward Propagation
		// 'y' has shape (L, C)
		// 't' (prototype) has shape (1, C)
		// We must broadcast 't' to shape (L, C) to match 'y' for the backward pass.

		rows, cols := y.Rows, y.Cols
		tSeq, err := bitsx.NewZerosMatrix(rows, cols)
		if err != nil {
			return err
		}

		// 't' is (1, C). We want to copy the active bits of t[0] to all rows of tSeq.
		tData := t.Data       // length = stride
		stride := t.Stride
		tSeqData := tSeq.Data // length = rows * stride

		for r := 0; r < rows; r++ {
			// Copy stride uint64s
			copy(tSeqData[r*stride:(r+1)*stride], tData)
		}

		_, err = backwards.Propagate(tSeq, deltas)
		return err
	})

	if err != nil {
		return nil, err
	}

	layerNum := len(m.Layers)
	signDeltas := make([]Delta, layerNum)

	for i := 0; i < layerNum; i++ {
		l := &m.Layers[i]
		signDeltas[i] = Delta{
			DataT1: make([]int16, len(l.H.DataT1)),
			DataT2: make([]int16, len(l.H.DataT2)),
			DataC1: make([]int16, len(l.H.DataC1)),
			DataC2: make([]int16, len(l.H.DataC2)),
		}

		// Accumulate
		for wId := 0; wId < p; wId++ {
			wd := m.deltasPerWorker[wId][i]
			for k, v := range wd.DataT1 {
				signDeltas[i].DataT1[k] += v
			}
			for k, v := range wd.DataT2 {
				signDeltas[i].DataT2[k] += v
			}
			for k, v := range wd.DataC1 {
				signDeltas[i].DataC1[k] += v
			}
			for k, v := range wd.DataC2 {
				signDeltas[i].DataC2[k] += v
			}
		}

		// Sign conversion
		convert := func(data []int16) {
			for k, v := range data {
				data[k] = int16(cmp.Compare(v, 0))
			}
		}
		convert(signDeltas[i].DataT1)
		convert(signDeltas[i].DataT2)
		convert(signDeltas[i].DataC1)
		convert(signDeltas[i].DataC2)
	}

	// Clear buffers
	for i := range m.deltasPerWorker {
		for j := range m.deltasPerWorker[i] {
			clear(m.deltasPerWorker[i][j].DataT1)
			clear(m.deltasPerWorker[i][j].DataT2)
			clear(m.deltasPerWorker[i][j].DataC1)
			clear(m.deltasPerWorker[i][j].DataC2)
		}
	}
	return signDeltas, nil
}

func (m *Model) UpdateWeight(deltas []Delta, prob float32, rng *rand.Rand) error {
	updateMat := func(dataH []int8, deltaData []int16, matW, matWT *bitsx.Matrix) error {
		rows, cols := matW.Rows, matW.Cols // W dimensions
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				idx := r*cols + c
				d := deltaData[idx]
				if d == 0 {
					continue
				}
				if rng.Float32() > prob {
					continue
				}

				old := dataH[idx]
				newVal := int(old) + int(2*d)
				clipped := int8(max(math.MinInt8, min(math.MaxInt8, newVal)))
				dataH[idx] = clipped

				// If sign changes, toggle the bit in W and WT
				isOldPlus := old >= 0
				isNewPlus := clipped >= 0

				if isOldPlus != isNewPlus {
					if err := matW.Toggle(r, c); err != nil {
						return err
					}
					if err := matWT.Toggle(c, r); err != nil {
						return err
					}
				}
			}
		}
		return nil
	}

	for i, delta := range deltas {
		l := &m.Layers[i]
		if err := updateMat(l.H.DataT1, delta.DataT1, &l.WT1, &l.WT1T); err != nil {
			return err
		}
		if err := updateMat(l.H.DataT2, delta.DataT2, &l.WT2, &l.WT2T); err != nil {
			return err
		}
		if err := updateMat(l.H.DataC1, delta.DataC1, &l.WC1, &l.WC1T); err != nil {
			return err
		}
		if err := updateMat(l.H.DataC2, delta.DataC2, &l.WC2, &l.WC2T); err != nil {
			return err
		}
	}
	return nil
}