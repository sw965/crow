package mixer_test

import (
	"fmt"
	"log"
	"math/rand/v2"
	"testing"

	"github.com/sw965/crow/dataset"
	"github.com/sw965/crow/model/binary/mlp/mixer" // パッケージパスは適切に調整してください
	"github.com/sw965/omw/mathx/bitsx"
	"github.com/sw965/omw/mathx/randx"
	"github.com/sw965/omw/slicesx"
)

// imageToSequence converts a flattened 784-bit image to a 28x28 binary sequence (bitsx.Matrix).
// Output is (L=28, C=28).
func imageToSequence(img bitsx.Matrix) (bitsx.Matrix, error) {
	if img.Cols != 784 {
		return bitsx.Matrix{}, fmt.Errorf("expected 784 cols, got %d", img.Cols)
	}

	seq, err := bitsx.NewZerosMatrix(28, 28)
	if err != nil {
		return bitsx.Matrix{}, err
	}

	for r := 0; r < 28; r++ {
		for c := 0; c < 28; c++ {
			flatIdx := r*28 + c
			bit, _ := img.Bit(0, flatIdx) 
			
			if bit == 1 {
				seq.Set(r, c)
			}
		}
	}
	return seq, nil
}

func TestMLPMixerMNIST(t *testing.T) {
	// --- Hyperparameters ---
	epochs := 1000
	batchSize := 128
	parallelism := 4
	lr := float32(0.01) // Update Probability

	L := 28        // Sequence Length (Rows of image)
	C := 28        // Input Channels (Cols of image)
	dimTok := 512   // Token Mixing Hidden Dim (Compact for test speed)
	dimChan := 512  // Channel Mixing Hidden Dim
	numLayers := 2 // Number of Mixer Blocks

	// 1. Load Data
	mnist, err := dataset.LoadFashionMnist()
	if err != nil {
		t.Fatal(err)
	}

	log.Println("Binarizing train images...")
	// Binarize using a simple threshold (e.g. 0.001 to capture non-background)
	xTrainRaw, err := dataset.BinarizeImages(mnist.TrainImages, 0.001)
	if err != nil {
		t.Fatal(err)
	}

	xTestRaw, err := dataset.BinarizeImages(mnist.TestImages, 0.001)
	if err != nil {
		t.Fatal(err)
	}

	// 2. Prepare Model
	rng := rand.New(rand.NewPCG(1, 2))
	model, err := mixer.NewModel(
		L, C, dimTok, dimChan, numLayers, parallelism,
		randx.NewPCGFromGlobalSeed(),
	)
	if err != nil {
		t.Fatal(err)
	}

	// 3. Prototypes (BEF - Binary Equiangular Frames)
	log.Println("Generating BEF prototypes...")
	// Prototypes in Mixer must match the output shape of the last layer.
	// In this implementation, the output of the mixer is (L x C).
	// Typically, prototypes for classification are vectors of size C (or L*C flattened).
	// The implementation of PredictLogits computes `y.Dot(proto)`.
	// Since `y` is (L x C), `proto` should be (1 x C) to broadcast dot-prod over L rows,
	// or (C x 1) if transposed.
	// Based on code: `y.Dot(proto)` returns counts array of length y.Rows (L).
	// We sum these up. So proto must have same Cols as y (which is C).
	// Proto Rows should be 1.
	
	protoDim := C
	protoBits, err := bitsx.NewBEFPrototypeMatrices(10, protoDim, 10000, rng)
	if err != nil {
		t.Fatal(err)
	}
	model.Prototypes = protoBits

	// 4. Prepare Targets
	log.Println("Preparing targets...")
	tTrain := make([]bitsx.Matrix, len(xTrainRaw))
	for i, label := range mnist.TrainLabels {
		tTrain[i] = model.Prototypes[int(label)]
	}

	// 5. Training Loop
	rngs := make([]*rand.Rand, parallelism)
	for i := range rngs {
		rngs[i] = randx.NewPCGFromGlobalSeed()
	}

	numTrain := len(xTrainRaw)
	numBatches := numTrain / batchSize

	log.Println("Start training...")
	for epoch := 0; epoch < epochs; epoch++ {
		perm := rng.Perm(numTrain)
		model.SetIsTraining(true)

		for b := 0; b < numBatches; b++ {
			start := b * batchSize

			batchXs := make([]bitsx.Matrix, batchSize)
			batchTs := make([]bitsx.Matrix, batchSize)

			for i := 0; i < batchSize; i++ {
				idx := perm[start+i]
				seq, err := imageToSequence(xTrainRaw[idx])
				if err != nil {
					t.Fatal(err)
				}
				batchXs[i] = seq
				batchTs[i] = tTrain[idx]
			}

			// Margin 0.5 is standard for BEP
			deltas, err := model.ComputeSignDeltas(batchXs, batchTs, 0.00001, rngs)
			if err != nil {
				t.Fatal(err)
			}

			if err := model.UpdateWeight(deltas, lr, rng); err != nil {
				t.Fatal(err)
			}
		}

		// Validation
		model.SetIsTraining(false)
		correct := 0
		testSamples := 1000 // Limit for speed
		for i := 0; i < testSamples; i++ {
			seq, _ := imageToSequence(xTestRaw[i])
			logits, err := model.PredictLogits(seq, rng)
			if err != nil {
				t.Fatal(err)
			}
			pred := slicesx.Argsort(logits)[len(logits)-1]
			if float32(pred) == mnist.TestLabels[i] {
				correct++
			}
		}
		acc := float32(correct) / float32(testSamples) * 100
		log.Printf("Epoch %d: Test Acc %.2f%%", epoch+1, acc)
	}
}