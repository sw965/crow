package main

import (
	"testing"
)

// setStates は、(word, bit) が有効な全リテラルについて states の値をビットスライスへ書き込む。
// states は states[word][bit] のレイアウト。
func setStates(c *clause, states *[numLitWords][64]uint8) {
	for p := range numPlanes {
		for w := range numLitWords {
			c.planes[p][w] = 0
		}
	}
	for w := range numLitWords {
		for b := range 64 {
			if validMask[w]>>b&1 == 0 {
				continue
			}
			s := states[w][b]
			for p := range numPlanes {
				if s>>p&1 == 1 {
					c.planes[p][w] |= 1 << b
				}
			}
		}
	}
	c.recount()
}

// getStates は setStates の逆変換。
func getStates(c *clause) [numLitWords][64]uint8 {
	var states [numLitWords][64]uint8
	for w := range numLitWords {
		for b := range 64 {
			var s uint8
			for p := range numPlanes {
				if c.planes[p][w]>>b&1 == 1 {
					s |= 1 << p
				}
			}
			states[w][b] = s
		}
	}
	return states
}

// refApply は参照実装: mask のビットが立っている有効リテラルの状態を±1(0/255で飽和)する。
func refApply(states *[numLitWords][64]uint8, mask *[numLitWords]uint64, inc bool) {
	for w := range numLitWords {
		m := mask[w] & validMask[w]
		for b := range 64 {
			if m>>b&1 == 0 {
				continue
			}
			if inc {
				if states[w][b] < 255 {
					states[w][b]++
				}
			} else {
				if states[w][b] > 0 {
					states[w][b]--
				}
			}
		}
	}
}

// checkIncDecAgainstReference は、シードから作った乱数状態・乱数マスクで
// incMasked / decMasked を繰り返し適用し、参照実装と完全一致する事を確認する。
func checkIncDecAgainstReference(t *testing.T, seed uint64) {
	t.Helper()
	r := newRng(seed)

	var states [numLitWords][64]uint8
	for w := range numLitWords {
		for b := range 64 {
			if validMask[w]>>b&1 == 0 {
				continue
			}
			// 飽和境界(0, 255)を高頻度で踏むように偏らせる
			switch r.intn(4) {
			case 0:
				states[w][b] = 0
			case 1:
				states[w][b] = 255
			case 2:
				states[w][b] = uint8(254 + r.intn(2)) // 254 or 255
			default:
				states[w][b] = uint8(r.intn(256))
			}
		}
	}

	var c clause
	setStates(&c, &states)

	for range 32 {
		var mask [numLitWords]uint64
		for w := range numLitWords {
			mask[w] = r.next() & validMask[w]
		}
		inc := r.intn(2) == 0
		if inc {
			c.incMasked(&mask)
		} else {
			c.decMasked(&mask)
		}
		refApply(&states, &mask, inc)

		got := getStates(&c)
		if got != states {
			t.Fatalf("ビットスライス実装と参照実装が不一致 (seed=%d, inc=%t)", seed, inc)
		}
	}
}

func TestIncDecMaskedMatchesReference(t *testing.T) {
	for seed := uint64(1); seed <= 8; seed++ {
		checkIncDecAgainstReference(t, seed)
	}
}

func FuzzIncDecMasked(f *testing.F) {
	f.Add(uint64(42))
	f.Fuzz(func(t *testing.T, seed uint64) {
		checkIncDecAgainstReference(t, seed)
	})
}

func TestFires(t *testing.T) {
	var states [numLitWords][64]uint8

	// 包含: 正リテラル0 (word0 bit0)、否定リテラル5 (word13 bit5)
	states[0][0] = 200
	states[wordsPerFtr][5] = 128

	var c clause
	setStates(&c, &states)
	if c.includeCnt != 2 {
		t.Fatalf("includeCntの不一致: got = %d, want = 2", c.includeCnt)
	}

	// x: 画素0=1, 画素5=0 → 正リテラル0は真、否定リテラル5も真 → 発火
	var lit [numLitWords]uint64
	lit[0] = 1 // x のワード
	for w := range wordsPerFtr {
		lit[wordsPerFtr+w] = ^lit[w] & validMask[wordsPerFtr+w] // ~x のワード
	}
	if !c.fires(&lit) {
		t.Error("発火するべき入力で発火しなかった")
	}

	// 画素5=1 にすると否定リテラル5が偽 → 不発火
	lit[0] |= 1 << 5
	lit[wordsPerFtr] &^= 1 << 5
	if c.fires(&lit) {
		t.Error("発火しないべき入力で発火した")
	}
}

func TestPackSamples(t *testing.T) {
	img := &oldMatrix{Rows: 1, Cols: numFeatures, Data: make([]uint64, wordsPerFtr)}
	img.Data[0] = 0b1011
	img.Data[12] = 0xFFFF // 有効な末尾16ビット全部1

	samples, err := packSamples(oldMatrices{img}, []int{7})
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	s := samples[0]

	if s.label != 7 {
		t.Errorf("labelの不一致: got = %d, want = 7", s.label)
	}
	if s.lit[0] != 0b1011 {
		t.Errorf("正リテラルの不一致: got = %b", s.lit[0])
	}
	if s.notLit[0] != ^uint64(0b1011) {
		t.Errorf("偽リテラル(正側)の不一致: got = %b", s.notLit[0])
	}

	for w := range numLitWords {
		if s.lit[w]&^validMask[w] != 0 || s.notLit[w]&^validMask[w] != 0 {
			t.Errorf("無効ビットが立っている: word = %d", w)
		}
		// 真リテラルと偽リテラルは有効ビット上で互いに補集合
		if s.lit[w]^s.notLit[w] != validMask[w] {
			t.Errorf("lit XOR notLit != validMask: word = %d", w)
		}
	}
}

// TestMintermConstructionRepresentsArbitraryMapping は、REPORT.md §5 の万能近似性の
// 核となる構成(コードごとのminterm節+極性割り当て)が、有限二値入力集合上の
// 任意のラベル割り当てを厳密に表現できる事を検証する。
//
//   - ランダムな784bitコード40個に無作為なラベル(0..9)を割り当てる
//   - 各コードについて「そのコードの真リテラル全784個を包含する節」(minterm節)を
//     ラベルのクラスの正極性スロット(偶数インデックス)へ直接構成する
//   - 全コードで、正解クラスの投票和がちょうど +1、他クラスは 0(argmaxが厳密一致)
//   - コードを1ビット反転した入力(コード集合外)では全クラスの投票和が 0
//     (UNIVERSAL_APPROXIMATION.md §4.2 のC限定構成と同じ「C外は既定動作」の確認)
//
// これは表現能力の検証であり、学習(trainEpoch)は使わない
// (表現能力と学習可能性の分離は REPORT.md §5 参照)。
func TestMintermConstructionRepresentsArbitraryMapping(t *testing.T) {
	r := newRng(7)

	const numCodes = 40
	type code struct {
		words [wordsPerFtr]uint64
		label int
	}
	codes := make([]code, numCodes)
	for i := range codes {
		for w := range wordsPerFtr {
			codes[i].words[w] = r.next() & validMask[w]
		}
		codes[i].label = int(r.intn(numClasses))
	}

	// クラスごとのminterm節数を数え、最大数に合わせて節数を確保する
	// (余りと奇数インデックスは空節のままにする。空節は推論では発火しない)
	perClass := make([]int, numClasses)
	for _, cd := range codes {
		perClass[cd.label]++
	}
	maxPer := 1
	for _, n := range perClass {
		maxPer = max(maxPer, n)
	}

	m := newTM(2*maxPer, 1, 3, false)
	next := make([]int, numClasses)
	for _, cd := range codes {
		var states [numLitWords][64]uint8
		for w := range wordsPerFtr {
			x := cd.words[w]
			for b := range 64 {
				if validMask[w]>>b&1 == 0 {
					continue
				}
				if x>>b&1 == 1 {
					states[w][b] = 255 // 画素=1: 正リテラルを包含
				} else {
					states[wordsPerFtr+w][b] = 255 // 画素=0: 否定リテラルを包含
				}
			}
		}
		j := 2 * next[cd.label] // 偶数インデックス = 正極性
		next[cd.label]++
		setStates(&m.clauses[cd.label][j], &states)
	}

	for i, cd := range codes {
		var lit [numLitWords]uint64
		for w := range wordsPerFtr {
			lit[w] = cd.words[w]
			lit[wordsPerFtr+w] = ^cd.words[w] & validMask[wordsPerFtr+w]
		}
		for cls := range numClasses {
			sum := m.classSumInfer(cls, &lit)
			want := int64(0)
			if cls == cd.label {
				want = 1
			}
			if sum != want {
				t.Fatalf("code %d: class %d の投票和が不一致: got = %d, want = %d", i, cls, sum, want)
			}
		}

		// 1ビット反転(コード集合外の入力)では全クラスの投票和が0
		flipped := int(r.intn(numFeatures))
		fw, fb := flipped/64, uint(flipped%64)
		lit[fw] ^= 1 << fb
		lit[wordsPerFtr+fw] ^= 1 << fb
		for cls := range numClasses {
			if sum := m.classSumInfer(cls, &lit); sum != 0 {
				t.Fatalf("code %d(1bit反転): class %d の投票和が0でない: got = %d", i, cls, sum)
			}
		}
	}
}

// TestTrainEpochLearnsSyntheticPatterns は、クラスごとの固定パターン+ビット反転ノイズという
// 単純な合成タスクを実際に学習できる事を確認する(決定的: シード固定・ワーカー2)。
func TestTrainEpochLearnsSyntheticPatterns(t *testing.T) {
	r := newRng(123)

	// クラスごとのプロトタイプ(784bit乱数)
	var protos [numClasses][wordsPerFtr]uint64
	for cls := range numClasses {
		for w := range wordsPerFtr {
			protos[cls][w] = r.next() & validMask[w]
		}
	}

	newSample := func(cls int) packedSample {
		var s packedSample
		for w := range wordsPerFtr {
			r1, r2 := r.next(), r.next()
			flip := r1 & r2 // 各ビット確率1/4で反転
			x := (protos[cls][w] ^ flip) & validMask[w]
			s.lit[w] = x
			s.lit[wordsPerFtr+w] = ^x & validMask[wordsPerFtr+w]
			s.notLit[w] = ^x & validMask[w]
			s.notLit[wordsPerFtr+w] = x
		}
		s.label = cls
		return s
	}

	train := make([]packedSample, 0, 2000)
	test := make([]packedSample, 0, 500)
	for i := range 2000 {
		train = append(train, newSample(i%numClasses))
	}
	for i := range 500 {
		test = append(test, newSample(i%numClasses))
	}

	m := newTM(64, 256, 3, true)
	for range 6 {
		m.trainEpoch(train, r, 2)
	}

	acc := m.accuracy(test, 2)
	if acc < 0.9 {
		t.Errorf("合成タスクの精度が低すぎる: got = %.4f, want >= 0.9", acc)
	}
}
