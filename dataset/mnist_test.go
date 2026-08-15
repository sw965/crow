package dataset_test

import (
	"testing"

	"github.com/sw965/crow/dataset"
)

func assertIntLabels(t *testing.T, name string, labels []int, minVal, maxVal int) {
	t.Helper()
	seen := make(map[int]bool)
	for _, l := range labels {
		if l < minVal || l > maxVal {
			t.Errorf("範囲外の %s: %d", name, l)
		}
		seen[l] = true
	}
	wantClasses := maxVal - minVal + 1
	if len(seen) != wantClasses {
		t.Errorf("%s のクラス数が %d ではありません: %d", name, wantClasses, len(seen))
	}
}

func testDatasetLoader(t *testing.T, loadFn func(dataset.LogFunc) (dataset.Binary[int], error)) {
	t.Helper()

	count := dataset.GetAndIncrementCounter(dataset.CounterFileName)
	if count >= dataset.CacheResetTestInterval {
		t.Logf("カウンターが %d に達したため、キャッシュを削除して再ダウンロードをテストします", count)
		if err := dataset.CleanCache(); err != nil {
			t.Fatalf("CleanCache失敗: %v", err)
		}
		defer dataset.ResetCounter(dataset.CounterFileName)
	}

	ds, err := loadFn(t.Logf)
	if err != nil {
		t.Fatalf("読み込み失敗: %v", err)
	}
	if len(ds.TrainInputs) != 60000 || len(ds.TrainLabels) != 60000 {
		t.Errorf("Train件数不一致: inputs=%d, labels=%d", len(ds.TrainInputs), len(ds.TrainLabels))
	}
	if len(ds.TestInputs) != 10000 || len(ds.TestLabels) != 10000 {
		t.Errorf("Test件数不一致: inputs=%d, labels=%d", len(ds.TestInputs), len(ds.TestLabels))
	}

	if ds.TrainInputs[0].Rows() != 1 || ds.TrainInputs[0].Cols() != 784 {
		t.Errorf("TrainInput形状不一致: got=(%d x %d), want=(1 x 784)", ds.TrainInputs[0].Rows(), ds.TrainInputs[0].Cols())
	}
	if ds.TestInputs[0].Rows() != 1 || ds.TestInputs[0].Cols() != 784 {
		t.Errorf("TestInput形状不一致: got=(%d x %d), want=(1 x 784)", ds.TestInputs[0].Rows(), ds.TestInputs[0].Cols())
	}

	assertIntLabels(t, "TrainLabels", ds.TrainLabels, 0, 9)
	assertIntLabels(t, "TestLabels", ds.TestLabels, 0, 9)
}

func TestLoadMNIST(t *testing.T) {
	testDatasetLoader(t, dataset.LoadMNIST)
}

func TestLoadFashionMNIST(t *testing.T) {
	testDatasetLoader(t, dataset.LoadFashionMNIST)
}
