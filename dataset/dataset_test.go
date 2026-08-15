package dataset

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCleanCache(t *testing.T) {
	count := GetAndIncrementCounter(CounterFileName)
	if count < CacheResetTestInterval {
		return
	}
	defer ResetCounter(CounterFileName)

	dir, err := getCacheDir()
	if err != nil {
		t.Fatalf("getCacheDir失敗: %v", err)
	}

	// 既存のファイルがあれば削除。既存のファイルがなくてもエラーは起きない
	if err := CleanCache(); err != nil {
		t.Fatalf("CleanCache失敗: %v", err)
	}

	// 2回CleanCacheを呼び出す事で、ファイルが存在しない状態でCleanCacheを呼び出してもエラーが起きない事をテスト
	if err := CleanCache(); err != nil {
		t.Errorf("存在しない状態でのCleanCache呼び出しでエラー: %v", err)
	}

	// ディレクトリを作成してCleanCacheで消えるか確認
	if err := os.Mkdir(dir, 0755); err != nil {
		t.Fatalf("ディレクトリ作成失敗: %v", err)
	}

	dummyFile := filepath.Join(dir, "dummy.txt")
	if err := os.WriteFile(dummyFile, []byte("test"), 0644); err != nil {
		t.Fatalf("ダミーファイル作成失敗: %v", err)
	}

	if err := CleanCache(); err != nil {
		t.Fatalf("CleanCache失敗: %v", err)
	}

	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Errorf("CleanCache実行後もディレクトリが存在しています: %s", dir)
	}
}

func TestPrintLog(t *testing.T) {
	PrintLog("test log message: %d\n", 123)
}
