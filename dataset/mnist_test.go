package dataset

import (
	"encoding/gob"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/sw965/omw/encoding/gobx"
	"github.com/sw965/omw/mathx/bitsx"
)

// 本物のデータセットはネットワーク経由でしか取得できない為、
// テスト方針に従い、httptestによる自作の互換環境(ローカルHTTPサーバー)で
// ダウンロードと読み込みの振る舞いを検証する。

func TestEnsureFile(t *testing.T) {
	t.Run("正常_ダウンロードして保存", func(t *testing.T) {
		content := []byte("dummy data")
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Write(content)
		}))
		defer server.Close()

		dir := t.TempDir()
		path := filepath.Join(dir, "data.gob")
		if err := ensureFile(path, server.URL+"/data.gob", nil); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}

		got, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if !slices.Equal(got, content) {
			t.Errorf("内容の不一致: got = %q, want = %q", got, content)
		}

		// 一時ファイルは残っておらず、保存先だけが残っている
		entries, err := os.ReadDir(dir)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if len(entries) != 1 || entries[0].Name() != "data.gob" {
			t.Errorf("一時ファイルが残っている: %v", entries)
		}
	})

	t.Run("正常_既に存在する場合はダウンロードしない", func(t *testing.T) {
		requested := false
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			requested = true
		}))
		defer server.Close()

		path := filepath.Join(t.TempDir(), "data.gob")
		want := []byte("cached")
		if err := os.WriteFile(path, want, 0644); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}

		if err := ensureFile(path, server.URL+"/data.gob", nil); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}

		if requested {
			t.Error("既存ファイルがあるのにダウンロードが行われた")
		}

		got, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if !slices.Equal(got, want) {
			t.Errorf("既存ファイルが上書きされた: got = %q, want = %q", got, want)
		}
	})

	t.Run("異常_HTTPステータスがOK以外", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.NotFound(w, r)
		}))
		defer server.Close()

		dir := t.TempDir()
		path := filepath.Join(dir, "data.gob")
		err := ensureFile(path, server.URL+"/data.gob", nil)
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}

		// 失敗時に、壊れたファイルや一時ファイルが残らない
		entries, readErr := os.ReadDir(dir)
		if readErr != nil {
			t.Fatalf("予期せぬエラー: %v", readErr)
		}
		if len(entries) != 0 {
			t.Errorf("ファイルが残っている: %v", entries)
		}
	})

	t.Run("異常_サーバーに接続できない", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "data.gob")
		// 存在しないサーバー
		err := ensureFile(path, "http://127.0.0.1:1/data.gob", nil)
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
	})
}

func TestLoadDataset(t *testing.T) {
	// サーバー側に4つのgobファイルを用意する
	serverDir := t.TempDir()

	newMatrix := func(setCols []int) *bitsx.Matrix {
		m, err := bitsx.NewZerosMatrix(1, 8)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		for _, c := range setCols {
			if err := m.Set(0, c); err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
		}
		return m
	}

	wantTrainImages := bitsx.Matrices{newMatrix([]int{0, 1})}
	wantTrainLabels := []int{7}
	wantTestImages := bitsx.Matrices{newMatrix([]int{2, 3})}
	wantTestLabels := []int{3}

	files := map[string]any{
		"train_imgs.gob":   wantTrainImages,
		"train_labels.gob": wantTrainLabels,
		"test_imgs.gob":    wantTestImages,
		"test_labels.gob":  wantTestLabels,
	}
	for name, data := range files {
		if err := gobx.Save(data, filepath.Join(serverDir, name)); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
	}

	server := httptest.NewServer(http.FileServer(http.Dir(serverDir)))
	defer server.Close()

	t.Run("正常_ダウンロードと読み込み", func(t *testing.T) {
		cacheDir := filepath.Join(t.TempDir(), "cache")
		got, err := loadDataset(server.URL+"/", cacheDir, "train_imgs.gob", "train_labels.gob", "test_imgs.gob", "test_labels.gob", t.Logf)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}

		if !slices.Equal(got.TrainLabels, wantTrainLabels) {
			t.Errorf("TrainLabelsの不一致: got = %v, want = %v", got.TrainLabels, wantTrainLabels)
		}
		if !slices.Equal(got.TestLabels, wantTestLabels) {
			t.Errorf("TestLabelsの不一致: got = %v, want = %v", got.TestLabels, wantTestLabels)
		}
		if !slices.Equal(got.TrainImages[0].Data, wantTrainImages[0].Data) {
			t.Errorf("TrainImagesの不一致: got = %v, want = %v", got.TrainImages[0].Data, wantTrainImages[0].Data)
		}
		if !slices.Equal(got.TestImages[0].Data, wantTestImages[0].Data) {
			t.Errorf("TestImagesの不一致: got = %v, want = %v", got.TestImages[0].Data, wantTestImages[0].Data)
		}

		// 2回目はキャッシュから読み込まれ、結果が同じになる
		got2, err := loadDataset(server.URL+"/", cacheDir, "train_imgs.gob", "train_labels.gob", "test_imgs.gob", "test_labels.gob", nil)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if !slices.Equal(got2.TrainLabels, wantTrainLabels) {
			t.Errorf("キャッシュからのTrainLabelsの不一致: got = %v, want = %v", got2.TrainLabels, wantTrainLabels)
		}
	})

	t.Run("異常_存在しないファイル", func(t *testing.T) {
		cacheDir := filepath.Join(t.TempDir(), "cache")
		_, err := loadDataset(server.URL+"/", cacheDir, "no_such.gob", "train_labels.gob", "test_imgs.gob", "test_labels.gob", nil)
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
	})
}

func TestLoadWithRecovery(t *testing.T) {
	t.Run("正常_初回でダウンロードして読み込める", func(t *testing.T) {
		requestCount := 0
		want := []int{1, 2, 3}
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			requestCount++
			enc := gob.NewEncoder(w)
			if err := enc.Encode(want); err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
		}))
		defer server.Close()

		path := filepath.Join(t.TempDir(), "data.gob")
		got, err := loadWithRecovery[[]int](path, server.URL+"/data.gob", nil)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if !slices.Equal(got, want) {
			t.Errorf("値の不一致: got = %v, want = %v", got, want)
		}
		if requestCount != 1 {
			t.Errorf("リクエスト回数の不一致: got = %d, want = 1", requestCount)
		}
	})

	t.Run("正常_破損キャッシュから回復する", func(t *testing.T) {
		requestCount := 0
		want := []int{4, 5, 6}
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			requestCount++
			enc := gob.NewEncoder(w)
			if err := enc.Encode(want); err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
		}))
		defer server.Close()

		path := filepath.Join(t.TempDir(), "data.gob")
		// 破損したキャッシュを事前に置いておく(存在するのでensureFileはダウンロードをスキップする)
		if err := os.WriteFile(path, []byte("broken"), 0644); err != nil {
			t.Fatalf("準備失敗: %v", err)
		}

		got, err := loadWithRecovery[[]int](path, server.URL+"/data.gob", nil)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if !slices.Equal(got, want) {
			t.Errorf("値の不一致: got = %v, want = %v", got, want)
		}
		// 破損検出後の再取得で1回だけリクエストが発生する
		if requestCount != 1 {
			t.Errorf("リクエスト回数の不一致: got = %d, want = 1", requestCount)
		}
	})

	t.Run("異常_再取得後も破損している場合は1回で諦める", func(t *testing.T) {
		requestCount := 0
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			requestCount++
			w.Write([]byte("still broken"))
		}))
		defer server.Close()

		path := filepath.Join(t.TempDir(), "data.gob")
		if err := os.WriteFile(path, []byte("broken"), 0644); err != nil {
			t.Fatalf("準備失敗: %v", err)
		}

		_, err := loadWithRecovery[[]int](path, server.URL+"/data.gob", nil)
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
		if requestCount != 1 {
			t.Errorf("再ダウンロード回数の不一致(繰り返し再取得しない事): got = %d, want = 1", requestCount)
		}
	})
}
