package dataset

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"

	"github.com/sw965/omw/atomicfile"
	"github.com/sw965/omw/encoding/gobx"
)

type LogFunc func(format string, a ...any)

var PrintLog LogFunc = func(format string, a ...any) {
	fmt.Printf(format, a...)
}

func CleanCache() error {
	dir, err := getCacheDir()
	if err != nil {
		return err
	}

	// ディレクトリが存在しない場合は何もしない
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		return nil
	}
	return os.RemoveAll(dir)
}

// loadWithRecovery は、ファイルを確保してgobとして読み込む。
// デコードに失敗した場合、破損したキャッシュとみなして削除し、再ダウンロードした上で
// 1回だけ再試行する（無限に再試行はしない）。
func loadWithRecovery[T any](path, url string, logf LogFunc) (T, error) {
	var zero T

	if err := ensureFile(path, url, logf); err != nil {
		return zero, err
	}

	data, err := gobx.Load[T](path)
	if err == nil {
		return data, nil
	}

	if removeErr := os.Remove(path); removeErr != nil {
		return zero, fmt.Errorf("キャッシュの読み込みに失敗し(%w)、破損したキャッシュの削除にも失敗しました: %w", err, removeErr)
	}

	if err := ensureFile(path, url, logf); err != nil {
		return zero, err
	}

	data, err = gobx.Load[T](path)
	if err != nil {
		return zero, fmt.Errorf("キャッシュが破損していた為、再取得しましたが、それでも読み込みに失敗しました: %w", err)
	}
	return data, nil
}

func getCacheDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("ホームディレクトリの取得に失敗: %w", err)
	}
	return filepath.Join(home, ".sw965_crow", "dataset"), nil
}

// ensureFile はファイルが存在しない場合のみURLからダウンロードします。
// ダウンロードが途中で失敗した場合に壊れたファイルが残らないよう、
// atomicfile.WriteFrom で一時ファイル経由の安全な保存を行います。
// レスポンスの内容は全体をメモリに保持せず、ストリーミングで一時ファイルへ書き込みます。
func ensureFile(path, url string, logf LogFunc) (err error) {
	if logf == nil {
		logf = func(string, ...any) {}
	}

	if _, err := os.Stat(path); err == nil {
		return nil // 既に存在するのでスキップ
	}

	logf("Downloading %s...\n", url)
	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("リクエストの作成に失敗: %w", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("ダウンロードに失敗: %w", err)
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			err = errors.Join(err, closeErr)
		}
	}()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTPステータスが不正: %s", resp.Status)
	}

	if err := atomicfile.WriteFrom(path, resp.Body, 0644); err != nil {
		return fmt.Errorf("ファイルの保存に失敗: %w", err)
	}
	return nil
}
