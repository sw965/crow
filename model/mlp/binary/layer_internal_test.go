package binary

import "testing"

func TestIsqrt(t *testing.T) {
	for n := range 10001 {
		r := isqrt(n)
		if r*r > n || (r+1)*(r+1) <= n {
			t.Fatalf("isqrt(%d) = %d: r*r <= n < (r+1)*(r+1) であるべき", n, r)
		}
	}
}

func TestMaxAbsNoiseForScale(t *testing.T) {
	t.Run("正常_倍率1/2は旧既定値の式と一致", func(t *testing.T) {
		// 旧既定値 isqrt(fanIn) * 866 / 1000 と、全ての fanIn で同じ値になる(1732/2000 = 866/1000)
		for fanIn := 1; fanIn <= 5000; fanIn++ {
			got, err := maxAbsNoiseForScale(fanIn, 1, 2)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			if want := isqrt(fanIn) * 866 / 1000; got != want {
				t.Fatalf("fanIn = %d: got = %d, want = %d", fanIn, got, want)
			}
		}
	})

	tests := []struct {
		name       string
		fanIn      int
		num, denom int
		want       int
		wantErr    bool
	}{
		// 旧実装の 0.5 × √3 × √fanIn(小数)を切り捨てた値
		{name: "正常_784入力_1/2", fanIn: 784, num: 1, denom: 2, want: 24},
		{name: "正常_512入力_1/2", fanIn: 512, num: 1, denom: 2, want: 19},
		{name: "正常_784入力_3/4", fanIn: 784, num: 3, denom: 4, want: 36},
		{name: "正常_784入力_1/1", fanIn: 784, num: 1, denom: 1, want: 48},
		{name: "正常_倍率0", fanIn: 784, num: 0, denom: 1, want: 0},
		{name: "正常_桁あふれしうる大きさの分子と分母", fanIn: 784, num: 1 << 60, denom: 1 << 61, want: 24},
		{name: "異常_分子が負", fanIn: 784, num: -1, denom: 2, wantErr: true},
		{name: "異常_分母が0", fanIn: 784, num: 1, denom: 0, wantErr: true},
		{name: "異常_入力数を超える", fanIn: 784, num: 17, denom: 1, wantErr: true},
		{name: "異常_桁あふれする大きさの分子", fanIn: 784, num: 1 << 62, denom: 1, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := maxAbsNoiseForScale(tt.fanIn, tt.num, tt.denom)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("エラーを期待したが、nilが返された: got = %d", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			if got != tt.want {
				t.Errorf("got = %d, want = %d", got, tt.want)
			}
		})
	}
}
