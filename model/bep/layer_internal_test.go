package bep

import "testing"

func TestIsqrt(t *testing.T) {
	for n := range 10001 {
		r := isqrt(n)
		if r*r > n || (r+1)*(r+1) <= n {
			t.Fatalf("isqrt(%d) = %d: r*r <= n < (r+1)*(r+1) であるべき", n, r)
		}
	}
}

func TestAbsZForScale(t *testing.T) {
	tests := []struct {
		name       string
		fanIn      int
		num, denom int
		want       int
		wantErr    bool
	}{
		{name: "正常_784入力_1/4", fanIn: 784, num: 1, denom: 4, want: 7},
		{name: "正常_512入力_1/4", fanIn: 512, num: 1, denom: 4, want: 5},
		{name: "正常_784入力_1/2", fanIn: 784, num: 1, denom: 2, want: 14},
		{name: "正常_倍率0", fanIn: 784, num: 0, denom: 1, want: 0},
		{name: "正常_|z|の最大値ちょうど", fanIn: 784, num: 56, denom: 1, want: 1568},
		{name: "正常_桁あふれしうる大きさの分子と分母", fanIn: 784, num: 1 << 60, denom: 1 << 62, want: 7},
		{name: "異常_分子が負", fanIn: 784, num: -1, denom: 4, wantErr: true},
		{name: "異常_分母が0", fanIn: 784, num: 1, denom: 0, wantErr: true},
		{name: "異常_|z|の最大値を超える", fanIn: 784, num: 57, denom: 1, wantErr: true},
		{name: "異常_桁あふれする大きさの分子", fanIn: 784, num: 1 << 62, denom: 1, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := absZForScale(tt.fanIn, tt.num, tt.denom)
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
