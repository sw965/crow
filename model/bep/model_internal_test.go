package bep

import "testing"

func TestArgmax(t *testing.T) {
	tests := []struct {
		name   string
		logits []int
		want   int
	}{
		{name: "正常_1要素", logits: []int{5}, want: 0},
		{name: "正常_最大が1つ", logits: []int{1, 9, 3}, want: 1},
		{name: "正常_同点なら後ろの添字", logits: []int{7, 2, 7, 7, 1}, want: 3},
		{name: "正常_負の値", logits: []int{-5, -1, -3}, want: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := argmax(tt.logits); got != tt.want {
				t.Errorf("argmax(%v) = %d, want = %d", tt.logits, got, tt.want)
			}
		})
	}
}
