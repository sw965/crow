package game_test

import (
	"math"
	"strings"
	"testing"

	"github.com/sw965/crow/game"
	"github.com/sw965/omw/mathx/randx"
	"github.com/sw965/omw/slicesx"
)

func TestPolicyValidateForLegalActions(t *testing.T) {
	tests := []struct {
		name           string
		policy         game.Policy[string]
		legalActions   []string
		checkUnique    bool
		wantErrMsgSubs []string
	}{
		{
			name:         "正常",
			policy:       game.Policy[string]{"グー": 0.5, "パー": 0.3, "チョキ": 0.2},
			legalActions: []string{"グー", "パー", "チョキ"},
			checkUnique:  true,
		},
		{
			name:           "異常_legalActionsが空",
			policy:         game.Policy[string]{},
			legalActions:   []string{},
			wantErrMsgSubs: []string{"legalActionsが空"},
		},
		{
			name:           "異常_要素数の不一致",
			policy:         game.Policy[string]{"グー": 0.5, "パー": 0.5},
			legalActions:   []string{"グー", "パー", "チョキ"},
			wantErrMsgSubs: []string{"要素数が不一致"},
		},
		{
			name:           "異常_合法手の確率が存在しない",
			policy:         game.Policy[string]{"グー": 0.5, "ビーム": 0.5},
			legalActions:   []string{"グー", "パー"},
			wantErrMsgSubs: []string{"確率が存在しません", "パー"},
		},
		{
			name:           "異常_負の確率",
			policy:         game.Policy[string]{"グー": 1.5, "パー": -0.5},
			legalActions:   []string{"グー", "パー"},
			wantErrMsgSubs: []string{"確率が不正"},
		},
		{
			name:           "異常_NaN",
			policy:         game.Policy[string]{"グー": float32(math.NaN()), "パー": 0.5},
			legalActions:   []string{"グー", "パー"},
			wantErrMsgSubs: []string{"確率が不正"},
		},
		{
			name:           "異常_合計が0",
			policy:         game.Policy[string]{"グー": 0.0, "パー": 0.0},
			legalActions:   []string{"グー", "パー"},
			wantErrMsgSubs: []string{"合計が0"},
		},
		{
			name:           "異常_合法手の重複",
			policy:         game.Policy[string]{"グー": 0.5, "パー": 0.5},
			legalActions:   []string{"グー", "グー"},
			checkUnique:    true,
			wantErrMsgSubs: []string{"重複"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.policy.ValidateForLegalActions(tc.legalActions, tc.checkUnique)
			if len(tc.wantErrMsgSubs) == 0 {
				if err != nil {
					t.Errorf("予期せぬエラー: %v", err)
				}
				return
			}

			if err == nil {
				t.Fatal("エラーを期待したが、nilが返された")
			}
			errMsg := err.Error()
			for _, sub := range tc.wantErrMsgSubs {
				if !strings.Contains(errMsg, sub) {
					t.Errorf("errMsg = %s, sub = %s", errMsg, sub)
				}
			}
		})
	}
}

func TestMaxSelectFunc(t *testing.T) {
	rng := randx.NewPCG()

	t.Run("正常_最大値が1つ", func(t *testing.T) {
		policy := game.Policy[string]{"戦う": 0.7, "逃げる": 0.2, "防御": 0.1}
		// 最大値が1つの場合、常にその行動が選ばれる
		for range 100 {
			got, err := game.MaxSelectFunc(policy, "勇者", rng)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			if got != "戦う" {
				t.Fatalf("値の不一致: got = %s, want = 戦う", got)
			}
		}
	})

	t.Run("統計_最大値が複数", func(t *testing.T) {
		policy := game.Policy[string]{"戦う": 0.4, "逃げる": 0.4, "防御": 0.2}
		n := 10000
		got := make([]string, n)
		for i := range n {
			v, err := game.MaxSelectFunc(policy, "勇者", rng)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			got[i] = v
		}

		counts := slicesx.Counts(got)
		if counts["防御"] != 0 {
			t.Errorf("最大値ではない行動が選ばれた: 防御 = %d回", counts["防御"])
		}

		// 同率最大の2つは、ほぼ均等に選ばれるはず
		const eps = 0.03
		for _, a := range []string{"戦う", "逃げる"} {
			ratio := float64(counts[a]) / float64(n)
			if math.Abs(ratio-0.5) > eps {
				t.Errorf("%s の選択比率の不一致: got = %.3f, want = 0.5(±%.3f)", a, ratio, eps)
			}
		}
	})

	t.Run("異常_空のpolicy", func(t *testing.T) {
		policy := game.Policy[string]{}
		_, err := game.MaxSelectFunc(policy, "勇者", rng)
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
		if !strings.Contains(err.Error(), "policyが空") {
			t.Errorf("エラーメッセージが不十分: %s", err.Error())
		}
	})
}

func TestWeightedRandomSelectFunc(t *testing.T) {
	rng := randx.NewPCG()

	t.Run("統計_重みに比例して選択", func(t *testing.T) {
		policy := game.Policy[string]{"グー": 0.6, "パー": 0.3, "チョキ": 0.1}
		n := 10000
		got := make([]string, n)
		for i := range n {
			v, err := game.WeightedRandomSelectFunc(policy, "プレイヤー", rng)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			got[i] = v
		}

		counts := slicesx.Counts(got)
		const eps = 0.03
		for a, want := range map[string]float64{"グー": 0.6, "パー": 0.3, "チョキ": 0.1} {
			ratio := float64(counts[a]) / float64(n)
			if math.Abs(ratio-want) > eps {
				t.Errorf("%s の選択比率の不一致: got = %.3f, want = %.3f(±%.3f)", a, ratio, want, eps)
			}
		}
	})

	t.Run("異常_空のpolicy", func(t *testing.T) {
		policy := game.Policy[string]{}
		_, err := game.WeightedRandomSelectFunc(policy, "プレイヤー", rng)
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
	})
}
