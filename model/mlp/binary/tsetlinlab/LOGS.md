# 実験の生ログ

実行日: 2026-08-14。環境: Windows 11 / Go 1.26.5 / 12論理コアCPU(12ワーカー)。
実験時の実装はリポジトリ外の一時作業ディレクトリに置いた作業コピーで、
本フォルダの main.go と機能的に同一
(同一シードで BEST が小数4桁まで一致する事を §6 で確認済み)。

データ: ~/.crow_dataset の二値化済みgob(crow/dataset と同一。1bit/画素)。
表示時刻はローカル時刻。train/eval はエポックあたりの秒数。

## 1. サニティチェック(MNIST 1万サンプル、128節)

```text
$ go run . -dataset mnist -clauses 128 -epochs 3 -trainsub 10000
dataset=mnist train=10000 test=10000 clauses/class=128 T=512 s=2^3 weighted=true workers=12
[00:11:20] epoch 1: test acc 0.8504 (best 0.8504) train 0.2s eval 0.1s
[00:11:20] epoch 2: test acc 0.8775 (best 0.8775) train 0.2s eval 0.0s
[00:11:20] epoch 3: test acc 0.8909 (best 0.8909) train 0.2s eval 0.0s
BEST 0.8909
```

## 2. MNIST 本実験(512節/クラス、シード1)

```text
$ go run . -dataset mnist -clauses 512 -epochs 20
dataset=mnist train=60000 test=10000 clauses/class=512 T=2048 s=2^3 weighted=true workers=12
[00:11:37] epoch 1: test acc 0.9327 (best 0.9327) train 2.4s eval 0.2s
[00:11:39] epoch 2: test acc 0.9463 (best 0.9463) train 2.1s eval 0.1s
[00:11:41] epoch 3: test acc 0.9524 (best 0.9524) train 2.1s eval 0.1s
[00:11:43] epoch 4: test acc 0.9601 (best 0.9601) train 2.0s eval 0.1s
[00:11:45] epoch 5: test acc 0.9625 (best 0.9625) train 2.0s eval 0.1s
[00:11:48] epoch 6: test acc 0.9623 (best 0.9625) train 2.0s eval 0.1s
[00:11:50] epoch 7: test acc 0.9644 (best 0.9644) train 2.0s eval 0.1s
[00:11:52] epoch 8: test acc 0.9664 (best 0.9664) train 2.0s eval 0.1s
[00:11:54] epoch 9: test acc 0.9688 (best 0.9688) train 1.9s eval 0.1s
[00:11:56] epoch 10: test acc 0.9685 (best 0.9688) train 1.9s eval 0.1s
[00:11:58] epoch 11: test acc 0.9698 (best 0.9698) train 2.0s eval 0.1s
[00:12:00] epoch 12: test acc 0.9692 (best 0.9698) train 1.9s eval 0.1s
[00:12:02] epoch 13: test acc 0.9714 (best 0.9714) train 1.9s eval 0.1s
[00:12:04] epoch 14: test acc 0.9691 (best 0.9714) train 1.9s eval 0.1s
[00:12:06] epoch 15: test acc 0.9716 (best 0.9716) train 1.9s eval 0.1s
[00:12:08] epoch 16: test acc 0.9706 (best 0.9716) train 1.9s eval 0.1s
[00:12:10] epoch 17: test acc 0.9724 (best 0.9724) train 1.8s eval 0.1s
[00:12:12] epoch 18: test acc 0.9709 (best 0.9724) train 1.8s eval 0.1s
[00:12:14] epoch 19: test acc 0.9712 (best 0.9724) train 1.8s eval 0.1s
[00:12:16] epoch 20: test acc 0.9733 (best 0.9733) train 1.9s eval 0.2s
BEST 0.9733
```

## 3. Fashion-MNIST 本実験(2048節/クラス、シード1)

```text
$ go run . -dataset fashion -clauses 2048 -slog2 4 -epochs 30
dataset=fashion train=60000 test=10000 clauses/class=2048 T=8192 s=2^4 weighted=true workers=12
[00:12:37] epoch 1: test acc 0.7966 (best 0.7966) train 9.1s eval 0.5s
[00:12:45] epoch 2: test acc 0.8247 (best 0.8247) train 6.9s eval 0.5s
[00:12:52] epoch 3: test acc 0.8368 (best 0.8368) train 6.6s eval 0.4s
[00:12:59] epoch 4: test acc 0.8410 (best 0.8410) train 6.3s eval 0.5s
[00:13:05] epoch 5: test acc 0.8477 (best 0.8477) train 6.3s eval 0.5s
[00:13:12] epoch 6: test acc 0.8514 (best 0.8514) train 6.1s eval 0.4s
[00:13:18] epoch 7: test acc 0.8554 (best 0.8554) train 5.6s eval 0.5s
[00:13:25] epoch 8: test acc 0.8572 (best 0.8572) train 6.0s eval 0.7s
[00:13:33] epoch 9: test acc 0.8637 (best 0.8637) train 7.2s eval 0.7s
[00:13:40] epoch 10: test acc 0.8620 (best 0.8637) train 6.6s eval 0.5s
[00:13:46] epoch 11: test acc 0.8627 (best 0.8637) train 5.9s eval 0.5s
[00:13:53] epoch 12: test acc 0.8651 (best 0.8651) train 6.2s eval 0.8s
[00:13:59] epoch 13: test acc 0.8654 (best 0.8654) train 5.6s eval 0.5s
[00:14:05] epoch 14: test acc 0.8685 (best 0.8685) train 5.6s eval 0.4s
[00:14:11] epoch 15: test acc 0.8673 (best 0.8685) train 5.3s eval 0.5s
[00:14:17] epoch 16: test acc 0.8680 (best 0.8685) train 5.6s eval 0.5s
[00:14:23] epoch 17: test acc 0.8687 (best 0.8687) train 5.6s eval 0.5s
[00:14:29] epoch 18: test acc 0.8692 (best 0.8692) train 5.3s eval 0.4s
[00:14:35] epoch 19: test acc 0.8675 (best 0.8692) train 5.2s eval 0.4s
[00:14:40] epoch 20: test acc 0.8733 (best 0.8733) train 5.2s eval 0.5s
[00:14:46] epoch 21: test acc 0.8700 (best 0.8733) train 5.6s eval 0.4s
[00:14:52] epoch 22: test acc 0.8697 (best 0.8733) train 5.7s eval 0.6s
[00:14:58] epoch 23: test acc 0.8681 (best 0.8733) train 4.7s eval 0.5s
[00:15:03] epoch 24: test acc 0.8708 (best 0.8733) train 4.6s eval 0.5s
[00:15:08] epoch 25: test acc 0.8697 (best 0.8733) train 4.6s eval 0.5s
[00:15:13] epoch 26: test acc 0.8735 (best 0.8735) train 4.7s eval 0.5s
[00:15:20] epoch 27: test acc 0.8742 (best 0.8742) train 6.2s eval 0.5s
[00:15:26] epoch 28: test acc 0.8702 (best 0.8742) train 5.6s eval 0.5s
[00:15:31] epoch 29: test acc 0.8724 (best 0.8742) train 5.3s eval 0.4s
[00:15:36] epoch 30: test acc 0.8730 (best 0.8742) train 4.5s eval 0.5s
BEST 0.8742
```

## 4. Fashion-MNIST 大規模構成(4096節/クラス、シード1)

```text
$ go run . -dataset fashion -clauses 4096 -slog2 4 -epochs 40
dataset=fashion train=60000 test=10000 clauses/class=4096 T=16384 s=2^4 weighted=true workers=12
[00:16:04] epoch 1: test acc 0.7931 (best 0.7931) train 18.3s eval 2.5s
[00:16:21] epoch 2: test acc 0.8263 (best 0.8263) train 14.9s eval 2.3s
[00:16:37] epoch 3: test acc 0.8391 (best 0.8391) train 13.5s eval 2.3s
[00:16:52] epoch 4: test acc 0.8453 (best 0.8453) train 12.2s eval 2.4s
[00:17:07] epoch 5: test acc 0.8515 (best 0.8515) train 13.4s eval 2.6s
[00:17:22] epoch 6: test acc 0.8553 (best 0.8553) train 12.5s eval 2.3s
[00:17:36] epoch 7: test acc 0.8578 (best 0.8578) train 11.3s eval 2.4s
[00:17:50] epoch 8: test acc 0.8583 (best 0.8583) train 12.2s eval 2.3s
[00:18:04] epoch 9: test acc 0.8637 (best 0.8637) train 11.0s eval 2.3s
[00:18:18] epoch 10: test acc 0.8635 (best 0.8637) train 11.8s eval 2.4s
[00:18:32] epoch 11: test acc 0.8636 (best 0.8637) train 11.2s eval 2.3s
[00:18:45] epoch 12: test acc 0.8661 (best 0.8661) train 11.5s eval 2.3s
[00:18:59] epoch 13: test acc 0.8700 (best 0.8700) train 10.8s eval 2.4s
[00:19:13] epoch 14: test acc 0.8685 (best 0.8700) train 12.1s eval 2.5s
[00:19:26] epoch 15: test acc 0.8684 (best 0.8700) train 10.8s eval 2.6s
[00:19:40] epoch 16: test acc 0.8685 (best 0.8700) train 11.0s eval 2.5s
[00:19:53] epoch 17: test acc 0.8715 (best 0.8715) train 11.0s eval 2.4s
[00:20:07] epoch 18: test acc 0.8719 (best 0.8719) train 11.4s eval 2.6s
[00:20:21] epoch 19: test acc 0.8715 (best 0.8719) train 11.1s eval 2.3s
[00:20:34] epoch 20: test acc 0.8711 (best 0.8719) train 10.7s eval 2.6s
[00:20:47] epoch 21: test acc 0.8724 (best 0.8724) train 10.5s eval 2.8s
[00:21:01] epoch 22: test acc 0.8740 (best 0.8740) train 10.8s eval 2.3s
[00:21:13] epoch 23: test acc 0.8722 (best 0.8740) train 10.1s eval 2.4s
[00:21:27] epoch 24: test acc 0.8744 (best 0.8744) train 11.2s eval 2.4s
[00:21:40] epoch 25: test acc 0.8732 (best 0.8744) train 10.9s eval 2.3s
[00:21:53] epoch 26: test acc 0.8740 (best 0.8744) train 10.5s eval 2.5s
[00:22:06] epoch 27: test acc 0.8734 (best 0.8744) train 10.5s eval 2.3s
[00:22:19] epoch 28: test acc 0.8716 (best 0.8744) train 10.5s eval 2.4s
[00:22:32] epoch 29: test acc 0.8727 (best 0.8744) train 10.5s eval 2.9s
[00:22:47] epoch 30: test acc 0.8728 (best 0.8744) train 11.9s eval 2.8s
[00:22:59] epoch 31: test acc 0.8742 (best 0.8744) train 9.9s eval 2.4s
[00:23:12] epoch 32: test acc 0.8761 (best 0.8761) train 10.5s eval 2.3s
[00:23:24] epoch 33: test acc 0.8740 (best 0.8761) train 10.0s eval 2.3s
[00:23:37] epoch 34: test acc 0.8757 (best 0.8761) train 10.3s eval 2.3s
[00:23:49] epoch 35: test acc 0.8759 (best 0.8761) train 10.4s eval 2.4s
[00:24:02] epoch 36: test acc 0.8750 (best 0.8761) train 9.8s eval 2.4s
[00:24:14] epoch 37: test acc 0.8732 (best 0.8761) train 10.0s eval 2.5s
[00:24:27] epoch 38: test acc 0.8777 (best 0.8777) train 10.6s eval 2.3s
[00:24:40] epoch 39: test acc 0.8744 (best 0.8777) train 10.0s eval 2.6s
[00:24:51] epoch 40: test acc 0.8778 (best 0.8778) train 9.4s eval 2.4s
BEST 0.8778
```

## 5. シード追試(シード2・3。エポック毎のログは `tail -1` で最終行のみ記録)

```text
$ for seed in 2 3; do go run . -dataset mnist -clauses 512 -epochs 20 -seed $seed | tail -1; done
$ for seed in 2 3; do go run . -dataset fashion -clauses 2048 -slog2 4 -epochs 30 -seed $seed | tail -1; done
=== mnist seed 2 ===
BEST 0.9728
=== mnist seed 3 ===
BEST 0.9710
=== fashion seed 2 ===
BEST 0.8734
=== fashion seed 3 ===
BEST 0.8756
```

集計(シード1/2/3):

- MNIST 512節/20ep: 0.9733 / 0.9728 / 0.9710 → 平均 0.9724、範囲 0.0023
- Fashion 2048節/30ep: 0.8742 / 0.8734 / 0.8756 → 平均 0.8744、範囲 0.0022

## 6. 本フォルダ版の再現確認

一時作業ディレクトリの作業コピーから本フォルダへ移植(デッドコード削除・ループ表記の
現代化のみ、乱数消費順序は不変)した後、シード1のMNIST実験を再実行し、
BEST が小数4桁まで一致する事を確認した。

```text
$ go run . -dataset mnist -clauses 512 -epochs 20
[00:43:24] epoch 19: test acc 0.9712 (best 0.9724) train 1.8s eval 0.1s
[00:43:26] epoch 20: test acc 0.9733 (best 0.9733) train 1.8s eval 0.1s
BEST 0.9733
```

## 参考: 比較基準(親フォルダの ../REPORT.md = BEP検証レポートより)

- crow BEP実装: MNIST 92.13±0.48、Fashion 83.78±0.03(3シード、50エポック)
- BEP論文忠実版: MNIST 87.99±1.32、Fashion 84.61±0.13
- BEP論文報告値: Fashion 85〜87%
