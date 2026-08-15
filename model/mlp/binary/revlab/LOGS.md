# 実験ログ

共通条件は20 epoch、train 54,000、validation 6,000、test 10,000、batch 1024、BEP margin 0.5、noise 0.5、可逆内部noise 0、state/output 1024、可逆2段です。表のテスト値は、検証精度が最高だったepochの値です。

## MNIST: バイアス無し

| mode | learnrev | seed | best validation | epoch | test |
|---|---:|---:|---:|---:|---:|
| dense | — | 1 | 0.8897 | 20 | 0.9042 |
| dense | — | 2 | 0.8963 | 18 | 0.8985 |
| dense | — | 3 | 0.8887 | 18 | 0.9018 |
| project | — | 1 | 0.8800 | 12 | 0.8906 |
| reversible | false | 1 | 0.9053 | 19 | 0.9147 |
| reversible | false | 2 | 0.9122 | 20 | 0.9161 |
| reversible | false | 3 | 0.9085 | 17 | 0.9197 |
| reversible | true | 1 | 0.9108 | 15 | 0.9201 |
| reversible | true | 2 | 0.9148 | 14 | 0.9162 |
| reversible | true | 3 | 0.9130 | 14 | 0.9196 |

## MNIST: 学習可逆 + 整数バイアス

`bias=true`、`biaschoice=0.1`。

| seed | best validation | epoch | test |
|---:|---:|---:|---:|
| 1 | 0.9160 | 19 | 0.9246 |
| 2 | 0.9160 | 19 | 0.9209 |
| 3 | 0.9118 | 20 | 0.9209 |

## Fashion-MNIST: seed 1

| mode | learnrev | bias | best validation | epoch | test |
|---|---:|---:|---:|---:|---:|
| dense | — | false | 0.8322 | 16 | 0.8225 |
| reversible | false | false | 0.8298 | 20 | 0.8207 |
| reversible | true | true (`biaschoice=0.1`) | 0.8350 | 19 | 0.8201 |

## 再現コマンド例

```powershell
go run ./model/mlp/binary/revlab --mode dense --dataset mnist --epochs 20 --bias=false --biaschoice 0 --seed 1 --threads 4
go run ./model/mlp/binary/revlab --mode reversible --blocks 2 --learnrev=false --dataset mnist --epochs 20 --bias=false --biaschoice 0 --seed 1 --threads 4
go run ./model/mlp/binary/revlab --mode reversible --blocks 2 --learnrev=true --dataset mnist --epochs 20 --bias=false --biaschoice 0 --seed 1 --threads 4
go run ./model/mlp/binary/revlab --mode reversible --blocks 2 --learnrev=true --dataset mnist --epochs 20 --bias=true --biaschoice 0.1 --seed 1 --threads 4
```
