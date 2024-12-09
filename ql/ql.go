package ql

func UpdateQ(q, nextMaxQ, reward, lr, discountRate float64) float64 {
	qRatio := 1.0 - lr
	newQ := (reward + discountRate * nextMaxQ)
	return (qRatio * q) + (lr * newQ)
}