package ql

func UpdateQ(q, nextMaxQ, reward, lr, gamma float64) float64 {
	qRatio := 1.0 - lr
	newQ := (reward + gamma * nextMaxQ)
	return (qRatio * q) + (lr * newQ)
}