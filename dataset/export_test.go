package dataset

import (
	"os"
	"strconv"
	"strings"
)

const (
	CacheResetTestInterval = 24
	CounterFileName        = ".test_counter.txt"
)

func GetAndIncrementCounter(fileName string) int {
	data, err := os.ReadFile(fileName)
	count := 0
	if err == nil {
		if c, err := strconv.Atoi(strings.TrimSpace(string(data))); err == nil {
			count = c
		}
	}
	count++
	if err := os.WriteFile(fileName, []byte(strconv.Itoa(count)), 0644); err != nil {
		panic(err)
	}
	return count
}

func ResetCounter(fileName string) {
	if err := os.WriteFile(fileName, []byte("0"), 0644); err != nil {
		panic(err)
	}
}
