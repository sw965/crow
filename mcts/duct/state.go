package duct

type StatePushFn[S any, A comparable] func(S, ...A) S