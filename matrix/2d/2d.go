package matrix2d

func Rotate90[Ss ~[]S, S ~[]E, E any](ss Ss) Ss {
    if len(ss) == 0 {
        return Ss{}
    }
    m := len(ss)
    n := len(ss[0])
    rotated := make(Ss, n)
    for i := range rotated {
        rotated[i] = make(S, m)
    }
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            rotated[j][m-1-i] = ss[i][j]
        }
    }
    return rotated
}

func Rotate180[Ss ~[]S, S ~[]E, E any](ss Ss) Ss {
    return Rotate90(Rotate90(ss))
}

func Rotate270[Ss ~[]S, S ~[]E, E any](ss Ss) Ss {
    return Rotate90(Rotate180(ss))
}