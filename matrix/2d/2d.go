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
    if len(ss) == 0 {
        return Ss{}
    }
    m := len(ss)
    n := len(ss[0])
    rotated := make(Ss, m)
    for i := range rotated {
        rotated[i] = make(S, n)
    }
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            rotated[m-1-i][n-1-j] = ss[i][j]
        }
    }
    return rotated
}

func Rotate270[Ss ~[]S, S ~[]E, E any](ss Ss) Ss {
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
            rotated[n-1-j][i] = ss[i][j]
        }
    }
    return rotated
}

func RotateAll[Ss ~[]S, S ~[]E, E any](ss Ss) (Ss, Ss, Ss) {
    rotated90 := Rotate90(ss)
    rotated180 := Rotate180(ss)
    rotated270 := Rotate270(ss)
    return rotated90, rotated180, rotated270
}