package tensor

type D4 []D3

func (d4 D4) AddScalar(s float64) {
	for i := range d4 {
		d4[i].AddScalar(s)
	}
}

func (d4 D4) Add(other D4) error {
	for i := range d4 {
		err := d4[i].Add(other[i])
		if err != nil {
			return err
		}
	}
	return nil
}

func (d4 D4) SubScalar(s float64) {
	for i := range d4 {
		d4[i].SubScalar(s)
	}
}

func (d4 D4) Sub(other D4) error {
	for i := range d4 {
		err := d4[i].Sub(other[i])
		if err != nil {
			return err
		}
	}
	return nil
}

func (d4 D4) MulScalar(s float64) {
	for i := range d4 {
		d4[i].MulScalar(s)
	}
}

func (d4 D4) Mul(other D4) error {
	for i := range d4 {
		err := d4[i].Mul(other[i])
		if err != nil {
			return err
		}
	}
	return nil
}

func (d4 D4) DivScalar(s float64) {
	for i := range d4 {
		d4[i].DivScalar(s)
	}
}

func (d4 D4) Div(other D4) error {
	for i := range d4 {
		err := d4[i].Div(other[i])
		if err != nil {
			return err
		}
	}
	return nil
}