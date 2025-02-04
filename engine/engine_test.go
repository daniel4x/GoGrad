package engine

import (
	"math"
	"testing"
)

const (
	TOL = 1e-6
	H   = 0.00000001
)

func eq1(a float64, b float64, c float64) (*Value, *Value, *Value, *Value, *Value, *Value) {
	_a := Value{data: a, label: "a"}
	_b := Value{data: b, label: "b"}
	_c := Value{data: c, label: "c"}
	_d := _a.Mul(&_b) // d = a * b
	_d.label = "d"
	_e := _d.Add(&_c) // e = d + c = a * b + c
	_e.label = "e"

	_L := _e.Tanh() // L = tanh(e) = tanh(d + c) = tanh(a * b + c)
	_L.label = "L"

	return &_a, &_b, &_c, _d, _e, _L
}

func eq2(a float64, b float64, c float64) (*Value, *Value, *Value, *Value, *Value, *Value) {
	_a := Value{data: a, label: "a"}
	_b := Value{data: b, label: "b"}
	_c := Value{data: c, label: "c"}
	_d := _a.Div(&_b) // d = a / b
	_d.label = "d"
	_e := _d.Sub(&_c) // e = d - c = a / b - c
	_e.label = "e"

	_L := _e.Tanh() // L = tanh(e) = tanh(d - c) = tanh((a / b) - c)
	_L.label = "L"

	return &_a, &_b, &_c, _d, _e, _L
}

func TestValueMulAdd(t *testing.T) {
	a, b, c, d, e, L := eq1(2.0, -4.0, 3.0)

	L.Backward()

	if L.grad != 1 {
		t.Errorf("L.grad = %f; want 1", L.grad)
	}

	eGradEstimation := (1 - math.Tanh(e.data)*math.Tanh(e.data)) * L.grad

	// dL/de = dL/dL * dL/de = 1 * dL/de = dL/de
	if math.Abs(e.grad-eGradEstimation) > TOL {
		t.Errorf("e.grad = %f; want %f", e.grad, eGradEstimation)
	}

	// dL/dd = dL/de * de/dd
	if math.Abs(d.grad-e.grad) > TOL {
		t.Errorf("d.grad = %f; want %f", d.grad, e.grad)
	}

	// dl/de * de/dc = 1 * de/dc = de/dc (de/dc = 1 because the derivative of c with respect to e is 1)
	if math.Abs(c.grad-e.grad) > TOL {
		t.Errorf("c.grad = %f; want %f", c.grad, e.grad)
	}

	// dl/da = dl/dd * dd/da = dl/dd * b
	if math.Abs(a.grad-d.grad*b.data) > TOL {
		t.Errorf("a.grad = %f; want %f", a.grad, d.grad*b.data)
	}

	// dl/db = dl/dd * dd/db = dl/dd * a
	if math.Abs(b.grad-d.grad*a.data) > TOL {
		t.Errorf("b.grad = %f; want %f", b.grad, d.grad*a.data)
	}
}

func TestValueMulAdd2(t *testing.T) {
	a, b, c, d, e, L := eq1(3.0, 4.0, -5.0)

	L.Backward()

	if L.grad != 1 {
		t.Errorf("L.grad = %f; want 1", L.grad)
	}

	eGradEstimation := (1 - math.Tanh(e.data)*math.Tanh(e.data)) * L.grad

	// dl/dl * dl/de = 1 * dl/de = dl/de
	if math.Abs(e.grad-eGradEstimation) > TOL {
		t.Errorf("e.grad = %f; want %f", e.grad, eGradEstimation)
	}

	// dl/de * de/dd = 1 * de/dd = de/dd (de/dd = 1 because the derivative of d with respect to e is 1)
	if math.Abs(d.grad-e.grad) > TOL {
		t.Errorf("d.grad = %f; want %f", d.grad, e.grad)
	}

	// dl/de * de/dc = 1 * de/dc = de/dc (de/dc = 1 because the derivative of c with respect to e is 1)
	if math.Abs(c.grad-e.grad) > TOL {
		t.Errorf("c.grad = %f; want %f", c.grad, e.grad)
	}

	// dl/da = dl/dd * dd/da = dl/dd * b
	if math.Abs(a.grad-d.grad*b.data) > TOL {
		t.Errorf("a.grad = %f; want %f", a.grad, d.grad*b.data)
	}

	// dl/db = dl/dd * dd/db = dl/dd * a
	if math.Abs(b.grad-d.grad*a.data) > TOL {
		t.Errorf("b.grad = %f; want %f", b.grad, d.grad*a.data)
	}
}

func TestValueDivSub(t *testing.T) {
	a, b, c, d, e, L := eq2(2.0, -4.0, 3.0)

	L.Backward()

	if L.grad != 1 {
		t.Errorf("L.grad = %f; want 1", L.grad)
	}

	eGradEstimation := (1 - math.Tanh(e.data)*math.Tanh(e.data)) * L.grad

	// dl/dl * dl/de = 1 * dl/de = dl/de
	if math.Abs(e.grad-eGradEstimation) > TOL {
		t.Errorf("e.grad = %f; want %f", e.grad, eGradEstimation)
	}

	// dl/de * de/dd = 1 * de/dd = de/dd (de/dd = 1 because the derivative of d with respect to e is 1)
	if math.Abs(d.grad-e.grad) > TOL {
		t.Errorf("d.grad = %f; want %f", d.grad, e.grad)
	}

	// dl/de * de/dc = -1 * de/dc = de/dc (de/dc = 1 because the derivative of c with respect to e is 1)
	if math.Abs(c.grad-(-1*e.grad)) > TOL {
		t.Errorf("c.grad = %f; want %f", c.grad, e.grad)
	}

	// dl/da = dl/dd * dd/da = dl/dd * b
	if math.Abs(a.grad-d.grad/b.data) > TOL {
		t.Errorf("a.grad = %f; want %f", a.grad, d.grad/b.data)
	}

	// dL/db = dL/dd * dd/db = dL/dd * -a * b^-2
	bGradEstimation := d.grad * -a.data * math.Pow(b.data, -2)
	if math.Abs(b.grad-bGradEstimation) > TOL {
		t.Errorf("b.grad = %f; want %f", b.grad, d.grad*a.data)
	}
}

func TestValueAddMulNumericalValidation(t *testing.T) {
	a, b, c, _, _, L := eq1(-2.0, 1.0, 5.0)

	L.Backward()
	L_orig := L.data
	aGrad := a.grad
	bGrad := b.grad
	cGrad := c.grad

	_, _, _, _, _, L = eq1(-2.0+H, 1.0, 5.0)
	aGradEstimation := (L.data - L_orig) / H // (L(a+h, b, c) - L(a, b, c)) / h

	_, _, _, _, _, L = eq1(-2.0, 1.0+H, 5.0)
	bGradEstimation := (L.data - L_orig) / H // (L(a, b+h, c) - L(a, b, c)) / h

	_, _, _, _, _, L = eq1(-2.0, 1.0, 5.0+H)
	cGradEstimation := (L.data - L_orig) / H // (L(a, b, c+h) - L(a, b, c)) / h

	if math.Abs(aGrad-aGradEstimation) > TOL {
		t.Errorf("a.grad = %f; want %f; aGrad-aGradEstimation=%f", aGrad, aGradEstimation, aGrad-aGradEstimation)
	}

	if math.Abs(bGrad-bGradEstimation) > TOL {
		t.Errorf("b.grad = %f; want %f; bGrad-bGradEstimation=%f", bGrad, bGradEstimation, bGrad-bGradEstimation)
	}

	if math.Abs(cGrad-cGradEstimation) > TOL {
		t.Errorf("c.grad = %f; want %f; cGrad-cGradEstimation=%f", cGrad, cGradEstimation, cGrad-cGradEstimation)
	}
}

func TestValueDivSubNumericalValidation(t *testing.T) {
	a, b, c, _, _, L := eq2(2.0, -4.0, 3.0)

	L.Backward()
	L_orig := L.data
	aGrad := a.grad
	bGrad := b.grad
	cGrad := c.grad

	_, _, _, _, _, L = eq2(2.0+H, -4.0, 3.0)
	aGradEstimation := (L.data - L_orig) / H // (L(a+h, b, c) - L(a, b, c)) / h

	_, _, _, _, _, L = eq2(2.0, -4.0+H, 3.0)
	bGradEstimation := (L.data - L_orig) / H // (L(a, b+h, c) - L(a, b, c)) / h

	_, _, _, _, _, L = eq2(2.0, -4.0, 3.0+H)
	cGradEstimation := (L.data - L_orig) / H // (L(a, b, c+h) - L(a, b, c)) / h

	if math.Abs(aGrad-aGradEstimation) > TOL {
		t.Errorf("a.grad = %f; want %f; aGrad-aGradEstimation=%f", aGrad, aGradEstimation, aGrad-aGradEstimation)
	}

	if math.Abs(bGrad-bGradEstimation) > TOL {
		t.Errorf("b.grad = %f; want %f; bGrad-bGradEstimation=%f", bGrad, bGradEstimation, bGrad-bGradEstimation)
	}

	if math.Abs(cGrad-cGradEstimation) > TOL {
		t.Errorf("c.grad = %f; want %f; cGrad-cGradEstimation=%f", cGrad, cGradEstimation, cGrad-cGradEstimation)
	}
}
