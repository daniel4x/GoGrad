package engine

import (
	"math"
	"testing"
)

func TestMLPSimpleScenario(t *testing.T) {
	x := makeValueMatrix(
		[][]float64{
			{2.0, 3.0, -1.0},
			{3.0, -1.0, 0.5},
			{0.5, 1.0, 1.0},
			{1.0, 1.0, -1.0},
		},
	)

	y := []float64{1, -1, -1, 1}

	nn := NewMLP(3, []int{4, 4, 1})

	epochs := 100
	alpha := 0.1

	for i := 0; i < epochs; i++ {
		y_model := make([]*Value, len(x))
		for j := 0; j < len(x); j++ {
			y_model[j] = nn.Call(x[j]) // forward
		}

		loss := y_model[0].Sub(y[0]).Pow(2)
		for j := 1; j < len(y_model); j++ {
			loss = loss.Add(y_model[j].Sub(y[j]).Pow(2))
		}

		// zero the gradients to avoid accumulation between epochs
		params := nn.Parameters()
		for j := 0; j < len(params); j++ {
			params[j].grad = 0
		}

		loss.Backward() // backward

		t.Log("epoch", i, "loss", loss.data)

		// update the parameters
		for j := 0; j < len(params); j++ {
			params[j].data -= alpha * params[j].grad
		}
	}

	t.Log("\n\n")
	t.Log("==== Test ====")
	// We just validate that the network fits the data
	for i := 0; i < len(x); i++ {
		y_model := nn.Call(x[i])
		t.Log("y", y[i], "y_model", y_model.data)
		if math.Abs(y_model.data-y[i]) > 0.1 {
			t.Error("Prediction error, y", y[i], "y_model", y_model.data, "Diff", math.Abs(y_model.data-y[i]))
		}
	}
}
