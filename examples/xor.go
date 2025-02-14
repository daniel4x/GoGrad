package main

import (
	"fmt"
	"math"

	e "github.com/daniel4x/GoGrad/engine"
)

func createXORData() ([][]*e.Value, []float64) {
	x := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	y := []float64{-1, 1, 1, -1} // setting False as -1 and True as 1 just to get a cleaner outputs from the model

	return e.MakeValueMatrix(x), y
}

func printData(X [][]*e.Value, y []float64) {
	for i := 0; i < len(X); i++ {
		fmt.Printf("(%v, %v) -> %v\n", X[i][0].Data(), X[i][1].Data(), y[i])
	}
}

func main() {
	// Create XOR dataset
	X, y := createXORData()
	fmt.Println("XOR dataset:")
	printData(X, y)

	// Define a two-layer MLP with 2 input neurons, 2 hidden layers with 4 neurons each, and 1 output neuron
	nn := e.NewMLP(2, []int{4, 4, 1})
	fmt.Println("\nMulti-layer Perceptron Definition:\n", nn)

	// Train the model
	epochs := 2000
	alpha := 0.01

	for i := 0; i < epochs; i++ {
		y_model := make([]*e.Value, len(X))

		// Forward pass
		// Feed in each data point
		for j := 0; j < len(X); j++ {
			y_model[j] = nn.Call(X[j])
		}

		// Compute the loss
		loss := y_model[0].Sub(y[0]).Pow(2)
		for j := 1; j < len(y_model); j++ {
			loss = loss.Add(y_model[j].Sub(y[j]).Pow(2))
		}

		// Backward pass
		// zero the gradients to avoid accumulation between epochs
		params := nn.Parameters()
		for j := 0; j < len(params); j++ {
			params[j].ZeroGrad()
		}

		loss.Backward() // backward

		// Update the parameters
		for j := 0; j < len(params); j++ {
			params[j].SetData(params[j].Data() - alpha*params[j].Grad())
		}

		if (i+1)%100 == 0 {
			// Print the loss every 100 epochs
			fmt.Println("epoch", i, "loss", loss.Data())
		}
	}

	// Test the model
	predictions := make([]float64, len(X))
	for i := 0; i < len(X); i++ {
		predictions[i] = nn.Call(X[i]).Data()
	}

	fmt.Println("\nTesting the model:")
	for i := 0; i < len(X); i++ {
		fmt.Printf("(%v, %v) -> Actual: %v Prediction %v\n", X[i][0].Data(), X[i][1].Data(), y[i], predictions[i])
	}

	// Raise error if the difference between the actual and predicted values is greater than 0.1
	for i := 0; i < len(X); i++ {
		if math.Abs(y[i]-predictions[i]) > 0.1 {
			panic(fmt.Sprintf("\nTest failed: (%v, %v) -> Actual: %v Prediction %v\n", X[i][0].Data(), X[i][1].Data(), y[i], predictions[i]))
		}
	}
}
