package engine

import (
	"fmt"
	"math/rand"
)

/**
 * The structs and functions in this file are used to create a simple feedforward neural network (MLP).
 * It follows the implementation of Andrej Karpathy in Python, which can be found at:
 * https://github.com/karpathy/micrograd
 **/

type Neuron struct {
	weights []*Value
	bias    *Value
}

func (n *Neuron) Call(x []*Value) *Value {
	ans := x[0].Mul(n.weights[0])
	i := 1
	for i < len(x) {
		ans = ans.Add(x[i].Mul(n.weights[i]))
		i++
	}
	ans = ans.Add(n.bias)
	return ans.Tanh()
}

func (n *Neuron) Parameters() []*Value {
	ans := make([]*Value, len(n.weights)+1)
	copy(ans, n.weights)
	ans[len(n.weights)] = n.bias
	return ans
}

func NewNeuron(in int) *Neuron {
	weights := make([]*Value, in)
	for i := 0; i < in; i++ {
		weights[i] = &Value{data: rand.Float64()*2 - 1}
	}

	bias := &Value{data: rand.Float64()*2 - 1}
	return &Neuron{weights: weights, bias: bias}
}

func (n *Neuron) String() string {
	ans := "Neuron{"
	ans += fmt.Sprintf("Weights=%v, ", len(n.weights))
	return ans + "Bias}"
}

type Layer struct {
	neurons []*Neuron
}

func (l *Layer) Call(x []*Value) []*Value {
	ans := make([]*Value, len(l.neurons))
	for i := 0; i < len(l.neurons); i++ {
		ans[i] = l.neurons[i].Call(x)
	}
	return ans
}

func (l *Layer) Parameters() []*Value {
	ans := make([]*Value, 0)
	for i := 0; i < len(l.neurons); i++ {
		ans = append(ans, l.neurons[i].Parameters()...)
	}
	return ans
}

func NewLayer(in, out int) *Layer {
	neurons := make([]*Neuron, out)
	for i := 0; i < out; i++ {
		neurons[i] = NewNeuron(in)
	}

	return &Layer{neurons: neurons}
}

func (l *Layer) String() string {
	ans := "Layer{\n"
	for i := 0; i < len(l.neurons); i++ {
		ans += "\t\t" + l.neurons[i].String() + ",\n"
	}
	ans += "\t}"
	return ans
}

type MLP struct {
	layers []*Layer
}

func (m *MLP) Call(x []*Value) *Value {
	ans := x
	for i := 0; i < len(m.layers); i++ {
		ans = m.layers[i].Call(ans)
	}

	if len(ans) != 1 {
		panic("MLP.Call: len(ans) != 1")
	}

	return ans[0]
}

func (m *MLP) Parameters() []*Value {
	ans := make([]*Value, 0)
	for i := 0; i < len(m.layers); i++ {
		ans = append(ans, m.layers[i].Parameters()...)
	}
	return ans
}

func NewMLP(in int, outs []int) *MLP {
	layers := make([]*Layer, len(outs))
	sizes := make([]int, len(outs)+1)
	sizes[0] = in
	copy(sizes[1:], outs)
	for i := 0; i < len(layers); i++ {
		layers[i] = NewLayer(sizes[i], sizes[i+1])
	}

	return &MLP{layers: layers}
}

func (m *MLP) String() string {
	ans := "MLP{\n"
	for i := 0; i < len(m.layers); i++ {
		ans += "\t" + m.layers[i].String() + ",\n"
	}
	ans += "}"
	return ans
}
