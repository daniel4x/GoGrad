package engine

/**
 * The Value struct represents a node in a computation graph for automatic differentiation.
 * It follows the implementation of Andrej Karpathy in Python, which can be found at:
 * https://github.com/karpathy/micrograd
 **/

import (
	"fmt"
	"math"
)

type Value struct {
	data     float64
	children []*Value
	op       string
	grad     float64
	label    string
	backward func()
}

func (v *Value) Add(other interface{}) *Value {
	otherValue := validateValue(other)
	ans := Value{data: v.data + otherValue.data, children: []*Value{v, otherValue}, op: "+", grad: 0}
	ans.backward = func() {
		v.grad += ans.grad
		otherValue.grad += ans.grad
	}

	return &ans
}

func (v *Value) Mul(other interface{}) *Value {
	otherValue := validateValue(other)
	ans := Value{data: v.data * otherValue.data, children: []*Value{v, otherValue}, op: "*", grad: 0}
	ans.backward = func() {
		v.grad += otherValue.data * ans.grad
		otherValue.grad += v.data * ans.grad
	}

	return &ans
}

func (v *Value) Pow(power float64) *Value {
	ans := Value{data: math.Pow(v.data, power), children: []*Value{v}, op: fmt.Sprintf("**%f", power), grad: 0}
	ans.backward = func() {
		v.grad += power * math.Pow(v.data, power-1) * ans.grad
	}

	return &ans
}

func (v *Value) Exp() *Value {
	ans := Value{data: math.Exp(v.data), children: []*Value{v}, op: "exp", grad: 0}
	ans.backward = func() {
		v.grad += math.Exp(v.data) * ans.grad
	}

	return &ans
}

func (v *Value) Tanh() *Value {
	exp2 := math.Exp(2 * v.data)
	tanh := (exp2 - 1) / (exp2 + 1)
	ans := Value{data: tanh, children: []*Value{v}, op: "tanh", grad: 0}
	ans.backward = func() {
		v.grad += (1 - tanh*tanh) * ans.grad
	}

	return &ans
}

func (v *Value) Neg() *Value {
	return v.Mul(-1.0)
}

func (v *Value) Sub(other interface{}) *Value {
	otherValue := validateValue(other)
	return v.Add(otherValue.Neg())
}

func (v *Value) Div(other interface{}) *Value {
	otherValue := validateValue(other)
	return v.Mul(otherValue.Pow(-1))
}

func (v *Value) Backward() {
	// topological order all of the children in the graph
	topo := []*Value{}
	visited := map[*Value]bool{}
	var dfs func(*Value)
	dfs = func(v *Value) {
		if !visited[v] {
			visited[v] = true
			for _, child := range v.children {
				dfs(child)
			}
			topo = append(topo, v)
		}
	}
	dfs(v)

	v.grad = 1
	// go back through the graph in reverse order
	for i := len(topo) - 1; i >= 0; i-- {
		if topo[i].backward != nil {
			topo[i].backward()
		}
	}
}

func (v Value) String() string {
	return fmt.Sprintf("Value(label=%s, data=%f, children=(%v), op=%s, grad=%f)", v.label, v.data, v.children, v.op, v.grad)
}

func validateValue(candidate interface{}) *Value {
	switch other := candidate.(type) {
	case *Value:
		return other
	case float64:
		return &Value{data: other, grad: 0}
	default:
		panic("Invalid input type! expected *Value or float64")
	}
}

// Helper functions to create Value slices and matrices (Tensor like objects)
func makeValues(data []float64) []*Value {
	/**
	 * Create a slice of Value pointers from a slice of float64.
	 **/
	ans := make([]*Value, len(data))
	for i := 0; i < len(data); i++ {
		ans[i] = &Value{data: data[i]}
	}
	return ans
}

func makeValueMatrix(data [][]float64) [][]*Value {
	/**
	 * Create a matrix of Value pointers from a matrix of float64.
	 **/
	ans := make([][]*Value, len(data))
	for i := 0; i < len(data); i++ {
		ans[i] = makeValues(data[i])
	}
	return ans
}
