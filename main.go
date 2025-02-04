package main

import (
	"fmt"
	"gograd/engine"
)

func main() {
	nn := engine.NewMLP(3, []int{4, 4, 1})
	fmt.Println(nn)
}
