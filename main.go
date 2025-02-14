package main

import (
	"fmt"

	"github.com/daniel4x/GoGrad/engine"
)

func main() {
	nn := engine.NewMLP(3, []int{4, 4, 1})
	fmt.Println(nn)
}
