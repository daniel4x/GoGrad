package main

import (
	"fmt"

	"github.com/daniel4x/gograd/engine"
)

func main() {
	nn := engine.NewMLP(3, []int{4, 4, 1})
	fmt.Println(nn)
}
