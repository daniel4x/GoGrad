# GoGrad - An Educational Project for Implementing Computational Graphs in Golang 

<p align="center">
  <img src="./artifacts/gograd.png" width="400" height="400">
</p>

## Introduction
This project is an educational project for implementing computational graphs in Golang. The project is inspired by the [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) by Andrej Karpathy. The project is intended to be a learning resource for understanding the basics of computational graphs and how they can be implemented in Golang. The project is not intended to be a full-fledged deep learning library.

## Why Another Deep Learning Library (and in Golang)!?
No particular reason. I just needed to learn some Golang for my work and I thought it would be a good idea to implement the backpropagation algorithm in Golang. I also always wanted to be hands-on with the implementation of backpropagation algorithm (rather than just using libraries like TensorFlow or PyTorch). So, I thought why not implement it in Golang?

## Running Tests
To run the tests, you can use the following command:
```bash
go test ./engine -v
```
*The `-v` is just for verbose output (if you want to see the loss values for each epoch).*