# Tensor Operations in C++: Implementation and Applications

## Introduction

Tensors have become a fundamental data structure in modern computing, particularly in areas such as scientific computing, machine learning, and deep learning. This document explores the implementation of tensor operations in C++, including representation strategies, mathematical operations, gradient computation, and backpropagation.

## What is a Tensor?

A tensor is a mathematical object that generalizes scalars, vectors, and matrices to higher dimensions. More precisely:

- A **0th-order tensor** is a scalar (a single value)
- A **1st-order tensor** is a vector (a one-dimensional array of values)
- A **2nd-order tensor** is a matrix (a two-dimensional array of values)
- A **3rd-order tensor** and beyond represent multi-dimensional arrays

In computational contexts, tensors are used to represent multi-dimensional data and the relationships between such data. They provide a unified mathematical framework for working with data of varying dimensions.

## Tensor Representation in C++

### N-Dimensional Arrays

In C++, tensors are typically represented as N-dimensional arrays. Unlike simple arrays, tensors require additional metadata to describe their structure:

1. **Shape**: A vector indicating the size along each dimension
2. **Data Buffer**: A contiguous memory buffer containing all elements
3. **Strides**: A vector describing how to navigate through the memory buffer

Here's our tensor class in C++:

```cpp
class Tensor {
private:
    std::vector<int> strides;
    std::vector<int> shape;

    // Back propagation
    bool isGrad;
    bool isLeaf = false;
    
    std::vector<Tensor*> previous;

    std::function<void()> gradFn;
    std::vector<float> data;
    std::vector<float> grad;

    std::string name;
    Tensor();
};
```

### Shape and Stride Concepts

#### Shape

The **shape** of a tensor defines its dimensions. For example:

- A tensor with shape `[3]` is a vector with 3 elements
- A tensor with shape `[2, 3]` is a 2×3 matrix
- A tensor with shape `[2, 3, 4]` is a 3-dimensional tensor with 2 planes, each having 3 rows and 4 columns

#### Stride

The **stride** of a tensor describes how many elements to skip in memory to move to the next element along a specific dimension. Strides are essential for efficiently navigating through the underlying memory buffer.

For a 3-dimensional tensor with shape `[2, 3, 4]` in row-major order (C/C++ standard):

- The stride for the last dimension would be `1` (move 1 element to get to the next column)
- The stride for the middle dimension would be `4` (move 4 elements to get to the next row)
- The stride for the first dimension would be `12` (move 12 elements to get to the next plane)

### Why Strides Are Useful

Strides provide several key benefits:

1. **Efficient Slicing**: Allow for creating views of tensor data without copying memory
2. **Different Memory Layouts**: Support both row-major (C-style) and column-major (Fortran-style) storage
3. **Broadcasting**: Enable operations between tensors of different shapes
4. **Transposition**: Permit matrix transposition without data copying by simply swapping strides

Our implementation of transposing a matrix by changing strides:

```cpp
Tensor* Tensor::transpose() 
{
    std::vector<int> newDims = {shape.rbegin(), shape.rend()};
    std::vector<int> newStrides = {strides.rbegin(), strides.rend()};
    Tensor* result = new Tensor(data, newDims, newStrides);
    return result;
}
```

## Mathematical Operations on Tensors

Our tensor operations library supports various mathematical operations:

### Element-wise Operations

```cpp
Tensor* operator+(Tensor* other);
Tensor* operator-(Tensor* other);
Tensor* operator*(Tensor* other);
Tensor* operator/(Tensor* other);
Tensor* operator+(float scalar);
Tensor* operator-(float scalar);
Tensor* operator*(float scalar);
Tensor* operator/(float scalar);
bool operator==(Tensor* other);
```

### Matrix Operations

```cpp
// Matrix multiplication
Tensor* Tensor::matmul2D_(Tensor* other) 
{
    std::vector<int> otherShape = other->getShape();
    std::vector<int> newDims = { shape[0], otherShape[1] };

    Tensor* result = new Tensor(newDims, isGrad);

    
    for(int i = 0; i < shape[0]; i++)
    {
        for(int j = 0; j < otherShape[1]; j++)
        {
            float sum = 0;
            for(int k = 0; k < shape[1]; k++)
            {
                sum += data[i * strides[0] + k] * other->data[k * other->strides[0] + j];
            }
            result->data[i * result->strides[0] + j] = sum;
        }
    }
    return result;
}
```

### Reduction Operations

```cpp
// Sum across specified dimension
Tensor* Tensor::sum(int dim, bool keepdim) 
{
    if (dim < 0 || dim >= shape.size()) {
        throw std::out_of_range("Dimension out of range");
    }

    std::vector<int> newDims = shape;
    if (keepdim) {
        newDims[dim] = 1; 
    } else {
        newDims.erase(newDims.begin() + dim); 
    }

    Tensor *result = new Tensor(newDims, isGrad);

    
    result->data.resize(result->numel(), 0.0f); 

    int outerStride = 1;
    for (int i = 0; i < dim; ++i) {
        outerStride *= shape[i];
    }
    int innerStride = 1;
    for (int i = dim + 1; i < shape.size(); ++i) {
        innerStride *= shape[i];
    }
    int dimSize = shape[dim];


    for (int outer = 0; outer < outerStride; ++outer) {
        for (int inner = 0; inner < innerStride; ++inner) {
            float sum = 0.0f;
            for (int d = 0; d < dimSize; ++d) {
                int index = outer * dimSize * innerStride + d * innerStride + inner;
                sum += data[index];
            }
            int resultIndex = outer * innerStride + inner;
            result->data[resultIndex] = sum;
        }
    }

    return result;
}

// Other reductions: mean, max, min, etc.
```

## Gradients and Backpropagation

For machine learning applications, our tensor operations support automatic differentiation and backpropagation using a computational graph approach.

### Computational Graph Representation

Each operation in a neural network can be represented as a node in a directed acyclic graph (DAG). In our implementation:

1. We store references to input tensors in the `previous` vector
2. We define gradient functions that know how to propagate gradients
3. We store both forward data and gradient data in each tensor

```cpp
class Tensor {
public:
    std::vector<Tensor*> previous;
    std::function<void()> gradFn;
    std::vector<float> data;
    std::vector<float> grad;
};
```

Example of our addition operation with gradient tracking:

```cpp
Tensor* Tensor::operator+(Tensor* other) {

    Tensor* result = new Tensor(shape, isGrad);

    if(isGrad) {
        result->addToPrevious(other);
        result->addToPrevious(this);
        result->gradFn = [result, this, other]() mutable {
            for (size_t i = 0; i < this->getGrad().size(); i++) {
                this->grad[i] += result->grad[i];
            }

            for (size_t i = 0; i < other->getGrad().size(); i++) {
                other->grad[i] += result->grad[i];
            }
        };
    }

    for (int i = 0; i < data.size(); i++) {
        result->data[i] = data[i] + other->data[i];
    }

    return result;
}
```

### Backpropagation Using Topological Sort

Our implementation uses topological sort for backpropagation, which computes gradients in the reverse order of the forward pass:

```cpp
void topologySort(std::vector<Tensor*>& visited, std::vector<Tensor*>& topoList, Tensor* t)
{
    if (std::find(visited.begin(), visited.end(), t) == visited.end()) 
    {
        visited.push_back(t); 
        for (Tensor* t1 : t->getPrevious())
        {
            topologySort(visited, topoList, t1);
        }
        topoList.push_back(t); 
    }
}


void Tensor::backward()
{
    this->grad[0] = 1.0;
    std::vector<Tensor*> visited;
    std::vector<Tensor*> topoList;
    topologySort(visited, topoList, this);
    for (int i = topoList.size() - 1; i >= 0; i--)
    {
        Tensor* t = topoList[i];
        if (t->gradFn != nullptr)
        {
            t->gradFn();
        }
    }
}
```

The algorithm works as follows:
1. Start with the loss tensor and set its gradient to 1.0
2. Build a topologically sorted list of all tensors in the computational graph
3. Traverse the list in reverse order, calling each tensor's gradient function
4. Each gradient function updates the gradients of its input tensors

## Implementation Considerations

When implementing a tensor operations package in C++, we've considered the following:

### Memory Management

Tensors can be large, so efficient memory management is crucial:

- Raw pointers for referencing tensors in the computational graph
- Vector-based storage for data and gradients
- Clear ownership semantics for tensor objects

### Optimization Techniques

Performance optimizations for tensor operations:

- **Vectorization**: Use SIMD instructions (SSE, AVX)
- **Parallelization**: Multi-threading for large operations
- **Kernel Fusion**: Combine multiple operations to reduce memory traffic
- **Memory Layout**: Optimize data layout for cache efficiency

### Hardware Acceleration

Support for various hardware:

- CPU optimizations with libraries like OpenBLAS or Intel MKL
- GPU support via CUDA or OpenCL
- Specialized hardware accelerators

## Example of a Complete Tensor Operation

Let's see our implementation of the ReLU activation function with gradient support:

```cpp
Tensor* Tensor::relu() {
    Tensor* result = new Tensor(shape, isGrad);
    if(isGrad)
    {
        result->addToPrevious(this);
        result->gradFn = [result, this]() mutable {
            for (size_t i = 0; i < this->getGrad().size(); i++) {
                this->grad[i] += result->grad[i] * (this->data[i] > 0 ? 1 : 0);
            }
        };
    }

    for (int i = 0; i < data.size(); i++) {
        result->data[i] = data[i] > 0 ? data[i] : 0;
    }
    return result;
}
```

In this example:
1. We create a new tensor with the same shape
2. If gradient tracking is enabled, we set up the gradient function
3. The gradient function applies the ReLU derivative (1 where input > 0, 0 elsewhere)
4. In the forward pass, we apply the ReLU function (max(0, x))

## Conclusion

Our C++ tensor operations package provides a solid foundation for numerical computing and machine learning applications. Key components include:

1. **Tensor Representation**: Using shapes and strides to efficiently represent N-dimensional arrays
2. **Mathematical Operations**: Element-wise, matrix, and reduction operations with operator overloading
3. **Computational Graph**: Dynamic graph construction for automatic differentiation
4. **Automatic Differentiation**: Lambda-based gradient functions for each operation
5. **Backpropagation**: Topological sort algorithm for correct gradient propagation

These components together enable the construction of complex neural networks and other tensor-based algorithms with automatic gradient computation, making it possible to implement advanced machine learning models entirely in C++.