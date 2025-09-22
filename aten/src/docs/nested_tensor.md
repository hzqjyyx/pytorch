# What is a Nested Tensor?

This document explains the concept of Nested Tensors in PyTorch, their motivation, internal representation, and usage.

## 1. The "Ragged" Data Problem

In a standard PyTorch `Tensor`, every piece of data in a batch must have a uniform shape. For example, to batch 10 images, you must resize them all to a consistent height and width (e.g., `(10, 3, 224, 224)`).

However, many real-world datasets are "ragged" or "jagged," meaning their dimensions are not uniform. The most common example is in Natural Language Processing (NLP), where a batch of sentences naturally have different lengths.

**Traditional Approach: Padding**
The conventional solution is to pad all the shorter tensors in the batch with a default value until they match the size of the largest tensor.

```python
# A batch of 3 sentences with different lengths
sentence1 = torch.tensor([10, 20, 30])         # len = 3
sentence2 = torch.tensor([40, 50])             # len = 2
sentence3 = torch.tensor([60, 70, 80, 90])   # len = 4

# To batch them, we pad them to the max length of 4
padded_batch = torch.tensor([
    [10, 20, 30, 0],  # <-- one 0 padded
    [40, 50, 0, 0],   # <-- two 0s padded
    [60, 70, 80, 90]
])
```

This approach has significant drawbacks:
*   **Memory Inefficiency**: A large portion of the resulting tensor can be meaningless padding, wasting memory.
*   **Computational Waste**: Operations are performed over the entire padded tensor, including the padded areas, leading to wasted computation.
*   **Complexity**: It often requires managing an additional `attention_mask` to tell downstream models which parts of the tensor are real data and which are padding.

## 2. Nested Tensors: The Solution

A **Nested Tensor** is a data structure designed to represent a batch of tensors with varying dimensions in a single, unified object, **without requiring padding**. It can be thought of as a "tensor of tensors."

Using a Nested Tensor, the batch of sentences from the example above can be represented directly:

```python
# Conceptually, a nested tensor represents this list directly
nested_tensor = torch.nested_tensor([sentence1, sentence2, sentence3])
```

This provides a much more efficient and semantically clean way to handle ragged data.

## 3. Internal Representation

A Nested Tensor is not simply a list of tensors. For performance, it is implemented with a more sophisticated internal structure:

1.  **Data Buffer**: All the data from the constituent tensors are concatenated and stored in a single, contiguous, one-dimensional `buffer`. This ensures the data is laid out efficiently in memory for fast access.

2.  **Metadata Tensors**: To interpret the flat `buffer`, the Nested Tensor stores metadata that describes the shape and layout of each constituent tensor. The two key metadata objects are:
    *   **`sizemat` (Size Matrix)**: A 2D CPU tensor where each **row** contains the sizes (shape) of one of the constituent tensors.
    *   **`stridemat` (Stride Matrix)**: A 2D CPU tensor where each **row** contains the strides of one of the constituent tensors.

For example, for `nested_tensor([torch.rand(2, 5), torch.rand(3, 5)])`, the metadata would look something like this:
*   `sizemat` would be `[[2, 5], [3, 5]]`.
*   `stridemat` would be `[[5, 1], [5, 1]]`.

This internal structure allows PyTorch to perform operations on ragged data efficiently while abstracting the complexity away from the user.

## 4. CPU and CUDA Support

Nested Tensors are a general-purpose feature in PyTorch and are **supported on both CPU and CUDA devices**. The core logic for manipulating nested tensor metadata is device-agnostic, allowing it to work seamlessly regardless of where the underlying data buffer is stored. This makes it a powerful tool for accelerating ragged data processing on any hardware.
