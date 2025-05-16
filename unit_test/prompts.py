## 目前这个是生成CPP的unitest test 没问题了，
## TODO: 1. 基于已有的可以运行的cpp 的unit test 生成cuda 版本的unit test.
## TODO: 2. 直接生成已有的cuda代码的unit test，根据生成的可以运行的cuda unit test生成cpp 版本的
CUDA2CPP_WRAPPER_PROMPT = """\
Please help me wrap this CUDA kernel to allow user to call it like a C++ function. The wrapper function should:

1. **Keep the input and output parameters and their orders as same the as the original CUDA kernel.**
2. Call the cuda kernel provided by user inside the wrapper function.
3. The generated code must be in the [CODE] and [/CODE] tags.

Here are examples for you:

Cuda Code:
[CODE]
```cuda
__global__ void add_100_kernel(int numElements, int* data) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {{
        data[idx] += 100;
    }}
}}
```
[/CODE]

Cuda Code Wrapper:
[CODE]
```cuda
void add_100_cuda_invoke_in_cpp(int numElements, int* data) {{
    int* d_data;
    cudaMalloc((void**)&d_data, numElements * sizeof(int));
    cudaMemcpy(d_data, data, numElements * sizeof(int), cudaMemcpyHostToDevice);
    add_100_kernel<<<numElements, 1>>>(numElements, d_data);
    cudaMemcpy(data, d_data, numElements * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}}
```
[/CODE]

Cuda Code:
[CODE]
```cuda
__global__ void set_sorting_offset ( const int nrows , const int ncols , int * offsets ) {{ 
    int tid = threadIdx . x + blockIdx . x * blockDim . x ; 
    if ( tid <= ncols ) 
        offsets [ tid ] = tid * nrows ; 
    return ; 
}}
```
[/CODE]

Cuda Code Wrapper:
[CODE]
```cuda
void set_sorting_offset_cuda_invoke_in_cpp(cosnt int nrows, const int ncols, int* offsets) {{
    int* d_offsets;
    cudaMalloc((void**)&d_offsets, ncols * sizeof(int));
    cudaMemcpy(d_offsets, offsets, ncols * sizeof(int), cudaMemcpyHostToDevice);
    set_sorting_offset<<<dim3((ncols + 255) / 256, 1, 1), 256>>>(nrows, ncols, d_offsets);
    cudaMemcpy(offsets, d_offsets, ncols * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_offsets);
}}
```
[/CODE]

Your task is to write a wrapper function for the following cuda kernel function, **The generated code must be in the [CODE] and [/CODE] tags**:

Cuda Code:
[CODE]
```cuda
{cuda_code}
```
[/CODE]

Cuda Code Wrapper:
"""
CUDA2CPP_WRAPPER_PROMPT_TRAINED = """\
Please help me wrap this CUDA kernel to allow user to call it like a C++ function. The wrapper function should:
1. **Keep the input and output parameters and their orders as same the as the original CUDA kernel.**
2. Call the cuda kernel provided by user inside the wrapper function.
3. Make sure the wrapper function can be compiled and run successfully.
"""

# TODO: 把根据CPP 和 Cuda生成unit test的prompt分开， 不要混在一起写。 , for cuda test cases you should use cudaMallocManaged
CODE_BASED_CPP_UNIT_TEST_INPUT_GENERATION_PROMPT = """Your task is to write 5 valid inputs to run the cpp function that performs a specific calculation.
The inputs must be between [INPUTS] and [/INPUTS] tags.
You must write the comment "//Input case n:" on a separate line directly above,
where n represents the input case number, starting from 1 and increasing by one for each subsequent input case.

Code: 
void add_100 ( int numElements , int * data ) {{ for ( int idx = 0 ; idx < numElements ; idx ++ ) {{ data [ idx ] += 100 ; }} }}
[INPUTS]
//Input case 1:
int data1[] = {{0}};
add_100(1, data1);

//Input case 2:
int data2[] = {{-100}};
add_100(1, data2);

//Input case 3:
int data3[] = {{1, 2, 3}};
add_100(3, data3);

//Input case 4:
int data4[] = {{INT_MAX - 100}};
add_100(1, data4);

//Input case 5:
int data5[] = {{-50, 0, 50}};
add_100(3, data5);
[/INPUTS]

Code: 
{code}
"""

CODE_BASED_CUDA_UNIT_TEST_INPUT_GENERATION_PROMPT = """Your task is to write 5 valid inputs to run the CUDA invoke in cpp function that performs a specific calculation.
The inputs must be between [INPUTS] and [/INPUTS] tags.
You must write the comment "//Input case n:" on a separate line directly above,
where n represents the input case number, starting from 1 and increasing by one for each subsequent input case.
Just call the CUDA invoke in cpp function provided by user, **DO NOT call the kernel function directly**.

Code: 
__global__ void add_100_kernel(int *data, int numElements) {{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {{
        data[idx] += 100;
    }}
}}

void add_100_cuda_invoke_in_cpp(int *data, int numElements) {{
    int *d_data;
    size_t size = numElements * sizeof(int);
    cudaMalloc((void **)&d_data, size);
    cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    int blocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    add_100_kernel<<<blocks, threadsPerBlock>>>(d_data, numElements);
    cudaDeviceSynchronize();
    cudaMemcpy(data, d_data, size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}}

[INPUTS]
//Input case 1:
int data1[] = {{0}};
add_100_cuda_invoke_in_cpp(1, data1);

//Input case 2:
int data2[] = {{-100}};
add_100_cuda_invoke_in_cpp(1, data2);

//Input case 3:
int data3[] = {{1, 2, 3}};
add_100_cuda_invoke_in_cpp(3, data3);

//Input case 4:
int data4[] = {{INT_MAX - 100}};
add_100_cuda_invoke_in_cpp(1, data4);

//Input case 5:
int data5[] = {{-50, 0, 50}};
add_100_cuda_invoke_in_cpp(3, data5);
[/INPUTS]

Code: 
{code}
"""

CODE_BASED_CPP_UNIT_TEST_INPUT_GENERATION_PROMPT_TRAINED = """Your task is to write 5 valid inputs to run the cpp function that performs a specific calculation.
You must write the comment "//Input case n:" on a separate line directly above,
where n represents the input case number, starting from 1 and increasing by one for each subsequent input case.
"""

CODE_BASED_CUDA_UNIT_TEST_INPUT_GENERATION_PROMPT_TRAINED = """Your task is to write 5 valid inputs to run the CUDA invoke in cpp function that performs a specific calculation.
You must write the comment "//Input case n:" on a separate line directly above,
where n represents the input case number, starting from 1 and increasing by one for each subsequent input case.
Just call the CUDA invoke in cpp function provided by user, **DO NOT call the kernel function directly**.
"""


CPP2CUDA_TRANSLATION_PROMPT ="""
Please help me convert this CPU code into equivalent CUDA kernel code. The converted code should:

1. Preserve the original functionality
2. Process data elements in the same order
3. Keep the same input parameters and data handling logic
4. The generated code must be in the  [CODE] and [/CODE] tags.


Here is an example for you:

Cpp Code:
```cpp
void add_100(int numElements, int *data) {{
    for (int idx = 0; idx < numElements; idx++) {{
        data[idx] += 100;
    }}
}}
```

Cuda Code:
[CODE]
```cuda
__global__ void add_100(int numElements, int *data) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {{
        data[idx] += 100;
    }}
}}
```
[/CODE] 

Your task is to write a equivalent cuda kernel function for the following cpp function:

Cpp Code:
```cpp
{cpp_code}
```

Cuda Code:
"""

CUDA2CPP_TRANSLATION_PROMPT ="""
Please help me convert this CUDA kernel code into equivalent CPU code. The converted code should:

1. Preserve the original functionality 
2. Process data elements in the same order
3. Keep the same input parameters and data handling logic
4. The generated code must be in the [CODE] and [/CODE] tags.

Here is an example for you:

Cuda Code:
```cuda
__global__ void add_100(int numElements, int *data) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {{
        data[idx] += 100;
    }}
}}
```

Cpp Code:
[CODE]
```cpp
void add_100(int numElements, int *data) {{
    for (int idx = 0; idx < numElements; idx++) {{
        data[idx] += 100;
    }}
}}
```
[/CODE]

Your task is to write an equivalent CPU function for the following CUDA kernel:

Cuda Code:
```cuda
{cuda_code}
```

Cpp Code:
"""