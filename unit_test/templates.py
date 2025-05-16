CPP_UNITTEST_TEMPLATES = """#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <type_traits>
#include <cassert> 
#include <iostream>
#include <tuple>
#include <utility>
#include <cfloat>
#include <cmath> 
#include <climits> 
using namespace std;

// 判断是否为数组的模板函数
template<typename T>
bool isArray(T param) {
    return false;
}

template<typename T, size_t N>
bool isArray(T (&param)[N]) {
    return true;
}

// 新增：二维数组的特化版本
template <typename T, size_t N, size_t M>
bool isArray(T (&param)[N][M])
{
    return true;
}


// 处理单个值的打印
template<typename T>
void print_var(T param) {
    if (!isArray(param)) {
        std::cout << param;
    }
}



// 处理数组的打印
template<typename T, size_t N>
void print_var(T (&param)[N]) {
    if (isArray(param)) {
        std::cout << "[ ";
        for (size_t i = 0; i < N; i++) {
            std::cout << param[i];
            if (i < N - 1) {
                std::cout << ", ";
            }
        }
        std::cout << " ]";
    }
}


// 新增：处理二维数组的打印
template <typename T, size_t N, size_t M>
void print_var(T (&param)[N][M])
{
    std::cout << "[\\n";
    for (size_t i = 0; i < N; i++)
    {
        std::cout << "  [ ";
        for (size_t j = 0; j < M; j++)
        {
            std::cout << param[i][j];
            if (j < M - 1)
            {
                std::cout << ", ";
            }
        }
        std::cout << " ]";
        if (i < N - 1)
        {
            std::cout << ",";
        }
        std::cout << "\\n";
    }
    std::cout << "]";
}


// Overload operator<< for std::vector<T>
template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (std::size_t i = 0; i < v.size(); ++i)
    {
        if (i != 0)
            os << ", ";
        os << v[i];
    }
    os << "]";
    return os;
}

template<typename Tuple, std::size_t... Is>
void print_tuple(const Tuple& t, std::index_sequence<Is...>)
{
    ((std::cout << (Is == 0 ? "" : ", "), print_var(std::get<Is>(t))), ...);
}

template<typename... Args>
void print_tuple(const std::tuple<Args...>& t)
{
    std::cout << "Arguments after function call: (";
    print_tuple(t, std::index_sequence_for<Args...>{});
    std::cout << ")" << std::endl;
}

template<typename Func, typename... Args>
auto wrapper(Func func, Args&&... args)
{
    // Create a tuple of references to arguments
    auto arg_tuple = std::forward_as_tuple(args...);

    // Get function return type
    using ReturnType = std::invoke_result_t<Func, Args&&...>;

    if constexpr (std::is_void_v<ReturnType>)
    {
        // If function returns void
        std::apply(func, arg_tuple);
        std::cout << "Return value: void" << std::endl;
    }
    else
    {
        // If function returns a value
        ReturnType result = std::apply(func, arg_tuple);
        std::cout << "Return value: ";
        print_var(result);
        std::cout << std::endl;
    }

    // Print arguments after function call
    print_tuple(arg_tuple);
}

// TO_FILL_FUNC

int main() {
    try {
    
        // TEST_CASE
    
    } catch (std::exception const& e) {
        std::cout << "Runtime Error:\\n" << e.what() << std::endl;
    }
    return 0;
}
"""

# nvcc -arch=sm_86 -std=c++17 --expt-relaxed-constexpr demo.cu -o demo
CUDA_UNITTEST_TEMPLATES = """#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <type_traits>
#include <cassert> 
#include <iostream>
#include <tuple>
#include <utility>
#include <cfloat> 
#include <cmath> 
#include <cuda_runtime.h>
using namespace std;

// 判断是否为数组的模板函数
template<typename T>
bool isArray(T param) {
    return false;
}

template<typename T, size_t N>
bool isArray(T (&param)[N]) {
    return true;
}

// 新增：二维数组的特化版本
template <typename T, size_t N, size_t M>
bool isArray(T (&param)[N][M])
{
    return true;
}


// 处理单个值的打印
template<typename T>
void print_var(T param) {
    if (!isArray(param)) {
        std::cout << param;
    }
}



// 处理数组的打印
template<typename T, size_t N>
void print_var(T (&param)[N]) {
    if (isArray(param)) {
        std::cout << "[ ";
        for (size_t i = 0; i < N; i++) {
            std::cout << param[i];
            if (i < N - 1) {
                std::cout << ", ";
            }
        }
        std::cout << " ]";
    }
}


// 新增：处理二维数组的打印
template <typename T, size_t N, size_t M>
void print_var(T (&param)[N][M])
{
    std::cout << "[\\n";
    for (size_t i = 0; i < N; i++)
    {
        std::cout << "  [ ";
        for (size_t j = 0; j < M; j++)
        {
            std::cout << param[i][j];
            if (j < M - 1)
            {
                std::cout << ", ";
            }
        }
        std::cout << " ]";
        if (i < N - 1)
        {
            std::cout << ",";
        }
        std::cout << "\\n";
    }
    std::cout << "]";
}


// Overload operator<< for std::vector<T>
template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (std::size_t i = 0; i < v.size(); ++i)
    {
        if (i != 0)
            os << ", ";
        os << v[i];
    }
    os << "]";
    return os;
}

template<typename Tuple, std::size_t... Is>
void print_tuple(const Tuple& t, std::index_sequence<Is...>)
{
    ((std::cout << (Is == 0 ? "" : ", "), print_var(std::get<Is>(t))), ...);
}

template<typename... Args>
void print_tuple(const std::tuple<Args...>& t)
{
    std::cout << "Arguments after function call: (";
    print_tuple(t, std::index_sequence_for<Args...>{});
    std::cout << ")" << std::endl;
}

template<typename Func, typename... Args>
auto wrapper(Func func, Args&&... args)
{
    // Create a tuple of references to arguments
    auto arg_tuple = std::forward_as_tuple(args...);

    // Get function return type
    using ReturnType = std::invoke_result_t<Func, Args&&...>;

    if constexpr (std::is_void_v<ReturnType>)
    {
        // If function returns void
        std::apply(func, arg_tuple);
        std::cout << "Return value: void" << std::endl;
    }
    else
    {
        // If function returns a value
        ReturnType result = std::apply(func, arg_tuple);
        std::cout << "Return value: ";
        print_var(result);
        std::cout << std::endl;
    }

    // Print arguments after function call
    print_tuple(arg_tuple);
}

// KERNEL_FUNC

// WRAPPER_FUNC

int main() {
    try {
    
        // TEST_CASE
    
    } catch (std::exception const& e) {
        std::cout << "Runtime Error:\\n" << e.what() << std::endl;
    }
    return 0;
}
"""


CPP_UNITTEST_TEMPLATES_FOR_COV = """#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <type_traits>
#include <cassert> 
#include <iostream>
#include <tuple>
#include <utility>
#include <cfloat>
#include <cmath> 
#include <climits> 
using namespace std;

// TO_FILL_FUNC

int main() {
    try {
    
        // TEST_CASE
    
    } catch (std::exception const& e) {
        std::cout << "Runtime Error:\\n" << e.what() << std::endl;
    }
    return 0;
}
"""