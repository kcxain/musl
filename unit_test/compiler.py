import tempfile
import os
import subprocess
import shutil
import time
from abc import ABC, abstractmethod
from pydantic import BaseModel
from enum import Enum
from typing import Tuple, Dict, Optional


class CompileMethod(Enum):
    LOCAL = "local"
    DOCKER = "docker"


class CompilingResult(str, Enum):
    # 编译成功且运行正常
    SUCCESS = "success"
    # 编译成功但运行时错误
    RUNTIME_ERROR = "runtime_error"
    # 编译错误
    COMPILE_ERROR = "compile_error"
    # 可选：其他可能的状态
    TIMEOUT = "timeout"  # 运行超时
    MEMORY_ERROR = "memory_error"  # 内存溢出
    UNKNOWN_ERROR = "unknown_error"  # 未知错误


class CompilerInterface(ABC):
    @abstractmethod
    def compile_and_run(self, cpp_code: str) -> Tuple[CompilingResult, str, Dict[str, Optional[float]]]:
        pass

    @abstractmethod
    def compile_and_run_with_cov(self, cpp_code: str) -> Tuple[CompilingResult, str, str, Dict[str, Optional[float]]]:
        pass


class LocalCompiler(CompilerInterface):
    def __init__(self, timeout: int = 60):
        self.timeout = timeout
        if not shutil.which("g++"):
            raise RuntimeError("g++ compiler not found")

    def compile_and_run(self, cpp_code: str) -> Tuple[CompilingResult, str, Dict[str, Optional[float]]]:
        metadata = {
            "compile_time": None,
            "execution_time": None
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            cpp_file = os.path.join(temp_dir, "program.cpp")
            exe_file = os.path.join(temp_dir, "program")

            with open(cpp_file, "w", encoding='utf-8') as f:
                f.write(cpp_code)

            try:
                compile_start = time.time()

                # 编译
                compile_result = subprocess.run(
                    ["g++", "-std=c++17", cpp_file, "-o", exe_file],
                    capture_output=True,
                    text=True,
                    encoding='utf-8'
                )
                metadata["compile_time"] = time.time() - compile_start


                if compile_result.returncode != 0:
                    return (
                        CompilingResult.COMPILE_ERROR,
                        f"Compilation Error: {compile_result.stderr}",
                        metadata
                    )

                # 运行
                execution_start = time.time()

                run_result = subprocess.run(
                    [exe_file], 
                    capture_output=True, 
                    text=True, 
                    timeout=self.timeout,
                    encoding='utf-8'
                )
                metadata["execution_time"] = time.time() - execution_start


                if run_result.returncode == 0:
                    return CompilingResult.SUCCESS, run_result.stdout, metadata
                else:
                    return (
                        CompilingResult.RUNTIME_ERROR,
                        f"Runtime Error: {run_result.stderr}",
                        metadata
                    )
                

            except subprocess.TimeoutExpired:
                return CompilingResult.TIMEOUT, "Error: Program execution timed out", metadata
            except Exception as e:
                return CompilingResult.UNKNOWN_ERROR, f"Error: {str(e)}", metadata


    def compile_and_run_with_cov(self, cpp_code):
        metadata = {
            "compile_time": None,
            "execution_time": None
        }

        # with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = "/tmp"
        cpp_file = os.path.join(temp_dir, "program.cpp")
        exe_file = os.path.join(temp_dir, "program")

        with open(cpp_file, "w", encoding='utf-8') as f:
            f.write(cpp_code)

        try:
            compile_start = time.time()
            # format
            format_result = subprocess.run(
                ["clang-format", "-i", cpp_file],
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            # 编译
            compile_result = subprocess.run(
                ["g++", "-std=c++17", "-fprofile-arcs", "-ftest-coverage", cpp_file, "-o", exe_file],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            metadata["compile_time"] = time.time() - compile_start

            if compile_result.returncode != 0:
                return (
                    CompilingResult.COMPILE_ERROR,
                    f"Compilation Error: {compile_result.stderr}",
                    "",
                    metadata
                )

            # 运行
            execution_start = time.time()

            run_result = subprocess.run(
                [exe_file], 
                cwd=temp_dir,
                capture_output=True, 
                text=True, 
                timeout=self.timeout,
                encoding='utf-8'
            )
            metadata["execution_time"] = time.time() - execution_start
            # gcov
            gcov_result = subprocess.run(
                ["gcov", "-a", "-f", "-b", "-m", cpp_file],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            if run_result.returncode == 0:
                return CompilingResult.SUCCESS, run_result.stdout, gcov_result.stdout, metadata
            else:
                return (
                    CompilingResult.RUNTIME_ERROR,
                    f"Runtime Error: {run_result.stderr}",
                    "",
                    metadata
                )

        except subprocess.TimeoutExpired:
            return CompilingResult.TIMEOUT, "Error: Program execution timed out", "", metadata
        except Exception as e:
            return CompilingResult.UNKNOWN_ERROR, f"Error: {str(e)}", "", metadata
            
        

class DockerCompiler(CompilerInterface):
    def __init__(
        self,
        mem_limit: str = "100m",
        cpu_period: int = 100000,
        cpu_quota: int = 50000,
        timeout: int = 60,
    ):
        self.mem_limit = mem_limit
        self.cpu_period = cpu_period
        self.cpu_quota = cpu_quota
        self.timeout = timeout

        try:
            import docker

            self.client = docker.from_client()
        except Exception as e:
            raise RuntimeError(f"Docker not available: {str(e)}")

    def compile_and_run(self, cpp_code: str) -> Tuple[CompilingResult, str]:
        with tempfile.TemporaryDirectory() as temp_dir:
            cpp_file = os.path.join(temp_dir, "program.cpp")
            with open(cpp_file, "w", encoding='utf-8') as f:
                f.write(cpp_code)

            dockerfile = """
            FROM gcc:latest
            WORKDIR /app
            COPY program.cpp .
            RUN g++ -std=c++17 -o program program.cpp
            CMD ["./program"]
            """

            dockerfile_path = os.path.join(temp_dir, "Dockerfile")
            with open(dockerfile_path, "w", encoding='utf-8') as f:
                f.write(dockerfile)

            try:
                # 构建Docker镜像
                image, _ = self.client.images.build(path=temp_dir, rm=True)

                # 运行容器
                container = self.client.containers.run(
                    image.id,
                    remove=True,
                    mem_limit=self.mem_limit,
                    cpu_period=self.cpu_period,
                    cpu_quota=self.cpu_quota,
                    network_mode="none",
                    timeout=self.timeout,
                )

                return CompilingResult.SUCCESS, container.decode("utf-8")

            except Exception as e:
                # 这里可以根据具体的异常类型返回不同的CompilingResult
                return CompilingResult.UNKNOWN_ERROR, f"Error: {str(e)}"


class CppCompiler:
    def __init__(self, method: CompileMethod = CompileMethod.LOCAL):
        self.method = method
        self._compiler = self._create_compiler(method)

    def _create_compiler(self, method: CompileMethod) -> CompilerInterface:
        if method == CompileMethod.LOCAL:
            return LocalCompiler()
        elif method == CompileMethod.DOCKER:
            return DockerCompiler()
        else:
            raise ValueError(f"Unsupported compilation method: {method}")

    def set_method(self, method: CompileMethod):
        """切换编译方式"""
        self.method = method
        self._compiler = self._create_compiler(method)

    def run_code(self, cpp_code: str) -> Tuple[CompilingResult, str, Dict[str, Optional[float]]]:
        """编译并运行C++代码"""
        return self._compiler.compile_and_run(cpp_code)
    
    def run_code_with_cov(self, cpp_code: str) -> Tuple[CompilingResult, str, str, Dict[str, Optional[float]]]:
        """编译并运行C++代码"""
        return self._compiler.compile_and_run_with_cov(cpp_code)

class CudaCompilerInterface(ABC):
    @abstractmethod
    def compile_and_run(self, cuda_code: str) -> Tuple[CompilingResult, str, Dict[str, Optional[float]]]:
        pass

class LocalCudaCompiler(CudaCompilerInterface):
    def __init__(self, timeout: int = 60, arch: str = "sm_86"):
        self.timeout = timeout
        self.arch = arch
        if not shutil.which("nvcc"):
            raise RuntimeError("NVIDIA CUDA Compiler (nvcc) not found")

    def compile_and_run(self, cuda_code: str) -> Tuple[CompilingResult, str, Dict[str, Optional[float]]]:
        metadata = {
            "compile_time": None,
            "execution_time": None
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            cuda_file = os.path.join(temp_dir, "program.cu")
            exe_file = os.path.join(temp_dir, "program")

            with open(cuda_file, "w", encoding='utf-8') as f:
                f.write(cuda_code)

            try:
                # Compile with nvcc
                compile_start = time.time()
                compile_result = subprocess.run(
                    [
                        "nvcc",
                        f"-arch={self.arch}",
                        "-std=c++17",
                        "--expt-relaxed-constexpr",
                        cuda_file,
                        "-o",
                        exe_file
                    ],
                    capture_output=True,
                    text=True,
                    encoding='utf-8'
                )
                metadata["compile_time"] = time.time() - compile_start

                if compile_result.returncode != 0:
                    # return (
                    #     CompilingResult.COMPILE_ERROR,
                    #     f"Compilation Error: {compile_result.stderr}",
                    # )
                    return CompilingResult.COMPILE_ERROR, f"Compilation Error: {compile_result.stderr}", metadata


                # Run the compiled program
                execution_start = time.time()
                run_result = subprocess.run(
                    [exe_file], 
                    capture_output=True, 
                    text=True, 
                    timeout=self.timeout,
                    encoding='utf-8'
                )
                metadata["execution_time"] = time.time() - execution_start

                # if run_result.returncode == 0:
                #     return CompilingResult.SUCCESS, run_result.stdout
                # else:
                #     return (
                #         CompilingResult.RUNTIME_ERROR,
                #         f"Runtime Error: {run_result.stderr}",
                #     )
                if run_result.returncode == 0:
                    return CompilingResult.SUCCESS, run_result.stdout, metadata
                else:
                    return CompilingResult.RUNTIME_ERROR, f"Runtime Error: {run_result.stderr}", metadata

            except subprocess.TimeoutExpired:
                return CompilingResult.TIMEOUT, "Error: Program execution timed out", metadata
            except Exception as e:
                return CompilingResult.UNKNOWN_ERROR, f"Error: {str(e)}", metadata

class DockerCudaCompiler(CudaCompilerInterface):
    def __init__(
        self,
        mem_limit: str = "2g",  # Increased for CUDA operations
        cpu_period: int = 100000,
        cpu_quota: int = 50000,
        timeout: int = 60,
        arch: str = "sm_86"
    ):
        self.mem_limit = mem_limit
        self.cpu_period = cpu_period
        self.cpu_quota = cpu_quota
        self.timeout = timeout
        self.arch = arch

        try:
            import docker
            self.client = docker.from_env()
        except Exception as e:
            raise RuntimeError(f"Docker not available: {str(e)}")

    def compile_and_run(self, cuda_code: str) -> Tuple[CompilingResult, str]:
        with tempfile.TemporaryDirectory() as temp_dir:
            cuda_file = os.path.join(temp_dir, "program.cu")
            with open(cuda_file, "w", encoding='utf-8') as f:
                f.write(cuda_code)

            dockerfile = f"""
            FROM nvidia/cuda:12.0.0-devel-ubuntu20.04
            WORKDIR /app
            COPY program.cu .
            RUN nvcc -arch={self.arch} -std=c++17 --expt-relaxed-constexpr program.cu -o program
            CMD ["./program"]
            """

            dockerfile_path = os.path.join(temp_dir, "Dockerfile")
            with open(dockerfile_path, "w", encoding='utf-8') as f:
                f.write(dockerfile)

            try:
                # Build Docker image
                image, _ = self.client.images.build(path=temp_dir, rm=True)

                # Run container with GPU support
                container = self.client.containers.run(
                    image.id,
                    remove=True,
                    mem_limit=self.mem_limit,
                    cpu_period=self.cpu_period,
                    cpu_quota=self.cpu_quota,
                    network_mode="none",
                    runtime="nvidia",  # Enable NVIDIA runtime
                    timeout=self.timeout,
                )

                return CompilingResult.SUCCESS, container.decode("utf-8")

            except Exception as e:
                return CompilingResult.UNKNOWN_ERROR, f"Error: {str(e)}"

class CudaCompiler:
    def __init__(self, method: CompileMethod = CompileMethod.LOCAL):
        self.method = method
        self._compiler = self._create_compiler(method)

    def _create_compiler(self, method: CompileMethod) -> CudaCompilerInterface:
        if method == CompileMethod.LOCAL:
            return LocalCudaCompiler()
        elif method == CompileMethod.DOCKER:
            return DockerCudaCompiler()
        else:
            raise ValueError(f"Unsupported compilation method: {method}")

    def set_method(self, method: CompileMethod):
        """Switch compilation method"""
        self.method = method
        self._compiler = self._create_compiler(method)

    def run_code(self, cuda_code: str) -> Tuple[CompilingResult, str, Dict[str, Optional[float]]]:
        """Compile and run CUDA code"""
        return self._compiler.compile_and_run(cuda_code)

# 使用示例
if __name__ == "__main__":
    # 测试代码
    code = """
#include <iostream>
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

void mul_Scalar_matrix ( float * a , float value , float * c , int N ) { for ( int idx = 0 ; idx < N ; idx ++ ) { c [ idx ] = a [ idx ] * value ; } }

int main() {
    try {
    
        float a4[] = {1.0, 0.0, 1.0};
float c4[3];
wrapper(mul_Scalar_matrix, a4, 0.0, c4, 3);

    
    } catch (std::exception const& e) {
        std::cout << "Runtime Error:" << e.what() << std::endl;
    }
    return 0;
}
    """

    try:
        # 创建编译器实例（默认使用本地编译）
        compiler = CppCompiler()

        # 使用本地编译
        print("Local compilation result:")
        print(compiler.run_code(code))

    except Exception as e:
        print(f"Error: {str(e)}")
