import json
from pathlib import Path
import re
from loguru import logger


def load_tok_file(tok_file_path: Path) -> list[str]:
    with tok_file_path.open("r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    if "test" or "valid" in tok_file_path.name:
        return lines
    if "mono" in tok_file_path.name:
        lines = [
            # 只分割第一个 '|'，规避 '||'
            line.split("|", 1)[1].strip() if line.strip() and "|" in line else line
            for line in lines
        ]
        lines = [line for line in lines if line]

    return lines

def count_nonempty_lines(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 直接读取整个文件内容并分割成行
            return len([line for line in f.read().splitlines() if line.strip()])
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在")
        return -1
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")
        return -1



def load_tok_file_as_map(tok_file_path: Path) -> dict:
    lines = load_tok_file(tok_file_path)
    target_kernel_code_map = {
        idx: line
        for idx, line in enumerate(lines)
    }
    return target_kernel_code_map

def format_dict_cases_as_str(cases: dict) -> str:
    test_cases_str = ""
    for idx, case in cases.items():
        test_cases_str += f"#Test case {idx}:\n{case}\n"
    return test_cases_str


def get_function_name_from_cpp_or_cuda_code(code: str) -> str:
    if not code or code == " ":
        return " "
    # TODO: cuda case maybe need to check further
    # remove the \n
    replaced_code = code.replace("\n", " ")
    front_code = replaced_code.split("(")[0].strip(" ")
    try:
        res = front_code.split()[-1]
    except:
        logger.error(f"get_function_name_from_cpp_or_cuda_code Error: {code}")
        return " "
    if res:
        return res
    return " "

def remove_comments(code):
    # 删除单行注释
    code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
    # 删除多行注释
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    return code

def replace_assert_statements(code: str) -> str:
    """
    Match and replace assert statements that end with semicolon.
    Example: "assert x > 0;" -> ""
    """

    # Pattern matches 'assert' followed by any characters until semicolon
    pattern = r"assert[^;]*;"
    # Replace all matching patterns with empty string
    result = re.sub(pattern, "", code)
    return result.strip()


def replace_cuda_free_statements(code: str) -> str:
    pattern = r"cudaFree\([^;]*\);"
    result = re.sub(pattern, "", code)
    return result


def wrapper_function_invoke_to_print_variables(code: str, function_name: str) -> str:
    """
    wrongfunc(args) => wrapper(func, args), func(args) is the last line of the code
    """
    # Pattern matches: function_name(any_params);
    pattern = r"(\w+)\s*\(([^;]*?)\);$"

    lines = code.strip().split('\n')
    last_line = lines[-1]
    match = re.search(pattern, last_line)
    if match:
        params_str = match.group(2)
        res = f"wrapper({function_name}, {params_str});"
        return "\n".join(lines[:-1] + [res])
    raise ValueError(f"Function call not found in the last line: {last_line}")

def replace_wrapper_invoke(code: str) -> str:
    """
    Replace all wrapper function calls with direct function calls:
    wrapper(func, arg1, arg2, ...) => func(arg1. arg2, ...)
    """
    def replacement(match):
        wrapper_call = match.group()
        params_str = re.search(r"wrapper\((\w+),\s*(.*)\)", wrapper_call)
        if params_str:
            function_name = params_str.group(1)
            params = params_str.group(2).strip()
            return f"{function_name}({params});"
        return wrapper_call

    pattern = r"wrapper\([^;]*?\);"
    return re.sub(pattern, replacement, code)


def replace_wrapper_invoke_back(code: str, function_name) -> str:
    """
    Replace all wrapper function calls with direct function calls:
    wrapper(func, args) => func(args)
    """
    def replacement(match):
        wrapper_call = match.group()
        params_str = re.search(rf"wrapper\({function_name},\s*(.*)\)", wrapper_call)
        if params_str:
            params = params_str.group(1).strip()
            return f"{function_name}({params});"
        return wrapper_call

    pattern = rf"wrapper\({function_name},[^;]*?\);"
    return re.sub(pattern, replacement, code)

def remove_code_block_lines(text):
    return "\n".join([line for line in text.split("\n") if not line.strip().startswith("```")])

def replace_wrapper_func_first_arg(wrapper_code: str, to_replace: str) -> str:
    """
    wrapper(set_sorting_offset, nrows1, ncols1, offsets1); = >  wrapper(to_replace, nrows1, ncols1, offsets1);
    """
    pattern = r"wrapper\((\w+),"
    try:
        res = re.sub(pattern, f"wrapper({to_replace},", wrapper_code)
        return res
    except:
        logger.error(f"replace_wrapper_func_first_arg Error: {wrapper_code}")
        return wrapper_code

def get_unit_test_output_statistics(unitest_output_dir: Path):
    case_counts = sum(1 for item in unitest_output_dir.iterdir() if item.is_dir())
    total_cases = 0
    status_counts = {}

    for sub_dir in unitest_output_dir.iterdir():
        if not sub_dir.is_dir():
            continue
        
        test_cases_json = sub_dir / "cpp_test_cases.json"
        if not test_cases_json.exists():
            test_cases_json = sub_dir / "cuda_test_cases.json"
            if not test_cases_json.exists():
                print("[warning]:json file not exist.")
                continue

        with open(test_cases_json) as f:
            data = json.load(f)

        if "status" in data:
            status_dict = data["status"]
            total_cases += len(status_dict)

            for status in status_dict.values():
                if status != "compile_error" and status != "success":
                    status = "other"
                status_counts[status] = status_counts.get(status, 0) + 1

    print("\nStatus statistics:")
    print(f"Total test functions: {case_counts}")
    print(f"Total test cases: {total_cases}")
    for status, count in status_counts.items():
        percentage = (count / total_cases) * 100 if total_cases > 0 else 0
        print(f"{status}: {count} ({percentage:.2f}%)")


def replace_kernel_function_in_wrapper(wrapper_code: str, to_function_name: str) -> str:
    # 修改正则表达式以处理函数名前后的空格
    # \s* 匹配任意数量的空白字符
    pattern = r'(\w+)\s*(?=\s*<<<)'
    return re.sub(pattern, to_function_name, wrapper_code)



if __name__ == "__main__":
    from unit_test.configs import RESOURCES_DIR, ROOT, OUTPUT_DIR
    from loguru import logger
    # ll = ['double old_arr1[] = {1.0};\ndouble new_arr1[1024];\nget_ev_cuda_invoke_in_cpp(old_arr1, new_arr1);', 'double old_arr2[] = {2.5};\ndouble new_arr2[1024];\nget_ev_cuda_invoke_in_cpp(old_arr2, new_arr2);', 'double old_arr3[] = {1.5, 2.0, 3.5};\ndouble new_arr3[1024];\nget_ev_cuda_invoke_in_cpp(old_arr3, new_arr3);', 'double old_arr4[] = {(get_ev_cuda_invoke_in_cpp, old_arr4, new_arr4);', 'double old_arr5[] = {1-1.0, 0.0, 1.0};\ndouble new_arr5[1024];\nget_ev_cuda_invoke_in_cpp(old_arr5, new_arr5);']
    
    # for code in ll:
    #     print(wrapper_function_invoke_to_print_variables(replace_assert_statements(remove_comments(code)), 'my_func'))
    test_replace_wrapper_func_first_arg()
    # get_unit_test_output_statistics(OUTPUT_DIR / "babel_test_dataset_unitest")

    # tok_file_path = RESOURCES_DIR / "cuda.mono.train.tok"
    # lines = load_tok_file(tok_file_path)
    # logger.info(f"Number of lines: {len(lines)}")
    # logger.info(f"last line: {lines[-1]}")

    # # check the get function name from cpp or cuda code
    # for code in lines[:10]:
    #     logger.info(get_function_name_from_cpp_or_cuda_code(code))

    # tok_file_path2 = ROOT / "BabelTower" / "dataset" / "cpp.para.test.tok"
    # lines2 = load_tok_file(tok_file_path2)
    # logger.info(f"Number of lines: {len(lines2)}")
    # logger.info(f"last line: {lines2[-1]}")
    # for code in lines2:
    #     logger.info(get_function_name_from_cpp_or_cuda_code(code))

    # tok_file_path3 = ROOT / "BabelTower" / "dataset" / "cpp.para.valid.tok"
    # lines3 = load_tok_file(tok_file_path3)
    # logger.info(f"Number of lines: {len(lines3)}")
    # logger.info(f"last line: {lines3[-1]}")
    # for code in lines3:
    #     logger.info(get_function_name_from_cpp_or_cuda_code(code))

    # tok_file_path4 = ROOT / "BabelTower" / "dataset" / "cuda.para.test.tok"
    # lines4 = load_tok_file(tok_file_path4)
    # logger.info(f"Number of lines: {len(lines4)}")
    # logger.info(f"last line: {lines4[-1]}")
    # for code in lines4:
    #     # logger.debug(code)
    #     logger.info(get_function_name_from_cpp_or_cuda_code(code))

    # tok_file_path3 = RESOURCES_DIR / "cpp.mono.train.tok"
    # lines3 = load_tok_file(tok_file_path3)
    # logger.info(f"Number of lines: {len(lines3)}")
    # logger.info(f"last line: {lines3[-1]}")
    # for code in lines3[:10]:
    #     logger.info(get_function_name_from_cpp_or_cuda_code(code))

    #
    # 2024-11-11 13:36:04.715 | INFO     | __main__:<module>:22 - Number of lines: 136025
    # 2024-11-11 13:36:04.715 | INFO     | __main__:<module>:23 - last line: __global__ void displayVelocityMag_k ( cData * part , float4 * pcolor , cData * v , int dx , int dy , float dt , int lb , size_t pitch ) { int gtidx = blockIdx . x * blockDim . x + threadIdx . x ; int gtidy = blockIdx . y * ( lb * blockDim . y ) + threadIdx . y * lb ; int p ; cData pterm , vterm ; float4 pcterm ; if ( gtidx < dx ) { for ( p = 0 ; p < lb ; p ++ ) { int fi = gtidy + p ; if ( fi < dy ) { int fj = fi * dx + gtidx ; pterm = part [ fj ] ; pcterm = pcolor [ fj ] ; int xvi = ( ( int ) ( pterm . x * dx ) ) ; int yvi = ( ( int ) ( pterm . y * dy ) ) ; vterm = * ( ( cData * ) ( ( char * ) v + yvi * pitch ) + xvi ) ; float CFL = 1.00f ; cData phiB = make_float2 ( vterm . x / ( CFL / ( float ) dx / dt ) , vterm . y / ( CFL / ( float ) dy / dt ) ) ; float umag = sqrtf ( phiB . x * phiB . x + phiB . y * phiB . y ) + 1E-7f ; pcterm . x = 0.5f * sinf ( 40 * umag ) + 0.5f ; pcterm . y = 0.5f * sinf ( 40 * umag + 2.0943f ) + 0.5f ; pcterm . z = 0.5f * sinf ( 40 * umag + 4.1887f ) + 0.5f ; pcterm . w = 1.0f ; pcolor [ fj ] = pcterm ; } } } }
    # 2024-11-11 13:36:04.716 | INFO     | __main__:<module>:27 - Number of lines: 180
    # 2024-11-11 13:36:04.716 | INFO     | __main__:<module>:28 - last line: void nlf_up_forward_cpu ( const int n , const float * filters , const int channel , const int height , const int width , const int wsize , float * top_data ) { for ( int index = 0 ; index < n ; index ++ ) { int step = height * width ; int base = index * step ; int fbase = index / channel * wsize * step ; for ( int row = height - 1 ; row >= 0 ; row -- ) { for ( int col = width - 1 ; col >= 0 ; col -- ) { float temp = 0 ; int r = row ; int c = col ; int shift = 0 * step + row * width + col ; temp += top_data [ base + r * width + c ] * filters [ fbase + shift ] ; r = row + 1 ; c = col ; shift = 1 * step + row * width + col ; if ( r < height ) temp += top_data [ base + r * width + c ] * filters [ fbase + shift ] ; else temp += top_data [ base + row * width + col ] * filters [ fbase + shift ] ; r = row + 1 ; c = col - 1 ; shift = 2 * step + row * width + col ; if ( r < height && c >= 0 ) temp += top_data [ base + r * width + c ] * filters [ fbase + shift ] ; else temp += top_data [ base + row * width + col ] * filters [ fbase + shift ] ; r = row + 1 ; c = col + 1 ; shift = 3 * step + row * width + col ; if ( r < height && c < width ) temp += top_data [ base + r * width + c ] * filters [ fbase + shift ] ; else temp += top_data [ base + row * width + col ] * filters [ fbase + shift ] ; r = row ; c = col + 1 ; shift = 4 * step + row * width + col ; if ( c < width ) temp += top_data [ base + r * width + c ] * filters [ fbase + shift ] ; else temp += top_data [ base + row * width + col ] * filters [ fbase + shift ] ; top_data [ base + row * width + col ] = temp ; } } } }
    # 2024-11-11 13:36:08.492 | INFO     | __main__:<module>:32 - Number of lines: 554497
    # 2024-11-11 13:36:08.493 | INFO     | __main__:<module>:33 - last line: static int ghost_event_proc ( GHOST_EventHandle evt , GHOST_TUserDataPtr C_void_ptr ) { bContext * C = C_void_ptr ; wmWindowManager * wm = CTX_wm_manager ( C ) ; GHOST_TEventType type = GHOST_GetEventType ( evt ) ; int time = GHOST_GetEventTime ( evt ) ; if ( type == GHOST_kEventQuit ) { WM_exit ( C ) ; } else { GHOST_WindowHandle ghostwin = GHOST_GetEventWindow ( evt ) ; GHOST_TEventDataPtr data = GHOST_GetEventData ( evt ) ; wmWindow * win ; if ( ( wm -> initialized & WM_INIT_WINDOW ) == 0 ) { return 1 ; } if ( ! ghostwin ) { puts ( " < ! > ▁ event ▁ has ▁ no ▁ window " ) ; return 1 ; } else if ( ! GHOST_ValidWindow ( g_system , ghostwin ) ) { puts ( " < ! > ▁ event ▁ has ▁ invalid ▁ window " ) ; return 1 ; } else { win = GHOST_GetWindowUserData ( ghostwin ) ; } switch ( type ) { case GHOST_kEventWindowDeactivate : wm_event_add_ghostevent ( wm , win , type , time , data ) ; win -> active = 0 ; win -> eventstate -> alt = 0 ; win -> eventstate -> ctrl = 0 ; win -> eventstate -> shift = 0 ; win -> eventstate -> oskey = 0 ; win -> eventstate -> keymodifier = 0 ; break ; case GHOST_kEventWindowActivate : { GHOST_TEventKeyData kdata ; wmEvent event ; int wx , wy ; const int keymodifier = ( ( query_qual ( SHIFT ) ? KM_SHIFT : 0 )
