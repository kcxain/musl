model_path="Qwen2.5-Coder-7B"
round=0


data_path="./BabelTower/dataset/cpp_mono_compile_pass.jsonl"
infer_dir="./round${round}/infer/"
cpp_output_path="./round${round}/infer/cpp_infer.jsonl"

python -m trans.vllm_predict \
        --model_path=${model_path} \
        --data_path=${data_path} \
        --output_path=${cpp_output_path} \
        --model_mode='trans'


data_path="./BabelTower/dataset/cuda_mono_compile_pass.jsonl"
cuda_output_path="./round${round}/infer/cuda_infer.jsonl"

python -m trans.vllm_predict \
        --model_path=${model_path} \
        --data_path=${data_path} \
        --output_path=${cuda_output_path} \
        --model_mode='trans'

python -m unit_test.validator \
        --cpp_file=${cpp_output_path} \
        --cuda_file=${cuda_output_path} \
        --test_prompt_mode='test' \
        --model_path=${model_path} \
        --round=${round}

