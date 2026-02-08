<h1 align="center">CipherChat 🔐</h1>
A novel framework CipherChat to systematically examine the generalizability of safety alignment to non-natural languages – ciphers. 
<br>   <br>

If you have any questions, please feel free to email the first author: [Youliang Yuan](https://github.com/YouliangYuan).
    
## 👉 Paper
For more details, please refer to our paper [ICLR 2024](https://openreview.net/forum?id=MbfAK4s61A).


<div align="center">
  <img src="paper/cover.png" alt="Logo" width="500">
</div>

<h3 align="center">LOVE💗 and Peace🌊</h3>
<h3 align="center">RESEARCH USE ONLY✅ NO MISUSE❌</h3>


## Our results
We provide our results (query-response pairs) in `experimental_results`, these files can be loaded by `torch.load()`. Then, you can get a list: the first element is the config and the rest of the elements are the query-response pairs.
```
result_data = torch.load(filename)
config = result_data[0]
pairs = result_data[1:]
```



## 🛠️ Usage
✨An example run:
```
python3 main.py \
 --model_name gpt-4-0613 \
--data_path data/data_en_zh.dict \
--encode_method caesar \
--instruction_type Crimes_And_Illegal_Activities \
--demonstration_toxicity toxic \
--language en
```

### How to Run

Requirements:
- `venv`
- `pip install vllm`
- `pip install openai`

In `run_experiments.py`, set the models in `--matrix_models` as needed:
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen3-14B`
- `mistralai/Mistral-7B-Instruct-v0.3`

Terminal 1 (serve model with vLLM):
```bash
module load python
conda activate venv
export https_proxy=http://proxy:80
export http_proxy=http://proxy:80
module load cuda/12.1
pip install -U "openai>=1.0"
```

Based on the model in `run_experiments.py`, launch the matching server. Example for Qwen 2.5:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dtype float16 \
  --gpu-memory-utilization 0.85
```

Terminal 2 (run experiments):
```bash
module load python
conda activate venv
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"

export OPENAI_API_BASE="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="local"

pip install "openai<1.0"

python3 run_experiments.py \
  --matrix_models "Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen3-14B,mistralai/Mistral-7B-Instruct-v0.3" \
  --matrix_instruction_types "Crimes_And_Illegal_Activities,Privacy_And_Property,Unsafe_Instruction_Topic,Role_Play_Instruction" \
  --matrix_encode_methods "unchange,caesar" \
  --matrix_use_demonstrations "true,false" \
  --matrix_debug_num 10 \
  --report_dir report_outputs
```

Generated report artifacts include:
- `report_outputs/overall_summary.csv` and `.md`
- `report_outputs/models/*_runs.csv` and `.md`
- `report_outputs/models/*_pivot.csv` and `.md` (toxicity/refusal/validity/grammar)
- `report_outputs/models/plots/*.png`
- `report_outputs/example_io.csv` and `.jsonl` (sample prompt/response pairs)
## 🔧 Argument Specification
1. `--model_name`: The name of the model to evaluate.

2. `--data_path`: Select the data to run. 

3. `--encode_method`: Select the cipher to use.

4. `--instruction_type`: Select the domain of data.

5. `--demonstration_toxicity`: Select the toxic or safe demonstrations.

6. `--language`: Select the language of the data.


## 💡Framework
<div align="center">
  <img src="paper/Overview.png" alt="Logo" width="500">
</div>

Our approach presumes that since human feedback and safety alignments are presented in natural language, using a human-unreadable cipher can potentially bypass the safety alignments effectively. Intuitively, we first teach the LLM to comprehend the cipher clearly by designating the LLM as a cipher expert, and elucidating the rules of enciphering and deciphering, supplemented with several demonstrations. We then convert the input into a cipher, which is less likely to be covered by the safety alignment of LLMs, before feeding it to the LLMs.  We finally employ a rule-based decrypter to convert the model output from a cipher format into the natural language form.  

## 📃Results
The query-responses pairs in our experiments are all stored in the form of a list in the "experimental_results" folder, and torch.load() can be used to load data.
<div align="center">
  <img src="paper/main_result_demo.jpg" alt="Logo" width="500">
</div>

### 🌰Case Study
<div align="center">
  <img src="paper/case.png" alt="Logo" width="500">
</div>

### 🫠Ablation Study
<div align="center">
  <img src="paper/ablation.png" alt="Logo" width="500">
</div>

### 🦙Other Models
<div align="center">
  <img src="paper/other_models.png" alt="Logo" width="500">
</div>




[![Star History Chart](https://api.star-history.com/svg?repos=RobustNLP/CipherChat&type=Date)](https://star-history.com/#RobustNLP/CipherChat&Date)

Community Discussion:
- Twitter: [AIDB](https://twitter.com/ai_database/status/1691655307892830417), [Jiao Wenxiang](https://twitter.com/WenxiangJiao/status/1691363450604457984)

## Citation

If you find our paper&tool interesting and useful, please feel free to give us a star and cite us through:
```bibtex
@inproceedings{yuan2024cipherchat,
  title={GPT-4 Is Too Smart To Be Safe: Stealthy Chat with LLMs via Cipher},
  author={Yuan, Youliang and Jiao, Wenxiang and Wang, Wenxuan and Huang, Jen-tse and He, Pinjia and Shi, Shuming and Tu, Zhaopeng},
  booktitle={The Twelfth International Conference on Learning Representations}
}
```
