# Credits

This recipe is a small bridge between several much larger pieces of work:

- [MiaAI-Lab's two-Spark DeepSeek V4 recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
  supplies the runtime architecture, launch lineage, NVFP4 DS-MLA path, dSpark
  integration, and the selected vLLM 0.27 backports used here.
- [Anemll's DGX Spark vLLM image](https://github.com/Anemll/dspark-vllm-gx10)
  is the pinned runtime base.
- Luke Alonso's [b12x](https://github.com/local-inference-lab/b12x) supplies the
  Blackwell EXL3/Trellis expert kernels. This image uses our
  [serving fork](https://github.com/tpurtell/sparkinfer-glmrt).
- [ModelCloud's GPTQModel](https://github.com/ModelCloud/GPTQModel) and
  [ExLlamaV3](https://github.com/turboderp-org/exllamav3) made the standard-HF
  EXL3 model possible. The quantization work used our
  [GPTQModel fork](https://github.com/tpurtell/GPTQModel).
- [0xSero's public DeepSeek V4 EXL3 work](https://huggingface.co/0xSero/DeepSeek-V4-Flash-0731-EXL3-3.0bpw)
  provided especially useful calibration and coverage comparisons while we
  developed the K2 artifact.

The Mia recipe also carries important work from
[Keys/drowzeys](https://github.com/drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash),
[Rafael Caricio](https://github.com/rafaelcaricio/spark_vllm_docker/pull/1),
[Fraser Price](https://github.com/fraserprice/dspark-vllm), and
[TonyD2Wild](https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark).

Finally, the whole stack rests on DeepSeek V4 Flash, vLLM, FlashInfer, CUDA,
NCCL, CUTLASS/CuTe DSL, PyTorch, and the Hugging Face ecosystem.

Repo-local scripts and docs follow this repository's MIT license. The vLLM,
b12x, model, container, CUDA, and other upstream components retain their own
licenses and terms.
