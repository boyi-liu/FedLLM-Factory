# FedLLM Factory

FedLLM Factory is a unified library for LoRA-based federated LLM fine-tuning.
Currently, it supports following 10+ baselines:

+ **FedIT**. Towards Building the Federated GPT: Federated Instruction Tuning. _ICASSP 2024_.
+ **FFA-LoRA**. Improving LoRA in Privacy-preserving Federated Learning. _ICLR 2024_.
+ **FLoRA**. FLoRA: Federated Fine-Tuning Large Language Models with Heterogeneous Low-Rank Adaptations. _NeurIPS 2024_.
+ **FlexLoRA**. Federated Fine-tuning of Large Language Models
under Heterogeneous Tasks and Client Resources. _NeurIPS 2024_.
+ **FedDPA**. Dual-Personalizing Adapter for Federated Foundation Models. _NeurIPS 2024_.
+ **FedSA-LoRA**. Selective Aggregation for Low-rank Adaptation in Federated Learning. _ICLR 2025_.
+ **FedEX-LoRA**. Exact Aggregation for Federated and Efficient
Fine-Tuning of Foundation Models. _ACL 2025_.
+ **FedSVD**. FedSVD: Adaptive Orthogonalization for Private Federated Learning with LoRA. _NeurIPS 2025_.
+ **RAVAN**. RAVAN: Multi-Head Low-Rank Adaptation for Federated Fine-Tuning. _NeurIPS 2025_.
+ **RoLoRA**. Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA. _NeurIPS 2025_.
+ **SLoRA**. SLoRA: Federated Parameter Efficient Fine-Tuning of Language Models. 2023.

Also, we support asynchronous version of federated LLM tuning.
To implement an asynchronous federated LLM fine-tuning algorithm, you can extend `asyncftbase.py`.

