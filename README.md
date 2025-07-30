
# ExplainGen: a Human-Centered LLM Assistant for Combating Misinformation

> **ExplainGen** is a web-based LLM system that helps users assess the credibility of online claims by generating **fact-grounded explanations**, rather than just labeling them as true or false.

---

## Why ExplainGen?

Large Language Models (LLMs) have shown potential in misinformation detection — but they:

- Struggle with **context-sensitive** or nuanced claims  
- May **hallucinate** or reflect bias in training data  
- Often act as **black-box classifiers**, giving answers without reasoning  

### Our Solution:
**ExplainGen** reframes LLMs as human-centered assistants:
- Provide **transparent, factual explanations**  
- Support **human judgment** over machine decisions  
- Fine-tuned on real-world fact-checking content

---

## Key Features

- Explanation generation instead of binary classification  
- Trained on **25k+ verified claims** with human-written justifications  
- Based on **LLaMA-3.1-Instruct**, optimized via LoRA + 4-bit quantization  
- Web interface with **Gradio** and **ngrok** tunneling  
- Extensible and lightweight — deployable on a single GPU  

---

## System Architecture

```
User Claim
    ↓
[ExplainGen LLM]
    ↓
Fact-grounded Explanation (w/ sources)
```

### Pipeline:
1. Scrape & clean fact-checking articles  
2. Structure into JSON format  
3. Fine-tune LLM with high-quality factual data  
4. Deploy with browser-based Web UI  

---

## Datasets Used

| Source          | Domain                       | # Samples |
|-----------------|------------------------------|-----------|
| **PolitiFact**      | Political & general claims   | 25,904    |
| **FactCheck.org**   | Science, COVID-19, archives  | ~900      |
| **LIAR-RAW**        | Filtered with justifications | 10,065    |

---

## Results

| Metric        | ExplainGen | Baseline LLM |
|---------------|------------|---------------|
| **ROUGE-1**   | 6.5% Higher| –             |
| **ROUGE-2**   | 8.2% Higher| –             |
| **ROUGE-L**   | 7.1% Higher| –             |
| **BERT F1**   | 7.0% Higher| –             |
| **BLEU**      | 14.3%Higher| –             |

---

## Future Work

- Conduct **user studies** on explanation helpfulness  
- Add **RAG (Retrieval-Augmented Generation)** to reduce hallucination  
- Gather more high-quality fact-checking data  
- Improve domain generalization for non-political claims  

---

## Author

**Zhicheng Yang**  

---

## Citation

If you use or reference ExplainGen in academic work:

```bibtex
@inproceedings{10.1145/3722570.3726897,
author = {Yang, Zhicheng and Jia, Xinle and Jiang, Xiaopeng},
title = {ExplainGen: a Human-Centered LLM Assistant for Combating Misinformation},
year = {2025},
isbn = {9798400716096},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3722570.3726897},
doi = {10.1145/3722570.3726897},
abstract = {While LLMs show promise in identifying misinformation, they often struggle with context-dependent cases and may even reinforce falsehoods due to biases in their training data. Instead of making final decisions on misinformation, we propose leveraging LLMs as human-centered assistants to generate context-based explanations that support human judgment in combating misinformation. This paper introduces ExplainGen, an LLM-based web app designed to provide fact-grounded explanations for assessing the credibility of statements. Due to limited transparency in LLM training data and the scarcity of high-quality fact-checking datasets, we scrape data from various domains and combine them with publicly available fact-checking instructions to fine-tune ExplainGen. Our evaluation shows that ExplainGen generates well-supported explanations that outperform the baseline models. As future work, we plan to conduct survey-based experiments to evaluate the effectiveness of ExplainGen for human decision-making, and incorporate retrieval-augmented generation (RAG) to reduce hallucinations of ExplainGen LLM. Our source code and datasets are available on our GitHub page: https://github.com/Accuracy98/ExplainGen.},
booktitle = {Proceedings of the 3rd International Workshop on Human-Centered Sensing, Modeling, and Intelligent Systems},
pages = {120–123},
numpages = {4},
keywords = {Human-centered Computing, Large Language Model, Misinformation},
location = {Irvine, CA, USA},
series = {HumanSys '25}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

