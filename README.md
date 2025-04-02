# ExplainGen
While LLMs show promise in identifying misinformation, they often struggle with context-dependent cases and may even reinforce falsehoods due to biases in their training data. Instead of making final decisions on misinformation, we propose leveraging LLMs as human-centered assistants to generate context-based explanations that support human judgment in combating misinformation.
This paper introduces ExplainGen, an LLM-based web app designed to provide fact-grounded explanations for assessing the credibility of statements. Due to limited transparency in LLM training data and the scarcity of high-quality fact-checking datasets, we scrape data from various domains and combine them with publicly available fact-checking instructions to fine-tune ExplainGen. Our evaluation shows that ExplainGen generates well-supported explanations that outperform the baseline models. As future work, we plan to conduct survey-based experiments to evaluate the effectiveness of ExplainGen for human decision-making, and incorporate retrieval-augmented generation (RAG) to reduce hallucinations of ExplainGen LLM.
