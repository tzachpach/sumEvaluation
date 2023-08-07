# sumEvaluation

In this repository, we present a novel approach to text summarization evaluation, addressing the limitations of traditional metrics and introducing a comprehensive methodology that integrates various techniques for a more accurate assessment.

------------------
To prepare the environment for improved_summac use

```
pip install -r requirements.txt
```

The notebook consists of all of the experiments that we ran and the final results.
------------------

We combine three approaches:
1. NLI - Natural Language Inference 
2. STS - Semantic Text Similarity
3. MLM - Masked Language Modeling

We rely on the main idea that was shown in the SummaC framework - we propose to change the granularities between different approaches:
* We test NLI, relying on 'MNLI', on the sentence level only (as it was shown superior by the SummaC authors)
* We test STS, relying on 'all-MiniLM-L6-v2', on the sentence level and on the paragraph level
* We test MLM, relying on 'BertForMaskedLM', on the paragraph level and on the document level

![Ben Gurion NLP](https://github.com/tzachpach/sumEvaluation/assets/58233980/89d7ea14-5a30-4361-a0eb-dabcf5765004)

--------------------
The datasets we have used are almost the same as were used in SummaC (except for CoGenSumm) and are as follows:

1. XSumFaith: is an extension of the XSum dataset, emphasizing the faithfulness of summarization models.
2. Polytope: presents a comprehensive typology of summarization errors.
3. FactCC: focuses on factual consistency in summaries.
4. SummEval: comprises summarizer outputs from a variety of models, labeled using a 5-point Likert scale for coherence, consistency, fluency, and relevance.
5. FRANK: annotates summarizers trained on CNN/DM and XSum datasets. 

-------------------

As our datasets are often imbalanced, we used Balanced Accuracy as our main KPI. Our results are as follows:

![Balanced Accuracy Results](https://i.imgur.com/0JgDXkv.png)
