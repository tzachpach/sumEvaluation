# sumVeracity

In this repository, we present a novel approach to text summarization evaluation, addressing the limitations of traditional metrics and introducing a comprehensive methodology that integrates various techniques for a more accurate assessment.

------------------
To install improved_summac use

'''
pip install -r requirements.txt
'''
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

Our results are as follows:
