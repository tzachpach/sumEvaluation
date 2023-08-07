# sumEvaluation

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

<table border="1" style="width:100%; text-align:center;">
    <caption><strong>Performance of Summary Inconsistency Detection models on the test set of the SUMMAC Benchmark (excluding CoGenSumm).</strong> Balanced accuracy is computed for each model on the five datasets in the benchmark, and the average is computed as the overall performance on the benchmark.</caption>
    <thead>
        <tr>
            <th></th>
            <th>xsumfaith</th>
            <th>polytope</th>
            <th>factcc</th>
            <th>summeval</th>
            <th>frank</th>
            <th>overall</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>model_name</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>MLM_DOC</td>
            <td style="background-color: lightgray;">0.555</td>
            <td style="background-color: darkgreen; color: white;"><strong>0.685</strong></td>
            <td style="background-color: lightgray;">0.573</td>
            <td style="background-color: mediumgray;">0.666</td>
            <td style="background-color: darkgreen;">0.785</td>
            <td style="background-color: mediumgray;">0.653</td>
        </tr>
        <tr>
            <td>MLM_PARA</td>
            <td style="background-color: mediumgray;">0.587</td>
            <td style="background-color: lightgreen;">0.589</td>
            <td style="background-color: lightgray;">0.573</td>
            <td style="background-color: mediumgray;">0.666</td>
            <td style="background-color: darkgreen;">0.785</td>
            <td style="background-color: mediumgray;">0.640</td>
        </tr>
        <tr>
            <td>NLI_BASELINE</td>
            <td style="background-color: darkgreen; color: white;"><strong>0.666</strong></td>
            <td style="background-color: mediumgray;">0.612</td>
            <td style="background-color: darkgreen; color: white;"><strong>0.902</strong></td>
            <td style="background-color: darkgreen;">0.818</td>
            <td style="background-color: darkgreen;">0.803</td>
            <td style="background-color: darkgreen;">0.760</td>
        </tr>
        <tr>
            <td>NLI_MLM_DOC</td>
            <td style="background-color: mediumgray;">0.643</td>
            <td style="background-color: mediumgray;">0.680</td>
            <td style="background-color: darkgreen;">0.888</td>
            <td style="background-color: darkgreen; color: white;"><strong>0.823</strong></td>
            <td style="background-color: darkgreen;">0.811</td>
            <td style="background-color: darkgreen; color: white;"><strong>0.769</strong></td>
        </tr>
        <tr>
            <td>NLI_MLM_PARA</td>
            <td style="background-color: mediumgray;">0.641</td>
            <td style="background-color: lightgreen;">0.576</td>
            <td style="background-color: mediumgray;">0.821</td>
            <td style="background-color: darkgreen;">0.822</td>
            <td style="background-color: darkgreen; color: white;"><strong>0.815</strong></td>
            <td style="background-color: mediumgray;">0.735</td>
        </tr>
        <tr>
            <td>NLI_STS_PARA</td>
            <td style="background-color: mediumgray;">0.632</td>
            <td style="background-color: lightgreen;">0.587</td>
            <td style="background-color: darkgreen;">0.861</td>
            <td style="background-color: mediumgray;">0.795</td>
            <td style="background-color: mediumgray;">0.794</td>
            <td style="background-color: mediumgray;">0.734</td>
        </tr>
        <tr>
            <td>NLI_STS_PARA_MLM_DOC</td>
            <td style="background-color: mediumgray;">0.626</td>
            <td style="background-color: mediumgray;">0.617</td>
            <td style="background-color: darkgreen;">0.852</td>
            <td style="background-color: mediumgray;">0.799</td>
            <td style="background-color: darkgreen;">0.806</td>
            <td style="background-color: mediumgray;">0.740</td>
        </tr>
        <tr>
            <td>NLI_STS_PARA_MLM_PARA</td>
            <td style="background-color: mediumgray;">0.646</td>
            <td style="background-color: lightgreen;">0.577</td>
            <td style="background-color: darkgreen;">0.852</td>
            <td style="background-color: mediumgray;">0.799</td>
            <td style="background-color: darkgreen;">0.806</td>
            <td style="background-color: mediumgray;">0.736</td>
        </tr>
        <tr>
            <td>NLI_STS_SENT</td>
            <td style="background-color: mediumgray;">0.655</td>
            <td style="background-color: lightgreen;">0.572</td>
            <td style="background-color: mediumgray;">0.809</td>
            <td style="background-color: mediumgray;">0.755</td>
            <td style="background-color: darkgreen;">0.787</td>
            <td style="background-color: mediumgray;">0.715</td>
        </tr>
        <tr>
            <td>NLI_STS_SENT_MLM_DOC</td>
            <td style="background-color: mediumgray;">0.655</td>
            <td style="background-color: mediumgray;">0.599</td>
            <td style="background-color: mediumgray;">0.822</td>
            <td style="background-color: mediumgray;">0.763</td>
            <td style="background-color: mediumgray;">0.795</td>
            <td style="background-color: mediumgray;">0.727</td>
        </tr>
        <tr>
            <td>NLI_STS_SENT_MLM_PARA</td>
            <td style="background-color: mediumgray;">0.655</td>
            <td style="background-color: lightgreen;">0.568</td>
            <td style="background-color: mediumgray;">0.822</td>
            <td style="background-color: mediumgray;">0.763</td>
            <td style="background-color: mediumgray;">0.795</td>
            <td style="background-color: mediumgray;">0.721</td>
        </tr>
        <tr>
            <td>STS_PARA</td>
            <td style="background-color: lightgray;">0.538</td>
            <td style="background-color: lightgray;">0.540</td>
            <td style="background-color: lightgray;">0.571</td>
            <td style="background-color: lightgray;">0.547</td>
            <td style="background-color: lightgray;">0.523</td>
            <td style="background-color: lightgray;">0.544</td>
        </tr>
        <tr>
            <td>STS_PARA_MLM_DOC</td>
            <td style="background-color: lightgray;">0.551</td>
            <td style="background-color: mediumgray;">0.594</td>
            <td style="background-color: lightgray;">0.576</td>
            <td style="background-color: lightgray;">0.592</td>
            <td style="background-color: mediumgray;">0.616</td>
            <td style="background-color: lightgray;">0.586</td>
        </tr>
        <tr>
            <td>STS_PARA_MLM_PARA</td>
            <td style="background-color: mediumgray;">0.586</td>
            <td style="background-color: lightgray;">0.563</td>
            <td style="background-color: lightgray;">0.576</td>
            <td style="background-color: lightgray;">0.592</td>
            <td style="background-color: mediumgray;">0.616</td>
            <td style="background-color: lightgray;">0.586</td>
        </tr>
        <tr>
            <td>STS_SENT</td>
            <td style="background-color: lightgray;">0.547</td>
            <td style="background-color: lightgray;">0.544</td>
            <td style="background-color: lightgray;">0.523</td>
            <td style="background-color: lightgray;">0.551</td>
            <td style="background-color: mediumgray;">0.595</td>
            <td style="background-color: lightgray;">0.552</td>
        </tr>
        <tr>
            <td>STS_SENT_MLM_DOC</td>
            <td style="background-color: lightgray;">0.547</td>
            <td style="background-color: mediumgray;">0.586</td>
            <td style="background-color: lightgray;">0.526</td>
            <td style="background-color: lightgray;">0.586</td>
            <td style="background-color: mediumgray;">0.658</td>
            <td style="background-color: lightgray;">0.581</td>
        </tr>
        <tr>
            <td>STS_SENT_MLM_PARA</td>
            <td style="background-color: mediumgray;">0.584</td>
            <td style="background-color: lightgray;">0.559</td>
            <td style="background-color: lightgray;">0.526</td>
            <td style="background-color: lightgray;">0.586</td>
            <td style="background-color: mediumgray;">0.658</td>
            <td style="background-color: lightgray;">0.583</td>
        </tr>
    </tbody>
</table>
