from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('distilbert-base-uncased')

# Two lists of sentences
sentences1 = ['Jeff joined Microsoft in 1992 to lead corporate developer evangelism for Windows NT']

sentences2 = ['Ted joined Google in 1991',
              'Jeff joined Microsoft in 1992',
              'Bill joined Microsoft in 1992']


#     document = "."
#     summary1 = "."
#Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

#Compute cosine-similarities
cosine_scores = util.cos_sim(embeddings1, embeddings2)

#Output the pairs with their score
for i in range(len(sentences1)):
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))