from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk, numpy as np, torch, os, json
from improved_summac.utils_misc import batcher
from blanc import BlancHelp

nltk.download('punkt')

model_map = {
    "snli-base": {"model_card": "boychaboy/SNLI_roberta-base", "entailment_idx": 0, "contradiction_idx": 2},
    "snli-large": {"model_card": "boychaboy/SNLI_roberta-large", "entailment_idx": 0, "contradiction_idx": 2},
    "mnli-base": {"model_card": "microsoft/deberta-base-mnli", "entailment_idx": 2, "contradiction_idx": 0},
    "mnli": {"model_card": "roberta-large-mnli", "entailment_idx": 2, "contradiction_idx": 0},
    "anli": {"model_card": "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli", "entailment_idx": 0, "contradiction_idx": 2},
    "vitc-base": {"model_card": "tals/albert-base-vitaminc-mnli", "entailment_idx": 0, "contradiction_idx": 1},
    "vitc": {"model_card": "tals/albert-xlarge-vitaminc-mnli", "entailment_idx": 0, "contradiction_idx": 1},
    "vitc-only": {"model_card": "tals/albert-xlarge-vitaminc", "entailment_idx": 0, "contradiction_idx": 1},
    "bart-mnli": {"model_card": "facebook/bart-large-mnli", "entailment_idx": 2, "contradiction_idx": 0}
    # "decomp": 0,
}


def card_to_name(card):
    card2name = {v["model_card"]: k for k, v in model_map.items()}
    if card in card2name:
        return card2name[card]
    return card


def name_to_card(name):
    if name in model_map:
        return model_map[name]["model_card"]
    return name


def get_neutral_idx(ent_idx, con_idx):
    return list(set([0, 1, 2]) - set([ent_idx, con_idx]))[0]


class SummaCImager:
    def __init__(self, model_type, model_name="vitc", granularity="paragraph", use_cache=True, max_doc_sents=100, device="cuda", **kwargs):

        self.grans = granularity.split("-")

        assert all(gran in ["paragraph", "sentence", "document", "2sents", "mixed"] for gran in self.grans) and len(self.grans) <= 2, "Unrecognized `granularity` %s" % (granularity)
        # assert model_name in model_map.keys(), "Unrecognized model name: `%s`" % (model_name)

        self.model_type = model_type
        if model_type == "nli":
            self.model_name = model_name
            if model_name != "decomp":
                self.model_card = name_to_card(model_name)
                self.entailment_idx = model_map[model_name]["entailment_idx"]
                self.contradiction_idx = model_map[model_name]["contradiction_idx"]
                self.neutral_idx = get_neutral_idx(self.entailment_idx, self.contradiction_idx)

        self.granularity = granularity
        self.use_cache = use_cache
        self.cache_folder = "summac_cache/"

        self.max_doc_sents = max_doc_sents
        self.max_input_length = 500
        self.device = device
        self.cache = {}
        self.model = None

    def load_nli(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_card)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_card).eval()
        self.model.to(self.device)
        if self.device == "cuda":
            self.model.half()

    def load_sts(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=self.device)

    def load_mlm(self, batch_size):
        self.model = BlancHelp(device=self.device, inference_batch_size=batch_size, show_progress_bar=False)

    def split_sentences(self, text):
        sentences = nltk.tokenize.sent_tokenize(text)
        sentences = [sent for sent in sentences if len(sent)>10]
        return sentences

    def split_2sents(self, text):
        sentences = nltk.tokenize.sent_tokenize(text)
        sentences = [sent for sent in sentences if len(sent)>10]
        two_sents = [" ".join(sentences[i:(i+2)]) for i in range(len(sentences))]
        return two_sents

    def split_paragraphs(self, text):
        if text.count("\n\n") > 0:
            paragraphs = [p.strip() for p in text.split("\n\n")]
        else:
            paragraphs = [p.strip() for p in text.split("\n")]
        return [p for p in paragraphs if len(p) > 10]

    def split_text(self, text, granularity="sentence"):
        if granularity == "document":
            return [text]
        elif granularity == "paragraph":
            return self.split_paragraphs(text)
        elif granularity == "sentence":
            return self.split_sentences(text)
        elif granularity == "2sents":
            return self.split_2sents(text)
        elif granularity == "mixed":
            return self.split_sentences(text) + self.split_paragraphs(text)

    def build_chunk_dataset(self, original, generated, pair_idx=None):
        if len(self.grans) == 1:
            gran_doc, gran_sum = self.grans[0], self.grans[0]
        else:
            gran_doc, gran_sum = self.grans[0], self.grans[1]

        original_chunks = self.split_text(original, granularity=gran_doc)[:self.max_doc_sents]
        generated_chunks = self.split_text(generated, granularity=gran_sum)

        N_ori, N_gen = len(original_chunks), len(generated_chunks)
        dataset = [{"premise": original_chunks[i], "hypothesis": generated_chunks[j], "doc_i": i, "gen_i": j, "pair_idx": pair_idx} for i in range(N_ori) for j in range(N_gen)]
        return dataset, N_ori, N_gen

    def build_images(self, originals, generateds, batch_size=128):
        todo_originals, todo_generateds = [], []
        for ori, gen in zip(originals, generateds):
            cache_key = (ori, gen)
            if cache_key not in self.cache:
                todo_originals.append(ori)
                todo_generateds.append(gen)
        
        total_dataset = []
        todo_images = []
        for pair_idx, (ori, gen) in enumerate(zip(todo_originals, todo_generateds)):
            dataset, N_ori, N_gen = self.build_chunk_dataset(ori, gen, pair_idx=pair_idx)
            if self.model_type == 'nli':
                n = 3
            else:
                n = 1

            if len(dataset) == 0:
                image = np.zeros((n, 1, 1))
            else:
                image = np.zeros((n, N_ori, N_gen))
            todo_images.append(image)
            total_dataset += dataset

        if len(total_dataset) > 0 and self.model is None: # Can't just rely on the cache
            if self.model_type == 'nli':
                self.load_nli()
            elif self.model_type == 'sts':
                self.load_sts()
            elif self.model_type == 'mlm':
                self.load_mlm(batch_size)

        for batch in batcher(total_dataset, batch_size=batch_size):
            batch_prems = [b["premise"] for b in batch]
            batch_hypos = [b["hypothesis"] for b in batch]

            if self.model_type == 'nli':
                batch_tokens = self.tokenizer.batch_encode_plus(list(zip(batch_prems, batch_hypos)), padding=True, truncation=True, max_length=self.max_input_length, return_tensors="pt", truncation_strategy="only_first")
                with torch.no_grad():
                    model_outputs = self.model(**{k: v.to(self.device) for k, v in batch_tokens.items()})
                batch_probs = torch.nn.functional.softmax(model_outputs["logits"], dim=-1)
                batch_evids = batch_probs[:, self.entailment_idx].tolist()
                batch_conts = batch_probs[:, self.contradiction_idx].tolist()
                batch_neuts = batch_probs[:, self.neutral_idx].tolist()
                zipped = zip(batch, batch_evids, batch_conts, batch_neuts)

            elif self.model_type == 'sts':
                # Get text embeddings for text pairs
                embeddings1 = self.model.encode(batch_prems, convert_to_tensor=True)
                embeddings2 = self.model.encode(batch_hypos, convert_to_tensor=True)

                # Compute cosine-similarities
                cosine_scores = util.cos_sim(embeddings1, embeddings2)
                batch_sts_score = cosine_scores.tolist()[0]
                zipped = zip(batch, batch_sts_score)

            elif self.model_type == 'mlm':
                batch_mlm_score = self.model.eval_pairs(batch_prems, batch_hypos)
                zipped = zip(batch, batch_mlm_score)

            for z in zipped:
                b = z[0]
                image = todo_images[b["pair_idx"]]

                if self.model_type == 'nli':
                    image[0, b["doc_i"], b["gen_i"]] = z[1]
                    image[1, b["doc_i"], b["gen_i"]] = z[2]
                    image[2, b["doc_i"], b["gen_i"]] = z[3]
                else:
                    image[0, b["doc_i"], b["gen_i"]] = z[1]

        for pair_idx, (ori, gen) in enumerate(zip(todo_originals, todo_generateds)):
            cache_key = (ori, gen)
            self.cache[cache_key] = todo_images[pair_idx]
        
        images = [self.cache[(ori, gen)] for ori, gen in zip(originals, generateds)]
        return images

    def get_cache_file(self):
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)
        return os.path.join(self.cache_folder, "cache_%s_%s.json" % (self.model_name, self.granularity))

    def save_cache(self):
        cache_cp = {"[///]".join(k): v.tolist() for k, v in self.cache.items()}
        with open(self.get_cache_file(), "w") as f:
            json.dump(cache_cp, f)

    def load_cache(self):
        cache_file = self.get_cache_file()
        if os.path.isfile(cache_file):
            with open(cache_file, "r") as f:
                cache_cp = json.load(f)
                self.cache = {tuple(k.split("[///]")): np.array(v) for k, v in cache_cp.items()}


class SummaCZS:
    def __init__(self, model_name="vitc", nli_granularity="sentence", sts_granularity="paragraph", mlm_granularity="document", op1="max", op2="mean", use_nli=True, use_ent=True, use_con=False, use_sts=False, use_mlm=False, imager_load_cache=True, device="cuda", **kwargs):
        assert op2 in ["min", "mean", "max"], "Unrecognized `op2`"
        assert op1 in ["max", "mean", "min"], "Unrecognized `op1`"
        self.device = device

        self.op2 = op2
        self.op1 = op1
        self.use_ent = use_ent
        self.use_con = use_con

        self.use_nli = use_nli
        self.use_sts = use_sts
        self.use_mlm = use_mlm

        if self.use_nli:
            self.nli_imager = SummaCImager("nli", model_name=model_name, granularity=nli_granularity, device=self.device, **kwargs)
        if self.use_sts:
            self.sts_imager = SummaCImager("sts", granularity=sts_granularity, device=self.device, **kwargs)
        if self.use_mlm:
            self.mlm_imager = SummaCImager("mlm", granularity=mlm_granularity, device=self.device, **kwargs)

    def save_imager_cache(self):
        self.nli_imager.save_cache()

    def image2score(self, image, model_type):
        scores = []
        for val in image:
            if self.op1 == "max":
                scores.append(np.max(val, axis=0))
            elif self.op1 == "mean":
                scores.append(np.mean(val, axis=0))
            elif self.op1 == "min":
                scores.append(np.min(val, axis=0))

        if model_type == 'nli':
            if self.use_ent and self.use_con:
                scores = scores[0] - scores[1]
            elif self.use_ent:
                scores = scores[0]
            elif self.use_con:
                scores = 1 - scores[1]
        else:
            scores = scores[0]

        if self.op2 == "mean":
            final_score = np.mean(scores)
        elif self.op2 == "min":
            final_score = np.min(scores)
        elif self.op2 == "max":
            final_score = np.max(scores)
        return final_score

    def score(self, sources, generateds, batch_size=128, **kwargs):
        scores = np.zeros(len(sources))
        if self.use_nli:
            nli_images = self.nli_imager.build_images(sources, generateds, batch_size=batch_size)
            scores += np.array([self.image2score(image, 'nli') for image in nli_images])
        if self.use_sts:
            sts_images = self.sts_imager.build_images(sources, generateds, batch_size=batch_size)
            scores += np.array([self.image2score(image, 'sts') for image in sts_images])
        if self.use_mlm:
            mlm_images = self.mlm_imager.build_images(sources, generateds, batch_size=batch_size)
            scores += np.array([self.image2score(image, 'mlm') for image in mlm_images])

        return {"scores": scores}


if __name__ == "__main__":
    model = SummaCZS(granularity="document", model_name="vitc", imager_load_cache=True, device="cpu") # Device can be `cpu` or `cuda` when GPU is available

    # document = "Jeff joined Microsoft in 1992 to lead corporate developer evangelism for Windows NT."
    # summary1 = "Jeff joined Microsoft in 1992."
    # summary2 = "ted joined Microsoft."
    #
    # print(model.score([document, document], [summary1, summary2])["scores"])
    #
    document = """Jeff joined Microsoft in 1992 to lead corporate developer evangelism for Windows NT. He then served as a Group Program manager in Microsoft's Internet Business Unit. In 1998, he led the creation of SharePoint Portal Server, which became one of Microsoft’s fastest-growing businesses, exceeding $2 billion in revenues. Jeff next served as Corporate Vice President for Program Management across Office 365 Services and Servers, which is the foundation of Microsoft's enterprise cloud leadership. He then led Corporate Strategy supporting Satya Nadella and Amy Hood on Microsoft's mobile-first/cloud-first transformation and acquisitions. Prior to joining Microsoft, Jeff was vice president for software development for an investment firm in New York. He leads Office shared experiences and core applications, as well as OneDrive and SharePoint consumer and business services in Office 365. Jeff holds a Master of Business Administration degree from Harvard Business School and a Bachelor of Science degree in information systems and finance from New York University."""
    summary = "Jeff joined Microsoft in 1992 to lead the company's corporate evangelism. He then served as a Group Manager in Microsoft's Internet Business Unit. In 1998, Jeff led Sharepoint Portal Server, which became the company's fastest-growing business, surpassing $3 million in revenue. Jeff next leads corporate strategy for SharePoint and Servers which is the basis of Microsoft's cloud-first strategy. He leads corporate strategy for Satya Nadella and Amy Hood on Microsoft's mobile-first."

    scores = model.score([document], [summary])["images"][0][0].T
    summary_sentences = model.nli_imager.split_text(summary)

    print(np.array2string(scores, precision=2))
    for score_row, sentence in zip(scores, summary_sentences):
        print("-----------")
        print("[SummaC score: %.3f; supporting sentence: %d] %s " % (np.max(score_row), np.argmax(score_row)+1, sentence))

