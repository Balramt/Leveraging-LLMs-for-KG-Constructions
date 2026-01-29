# from __future__ import annotations

# import json
# import re

# import nltk
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize


# nltk.download("punkt")
# nltk.download("wordnet")


# def lemmatize_and_normalize(text, lemmatizer):
#     text = text.lower()
#     text = text.replace("_", " ")
#     text = re.sub(r"[^a-z0-9 ]", "", text)
#     text = text.strip()
#     tokens = word_tokenize(text)
#     lemmatized = "".join(lemmatizer.lemmatize(token) for token in tokens)
#     return lemmatized


# def normalize_triple(sub_label, rel_label, obj_label, lemmatizer, stem_rel=False):
#     sub_label = lemmatize_and_normalize(sub_label, lemmatizer)
#     obj_label = lemmatize_and_normalize(obj_label, lemmatizer)

#     rel_label_clean = rel_label.lower().replace("_", " ")
#     rel_label_clean = re.sub(r"[^a-z0-9 ]", "", rel_label_clean).strip()
#     if stem_rel:
#         rel_label_clean = " ".join([lemmatizer.lemmatize(w) for w in word_tokenize(rel_label_clean)])
#     rel_label_clean = re.sub(r"\s+", "", rel_label_clean)

#     return f"{sub_label}{rel_label_clean}{obj_label}"


# def calculate_precision_recall_f1(gold_set, pred_set):
#     if not pred_set:
#         return 0.0, 0.0, 0.0
#     intersection = gold_set.intersection(pred_set)
#     p = len(intersection) / len(pred_set)
#     r = len(intersection) / len(gold_set)
#     f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
#     return p, r, f1


# def get_subject_object_hallucinations(lemmatizer, ontology, sentence, triples):
#     if not triples:
#         return 0, 0
#     extended_sentence = sentence + " " + " ".join([c["label"] for c in ontology["concepts"]])
#     normalized_sentence = lemmatize_and_normalize(extended_sentence, lemmatizer)

#     subj_halluc, obj_halluc = 0, 0
#     for sub, rel, obj in triples:
#         norm_sub = lemmatize_and_normalize(sub, lemmatizer)
#         norm_obj = lemmatize_and_normalize(obj, lemmatizer)
#         if norm_sub not in normalized_sentence:
#             subj_halluc += 1
#         if norm_obj not in normalized_sentence:
#             obj_halluc += 1

#     return subj_halluc / len(triples), obj_halluc / len(triples)


# def get_ontology_conformance(ontology, triples):
#     if not triples:
#         return 1, 0
#     lemmatizer = WordNetLemmatizer()
#     ont_rels = {lemmatize_and_normalize(rel["label"], lemmatizer) for rel in ontology["relations"]}
#     num_conformant = sum(1 for tr in triples if lemmatize_and_normalize(tr[1], lemmatizer) in ont_rels)
#     conformance = num_conformant / len(triples)
#     return conformance, 1 - conformance


# def evaluate_and_save_results(ground_truth_data, ontology, model_data, output_file):
#     lemmatizer = WordNetLemmatizer()
#     results = []

#     for gt_entry, model_entry in zip(ground_truth_data, model_data):
#         if not gt_entry.get("triples"):
#             continue

#         gt_triples = [[tr["sub"], tr["rel"], tr["obj"]] for tr in gt_entry["triples"]]
#         system_triples = [[tr["sub"], tr["rel"], tr["obj"]] for tr in model_entry["triples"]]

#         gt_relations = {lemmatize_and_normalize(tr[1], lemmatizer) for tr in gt_triples}
#         filtered_system_triples = [
#             tr for tr in system_triples if lemmatize_and_normalize(tr[1], lemmatizer) in gt_relations
#         ]

#         normalized_gt_triples = {normalize_triple(tr[0], tr[1], tr[2], lemmatizer) for tr in gt_triples}
#         normalized_system_triples = {
#             normalize_triple(tr[0], tr[1], tr[2], lemmatizer) for tr in filtered_system_triples
#         }

#         precision, recall, f1 = calculate_precision_recall_f1(normalized_gt_triples, normalized_system_triples)
#         ont_conformance, rel_hallucination = get_ontology_conformance(ontology, system_triples)
#         subj_hallucination, obj_hallucination = get_subject_object_hallucinations(
#             lemmatizer, ontology, gt_entry["sent"], system_triples
#         )

#         result = {
#             "id": gt_entry["id"],
#             "precision": f"{precision:.2f}",
#             "recall": f"{recall:.2f}",
#             "f1": f"{f1:.2f}",
#             "onto_conf": f"{ont_conformance:.2f}",
#             "rel_halluc": f"{rel_hallucination:.2f}",
#             "sub_halluc": f"{subj_hallucination:.2f}",
#             "obj_halluc": f"{obj_hallucination:.2f}",
#             "llm_triples": system_triples,
#             "filtered_llm_triples": filtered_system_triples,
#             "gt_triples": gt_triples,
#             "sent": gt_entry["sent"],
#         }

#         results.append(result)

#     with open(output_file, "w") as f:
#         for res in results:
#             f.write(json.dumps(res) + "\n")


# def read_jsonl(file_path, required_keys=None):
#     with open(file_path, "r") as file:
#         data = [json.loads(line) for line in file]
#         if required_keys:
#             for entry in data:
#                 if not all(key in entry for key in required_keys):
#                     raise ValueError(f"Missing keys in entry: {entry}")
#         return data


# def read_ontology_json(json_path):
#     with open(json_path) as file:
#         return json.load(file)

# def run_evaluations_for_all_categories_dbpedia():
#     categories = [
#         "university",
#         "musicalwork",
#         "airport",
#         "building",
#         "athlete",
#         "politician",
#         "company",
#         "celestialbody",
#         "astronaut",
#         "comicscharacter",
#         "meanoftransportation",
#         "monument",
#         "food",
#         "writtenwork",
#         "sportsteam",
#         "city",
#         "artist",
#         "scientist",
#         "film",
#     ]

#     for i, category in enumerate(categories, start=1):
#         print(f"\n=== Running Evaluation for Category: {i} - {category} ===")

#         output_filepath = (
#             f"/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Mistral/"
#             f"ont_{i}_{category}_llm_stats_improved.jsonl"
#         )
#         ground_truth_filepath = (
#             f"/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/input/dbpedia/ground_truth/"
#             f"ont_{i}_{category}_ground_truth.jsonl"
#         )
#         ontology_filepath = (
#             f"/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/input/dbpedia/ontology/"
#             f"{i}_{category}_ontology.json"
#         )
#         model_response_filepath = (
#             f"/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/llm_responses/Mistral/"
#             f"ont_{i}_{category}_llm_response_improved.jsonl"
#         )

#         try:
#             ground_truth_data = read_jsonl(ground_truth_filepath)
#             ontology_data = read_ontology_json(ontology_filepath)
#             model_data = read_jsonl(model_response_filepath, required_keys=["id", "triples"])

#             evaluate_and_save_results(ground_truth_data, ontology_data, model_data, output_filepath)
#             print(f"✅ Successfully evaluated and saved results for {category}")
#         except Exception as e:
#             print(f"❌ Error processing category '{category}': {e}")


# if __name__ == "__main__":
#     run_evaluations_for_all_categories_dbpedia()



from __future__ import annotations

import json
import re

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Downloads (safe to keep here; consider moving into main() if you don't want it on import)
nltk.download("punkt")
nltk.download("wordnet")


def lemmatize_and_normalize(text: str, lemmatizer: WordNetLemmatizer) -> str:
    text = text.lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = text.strip()
    tokens = word_tokenize(text)
    lemmatized = "".join(lemmatizer.lemmatize(token) for token in tokens)
    return lemmatized


def normalize_triple(
    sub_label: str,
    rel_label: str,
    obj_label: str,
    lemmatizer: WordNetLemmatizer,
    stem_rel: bool = False,
) -> str:
    sub_label = lemmatize_and_normalize(sub_label, lemmatizer)
    obj_label = lemmatize_and_normalize(obj_label, lemmatizer)

    rel_label_clean = rel_label.lower().replace("_", " ")
    rel_label_clean = re.sub(r"[^a-z0-9 ]", "", rel_label_clean).strip()
    if stem_rel:
        rel_label_clean = " ".join(lemmatizer.lemmatize(w) for w in word_tokenize(rel_label_clean))
    rel_label_clean = re.sub(r"\s+", "", rel_label_clean)

    return f"{sub_label}{rel_label_clean}{obj_label}"


def calculate_precision_recall_f1(gold_set: set[str], pred_set: set[str]) -> tuple[float, float, float]:
    if not pred_set:
        return 0.0, 0.0, 0.0
    intersection = gold_set.intersection(pred_set)
    p = len(intersection) / len(pred_set)
    r = len(intersection) / len(gold_set) if gold_set else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def get_subject_object_hallucinations(
    lemmatizer: WordNetLemmatizer,
    ontology: dict,
    sentence: str,
    triples: list[list[str]],
) -> tuple[float, float]:
    if not triples:
        return 0.0, 0.0

    extended_sentence = sentence + " " + " ".join(c["label"] for c in ontology.get("concepts", []))
    normalized_sentence = lemmatize_and_normalize(extended_sentence, lemmatizer)

    subj_halluc, obj_halluc = 0, 0
    for sub, _rel, obj in triples:
        norm_sub = lemmatize_and_normalize(sub, lemmatizer)
        norm_obj = lemmatize_and_normalize(obj, lemmatizer)
        if norm_sub not in normalized_sentence:
            subj_halluc += 1
        if norm_obj not in normalized_sentence:
            obj_halluc += 1

    return subj_halluc / len(triples), obj_halluc / len(triples)


def get_ontology_conformance(ontology: dict, triples: list[list[str]]) -> tuple[float, float]:
    if not triples:
        return 1.0, 0.0

    lemmatizer = WordNetLemmatizer()
    ont_rels = {lemmatize_and_normalize(rel["label"], lemmatizer) for rel in ontology.get("relations", [])}
    num_conformant = sum(1 for tr in triples if lemmatize_and_normalize(tr[1], lemmatizer) in ont_rels)
    conformance = num_conformant / len(triples)
    return conformance, 1.0 - conformance


def evaluate_and_save_results(
    ground_truth_data: list[dict],
    ontology: dict,
    model_data: list[dict],
    output_file: str,
) -> None:
    lemmatizer = WordNetLemmatizer()
    results: list[dict] = []

    for gt_entry, model_entry in zip(ground_truth_data, model_data):
        if not gt_entry.get("triples"):
            continue

        gt_triples = [[tr["sub"], tr["rel"], tr["obj"]] for tr in gt_entry["triples"]]
        system_triples = [[tr["sub"], tr["rel"], tr["obj"]] for tr in model_entry["triples"]]

        gt_relations = {lemmatize_and_normalize(tr[1], lemmatizer) for tr in gt_triples}
        filtered_system_triples = [
            tr for tr in system_triples if lemmatize_and_normalize(tr[1], lemmatizer) in gt_relations
        ]

        normalized_gt_triples = {normalize_triple(tr[0], tr[1], tr[2], lemmatizer) for tr in gt_triples}
        normalized_system_triples = {
            normalize_triple(tr[0], tr[1], tr[2], lemmatizer) for tr in filtered_system_triples
        }

        precision, recall, f1 = calculate_precision_recall_f1(normalized_gt_triples, normalized_system_triples)
        ont_conformance, rel_hallucination = get_ontology_conformance(ontology, system_triples)
        subj_hallucination, obj_hallucination = get_subject_object_hallucinations(
            lemmatizer, ontology, gt_entry["sent"], system_triples
        )

        result = {
            "id": gt_entry["id"],
            "precision": f"{precision:.2f}",
            "recall": f"{recall:.2f}",
            "f1": f"{f1:.2f}",
            "onto_conf": f"{ont_conformance:.2f}",
            "rel_halluc": f"{rel_hallucination:.2f}",
            "sub_halluc": f"{subj_hallucination:.2f}",
            "obj_halluc": f"{obj_hallucination:.2f}",
            "llm_triples": system_triples,
            "filtered_llm_triples": filtered_system_triples,
            "gt_triples": gt_triples,
            "sent": gt_entry["sent"],
        }
        results.append(result)

    with open(output_file, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")


def read_jsonl(file_path: str, required_keys: list[str] | None = None) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]
    if required_keys:
        for entry in data:
            if not all(key in entry for key in required_keys):
                raise ValueError(f"Missing keys in entry: {entry}")
    return data


def read_ontology_json(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def run_evaluations_for_all_categories_dbpedia() -> None:
    categories = ["movie", "music", "sport", "book", "military", "computer", "space", "politics", "nature", "culture"]

    for i, category in enumerate(categories, start=1):
        print(f"\n=== Running Evaluation for Category: {i} - {category} ===")

        output_filepath = (
            "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/wikidata/individual_statics/Mistral/"
            f"ont_{i}_{category}_llm_stats_improved.jsonl"
        )
        ground_truth_filepath = (
            "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/input/wikidata/ground_truth/"
            f"ont_{i}_{category}_ground_truth.jsonl"
        )
        ontology_filepath = (
            "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/input/wikidata/ontology/"
            f"{i}_{category}_ontology.json"
        )
        model_response_filepath = (
            "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/wikidata/llm_responses/Mistral/"
            f"ont_{i}_{category}_llm_response_improved.jsonl"
        )

        try:
            ground_truth_data = read_jsonl(ground_truth_filepath)
            ontology_data = read_ontology_json(ontology_filepath)
            model_data = read_jsonl(model_response_filepath, required_keys=["id", "triples"])

            evaluate_and_save_results(ground_truth_data, ontology_data, model_data, output_filepath)
            print(f"✅ Successfully evaluated and saved results for {category}")
        except Exception as e:
            print(f"❌ Error processing category '{category}': {e}")


if __name__ == "__main__":
    run_evaluations_for_all_categories_dbpedia()
