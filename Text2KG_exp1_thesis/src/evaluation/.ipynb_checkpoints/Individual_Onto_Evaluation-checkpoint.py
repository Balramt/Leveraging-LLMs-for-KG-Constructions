from __future__ import annotations

import argparse
import json
import re
from typing import Any, Dict, List, Tuple

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def calculate_precision_recall_f1(gold, pred):
    if len(pred) == 0:
        return 0, 0, 0
    p = len(gold.intersection(pred)) / len(pred)
    r = len(gold.intersection(pred)) / len(gold)
    f1 = 2 * ((p * r) / (p + r)) if (p + r) > 0 else 0
    return p, r, f1


def clean_entity_string(ps, entity):
    stemmed_entity = "".join([ps.stem(word) for word in word_tokenize(entity)])
    normalized_stemmed_entity = re.sub(r"(_|\s+)", "", stemmed_entity).lower()
    return normalized_stemmed_entity


def get_subject_object_hallucinations(ps, test_sentence, triples):
    if len(triples) == 0:
        return 0, 0

    stemmed_sentence = "".join([ps.stem(word) for word in word_tokenize(test_sentence)])
    normalized_stemmed_sentence = re.sub(r"(_|\s+)", "", stemmed_sentence).lower()

    num_subj_hallucinations, num_obj_hallucinations = 0, 0
    for triple in triples:
        normalized_stemmed_subject = clean_entity_string(ps, triple[0])
        normalized_stemmed_object = clean_entity_string(ps, triple[2])

        if normalized_stemmed_sentence.find(normalized_stemmed_subject) == -1:
            num_subj_hallucinations += 1
        if normalized_stemmed_sentence.find(normalized_stemmed_object) == -1:
            num_obj_hallucinations += 1

    subj_hallucination = num_subj_hallucinations / len(triples)
    obj_hallucination = num_obj_hallucinations / len(triples)
    return subj_hallucination, obj_hallucination


def get_ontology_conformance(ontology_rels, triples):
    if len(triples) == 0:
        return 1, 0
    num_rels_conformant = len([tr for tr in triples if tr[1] in ontology_rels])
    ont_conformance = num_rels_conformant / len(triples)
    rel_hallucination = 1 - ont_conformance
    return ont_conformance, rel_hallucination


def normalize_triple(sub_label, rel_label, obj_label):
    sub_label = re.sub(r"(_|\s+)", "", sub_label).lower()
    rel_label = re.sub(r"(_|\s+)", "", rel_label).lower()
    obj_label = re.sub(r"(_|\s+)", "", obj_label).lower()
    return f"{sub_label}{rel_label}{obj_label}"


def evaluate_and_save_results(ground_truth_data, model_data, output_file):
    ps = PorterStemmer()

    results = []
    for gt_entry, model_entry in zip(ground_truth_data, model_data):
        gt_triples = [[tr["sub"], tr["rel"], tr["obj"]] for tr in gt_entry["triples"]]
        system_triples = [[tr["sub"], tr["rel"], tr["obj"]] for tr in model_entry["triples"]]

        gt_relations = {tr[1].replace(" ", "_") for tr in gt_triples}
        filtered_system_triples = [tr for tr in system_triples if tr[1] in gt_relations]

        normalized_gt_triples = {normalize_triple(tr[0], tr[1], tr[2]) for tr in gt_triples}
        normalized_system_triples = {normalize_triple(tr[0], tr[1], tr[2]) for tr in filtered_system_triples}

        precision, recall, f1 = calculate_precision_recall_f1(normalized_gt_triples, normalized_system_triples)
        ont_conformance, rel_hallucination = get_ontology_conformance(gt_relations, system_triples)
        subj_hallucination, obj_hallucination = get_subject_object_hallucinations(ps, gt_entry["sent"], system_triples)

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

    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def read_ground_truth_jsonl(file_path):
    with open(file_path, "r") as file:
        data = [json.loads(line) for line in file]
        for entry in data:
            if not all(key in entry for key in ["id", "sent", "triples"]):
                raise ValueError(f"Entry missing required keys: {entry}")
        return data


def read_model_jsonl(file_path):
    with open(file_path, "r") as file:
        data = [json.loads(line) for line in file]
        for entry in data:
            if not all(key in entry for key in ["id", "triples"]):
                raise ValueError(f"Entry missing required keys: {entry}")
        return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="Wikidata/Evaluation_Statistics/ont_9_nature_llm_stats.jsonl")
    parser.add_argument("--ground_truth", default="Wikidata/Ground_Truth/ont_9_nature_ground_truth.jsonl")
    parser.add_argument("--model_response", default="Wikidata/Response/ont_9_nature_llm_response.jsonl")
    parser.add_argument("--download_punkt_tab", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.download_punkt_tab:
        nltk.download("punkt_tab")

    ground_truth_data = read_ground_truth_jsonl(args.ground_truth)
    model_data = read_model_jsonl(args.model_response)
    evaluate_and_save_results(ground_truth_data, model_data, args.output)


if __name__ == "__main__":
    main()
