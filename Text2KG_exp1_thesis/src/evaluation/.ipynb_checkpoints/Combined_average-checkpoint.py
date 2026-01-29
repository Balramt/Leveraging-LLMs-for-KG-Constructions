from __future__ import annotations

import json


def read_jsonl_file(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


def extract_numerical_data(data):
    numerical_data = {
        "precision": [],
        "recall": [],
        "f1": [],
        "onto_conf": [],
        "rel_halluc": [],
        "sub_halluc": [],
        "obj_halluc": [],
    }

    for entry in data:
        numerical_data["precision"].append(float(entry.get("precision", 0.0)))
        numerical_data["recall"].append(float(entry.get("recall", 0.0)))
        numerical_data["f1"].append(float(entry.get("f1", 0.0)))
        numerical_data["onto_conf"].append(float(entry.get("onto_conf", 0.0)))
        numerical_data["rel_halluc"].append(float(entry.get("rel_halluc", 0.0)))
        numerical_data["sub_halluc"].append(float(entry.get("sub_halluc", 0.0)))
        numerical_data["obj_halluc"].append(float(entry.get("obj_halluc", 0.0)))

    return numerical_data


def calculate_averages(numerical_data):
    averages = {
        "avg_precision": 0.0,
        "avg_recall": 0.0,
        "avg_f1": 0.0,
        "avg_onto_conf": 0.0,
        "avg_rel_halluc": 0.0,
        "avg_sub_halluc": 0.0,
        "avg_obj_halluc": 0.0,
    }

    for key, values in numerical_data.items():
        avg_key = f"avg_{key}"
        if values:
            averages[avg_key] = sum(values) / len(values)

    return averages


def save_to_jsonl(data, output_filepath):
    with open(output_filepath, "w", encoding="utf-8") as file:
        for record in data:
            json.dump(record, file)
            file.write("\n")


def process_multiple_files(files, output_filepath):
    all_results = []

    for filepath, ontology in files:
        data = read_jsonl_file(filepath)
        numerical_data = extract_numerical_data(data)

        averages_all = calculate_averages(numerical_data)
        averages_all.update({"onto": ontology, "type": "all_test_cases"})
        all_results.append(averages_all)

    save_to_jsonl(all_results, output_filepath)


files = [
    ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/evaluations_statistics/individual_statics/ont_1_university_llm_stats.jsonl", "1_university"),
    ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/evaluations_statistics/individual_statics/ont_2_musicalwork_llm_stats.jsonl", "2_musicalwork"),
    ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/evaluations_statistics/individual_statics/ont_3_airport_llm_stats.jsonl", "3_airport"),
    ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/evaluations_statistics/individual_statics/ont_4_building_llm_stats.jsonl", "4_building"),
    ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/evaluations_statistics/individual_statics/ont_5_athlete_llm_stats.jsonl", "5_athlete"),
    ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/evaluations_statistics/individual_statics/ont_6_politician_llm_stats.jsonl", "6_politician"),
    ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/evaluations_statistics/individual_statics/ont_7_company_llm_stats.jsonl", "7_company"),
    ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/evaluations_statistics/individual_statics/ont_8_celestialbody_llm_stats.jsonl", "8_celestialbody"),
    ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/evaluations_statistics/individual_statics/ont_9_astronaut_llm_stats.jsonl", "9_astronaut"),
    ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/evaluations_statistics/individual_statics/ont_10_comicscharacter_llm_stats.jsonl", "10_comicscharacter"),
    ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/evaluations_statistics/individual_statics/ont_11_meanoftransportation_llm_stats.jsonl", "11_meanoftransportation"),
    ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/evaluations_statistics/individual_statics/ont_12_monument_llm_stats.jsonl", "12_monument"),
    ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/evaluations_statistics/individual_statics/ont_13_food_llm_stats.jsonl", "13_food"),
    ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/evaluations_statistics/individual_statics/ont_14_writtenwork_llm_stats.jsonl", "14_writtenwork"),
    ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/evaluations_statistics/individual_statics/ont_15_sportsteam_llm_stats.jsonl", "15_sportsteam"),
    ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/evaluations_statistics/individual_statics/ont_16_city_llm_stats.jsonl", "16_city"),
    ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/evaluations_statistics/individual_statics/ont_17_artist_llm_stats.jsonl", "17_artist"),
    ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/evaluations_statistics/individual_statics/ont_18_scientist_llm_stats.jsonl", "18_scientist"),
    ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/evaluations_statistics/individual_statics/ont_19_film_llm_stats.jsonl", "19_film"),
]


OUTPUT_FILEPATH = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/evaluations_statistics/combined_averages1.jsnol"

process_multiple_files(files, OUTPUT_FILEPATH)
