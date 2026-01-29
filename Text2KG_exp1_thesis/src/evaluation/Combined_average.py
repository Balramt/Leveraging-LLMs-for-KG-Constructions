# Below code for Dbpedia LLama dataset


# from __future__ import annotations

# import json


# def read_jsonl_file(filepath):
#     data = []
#     with open(filepath, "r", encoding="utf-8") as file:
#         for line in file:
#             data.append(json.loads(line.strip()))
#     return data


# def extract_numerical_data(data):
#     numerical_data = {
#         "precision": [],
#         "recall": [],
#         "f1": [],
#         "onto_conf": [],
#         "rel_halluc": [],
#         "sub_halluc": [],
#         "obj_halluc": [],
#     }

#     for entry in data:
#         numerical_data["precision"].append(float(entry.get("precision", 0.0)))
#         numerical_data["recall"].append(float(entry.get("recall", 0.0)))
#         numerical_data["f1"].append(float(entry.get("f1", 0.0)))
#         numerical_data["onto_conf"].append(float(entry.get("onto_conf", 0.0)))
#         numerical_data["rel_halluc"].append(float(entry.get("rel_halluc", 0.0)))
#         numerical_data["sub_halluc"].append(float(entry.get("sub_halluc", 0.0)))
#         numerical_data["obj_halluc"].append(float(entry.get("obj_halluc", 0.0)))

#     return numerical_data


# def calculate_averages(numerical_data):
#     averages = {
#         "avg_precision": 0.0,
#         "avg_recall": 0.0,
#         "avg_f1": 0.0,
#         "avg_onto_conf": 0.0,
#         "avg_rel_halluc": 0.0,
#         "avg_sub_halluc": 0.0,
#         "avg_obj_halluc": 0.0,
#     }

#     for key, values in numerical_data.items():
#         avg_key = f"avg_{key}"
#         if values:
#             averages[avg_key] = sum(values) / len(values)

#     return averages


# def save_to_jsonl(data, output_filepath):
#     with open(output_filepath, "w", encoding="utf-8") as file:
#         for record in data:
#             json.dump(record, file)
#             file.write("\n")


# def process_multiple_files(files, output_filepath):
#     all_results = []

#     for filepath, ontology in files:
#         data = read_jsonl_file(filepath)
#         numerical_data = extract_numerical_data(data)
#         averages_all = calculate_averages(numerical_data)
#         averages_all.update({"onto": ontology, "type": "all_test_cases"})
#         all_results.append(averages_all)

#     save_to_jsonl(all_results, output_filepath)


# files = [
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Llama/ont_1_university_llm_stats_improved.jsonl", "1_university"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Llama/ont_2_musicalwork_llm_stats_improved.jsonl", "2_musicalwork"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Llama/ont_3_airport_llm_stats_improved.jsonl", "3_airport"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Llama/ont_4_building_llm_stats_improved.jsonl", "4_building"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Llama/ont_5_athlete_llm_stats_improved.jsonl", "5_athlete"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Llama/ont_6_politician_llm_stats_improved.jsonl", "6_politician"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Llama/ont_7_company_llm_stats_improved.jsonl", "7_company"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Llama/ont_8_celestialbody_llm_stats_improved.jsonl", "8_celestialbody"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Llama/ont_9_astronaut_llm_stats_improved.jsonl", "9_astronaut"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Llama/ont_10_comicscharacter_llm_stats_improved.jsonl", "10_comicscharacter"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Llama/ont_11_meanoftransportation_llm_stats_improved.jsonl", "11_meanoftransportation"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Llama/ont_12_monument_llm_stats_improved.jsonl", "12_monument"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Llama/ont_13_food_llm_stats_improved.jsonl", "13_food"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Llama/ont_14_writtenwork_llm_stats_improved.jsonl", "14_writtenwork"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Llama/ont_15_sportsteam_llm_stats_improved.jsonl", "15_sportsteam"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Llama/ont_16_city_llm_stats_improved.jsonl", "16_city"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Llama/ont_17_artist_llm_stats_improved.jsonl", "17_artist"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Llama/ont_18_scientist_llm_stats_improved.jsonl", "18_scientist"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Llama/ont_19_film_llm_stats_improved.jsonl", "19_film"),
# ]

# OUTPUT_FILEPATH = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Llama_overall_average2.jsonl"


# if __name__ == "__main__":
#     process_multiple_files(files, OUTPUT_FILEPATH)



#below code is for dbpedia Mistral dataset

# from __future__ import annotations

# import json


# def read_jsonl_file(filepath):
#     data = []
#     with open(filepath, "r", encoding="utf-8") as file:
#         for line in file:
#             data.append(json.loads(line.strip()))
#     return data


# def extract_numerical_data(data):
#     numerical_data = {
#         "precision": [],
#         "recall": [],
#         "f1": [],
#         "onto_conf": [],
#         "rel_halluc": [],
#         "sub_halluc": [],
#         "obj_halluc": [],
#     }

#     for entry in data:
#         numerical_data["precision"].append(float(entry.get("precision", 0.0)))
#         numerical_data["recall"].append(float(entry.get("recall", 0.0)))
#         numerical_data["f1"].append(float(entry.get("f1", 0.0)))
#         numerical_data["onto_conf"].append(float(entry.get("onto_conf", 0.0)))
#         numerical_data["rel_halluc"].append(float(entry.get("rel_halluc", 0.0)))
#         numerical_data["sub_halluc"].append(float(entry.get("sub_halluc", 0.0)))
#         numerical_data["obj_halluc"].append(float(entry.get("obj_halluc", 0.0)))

#     return numerical_data


# def calculate_averages(numerical_data):
#     averages = {
#         "avg_precision": 0.0,
#         "avg_recall": 0.0,
#         "avg_f1": 0.0,
#         "avg_onto_conf": 0.0,
#         "avg_rel_halluc": 0.0,
#         "avg_sub_halluc": 0.0,
#         "avg_obj_halluc": 0.0,
#     }

#     for key, values in numerical_data.items():
#         avg_key = f"avg_{key}"
#         if values:
#             averages[avg_key] = sum(values) / len(values)

#     return averages


# def save_to_jsonl(data, output_filepath):
#     with open(output_filepath, "w", encoding="utf-8") as file:
#         for record in data:
#             json.dump(record, file)
#             file.write("\n")


# def process_multiple_files(files, output_filepath):
#     all_results = []

#     for filepath, ontology in files:
#         data = read_jsonl_file(filepath)
#         numerical_data = extract_numerical_data(data)
#         averages_all = calculate_averages(numerical_data)
#         averages_all.update({"onto": ontology, "type": "all_test_cases"})
#         all_results.append(averages_all)

#     save_to_jsonl(all_results, output_filepath)


# files = [
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/wikidata/individual_statics/Mistral/ont_1_university_llm_stats_improved.jsonl", "1_university"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/wikidata/individual_statics/Mistral/ont_2_musicalwork_llm_stats_improved.jsonl", "2_musicalwork"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/wikidata/individual_statics/Mistral/ont_3_airport_llm_stats_improved.jsonl", "3_airport"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/wikidata/individual_statics/Mistral/ont_4_building_llm_stats_improved.jsonl", "4_building"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Mistral/ont_5_athlete_llm_stats_improved.jsonl", "5_athlete"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Mistral/ont_6_politician_llm_stats_improved.jsonl", "6_politician"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Mistral/ont_7_company_llm_stats_improved.jsonl", "7_company"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Mistral/ont_8_celestialbody_llm_stats_improved.jsonl", "8_celestialbody"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Mistral/ont_9_astronaut_llm_stats_improved.jsonl", "9_astronaut"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Mistral/ont_10_comicscharacter_llm_stats_improved.jsonl", "10_comicscharacter"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Mistral/ont_11_meanoftransportation_llm_stats_improved.jsonl", "11_meanoftransportation"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Mistral/ont_12_monument_llm_stats_improved.jsonl", "12_monument"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Mistral/ont_13_food_llm_stats_improved.jsonl", "13_food"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Mistral/ont_14_writtenwork_llm_stats_improved.jsonl", "14_writtenwork"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Mistral/ont_15_sportsteam_llm_stats_improved.jsonl", "15_sportsteam"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Mistral/ont_16_city_llm_stats_improved.jsonl", "16_city"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Mistral/ont_17_artist_llm_stats_improved.jsonl", "17_artist"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Mistral/ont_18_scientist_llm_stats_improved.jsonl", "18_scientist"),
#     ("/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/individual_statics/Mistral/ont_19_film_llm_stats_improved.jsonl", "19_film"),
# ]

# OUTPUT_FILEPATH = (
#     "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/dbpedia/"
#     "individual_statics/Mistral_overall_average.jsonl"
# )


# if __name__ == "__main__":
#     process_multiple_files(files, OUTPUT_FILEPATH)


#Below code is for wikidatafrom __future__ import annotations

import json


def read_jsonl_file(filepath: str) -> list[dict]:
    data = []
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


def extract_numerical_data(data: list[dict]) -> dict[str, list[float]]:
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


def calculate_averages(numerical_data: dict[str, list[float]]) -> dict[str, float]:
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


def save_to_jsonl(data: list[dict], output_filepath: str) -> None:
    with open(output_filepath, "w", encoding="utf-8") as file:
        for record in data:
            json.dump(record, file)
            file.write("\n")


def process_multiple_files(files: list[tuple[str, str]], output_filepath: str) -> None:
    all_results = []

    for filepath, ontology in files:
        data = read_jsonl_file(filepath)
        numerical_data = extract_numerical_data(data)
        averages_all = calculate_averages(numerical_data)
        averages_all.update({"onto": ontology, "type": "all_test_cases"})
        all_results.append(averages_all)

    save_to_jsonl(all_results, output_filepath)


# Wikidata categories (10 files total)
categories = [
    "movie",
    "music",
    "sport",
    "book",
    "military",
    "computer",
    "space",
    "politics",
    "nature",
    "culture",
]

# Build the 10 wikidata stats file paths
files = [
    (
        "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/wikidata/individual_statics/Llama/"
        f"ont_{i}_{category}_llm_stats_improved.jsonl",
        f"{i}_{category}",
    )
    for i, category in enumerate(categories, start=1)
]

OUTPUT_FILEPATH = (
    "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/wikidata/"
    "individual_statics/Llama_overall_average.jsonl"
)

if __name__ == "__main__":
    process_multiple_files(files, OUTPUT_FILEPATH)
    print(f"✅ Saved overall averages for Wikidata (10 files) to:\n{OUTPUT_FILEPATH}")


