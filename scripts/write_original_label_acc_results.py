"""
Write original label ACC Table
"""

import json

# FIXME: update
file_names = [
    "results_original_label_acc_after_cleanse",
]

for file_name in file_names:

    input_file = f"results/{file_name}.json"
    with open(input_file, "r", encoding="utf-8") as jf:
        table = json.load(jf)

    output_file_acc_asr = f"results/{file_name}.txt"

    output_acc_asr_file_handle = open(output_file_acc_asr, "w", encoding="utf-8")

    classifiers = ["knn", "linear"]
    ssl_methods = ["byol", "mocov2", "simclr"]
    datasets = ["imagenet100", "cifar10", "cifar100"]
    triggers = ["htba", "ftrojan"]

    for classifier in classifiers:
        # output_acc_asr_file_handle.write(f"{classifier}\n")
        for method in ssl_methods:
            # output_acc_asr_file_handle.write(f"{method}\n")

            uncleansed_results = []
            cleansed_results = []
            for dataset in datasets:

                if classifier == "knn":
                    result_key_uncleansed = "knn_original_label_acc_800"
                    result_key_cleansed = "knn_original_label_acc"
                elif classifier == "linear":
                    result_key_uncleansed = "linear_original_label_acc_800"
                    result_key_cleansed = "linear_original_label_acc"

                for trigger in triggers:
                    # find the corresponding entry in the table
                    entity = next(
                        (
                            v
                            for v in table.values()
                            if v.get("method") == method
                            and v.get("dataset") == dataset
                            and v.get("trigger_type") == trigger
                        ),
                        None,
                    )
                    if not entity:
                        print(
                            f"Warning: No entry found for method={method}, dataset={dataset}, trigger={trigger}"
                        )
                        continue

                    entity_result_uncleansed = entity.get(result_key_uncleansed)
                    entity_result_cleansed = entity.get(result_key_cleansed)
                    if entity_result_uncleansed is None:
                        print(
                            f"Warning: No result for key {entity_result_uncleansed} in entry for method={method}, dataset={dataset}, trigger={trigger}"
                        )
                        continue
                    if entity_result_cleansed is None:
                        print(
                            f"Warning: No result for key {entity_result_cleansed} in entry for method={method}, dataset={dataset}, trigger={trigger}"
                        )
                        continue

                    uncleansed_results.append(
                        entity_result_uncleansed
                        if entity_result_uncleansed is not None
                        else "N/A"
                    )
                    cleansed_results.append(
                        entity_result_cleansed
                        if entity_result_cleansed is not None
                        else "N/A"
                    )

            output_acc_asr_file_handle.write(
                "\t".join(map(str, uncleansed_results)) + "\n"
            )
            output_acc_asr_file_handle.write(
                "\t".join(map(str, cleansed_results)) + "\n"
            )

        # Separate different classifiers with an extra newline
        output_acc_asr_file_handle.write("\n")
    output_acc_asr_file_handle.close()
