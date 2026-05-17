"""
Write ACC ASR Table -- write data
"""

import json

from pyparsing import Dict, Optional

input_file = "results/results_ours_poi8.json"  # FIXME: update
with open(input_file, "r", encoding="utf-8") as jf:
    table = json.load(jf)  # type: Dict[str, Dict[str, Optional[object]]]


output_file_acc_asr = "results/uncleansed.txt"  # FIXME: update
output_acc_asr_file_handle = open(output_file_acc_asr, "w", encoding="utf-8")

classifiers = ["knn", "linear"]
ssl_methods = ["byol", "mocov2", "simclr"]
datasets = ["imagenet100", "cifar10", "cifar100"]
triggers = ["htba", "ftrojan"]
metrics = ["clean_acc", "back_asr"]

for classifier in classifiers:
    # output_acc_asr_file_handle.write(f"{classifier}\n")
    for method in ssl_methods:
        # output_acc_asr_file_handle.write(f"{method}\n")

        for dataset in datasets:
            for metric in metrics:

                # if classifier == "knn":
                #     if metric == "clean_acc":
                #         result_key = "clean_acc_800"
                #     elif metric == "back_asr":
                #         result_key = "back_acc_800"
                # elif classifier == "linear":
                #     if metric == "clean_acc":
                #         result_key = "linear_ACC"
                #     elif metric == "back_asr":
                #         result_key = "linear_ASR"

                result_key = f"{classifier}_{metric}"

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

                    entity_result = entity.get(result_key)
                    if entity_result is None:
                        print(
                            f"Warning: No result for key {result_key} in entry for method={method}, dataset={dataset}, trigger={trigger}"
                        )
                        continue

                    # mean_val = table[dataset][trigger][method][classifier][metric][
                    #     "mean"
                    # ]
                    # std_val = table[dataset][trigger][method][classifier][metric]["std"]

                    # if isinstance(mean_val, (int, float)) and isinstance(
                    #     std_val, (int, float)
                    # ):

                    output_acc_asr_file_handle.write(f"{entity_result}\t")

        # Newline after each method
        output_acc_asr_file_handle.write("\n")

    # Separate different classifiers with an extra newline
    output_acc_asr_file_handle.write("\n")
