"""
Write ACC ASR Table -- write data
"""

import json
from pyparsing import Dict, Optional

file_names = [
    "results_ablation_aug_1",
    "results_ablation_aug_2",
    "results_ablation_aug_3",
    "results_ablation_aug_4",
    "results_ablation_aug_5",
    "results_ablation_aug_6",
    "results_ablation_impure_p0c8",
    "results_ablation_impure_p1c7",
    "results_ablation_impure_p2c6",
    "results_ablation_impure_p3c5",
    "results_ablation_impure_p4c4",
    "results_ablation_impure_p5c3",
    "results_ablation_impure_p6c2",
    "results_ablation_impure_p7c1",
    "results_ablation_rm_channel_40",
    "results_ablation_rm_channel_100",
    "results_ablation_rm_channel_130",
    "results_ablation_view_1",
    "results_ablation_view_4",
    "results_ablation_view_16",
    "results_ablation_view_32",
    "results_ablation_view_128",
    "results_ablation_vote_channel_10",
    "results_ablation_vote_channel_20",
    "results_ablation_vote_channel_40",
    "results_ablation_vote_channel_50",
]  # FIXME: update

for file_name in file_names:

    input_file = f"results/{file_name}.json"
    with open(input_file, "r", encoding="utf-8") as jf:
        table = json.load(jf)  # type: Dict[str, Dict[str, Optional[object]]]

    output_file_acc_asr = f"results/{file_name}.txt"
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
    output_acc_asr_file_handle.close()
