import torch

def display_results(predictions):
    print("\n--- Output Layer ---")
    for task, result in predictions.items():
        if isinstance(result, torch.Tensor):
            # Convert tensor to a list or a single value
            result_to_display = result.tolist()
            if len(result_to_display) == 1:
                # If it's a single-element list, just show the element
                result_to_display = result_to_display[0]
        else:
            result_to_display = result
        print(f"{task}: {result_to_display}")