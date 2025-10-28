def display_results(predictions):
    print("\n--- Output Layer ---")
    for task, result in predictions.items():
        print(f"{task}: {result}")