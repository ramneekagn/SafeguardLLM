from filters.internal.simple_internal_detector import SimpleInternalFilter
import torch
# --- Test Script --- AI 
def test_simple_internal_filter():
    batch_size = 4
    seq_len = 1
    n = 3  
    num_steps = 3  

    filter_instance = SimpleInternalFilter(target_layer="transformer.layers.0")

    dummy_outputs = [
        torch.ones(batch_size, seq_len, n) * (i + 1) 
        for i in range(num_steps)
    ]

    print("--- Simulating Hook Calls ---")
    for i, dummy_output in enumerate(dummy_outputs):
        print(f"Step {i+1}:")
        filter_instance.hook(module=None, input=None, output=dummy_output)
        
    assert len(filter_instance.activation_cache) == num_steps
    assert filter_instance.pos == num_steps

    print("\n--- Validating ---")
    results = filter_instance.validate()
    print(results)

if __name__ == "__main__":
    test_simple_internal_filter()