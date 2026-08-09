
# Fault Injection

This folder contains the implementation of the fault injection framework used in the paper.

The proposed implementation injects faults into the quantized INT8 activation tensors of the neural network using the PyTorch FX graph. The framework supports three fault models:

- Stuck-at-0 (SA-0)
- Stuck-at-1 (SA-1)
- Bit Flip (Transient Upset)

## Injection Method

Faults are injected after feature extraction by modifying the INT8 activation tensor using `torch.int_repr()`. The modified tensor is then reconstructed using the original quantization parameters (scale and zero-point) before continuing the inference process.

The fault injection process consists of:

1. Loading the trained quantized CNN.
2. Selecting the target activation tensor.
3. Randomly selecting the affected activation channels.
4. Generating the fault mask according to the selected fault model.
5. Injecting the fault using bitwise operations.
6. Reconstructing the quantized tensor.
7. Saving the faulty model for evaluation.

## Configurable Parameters

The following parameters can be configured before each fault injection campaign:

- Fault model (SA-0, SA-1, or Bit Flip)
- Target bit position(s)
- Percentage of affected activation channels
- Random seed
- Target CNN architecture

## Supported CNN Models

- ResNet18
- MobileNetV2
- RegNet-Y-800MF

## More Info

Tensor Insertion Point:

The fault injection is strategically placed after the flatten layer of the model. This is determined by the line if "flatten" in node.name: within the FX graph modification loop for different models. This location represents the consolidated feature vector passed to the decision-making (ANN) layers, making it a sensitive point for error propagation.

Bit-Numbering Convention:

The FI targets specific bit positions within the 8-bit unsigned integer (uint8) representation of the quantized activations. The target_bits variable  defines these positions. The convention used is standard: bit 0 is the Least Significant Bit (LSB), and bit 7 is the Most Significant Bit (MSB). The bit_val is calculated as sum(2**b for b in target_bits), which creates a mask where the specified bit positions are set to 1.

Signed INT8 Handling:

PyTorch's QNNPACK backend typically uses torch.qint8 for quantized tensors, which is a signed 8-bit integer. However, bitwise operations are most naturally performed on unsigned integers. The code handles this by first converting the quantized tensor's raw integer representation (torch.qint8) to its unsigned uint8 equivalent using int_repr_node = graph.call_method('int_repr', args=(node,)). This method extracts the underlying 8-bit integer values (which are effectively stored as unsigned, even if interpreted as signed in qint8 for arithmetic). The bitwise operations (OR, XOR, AND) are then applied to these uint8 values. After the fault is injected, the values are re-quantized, which implicitly handles the conversion back to qint8 if necessary.

Scale and Zero-Point Reconstruction:

After injecting the fault into the uint8 representation, the modified integer values must be converted back into a valid quantized tensor format. This is done using torch._make_per_tensor_quantized_tensor. This function requires the original scale and zero_point parameters that were determined during the initial quantization (PTQ or QAT). The code retrieves these from variables like target_scale and target_zero_point (e.g., target_scale = 0.05534251034259796, target_zero_point = 0). By reusing these original parameters, the re-quantized tensor (requant_node) retains its correct mapping to real-world floating-point values, ensuring compatibility with subsequent quantized layers in the model.

Clipping and Rounding Rules:

The torch._make_per_tensor_quantized_tensor function, when used for re-quantization, implicitly adheres to the clipping and rounding rules defined by the underlying quantization scheme (QNNPACK in this case). When a floating-point value is quantized to an integer, it's typically multiplied by the inverse of the scale, has the zero-point added, and then rounded to the nearest integer. If the resulting integer falls outside the valid range for the target integer type (e.g., 0-255 for uint8 or -128 to 127 for qint8), it is clipped to the minimum or maximum value of that range. The fault injection modifies the integer representation directly; when these modified integers are re-converted, they are within the valid 8-bit range, so further clipping is generally not needed on the integer side. The torch._make_per_tensor_quantized_tensor operation primarily handles mapping these integers back to the quantized tensor format with the correct scale and zero-point.

Fault-Selection Logic:

num_channels: This defines the total number of values in the 1D feature vector after the flatten layer (e.g., num_channels = 512). This number is specific to the model and dataset used.
percentage: This variable determines the proportion of the num_channels that will be affected by the fault.
num_to_fault: Calculated as math.ceil(percentage * num_channels), this is the actual count of channels to be faulty. math.ceil ensures at least one channel is selected if the percentage is very small.
target_channels: A list of indices of the channels where faults will be injected. These are selected randomly using torch.randperm(num_channels)[:num_to_fault].tolist(), ensuring a random but reproducible selection due to the seed being set.
fault_mask: A torch.uint8 tensor of shape (1, num_channels). For each ch in target_channels, the pre-calculated bit_val (representing the specific bits to flip/set) is placed into the fault_mask at that channel's position. This fault_mask is then registered as a buffer in the fault_model using fault_model.register_buffer('fault_mask', fault_mask.to(torch.uint8)) and used during the bitwise operations (OR, XOR, AND) to inject the faults.
