// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32" %s | FileCheck %s

func.func @main(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>) -> tensor<4xf64> {
  // CHECK-LABEL: @main
  // CHECK: %[[LHS_CONCAT:.*]] = stablehlo.concatenate %arg0, %arg0, dim = 0 : (tensor<2xf64>, tensor<2xf64>) -> tensor<4xf64>
  // CHECK: %[[LHS_CAST:.*]] = builtin.unrealized_conversion_cast %[[LHS_CONCAT]] : tensor<4xf64> to tensor<2x4xf32>
  // CHECK: %[[RHS_CONCAT:.*]] = stablehlo.concatenate %arg1, %arg1, dim = 0 : (tensor<2xf64>, tensor<2xf64>) -> tensor<4xf64>
  // CHECK: %[[RHS_CAST:.*]] = builtin.unrealized_conversion_cast %[[RHS_CONCAT]] : tensor<4xf64> to tensor<2x4xf32>
  // CHECK: %[[LHS_HI:.*]] = stablehlo.slice %[[LHS_CAST]] [0:1, 0:4] : (tensor<2x4xf32>) -> tensor<1x4xf32>
  // CHECK: %[[RHS_HI:.*]] = stablehlo.slice %[[RHS_CAST]] [0:1, 0:4] : (tensor<2x4xf32>) -> tensor<1x4xf32>
  // CHECK: %[[ADD_HI:.*]] = stablehlo.add %[[LHS_HI]], %[[RHS_HI]] : tensor<1x4xf32>
  // Wait, my mental model for AddOpConversion was:
  // It extracts limbs, performs twoSum/fastTwoSum, and packs limbs!
  // If the input is are ConcatenateOp (which we just optimized), it will extract limbs from the concatenated inputs!
  // If we optimize it, the ConcatenateOp is on the INPUTS to the AddOp!
  // So `AddOp` operates on concatenated inputs!
  // And `AddOpConversion` will extract limbs from these inputs!
  // If the inputs were limb tensors (e.g. from functions args), they are `UnrealizedConversionCast`!
  // Let's see what happens!
  // Let's just create the test and see what it outputs!
  %0 = stablehlo.add %arg0, %arg1 : tensor<2xf64>
  %1 = stablehlo.add %arg0, %arg1 : tensor<2xf64>
  %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<2xf64>, tensor<2xf64>) -> tensor<4xf64>
  return %2 : tensor<4xf64>
}
