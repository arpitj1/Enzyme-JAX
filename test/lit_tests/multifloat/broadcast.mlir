// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32" %s | FileCheck %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=CHECK-LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=CHECK-TUPLE %s

func.func @main(%arg0: tensor<4xf64>) -> tensor<4x5xf64> {
  // CHECK-LABEL: @main
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<4xf64> to tensor<2x4xf32>
  // CHECK: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[CAST]], dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4x5xf32>
  // CHECK: %[[OUT:.*]] = builtin.unrealized_conversion_cast %[[BCAST]] : tensor<2x4x5xf32> to tensor<4x5xf64>
  // CHECK: return %[[OUT]] : tensor<4x5xf64>

  // CHECK-LAST-LABEL: @main
  // CHECK-LAST: %[[CAST:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<4xf64> to tensor<4x2xf32>
  // CHECK-LAST: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[CAST]], dims = [0, 2] : (tensor<4x2xf32>) -> tensor<4x5x2xf32>
  // CHECK-LAST: %[[OUT:.*]] = builtin.unrealized_conversion_cast %[[BCAST]] : tensor<4x5x2xf32> to tensor<4x5xf64>
  // CHECK-LAST: return %[[OUT]] : tensor<4x5xf64>

  // CHECK-TUPLE-LABEL: @main
  // CHECK-TUPLE: %[[CAST:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<4xf64> to tuple<tensor<4xf32>, tensor<4xf32>>
  // CHECK-TUPLE: %[[HIGH:.*]] = stablehlo.get_tuple_element %[[CAST]][0] : (tuple<tensor<4xf32>, tensor<4xf32>>) -> tensor<4xf32>
  // CHECK-TUPLE: %[[LOW:.*]] = stablehlo.get_tuple_element %[[CAST]][1] : (tuple<tensor<4xf32>, tensor<4xf32>>) -> tensor<4xf32>
  // CHECK-TUPLE: %[[BCAST_HIGH:.*]] = stablehlo.broadcast_in_dim %[[HIGH]], dims = [0] : (tensor<4xf32>) -> tensor<4x5xf32>
  // CHECK-TUPLE: %[[BCAST_LOW:.*]] = stablehlo.broadcast_in_dim %[[LOW]], dims = [0] : (tensor<4xf32>) -> tensor<4x5xf32>
  // CHECK-TUPLE: %[[PACKED:.*]] = stablehlo.tuple %[[BCAST_HIGH]], %[[BCAST_LOW]] : tuple<tensor<4x5xf32>, tensor<4x5xf32>>
  // CHECK-TUPLE: %[[OUT:.*]] = builtin.unrealized_conversion_cast %[[PACKED]] : tuple<tensor<4x5xf32>, tensor<4x5xf32>> to tensor<4x5xf64>
  // CHECK-TUPLE: return %[[OUT]] : tensor<4x5xf64>
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<4xf64>) -> tensor<4x5xf64>
  return %0 : tensor<4x5xf64>
}
