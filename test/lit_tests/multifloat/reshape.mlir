// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32" %s | FileCheck %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=CHECK-LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=CHECK-TUPLE %s

func.func @main(%arg0: tensor<4xf64>) -> tensor<2x2xf64> {
  // CHECK-LABEL: @main
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<4xf64> to tensor<2x4xf32>
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %[[CAST]] : (tensor<2x4xf32>) -> tensor<2x2x2xf32>
  // CHECK: %[[OUT:.*]] = builtin.unrealized_conversion_cast %[[RESHAPE]] : tensor<2x2x2xf32> to tensor<2x2xf64>
  // CHECK: return %[[OUT]] : tensor<2x2xf64>

  // CHECK-LAST-LABEL: @main
  // CHECK-LAST: %[[CAST:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<4xf64> to tensor<4x2xf32>
  // CHECK-LAST: %[[RESHAPE:.*]] = stablehlo.reshape %[[CAST]] : (tensor<4x2xf32>) -> tensor<2x2x2xf32>
  // CHECK-LAST: %[[OUT:.*]] = builtin.unrealized_conversion_cast %[[RESHAPE]] : tensor<2x2x2xf32> to tensor<2x2xf64>
  // CHECK-LAST: return %[[OUT]] : tensor<2x2xf64>

  // CHECK-TUPLE-LABEL: @main
  // CHECK-TUPLE: %[[CAST:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<4xf64> to tuple<tensor<4xf32>, tensor<4xf32>>
  // CHECK-TUPLE: %[[HIGH:.*]] = stablehlo.get_tuple_element %[[CAST]][0] : (tuple<tensor<4xf32>, tensor<4xf32>>) -> tensor<4xf32>
  // CHECK-TUPLE: %[[LOW:.*]] = stablehlo.get_tuple_element %[[CAST]][1] : (tuple<tensor<4xf32>, tensor<4xf32>>) -> tensor<4xf32>
  // CHECK-TUPLE: %[[RESHAPE_HIGH:.*]] = stablehlo.reshape %[[HIGH]] : (tensor<4xf32>) -> tensor<2x2xf32>
  // CHECK-TUPLE: %[[RESHAPE_LOW:.*]] = stablehlo.reshape %[[LOW]] : (tensor<4xf32>) -> tensor<2x2xf32>
  // CHECK-TUPLE: %[[PACKED:.*]] = stablehlo.tuple %[[RESHAPE_HIGH]], %[[RESHAPE_LOW]] : tuple<tensor<2x2xf32>, tensor<2x2xf32>>
  // CHECK-TUPLE: %[[OUT:.*]] = builtin.unrealized_conversion_cast %[[PACKED]] : tuple<tensor<2x2xf32>, tensor<2x2xf32>> to tensor<2x2xf64>
  // CHECK-TUPLE: return %[[OUT]] : tensor<2x2xf64>
  %0 = stablehlo.reshape %arg0 : (tensor<4xf64>) -> tensor<2x2xf64>
  return %0 : tensor<2x2xf64>
}
