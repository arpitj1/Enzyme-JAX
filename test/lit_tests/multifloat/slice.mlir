// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32" %s | FileCheck %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=CHECK-LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=CHECK-TUPLE %s

func.func @main(%arg0: tensor<5xf64>) -> tensor<3xf64> {
  // CHECK-LABEL: @main
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<5xf64> to tensor<2x5xf32>
  // CHECK: %[[SLICE:.*]] = stablehlo.slice %[[CAST]] [0:2, 1:4] : (tensor<2x5xf32>) -> tensor<2x3xf32>
  // CHECK: %[[OUT:.*]] = builtin.unrealized_conversion_cast %[[SLICE]] : tensor<2x3xf32> to tensor<3xf64>
  // CHECK: return %[[OUT]] : tensor<3xf64>

  // CHECK-LAST-LABEL: @main
  // CHECK-LAST: %[[CAST:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<5xf64> to tensor<5x2xf32>
  // CHECK-LAST: %[[SLICE:.*]] = stablehlo.slice %[[CAST]] [1:4, 0:2] : (tensor<5x2xf32>) -> tensor<3x2xf32>
  // CHECK-LAST: %[[OUT:.*]] = builtin.unrealized_conversion_cast %[[SLICE]] : tensor<3x2xf32> to tensor<3xf64>
  // CHECK-LAST: return %[[OUT]] : tensor<3xf64>

  // CHECK-TUPLE-LABEL: @main
  // CHECK-TUPLE: %[[CAST:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<5xf64> to tuple<tensor<5xf32>, tensor<5xf32>>
  // CHECK-TUPLE: %[[HIGH:.*]] = stablehlo.get_tuple_element %[[CAST]][0] : (tuple<tensor<5xf32>, tensor<5xf32>>) -> tensor<5xf32>
  // CHECK-TUPLE: %[[LOW:.*]] = stablehlo.get_tuple_element %[[CAST]][1] : (tuple<tensor<5xf32>, tensor<5xf32>>) -> tensor<5xf32>
  // CHECK-TUPLE: %[[SLICE_HIGH:.*]] = stablehlo.slice %[[HIGH]] [1:4] : (tensor<5xf32>) -> tensor<3xf32>
  // CHECK-TUPLE: %[[SLICE_LOW:.*]] = stablehlo.slice %[[LOW]] [1:4] : (tensor<5xf32>) -> tensor<3xf32>
  // CHECK-TUPLE: %[[PACKED:.*]] = stablehlo.tuple %[[SLICE_HIGH]], %[[SLICE_LOW]] : tuple<tensor<3xf32>, tensor<3xf32>>
  // CHECK-TUPLE: %[[OUT:.*]] = builtin.unrealized_conversion_cast %[[PACKED]] : tuple<tensor<3xf32>, tensor<3xf32>> to tensor<3xf64>
  // CHECK-TUPLE: return %[[OUT]] : tensor<3xf64>
  %0 = stablehlo.slice %arg0 [1:4] : (tensor<5xf64>) -> tensor<3xf64>
  return %0 : tensor<3xf64>
}
