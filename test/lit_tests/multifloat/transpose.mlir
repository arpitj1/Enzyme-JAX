// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32" %s | FileCheck %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=CHECK-LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=CHECK-TUPLE %s

func.func @main(%arg0: tensor<2x3xf64>) -> tensor<3x2xf64> {
  // CHECK-LABEL: @main
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<2x3xf64> to tensor<2x2x3xf32>
  // CHECK: %[[TRANS:.*]] = stablehlo.transpose %[[CAST]], dims = [0, 2, 1] : (tensor<2x2x3xf32>) -> tensor<2x3x2xf32>
  // CHECK: %[[OUT:.*]] = builtin.unrealized_conversion_cast %[[TRANS]] : tensor<2x3x2xf32> to tensor<3x2xf64>
  // CHECK: return %[[OUT]] : tensor<3x2xf64>

  // CHECK-LAST-LABEL: @main
  // CHECK-LAST: %[[CAST:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<2x3xf64> to tensor<2x3x2xf32>
  // CHECK-LAST: %[[TRANS:.*]] = stablehlo.transpose %[[CAST]], dims = [1, 0, 2] : (tensor<2x3x2xf32>) -> tensor<3x2x2xf32>
  // CHECK-LAST: %[[OUT:.*]] = builtin.unrealized_conversion_cast %[[TRANS]] : tensor<3x2x2xf32> to tensor<3x2xf64>
  // CHECK-LAST: return %[[OUT]] : tensor<3x2xf64>

  // CHECK-TUPLE-LABEL: @main
  // CHECK-TUPLE: %[[CAST:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<2x3xf64> to tuple<tensor<2x3xf32>, tensor<2x3xf32>>
  // CHECK-TUPLE: %[[HIGH:.*]] = stablehlo.get_tuple_element %[[CAST]][0] : (tuple<tensor<2x3xf32>, tensor<2x3xf32>>) -> tensor<2x3xf32>
  // CHECK-TUPLE: %[[LOW:.*]] = stablehlo.get_tuple_element %[[CAST]][1] : (tuple<tensor<2x3xf32>, tensor<2x3xf32>>) -> tensor<2x3xf32>
  // CHECK-TUPLE: %[[TRANS_HIGH:.*]] = stablehlo.transpose %[[HIGH]], dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
  // CHECK-TUPLE: %[[TRANS_LOW:.*]] = stablehlo.transpose %[[LOW]], dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
  // CHECK-TUPLE: %[[PACKED:.*]] = stablehlo.tuple %[[TRANS_HIGH]], %[[TRANS_LOW]] : tuple<tensor<3x2xf32>, tensor<3x2xf32>>
  // CHECK-TUPLE: %[[OUT:.*]] = builtin.unrealized_conversion_cast %[[PACKED]] : tuple<tensor<3x2xf32>, tensor<3x2xf32>> to tensor<3x2xf64>
  // CHECK-TUPLE: return %[[OUT]] : tensor<3x2xf64>
  %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x3xf64>) -> tensor<3x2xf64>
  return %0 : tensor<3x2xf64>
}
