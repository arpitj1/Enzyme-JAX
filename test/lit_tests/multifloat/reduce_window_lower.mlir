// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=1" %s | FileCheck %s

func.func @reduce_window_sum(%arg0: tensor<4x4xf64>) -> tensor<4x4xf64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 1, 1>,
    padding = dense<[[2, 0], [0, 0]]> : tensor<2x2xi64>
  }> ({
  ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
    %1 = stablehlo.add %arg1, %arg2 : tensor<f64>
    stablehlo.return %1 : tensor<f64>
  }) : (tensor<4x4xf64>, tensor<f64>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}

// CHECK-LABEL: func.func @reduce_window_sum
// CHECK: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK: %[[PAD_VAL:.*]] = builtin.unrealized_conversion_cast %[[CST]] : tensor<f64> to tensor<f32>
// CHECK: %[[PADDED:.*]] = stablehlo.pad %{{.*}}, %[[PAD_VAL]], low = [2, 0], high = [0, 0], interior = [0, 0]
// CHECK: %[[S0:.*]] = stablehlo.slice %[[PADDED]] [0:4, 0:4]
// CHECK: %[[S1:.*]] = stablehlo.slice %[[PADDED]] [1:5, 0:4]
// CHECK: %[[S2:.*]] = stablehlo.slice %[[PADDED]] [2:6, 0:4]
// CHECK: %[[ADD0:.*]] = stablehlo.add %[[S0]], %[[S1]]
// CHECK: %[[ADD1:.*]] = stablehlo.add %[[ADD0]], %[[S2]]
// CHECK: return %{{.*}}
