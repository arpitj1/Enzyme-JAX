// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first expansion-size=2" %s | FileCheck %s

func.func @convert_f64_to_f32(%arg0: tensor<4x4xf64>) -> tensor<4x4xf32> {
  %0 = stablehlo.convert %arg0 : (tensor<4x4xf64>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
// CHECK-LABEL: func.func @convert_f64_to_f32
// CHECK: %[[EXTRACTED:.*]] = stablehlo.slice %{{.*}} [0:1, 0:4, 0:4]
// CHECK: %[[RESHAPED:.*]] = stablehlo.reshape %[[EXTRACTED]]
// CHECK: return %[[RESHAPED]]

func.func @convert_f32_to_f64(%arg0: tensor<4x4xf32>) -> tensor<4x4xf64> {
  %0 = stablehlo.convert %arg0 : (tensor<4x4xf32>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}
// CHECK-LABEL: func.func @convert_f32_to_f64
// CHECK: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<1x4x4xf32>
// CHECK: %[[RESHAPED:.*]] = stablehlo.reshape %{{.*}} : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
// CHECK: %[[PACKED:.*]] = stablehlo.concatenate %[[RESHAPED]], %[[ZERO]], dim = 0 : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<2x4x4xf32>
// CHECK: %[[HIGH_EXT:.*]] = stablehlo.slice %[[PACKED]] [0:1, 0:4, 0:4] : (tensor<2x4x4xf32>) -> tensor<1x4x4xf32>
// CHECK: %[[LOW_EXT:.*]] = stablehlo.slice %[[PACKED]] [1:2, 0:4, 0:4] : (tensor<2x4x4xf32>) -> tensor<1x4x4xf32>
// CHECK: %[[HIGH_RESHAPED:.*]] = stablehlo.reshape %[[HIGH_EXT]] : (tensor<1x4x4xf32>) -> tensor<4x4xf32>
// CHECK: %[[LOW_RESHAPED:.*]] = stablehlo.reshape %[[LOW_EXT]] : (tensor<1x4x4xf32>) -> tensor<4x4xf32>
// CHECK: %[[HIGH_64:.*]] = stablehlo.convert %[[HIGH_RESHAPED]] : (tensor<4x4xf32>) -> tensor<4x4xf64>
// CHECK: %[[LOW_64:.*]] = stablehlo.convert %[[LOW_RESHAPED]] : (tensor<4x4xf32>) -> tensor<4x4xf64>
// CHECK: %[[SUM:.*]] = stablehlo.add %[[HIGH_64]], %[[LOW_64]] : tensor<4x4xf64>
// CHECK: return %[[SUM]]

func.func @convert_i32_to_f64(%arg0: tensor<4x4xi32>) -> tensor<4x4xf64> {
  %0 = stablehlo.convert %arg0 : (tensor<4x4xi32>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}
// CHECK-LABEL: func.func @convert_i32_to_f64
// CHECK: %[[HIGH:.*]] = stablehlo.convert %{{.*}} : (tensor<4x4xi32>) -> tensor<4x4xf32>
// CHECK: %[[HIGH_BACK:.*]] = stablehlo.convert %[[HIGH]] : (tensor<4x4xf32>) -> tensor<4x4xi32>
// CHECK: %[[REM:.*]] = stablehlo.subtract %{{.*}}, %[[HIGH_BACK]]
// CHECK: %[[LOW:.*]] = stablehlo.convert %[[REM]] : (tensor<4x4xi32>) -> tensor<4x4xf32>
// CHECK: %[[HIGH_RESHAPED:.*]] = stablehlo.reshape %[[HIGH]] : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
// CHECK: %[[LOW_RESHAPED:.*]] = stablehlo.reshape %[[LOW]] : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
// CHECK: %[[PACKED:.*]] = stablehlo.concatenate %[[HIGH_RESHAPED]], %[[LOW_RESHAPED]], dim = 0 : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<2x4x4xf32>
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[PACKED]] : tensor<2x4x4xf32> to tensor<4x4xf64>
// CHECK: return %[[CAST]]

func.func @convert_f64_to_i32(%arg0: tensor<4x4xf64>) -> tensor<4x4xi32> {
  %0 = stablehlo.convert %arg0 : (tensor<4x4xf64>) -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}
// CHECK-LABEL: func.func @convert_f64_to_i32
// CHECK: %[[EXTRACTED:.*]] = stablehlo.slice %{{.*}} [0:1, 0:4, 0:4]
// CHECK: %[[RESHAPED:.*]] = stablehlo.reshape %[[EXTRACTED]]
// CHECK: %[[CONVERTED:.*]] = stablehlo.convert %[[RESHAPED]] : (tensor<4x4xf32>) -> tensor<4x4xi32>
// CHECK: return %[[CONVERTED]]
