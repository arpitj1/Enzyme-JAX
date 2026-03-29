// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck %s

func.func @abs(%arg0: tensor<2xf64>) -> tensor<2xf64> {
  %0 = stablehlo.abs %arg0 : tensor<2xf64>
  return %0 : tensor<2xf64>
}
// CHECK: func @abs(%arg0: tensor<2xf64>) -> tensor<2xf64> {
// CHECK: %{{.*}} = builtin.unrealized_conversion_cast %arg0 : tensor<2xf64> to tensor<2x2xf32>
// CHECK: %{{.*}} = stablehlo.compare  GE, %{{.*}}, %{{.*}} : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x2xi1>
// CHECK: %{{.*}} = stablehlo.negate %{{.*}} : tensor<1x2xf32>
// CHECK: %{{.*}} = stablehlo.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<1x2xi1>, tensor<1x2xf32>
// CHECK: %{{.*}} = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : tensor<2x2xf32> to tensor<2xf64>
// CHECK: return %{{.*}} : tensor<2xf64>
