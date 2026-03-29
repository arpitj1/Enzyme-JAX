// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck %s

func.func @reverse(%arg0: tensor<2xf64>) -> tensor<2xf64> {
  %0 = stablehlo.reverse %arg0, dims = [0] : tensor<2xf64>
  return %0 : tensor<2xf64>
}
// CHECK: func @reverse(%arg0: tensor<2xf64>) -> tensor<2xf64> {
// CHECK: %{{.*}} = builtin.unrealized_conversion_cast %arg0 : tensor<2xf64> to tensor<2x2xf32>
// CHECK: %{{.*}} = stablehlo.reverse %{{.*}}, dims = [1] : tensor<2x2xf32>
// CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : tensor<2x2xf32> to tensor<2xf64>
// CHECK: return %{{.*}} : tensor<2xf64>
