// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck %s

func.func @sqrt(%arg0: tensor<2xf64>) -> tensor<2xf64> {
  %0 = stablehlo.sqrt %arg0 : tensor<2xf64>
  return %0 : tensor<2xf64>
}
// CHECK: func @sqrt(%arg0: tensor<2xf64>) -> tensor<2xf64> {
// CHECK: %{{.*}} = builtin.unrealized_conversion_cast %arg0 : tensor<2xf64> to tensor<2x2xf32>
// CHECK: %{{.*}} = stablehlo.sqrt %{{.*}} : tensor<1x2xf32>
// CHECK: %{{.*}} = stablehlo.multiply %{{.*}}, %{{.*}} : tensor<1x2xf32>
// CHECK: %{{.*}} = stablehlo.subtract %{{.*}}, %{{.*}} : tensor<1x2xf32>
// CHECK: %{{.*}} = stablehlo.add %{{.*}}, %{{.*}} : tensor<1x2xf32>
// CHECK: %{{.*}} = stablehlo.divide %{{.*}}, %{{.*}} : tensor<1x2xf32>
// CHECK: %{{.*}} = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : tensor<2x2xf32> to tensor<2xf64>
// CHECK: return %{{.*}} : tensor<2xf64>
