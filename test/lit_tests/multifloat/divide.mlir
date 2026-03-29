// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | FileCheck %s

func.func @divide(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>) -> tensor<2xf64> {
  // CHECK-LABEL: @divide
  // CHECK: %[[X_HI:.*]] = stablehlo.slice %{{.*}} [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
  // CHECK: %[[X_LO:.*]] = stablehlo.slice %{{.*}} [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
  // CHECK: %[[Y_HI:.*]] = stablehlo.slice %{{.*}} [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
  // CHECK: %[[Y_LO:.*]] = stablehlo.slice %{{.*}} [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
  // CHECK: %[[Z_HI:.*]] = stablehlo.divide %[[X_HI]], %[[Y_HI]]
  // CHECK: %[[P_HI:.*]], %[[P_LO:.*]] = "twoProdDekker"
  // CHECK: %{{.*}} = stablehlo.subtract %[[X_HI]], %{{.*}}
  // CHECK: %{{.*}} = stablehlo.divide %{{.*}}, %[[Y_HI]]
  // CHECK: %{{.*}} = stablehlo.add %[[Z_HI]], %{{.*}}
  
  %0 = stablehlo.divide %arg0, %arg1 : tensor<2xf64>
  return %0 : tensor<2xf64>
}
