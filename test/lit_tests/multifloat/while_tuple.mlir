// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck %s

func.func @while(%arg0: tensor<f64>) -> tensor<f64> {
  %cst = stablehlo.constant dense<1.000000e+01> : tensor<f64>
  %0:2 = stablehlo.while(%iterArg0 = %arg0, %iterArg1 = %cst) : tensor<f64>, tensor<f64>
    cond {
      %1 = stablehlo.compare LT, %iterArg0, %iterArg1 : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg0, %iterArg0 : tensor<f64>
      stablehlo.return %1, %iterArg1 : tensor<f64>, tensor<f64>
    }
  return %0#0 : tensor<f64>
}

// CHECK-LABEL: func.func @while
// CHECK-DAG: %[[CST:.*]] = stablehlo.constant dense<1.000000e+01> : tensor<f64>
// CHECK-DAG: %[[ARG_CAST:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<f64> to tuple<tensor<f32>, tensor<f32>>
// CHECK-DAG: %[[ARG_HI:.*]] = stablehlo.get_tuple_element %[[ARG_CAST]][0] : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
// CHECK-DAG: %[[ARG_LO:.*]] = stablehlo.get_tuple_element %[[ARG_CAST]][1] : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
// CHECK-DAG: %[[CST_CAST:.*]] = builtin.unrealized_conversion_cast %[[CST]] : tensor<f64> to tuple<tensor<f32>, tensor<f32>>
// CHECK-DAG: %[[CST_HI:.*]] = stablehlo.get_tuple_element %[[CST_CAST]][0] : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
// CHECK-DAG: %[[CST_LO:.*]] = stablehlo.get_tuple_element %[[CST_CAST]][1] : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>

// CHECK: stablehlo.while
// CHECK-NEXT: cond {
// CHECK:        %[[COMP_CAST1:.*]] = builtin.unrealized_conversion_cast %{{.*}}, %{{.*}} : tensor<f32>, tensor<f32> to tensor<f64>
// CHECK:        %[[COMP_CAST2:.*]] = builtin.unrealized_conversion_cast %{{.*}}, %{{.*}} : tensor<f32>, tensor<f32> to tensor<f64>
// CHECK:        %[[CMP:.*]] = stablehlo.compare LT, %{{.*}}, %{{.*}} : (tensor<f64>, tensor<f64>) -> tensor<i1>
// CHECK:        stablehlo.return %[[CMP]] : tensor<i1>
// CHECK-NEXT: } do {
// CHECK:        stablehlo.return %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
// CHECK-NEXT: }
