// RUN: mlir-opt -split-input-file -convert-stablehlo-gather-to-slice %s | FileCheck %s

// Original example:
// %c_1179 = stablehlo.constant dense<"0x00000000000000006B00000000000000070000000000000000..."> : tensor<180x3xi64>
// %2803 = stablehlo.dynamic_update_slice %2715, %2802, %c_1336, %c_1308, %c_1329 : (tensor<1x128x194xf64>, tensor<1x1x180xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x128x194xf64>
// %2804 = "stablehlo.gather"(%2803, %c_1179) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2],
//   start_index_map = [0, 1, 2], index_vector_dim = 1>, indices_are_sorted = false, 
//   slice_sizes = array<i64: 1, 1, 1>}> : (tensor<1x128x194xf64>, tensor<180x3xi64>) -> tensor<180xf64>

// CHECK-LABEL: func @gather_to_slice
//func.func @gather_to_slice_wrapped_around(%arg0: tensor<1x128x194xf64>) -> tensor<180xf64> {
//  %indices = stablehlo.constant dense<"0x00000000000000006B00000000000000070000000000000000000000000000006B00000000000000BA0000000000000000000000000000006B00000000000000B90000000000000000000000000000006B00000000000000B80000000000000000000000000000006B00000000000000B70000000000000000000000000000006B00000000000000B60000000000000000000000000000006B00000000000000B50000000000000000000000000000006B00000000000000B40000000000000000000000000000006B00000000000000B30000000000000000000000000000006B00000000000000B20000000000000000000000000000006B00000000000000B10000000000000000000000000000006B00000000000000B00000000000000000000000000000006B00000000000000AF0000000000000000000000000000006B00000000000000AE0000000000000000000000000000006B00000000000000AD0000000000000000000000000000006B00000000000000AC0000000000000000000000000000006B00000000000000AB0000000000000000000000000000006B00000000000000AA0000000000000000000000000000006B00000000000000A90000000000000000000000000000006B00000000000000A80000000000000000000000000000006B00000000000000A70000000000000000000000000000006B00000000000000A60000000000000000000000000000006B00000000000000A50000000000000000000000000000006B00000000000000A40000000000000000000000000000006B00000000000000A30000000000000000000000000000006B00000000000000A20000000000000000000000000000006B00000000000000A10000000000000000000000000000006B00000000000000A00000000000000000000000000000006B000000000000009F0000000000000000000000000000006B000000000000009E0000000000000000000000000000006B000000000000009D0000000000000000000000000000006B000000000000009C0000000000000000000000000000006B000000000000009B0000000000000000000000000000006B000000000000009A0000000000000000000000000000006B00000000000000990000000000000000000000000000006B00000000000000980000000000000000000000000000006B00000000000000970000000000000000000000000000006B00000000000000960000000000000000000000000000006B00000000000000950000000000000000000000000000006B00000000000000940000000000000000000000000000006B00000000000000930000000000000000000000000000006B00000000000000920000000000000000000000000000006B00000000000000910000000000000000000000000000006B00000000000000900000000000000000000000000000006B000000000000008F0000000000000000000000000000006B000000000000008E0000000000000000000000000000006B000000000000008D0000000000000000000000000000006B000000000000008C0000000000000000000000000000006B000000000000008B0000000000000000000000000000006B000000000000008A0000000000000000000000000000006B00000000000000890000000000000000000000000000006B00000000000000880000000000000000000000000000006B00000000000000870000000000000000000000000000006B00000000000000860000000000000000000000000000006B00000000000000850000000000000000000000000000006B00000000000000840000000000000000000000000000006B00000000000000830000000000000000000000000000006B00000000000000820000000000000000000000000000006B00000000000000810000000000000000000000000000006B00000000000000800000000000000000000000000000006B000000000000007F0000000000000000000000000000006B000000000000007E0000000000000000000000000000006B000000000000007D0000000000000000000000000000006B000000000000007C0000000000000000000000000000006B000000000000007B0000000000000000000000000000006B000000000000007A0000000000000000000000000000006B00000000000000790000000000000000000000000000006B00000000000000780000000000000000000000000000006B00000000000000770000000000000000000000000000006B00000000000000760000000000000000000000000000006B00000000000000750000000000000000000000000000006B00000000000000740000000000000000000000000000006B00000000000000730000000000000000000000000000006B00000000000000720000000000000000000000000000006B00000000000000710000000000000000000000000000006B00000000000000700000000000000000000000000000006B000000000000006F0000000000000000000000000000006B000000000000006E0000000000000000000000000000006B000000000000006D0000000000000000000000000000006B000000000000006C0000000000000000000000000000006B000000000000006B0000000000000000000000000000006B000000000000006A0000000000000000000000000000006B00000000000000690000000000000000000000000000006B00000000000000680000000000000000000000000000006B00000000000000670000000000000000000000000000006B00000000000000660000000000000000000000000000006B00000000000000650000000000000000000000000000006B00000000000000640000000000000000000000000000006B00000000000000630000000000000000000000000000006B00000000000000620000000000000000000000000000006B00000000000000610000000000000000000000000000006B00000000000000600000000000000000000000000000006B000000000000005F0000000000000000000000000000006B000000000000005E0000000000000000000000000000006B000000000000005D0000000000000000000000000000006B000000000000005C0000000000000000000000000000006B000000000000005B0000000000000000000000000000006B000000000000005A0000000000000000000000000000006B00000000000000590000000000000000000000000000006B00000000000000580000000000000000000000000000006B00000000000000570000000000000000000000000000006B00000000000000560000000000000000000000000000006B00000000000000550000000000000000000000000000006B00000000000000540000000000000000000000000000006B00000000000000530000000000000000000000000000006B00000000000000520000000000000000000000000000006B00000000000000510000000000000000000000000000006B00000000000000500000000000000000000000000000006B000000000000004F0000000000000000000000000000006B000000000000004E0000000000000000000000000000006B000000000000004D0000000000000000000000000000006B000000000000004C0000000000000000000000000000006B000000000000004B0000000000000000000000000000006B000000000000004A0000000000000000000000000000006B00000000000000490000000000000000000000000000006B00000000000000480000000000000000000000000000006B00000000000000470000000000000000000000000000006B00000000000000460000000000000000000000000000006B00000000000000450000000000000000000000000000006B00000000000000440000000000000000000000000000006B00000000000000430000000000000000000000000000006B00000000000000420000000000000000000000000000006B00000000000000410000000000000000000000000000006B00000000000000400000000000000000000000000000006B000000000000003F0000000000000000000000000000006B000000000000003E0000000000000000000000000000006B000000000000003D0000000000000000000000000000006B000000000000003C0000000000000000000000000000006B000000000000003B0000000000000000000000000000006B000000000000003A0000000000000000000000000000006B00000000000000390000000000000000000000000000006B00000000000000380000000000000000000000000000006B00000000000000370000000000000000000000000000006B00000000000000360000000000000000000000000000006B00000000000000350000000000000000000000000000006B00000000000000340000000000000000000000000000006B00000000000000330000000000000000000000000000006B00000000000000320000000000000000000000000000006B00000000000000310000000000000000000000000000006B00000000000000300000000000000000000000000000006B000000000000002F0000000000000000000000000000006B000000000000002E0000000000000000000000000000006B000000000000002D0000000000000000000000000000006B000000000000002C0000000000000000000000000000006B000000000000002B0000000000000000000000000000006B000000000000002A0000000000000000000000000000006B00000000000000290000000000000000000000000000006B00000000000000280000000000000000000000000000006B00000000000000270000000000000000000000000000006B00000000000000260000000000000000000000000000006B00000000000000250000000000000000000000000000006B00000000000000240000000000000000000000000000006B00000000000000230000000000000000000000000000006B00000000000000220000000000000000000000000000006B00000000000000210000000000000000000000000000006B00000000000000200000000000000000000000000000006B000000000000001F0000000000000000000000000000006B000000000000001E0000000000000000000000000000006B000000000000001D0000000000000000000000000000006B000000000000001C0000000000000000000000000000006B000000000000001B0000000000000000000000000000006B000000000000001A0000000000000000000000000000006B00000000000000190000000000000000000000000000006B00000000000000180000000000000000000000000000006B00000000000000170000000000000000000000000000006B00000000000000160000000000000000000000000000006B00000000000000150000000000000000000000000000006B00000000000000140000000000000000000000000000006B00000000000000130000000000000000000000000000006B00000000000000120000000000000000000000000000006B00000000000000110000000000000000000000000000006B00000000000000100000000000000000000000000000006B000000000000000F0000000000000000000000000000006B000000000000000E0000000000000000000000000000006B000000000000000D0000000000000000000000000000006B000000000000000C0000000000000000000000000000006B000000000000000B0000000000000000000000000000006B000000000000000A0000000000000000000000000000006B00000000000000090000000000000000000000000000006B000000000000000800000000000000"> : tensor<180x3xi64>
//  
//  // CHECK: %[[SLICE:.*]] = "stablehlo.slice"(%arg0)
//  // CHECK-SAME: start_indices = array<i64: 0, 107, 186>
//  // CHECK-SAME: limit_indices = array<i64: 1, 107, 8>
//  // CHECK-SAME: strides = array<i64: 1, 1, 1>
//  // CHECK: return %[[SLICE]] : tensor<180xf64>
//  %result = "stablehlo.gather"(%arg0, %indices) {
//    dimension_numbers = #stablehlo.gather<
//      collapsed_slice_dims = [0, 1, 2],
//      start_index_map = [0, 1, 2],
//      index_vector_dim = 1
//    >, 
//    indices_are_sorted = false, 
//    slice_sizes = array<i64: 1, 1, 1>
//  } : (tensor<1x128x194xf64>, tensor<180x3xi64>) -> tensor<180xf64>
//  
//  return %result : tensor<180xf64>
//}

// -----

// Example of multi dim strided slice op (for reference):
//  %1 = "stablehlo.slice"(%arg0)
//  start_indices = array<i64: 0, 186, 107>
//  limit_indices = array<i64: 0, 8, 107>
//  strides = array<i64: 1, 1, 1>
//  } : tensor<1x128x194xf64> -> tensor<180xf64>
//

func.func @gather_to_slice_reverse(%arg0: tensor<1x128x194xf64>) -> tensor<179xf64> {
  %indices = stablehlo.constant dense<"0x00000000000000006B00000000000000BA0000000000000000000000000000006B00000000000000B90000000000000000000000000000006B00000000000000B80000000000000000000000000000006B00000000000000B70000000000000000000000000000006B00000000000000B60000000000000000000000000000006B00000000000000B50000000000000000000000000000006B00000000000000B40000000000000000000000000000006B00000000000000B30000000000000000000000000000006B00000000000000B20000000000000000000000000000006B00000000000000B10000000000000000000000000000006B00000000000000B00000000000000000000000000000006B00000000000000AF0000000000000000000000000000006B00000000000000AE0000000000000000000000000000006B00000000000000AD0000000000000000000000000000006B00000000000000AC0000000000000000000000000000006B00000000000000AB0000000000000000000000000000006B00000000000000AA0000000000000000000000000000006B00000000000000A90000000000000000000000000000006B00000000000000A80000000000000000000000000000006B00000000000000A70000000000000000000000000000006B00000000000000A60000000000000000000000000000006B00000000000000A50000000000000000000000000000006B00000000000000A40000000000000000000000000000006B00000000000000A30000000000000000000000000000006B00000000000000A20000000000000000000000000000006B00000000000000A10000000000000000000000000000006B00000000000000A00000000000000000000000000000006B000000000000009F0000000000000000000000000000006B000000000000009E0000000000000000000000000000006B000000000000009D0000000000000000000000000000006B000000000000009C0000000000000000000000000000006B000000000000009B0000000000000000000000000000006B000000000000009A0000000000000000000000000000006B00000000000000990000000000000000000000000000006B00000000000000980000000000000000000000000000006B00000000000000970000000000000000000000000000006B00000000000000960000000000000000000000000000006B00000000000000950000000000000000000000000000006B00000000000000940000000000000000000000000000006B00000000000000930000000000000000000000000000006B00000000000000920000000000000000000000000000006B00000000000000910000000000000000000000000000006B00000000000000900000000000000000000000000000006B000000000000008F0000000000000000000000000000006B000000000000008E0000000000000000000000000000006B000000000000008D0000000000000000000000000000006B000000000000008C0000000000000000000000000000006B000000000000008B0000000000000000000000000000006B000000000000008A0000000000000000000000000000006B00000000000000890000000000000000000000000000006B00000000000000880000000000000000000000000000006B00000000000000870000000000000000000000000000006B00000000000000860000000000000000000000000000006B00000000000000850000000000000000000000000000006B00000000000000840000000000000000000000000000006B00000000000000830000000000000000000000000000006B00000000000000820000000000000000000000000000006B00000000000000810000000000000000000000000000006B00000000000000800000000000000000000000000000006B000000000000007F0000000000000000000000000000006B000000000000007E0000000000000000000000000000006B000000000000007D0000000000000000000000000000006B000000000000007C0000000000000000000000000000006B000000000000007B0000000000000000000000000000006B000000000000007A0000000000000000000000000000006B00000000000000790000000000000000000000000000006B00000000000000780000000000000000000000000000006B00000000000000770000000000000000000000000000006B00000000000000760000000000000000000000000000006B00000000000000750000000000000000000000000000006B00000000000000740000000000000000000000000000006B00000000000000730000000000000000000000000000006B00000000000000720000000000000000000000000000006B00000000000000710000000000000000000000000000006B00000000000000700000000000000000000000000000006B000000000000006F0000000000000000000000000000006B000000000000006E0000000000000000000000000000006B000000000000006D0000000000000000000000000000006B000000000000006C0000000000000000000000000000006B000000000000006B0000000000000000000000000000006B000000000000006A0000000000000000000000000000006B00000000000000690000000000000000000000000000006B00000000000000680000000000000000000000000000006B00000000000000670000000000000000000000000000006B00000000000000660000000000000000000000000000006B00000000000000650000000000000000000000000000006B00000000000000640000000000000000000000000000006B00000000000000630000000000000000000000000000006B00000000000000620000000000000000000000000000006B00000000000000610000000000000000000000000000006B00000000000000600000000000000000000000000000006B000000000000005F0000000000000000000000000000006B000000000000005E0000000000000000000000000000006B000000000000005D0000000000000000000000000000006B000000000000005C0000000000000000000000000000006B000000000000005B0000000000000000000000000000006B000000000000005A0000000000000000000000000000006B00000000000000590000000000000000000000000000006B00000000000000580000000000000000000000000000006B00000000000000570000000000000000000000000000006B00000000000000560000000000000000000000000000006B00000000000000550000000000000000000000000000006B00000000000000540000000000000000000000000000006B00000000000000530000000000000000000000000000006B00000000000000520000000000000000000000000000006B00000000000000510000000000000000000000000000006B00000000000000500000000000000000000000000000006B000000000000004F0000000000000000000000000000006B000000000000004E0000000000000000000000000000006B000000000000004D0000000000000000000000000000006B000000000000004C0000000000000000000000000000006B000000000000004B0000000000000000000000000000006B000000000000004A0000000000000000000000000000006B00000000000000490000000000000000000000000000006B00000000000000480000000000000000000000000000006B00000000000000470000000000000000000000000000006B00000000000000460000000000000000000000000000006B00000000000000450000000000000000000000000000006B00000000000000440000000000000000000000000000006B00000000000000430000000000000000000000000000006B00000000000000420000000000000000000000000000006B00000000000000410000000000000000000000000000006B00000000000000400000000000000000000000000000006B000000000000003F0000000000000000000000000000006B000000000000003E0000000000000000000000000000006B000000000000003D0000000000000000000000000000006B000000000000003C0000000000000000000000000000006B000000000000003B0000000000000000000000000000006B000000000000003A0000000000000000000000000000006B00000000000000390000000000000000000000000000006B00000000000000380000000000000000000000000000006B00000000000000370000000000000000000000000000006B00000000000000360000000000000000000000000000006B00000000000000350000000000000000000000000000006B00000000000000340000000000000000000000000000006B00000000000000330000000000000000000000000000006B00000000000000320000000000000000000000000000006B00000000000000310000000000000000000000000000006B00000000000000300000000000000000000000000000006B000000000000002F0000000000000000000000000000006B000000000000002E0000000000000000000000000000006B000000000000002D0000000000000000000000000000006B000000000000002C0000000000000000000000000000006B000000000000002B0000000000000000000000000000006B000000000000002A0000000000000000000000000000006B00000000000000290000000000000000000000000000006B00000000000000280000000000000000000000000000006B00000000000000270000000000000000000000000000006B00000000000000260000000000000000000000000000006B00000000000000250000000000000000000000000000006B00000000000000240000000000000000000000000000006B00000000000000230000000000000000000000000000006B00000000000000220000000000000000000000000000006B00000000000000210000000000000000000000000000006B00000000000000200000000000000000000000000000006B000000000000001F0000000000000000000000000000006B000000000000001E0000000000000000000000000000006B000000000000001D0000000000000000000000000000006B000000000000001C0000000000000000000000000000006B000000000000001B0000000000000000000000000000006B000000000000001A0000000000000000000000000000006B00000000000000190000000000000000000000000000006B00000000000000180000000000000000000000000000006B00000000000000170000000000000000000000000000006B00000000000000160000000000000000000000000000006B00000000000000150000000000000000000000000000006B00000000000000140000000000000000000000000000006B00000000000000130000000000000000000000000000006B00000000000000120000000000000000000000000000006B00000000000000110000000000000000000000000000006B00000000000000100000000000000000000000000000006B000000000000000F0000000000000000000000000000006B000000000000000E0000000000000000000000000000006B000000000000000D0000000000000000000000000000006B000000000000000C0000000000000000000000000000006B000000000000000B0000000000000000000000000000006B000000000000000A0000000000000000000000000000006B00000000000000090000000000000000000000000000006B000000000000000800000000000000"> : tensor<179x3xi64>
  // CHECK-NOT: stablehlo.gather
  // CHECK: %[[SLICE:.*]] = stablehlo.slice %[[ARG0]] [0:1, 107:108, 8:187] 
  // CHECK-SAME: : (tensor<1x128x194xf64>) -> tensor<1x1x179xf64>
  // CHECK: %[[REVERSED:.*]] = stablehlo.reverse %[[SLICE]], dims = [2] 
  // CHECK-SAME: : tensor<1x1x179xf64>
  // CHECK: %[[RESHAPED:.*]] = stablehlo.reshape %[[REVERSED]] 
  // CHECK-SAME: : (tensor<1x1x179xf64>) -> tensor<179xf64>
  %result = "stablehlo.gather"(%arg0, %indices) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 1, 2],
      start_index_map = [0, 1, 2],
      index_vector_dim = 1
    >, 
    indices_are_sorted = false, 
    slice_sizes = array<i64: 1, 1, 1>
  } : (tensor<1x128x194xf64>, tensor<179x3xi64>) -> tensor<179xf64>
  
  return %result : tensor<179xf64>
}

func.func @gather_to_slice_collapse_dims(%arg0: tensor<1x128x194xf64>) -> tensor<3xf64> {
  %indices = stablehlo.constant dense<[
  [0, 10, 4],
  [0, 10, 5],
  [0, 10, 6]
]> : tensor<3x3xi64>
  // CHECK-NOT: stablehlo.gather
  // CHECK: %[[SLICE:.*]] = stablehlo.slice %[[ARG0]] [0:1, 10:11, 4:7] 
  // CHECK-SAME: : (tensor<1x128x194xf64>) -> tensor<1x1x3xf64>
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %[[SLICE]] 
  // CHECK-SAME: : (tensor<1x1x3xf64>) -> tensor<3xf64>  
  %result = "stablehlo.gather"(%arg0, %indices) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 1, 2],
      start_index_map = [0, 1, 2],
      index_vector_dim = 1
    >, 
    indices_are_sorted = false, 
    slice_sizes = array<i64: 1, 1, 1>
  } : (tensor<1x128x194xf64>, tensor<3x3xi64>) -> tensor<3xf64>
  
  return %result : tensor<3xf64>
}

