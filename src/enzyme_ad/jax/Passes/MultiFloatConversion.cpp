//===- MultiFloatConversion.cpp - Multi-Float Conversion Pass -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include <optional>
#include "stablehlo/dialect/StablehloOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_MULTIFLOATCONVERSIONPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

namespace {

Type getFloatTypeFromString(StringRef typeStr, MLIRContext *context) {
  if (typeStr == "f64") return Float64Type::get(context);
  if (typeStr == "f32") return Float32Type::get(context);
  if (typeStr == "f16") return Float16Type::get(context);
  if (typeStr == "bf16") return BFloat16Type::get(context);
  if (typeStr == "f8E4M3FN" || typeStr == "fp8") return Float8E4M3FNType::get(context);
  if (typeStr == "f8E5M2") return Float8E5M2Type::get(context);
  return nullptr;
}

Value extractLimb(Value tensor, int limbIndex, OpBuilder &builder, Location loc, StringRef concatDimension) {
  bool isTuple = concatDimension == "tuple";
  bool isFirst = concatDimension == "first";
  if (isTuple) {
    if (auto tupleOp = tensor.getDefiningOp<stablehlo::TupleOp>()) {
      return tupleOp.getOperand(limbIndex);
    }
    return builder.create<stablehlo::GetTupleElementOp>(loc, tensor, builder.getI32IntegerAttr(limbIndex));
  }
  
  if (auto concatOp = tensor.getDefiningOp<stablehlo::ConcatenateOp>()) {
    auto type = cast<RankedTensorType>(tensor.getType()); // use result type to get rank
    int concatDim = isFirst ? 0 : type.getRank() - 1;
    if (concatOp.getDimension() == concatDim && concatOp.getOperands().size() == 2) {
      return concatOp.getOperand(limbIndex);
    }
  }

  auto type = cast<RankedTensorType>(tensor.getType());
  SmallVector<int64_t> sliceShape = llvm::to_vector(type.getShape());
  SmallVector<int64_t> startIndices(type.getRank(), 0);
  SmallVector<int64_t> limitIndices = llvm::to_vector(type.getShape());
  
  int dimToSlice = isFirst ? 0 : type.getRank() - 1;
  
  sliceShape[dimToSlice] = 1;
  startIndices[dimToSlice] = limbIndex;
  limitIndices[dimToSlice] = limbIndex + 1;
  
  SmallVector<int64_t> strides(type.getRank(), 1);
  
  auto sliceOp = builder.create<stablehlo::SliceOp>(
      loc, RankedTensorType::get(sliceShape, type.getElementType()),
      tensor, builder.getDenseI64ArrayAttr(startIndices),
      builder.getDenseI64ArrayAttr(limitIndices),
      builder.getDenseI64ArrayAttr(strides));
      
  return sliceOp;
}



Value packLimbs(Value high, Value low, OpBuilder &builder, Location loc, StringRef concatDimension) {
  bool isTuple = concatDimension == "tuple";
  bool isFirst = concatDimension == "first";
  if (isTuple) {
    return builder.create<stablehlo::TupleOp>(loc, TupleType::get(high.getContext(), {high.getType(), low.getType()}), ValueRange{high, low});
  }
  auto type = cast<RankedTensorType>(high.getType());
  
  // Expect high and low to be tensors with a size-1 dimension at the concatenation axis.
  // We just concatenate them along that axis.
  
  int concatDim = isFirst ? 0 : type.getRank() - 1;
  
  SmallVector<int64_t> outShape = llvm::to_vector(type.getShape());
  outShape[concatDim] = 2;
  
  auto concatOp = builder.create<stablehlo::ConcatenateOp>(
      loc, RankedTensorType::get(outShape, type.getElementType()),
      ValueRange{high, low}, concatDim);
      
  return concatOp;
}



std::pair<Value, Value> twoSum(Value a, Value b, OpBuilder &builder, Location loc) {
  Value sum = builder.create<stablehlo::AddOp>(loc, a, b);
  Value a_prime = builder.create<stablehlo::SubtractOp>(loc, sum, b);
  Value b_prime = builder.create<stablehlo::SubtractOp>(loc, sum, a_prime);
  Value a_err = builder.create<stablehlo::SubtractOp>(loc, a, a_prime);
  Value b_err = builder.create<stablehlo::SubtractOp>(loc, b, b_prime);
  Value err = builder.create<stablehlo::AddOp>(loc, a_err, b_err);
  return {sum, err};
}

std::pair<Value, Value> fastTwoSum(Value a, Value b, OpBuilder &builder, Location loc) {
  Value sum = builder.create<stablehlo::AddOp>(loc, a, b);
  Value b_prime = builder.create<stablehlo::SubtractOp>(loc, sum, a);
  Value b_err = builder.create<stablehlo::SubtractOp>(loc, b, b_prime);
  return {sum, b_err};
}

Value getSplitConstant(Type type, OpBuilder &builder, Location loc) {
  auto tensorTy = cast<RankedTensorType>(type);
  auto floatTy = cast<FloatType>(tensorTy.getElementType());
  int precision = floatTy.getWidth() == 64 ? 53 :
                   floatTy.getWidth() == 32 ? 24 :
                   floatTy.getWidth() == 16 ? (floatTy.isF16() ? 11 : 8) :
                   floatTy.getWidth() == 8 ? (isa<Float8E4M3FNType>(floatTy) ? 4 : 3) : 0;
  if (precision == 0) return nullptr;
  int k = (precision + 1) / 2;
  double val = std::pow(2.0, k) + 1.0;
  
  auto attr = builder.getFloatAttr(floatTy, val);
  auto splatAttr = SplatElementsAttr::get(tensorTy, attr);
  return builder.create<stablehlo::ConstantOp>(loc, splatAttr);
}

std::pair<Value, Value> split(Value a, OpBuilder &builder, Location loc) {
  Value c_const = getSplitConstant(a.getType(), builder, loc);
  Value c = builder.create<stablehlo::MulOp>(loc, a, c_const);
  Value a_big = builder.create<stablehlo::SubtractOp>(loc, c, a);
  Value a_hi = builder.create<stablehlo::SubtractOp>(loc, c, a_big);
  Value a_lo = builder.create<stablehlo::SubtractOp>(loc, a, a_hi);
  return {a_hi, a_lo};
}

std::pair<Value, Value> twoProdDekker(Value a, Value b, OpBuilder &builder, Location loc) {
  Value p = builder.create<stablehlo::MulOp>(loc, a, b);
  auto [a_hi, a_lo] = split(a, builder, loc);
  auto [b_hi, b_lo] = split(b, builder, loc);
  
  Value p1 = builder.create<stablehlo::MulOp>(loc, a_hi, b_hi);
  Value p2 = builder.create<stablehlo::MulOp>(loc, a_hi, b_lo);
  Value p3 = builder.create<stablehlo::MulOp>(loc, a_lo, b_hi);
  Value p4 = builder.create<stablehlo::MulOp>(loc, a_lo, b_lo);
  
  Value err1 = builder.create<stablehlo::SubtractOp>(loc, p1, p);
  Value err2 = builder.create<stablehlo::AddOp>(loc, err1, p2);
  Value err3 = builder.create<stablehlo::AddOp>(loc, err2, p3);
  Value err4 = builder.create<stablehlo::AddOp>(loc, err3, p4);
  
  return {p, err4};
}

struct AddOpConversion : public OpConversionPattern<stablehlo::AddOp> {
  AddOpConversion(TypeConverter &typeConverter, MLIRContext *context, StringRef concatDimension)
      : OpConversionPattern<stablehlo::AddOp>(typeConverter, context), concatDimension(concatDimension) {}

  StringRef concatDimension;

  LogicalResult matchAndRewrite(stablehlo::AddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value x1 = extractLimb(adaptor.getOperands()[0], 0, rewriter, loc, concatDimension);
    Value x2 = extractLimb(adaptor.getOperands()[0], 1, rewriter, loc, concatDimension);
    Value y1 = extractLimb(adaptor.getOperands()[1], 0, rewriter, loc, concatDimension);
    Value y2 = extractLimb(adaptor.getOperands()[1], 1, rewriter, loc, concatDimension);


    auto [a, b] = twoSum(x1, y1, rewriter, loc);
    auto [c, d] = twoSum(x2, y2, rewriter, loc);
    auto [new_a, new_c] = fastTwoSum(a, c, rewriter, loc);
    Value b2 = rewriter.create<stablehlo::AddOp>(loc, b, d);
    Value b3 = rewriter.create<stablehlo::AddOp>(loc, b2, new_c);
    auto [final_a, final_b] = fastTwoSum(new_a, b3, rewriter, loc);
    
    Value packed = packLimbs(final_a, final_b, rewriter, loc, concatDimension);
    rewriter.replaceOp(op, packed);
    return success();
  }
};

struct MulOpConversion : public OpConversionPattern<stablehlo::MulOp> {
  MulOpConversion(TypeConverter &typeConverter, MLIRContext *context, StringRef concatDimension)
      : OpConversionPattern<stablehlo::MulOp>(typeConverter, context), concatDimension(concatDimension) {}

  StringRef concatDimension;

  LogicalResult matchAndRewrite(stablehlo::MulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value x1 = extractLimb(adaptor.getOperands()[0], 0, rewriter, loc, concatDimension);
    Value x2 = extractLimb(adaptor.getOperands()[0], 1, rewriter, loc, concatDimension);
    Value y1 = extractLimb(adaptor.getOperands()[1], 0, rewriter, loc, concatDimension);
    Value y2 = extractLimb(adaptor.getOperands()[1], 1, rewriter, loc, concatDimension);


    auto [p00, e00] = twoProdDekker(x1, y1, rewriter, loc);
    Value p01 = rewriter.create<stablehlo::MulOp>(loc, x1, y2);
    Value p10 = rewriter.create<stablehlo::MulOp>(loc, x2, y1);
    Value p01_p10 = rewriter.create<stablehlo::AddOp>(loc, p01, p10);
    Value e00_new = rewriter.create<stablehlo::AddOp>(loc, e00, p01_p10);
    auto [final_p, final_e] = fastTwoSum(p00, e00_new, rewriter, loc);
    
    Value packed = packLimbs(final_p, final_e, rewriter, loc, concatDimension);
    rewriter.replaceOp(op, packed);
    return success();
  }
};

struct SubOpConversion : public OpConversionPattern<stablehlo::SubtractOp> {
  SubOpConversion(TypeConverter &typeConverter, MLIRContext *context, StringRef concatDimension)
      : OpConversionPattern<stablehlo::SubtractOp>(typeConverter, context), concatDimension(concatDimension) {}

  StringRef concatDimension;

  LogicalResult matchAndRewrite(stablehlo::SubtractOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value x1 = extractLimb(adaptor.getOperands()[0], 0, rewriter, loc, concatDimension);
    Value x2 = extractLimb(adaptor.getOperands()[0], 1, rewriter, loc, concatDimension);
    Value y1 = extractLimb(adaptor.getOperands()[1], 0, rewriter, loc, concatDimension);
    Value y2 = extractLimb(adaptor.getOperands()[1], 1, rewriter, loc, concatDimension);


    Value neg_y1 = rewriter.create<stablehlo::NegOp>(loc, y1);
    Value neg_y2 = rewriter.create<stablehlo::NegOp>(loc, y2);

    auto [a, b] = twoSum(x1, neg_y1, rewriter, loc);
    auto [c, d] = twoSum(x2, neg_y2, rewriter, loc);
    auto [new_a, new_c] = fastTwoSum(a, c, rewriter, loc);
    Value b2 = rewriter.create<stablehlo::AddOp>(loc, b, d);
    Value b3 = rewriter.create<stablehlo::AddOp>(loc, b2, new_c);
    auto [final_a, final_b] = fastTwoSum(new_a, b3, rewriter, loc);
    
    Value packed = packLimbs(final_a, final_b, rewriter, loc, concatDimension);
    rewriter.replaceOp(op, packed);
    return success();
  }
};

struct SelectOpConversion : public OpConversionPattern<stablehlo::SelectOp> {
  StringRef concatDimension;

  SelectOpConversion(TypeConverter &typeConverter, MLIRContext *context, StringRef concatDimension)
      : OpConversionPattern<stablehlo::SelectOp>(typeConverter, context), concatDimension(concatDimension) {}

  LogicalResult matchAndRewrite(stablehlo::SelectOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    bool isTuple = concatDimension == "tuple";
    bool isFirst = concatDimension == "first";

    if (isTuple) {
      Value onTrueHigh = extractLimb(adaptor.getOperands()[1], 0, rewriter, loc, concatDimension);
      Value onTrueLow = extractLimb(adaptor.getOperands()[1], 1, rewriter, loc, concatDimension);
      Value onFalseHigh = extractLimb(adaptor.getOperands()[2], 0, rewriter, loc, concatDimension);
      Value onFalseLow = extractLimb(adaptor.getOperands()[2], 1, rewriter, loc, concatDimension);

      auto selectHigh = rewriter.create<stablehlo::SelectOp>(loc, adaptor.getOperands()[0], onTrueHigh, onFalseHigh);
      auto selectLow = rewriter.create<stablehlo::SelectOp>(loc, adaptor.getOperands()[0], onTrueLow, onFalseLow);

      Value packed = packLimbs(selectHigh, selectLow, rewriter, loc, concatDimension);
      rewriter.replaceOp(op, packed);
      return success();
    }

    auto valuesType = cast<RankedTensorType>(adaptor.getOperands()[1].getType());
    auto predType = cast<RankedTensorType>(adaptor.getOperands()[0].getType());
    
    SmallVector<int64_t> broadcastDims;
    if (isFirst) {
      for (int i = 0; i < predType.getRank(); ++i) {
        broadcastDims.push_back(i + 1);
      }
    } else {
      for (int i = 0; i < predType.getRank(); ++i) {
        broadcastDims.push_back(i);
      }
    }

    auto outType = RankedTensorType::get(valuesType.getShape(), predType.getElementType());
    auto broadcastPredicate = rewriter.create<stablehlo::BroadcastInDimOp>(
        loc, outType, adaptor.getOperands()[0], rewriter.getDenseI64ArrayAttr(broadcastDims));

    auto selectOp = rewriter.create<stablehlo::SelectOp>(
        loc, valuesType, broadcastPredicate, adaptor.getOperands()[1], adaptor.getOperands()[2]);

    rewriter.replaceOp(op, selectOp);
    return success();
  }
};

struct SliceOpConversion : public OpConversionPattern<stablehlo::SliceOp> {
  StringRef concatDimension;

  SliceOpConversion(TypeConverter &typeConverter, MLIRContext *context, StringRef concatDimension)
      : OpConversionPattern<stablehlo::SliceOp>(typeConverter, context), concatDimension(concatDimension) {}

  LogicalResult matchAndRewrite(stablehlo::SliceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    bool isTuple = concatDimension == "tuple";
    bool isFirst = concatDimension == "first";

    if (isTuple) {
      Value high = extractLimb(adaptor.getOperands()[0], 0, rewriter, loc, concatDimension);
      Value low = extractLimb(adaptor.getOperands()[0], 1, rewriter, loc, concatDimension);

      auto sliceHigh = rewriter.create<stablehlo::SliceOp>(loc, high, op.getStartIndices(), op.getLimitIndices(), op.getStrides());
      auto sliceLow = rewriter.create<stablehlo::SliceOp>(loc, low, op.getStartIndices(), op.getLimitIndices(), op.getStrides());

      Value packed = packLimbs(sliceHigh, sliceLow, rewriter, loc, concatDimension);
      rewriter.replaceOp(op, packed);
      return success();
    }

    SmallVector<int64_t> startIndices = llvm::to_vector(op.getStartIndices());
    SmallVector<int64_t> limitIndices = llvm::to_vector(op.getLimitIndices());
    SmallVector<int64_t> strides = llvm::to_vector(op.getStrides());

    if (isFirst) {
      startIndices.insert(startIndices.begin(), 0);
      limitIndices.insert(limitIndices.begin(), 2);
      strides.insert(strides.begin(), 1);
    } else {
      startIndices.push_back(0);
      limitIndices.push_back(2);
      strides.push_back(1);
    }

    auto outType = cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));

    auto sliceOp = rewriter.create<stablehlo::SliceOp>(
        loc, outType, adaptor.getOperands()[0],
        rewriter.getDenseI64ArrayAttr(startIndices),
        rewriter.getDenseI64ArrayAttr(limitIndices),
        rewriter.getDenseI64ArrayAttr(strides));

    rewriter.replaceOp(op, sliceOp);
    return success();
  }
};struct BroadcastInDimOpConversion : public OpConversionPattern<stablehlo::BroadcastInDimOp> {
  StringRef concatDimension;

  BroadcastInDimOpConversion(TypeConverter &typeConverter, MLIRContext *context, StringRef concatDimension)
      : OpConversionPattern<stablehlo::BroadcastInDimOp>(typeConverter, context), concatDimension(concatDimension) {}

  LogicalResult matchAndRewrite(stablehlo::BroadcastInDimOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    bool isTuple = concatDimension == "tuple";
    bool isFirst = concatDimension == "first";

    if (isTuple) {
      Value high = extractLimb(adaptor.getOperands()[0], 0, rewriter, loc, concatDimension);
      Value low = extractLimb(adaptor.getOperands()[0], 1, rewriter, loc, concatDimension);

      auto outType = cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
      auto partType = cast<TupleType>(outType).getType(0);

      auto bcastHigh = rewriter.create<stablehlo::BroadcastInDimOp>(loc, partType, high, op.getBroadcastDimensions());
      auto bcastLow = rewriter.create<stablehlo::BroadcastInDimOp>(loc, partType, low, op.getBroadcastDimensions());

      Value packed = packLimbs(bcastHigh, bcastLow, rewriter, loc, concatDimension);
      rewriter.replaceOp(op, packed);
      return success();
    }

    SmallVector<int64_t> broadcastDims = llvm::to_vector(op.getBroadcastDimensions());
    auto outType = cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
    
    if (isFirst) {
      SmallVector<int64_t> newBroadcastDims;
      newBroadcastDims.push_back(0);
      for (auto dim : broadcastDims) {
        newBroadcastDims.push_back(dim + 1);
      }
      broadcastDims = std::move(newBroadcastDims);
    } else {
      broadcastDims.push_back(outType.getRank() - 1);
    }

    auto bcastOp = rewriter.create<stablehlo::BroadcastInDimOp>(
        loc, outType, adaptor.getOperands()[0],
        rewriter.getDenseI64ArrayAttr(broadcastDims));

    rewriter.replaceOp(op, bcastOp);
    return success();
  }
};

struct TransposeOpConversion : public OpConversionPattern<stablehlo::TransposeOp> {
  StringRef concatDimension;

  TransposeOpConversion(TypeConverter &typeConverter, MLIRContext *context, StringRef concatDimension)
      : OpConversionPattern<stablehlo::TransposeOp>(typeConverter, context), concatDimension(concatDimension) {}

  LogicalResult matchAndRewrite(stablehlo::TransposeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    bool isTuple = concatDimension == "tuple";
    bool isFirst = concatDimension == "first";

    if (isTuple) {
      Value high = extractLimb(adaptor.getOperands()[0], 0, rewriter, loc, concatDimension);
      Value low = extractLimb(adaptor.getOperands()[0], 1, rewriter, loc, concatDimension);

      auto outType = cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
      auto partType = cast<TupleType>(outType).getType(0);

      auto transHigh = rewriter.create<stablehlo::TransposeOp>(loc, partType, high, op.getPermutation());
      auto transLow = rewriter.create<stablehlo::TransposeOp>(loc, partType, low, op.getPermutation());

      Value packed = packLimbs(transHigh, transLow, rewriter, loc, concatDimension);
      rewriter.replaceOp(op, packed);
      return success();
    }

    SmallVector<int64_t> permutation = llvm::to_vector(op.getPermutation());
    auto outType = cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
    
    if (isFirst) {
      SmallVector<int64_t> newPerm;
      newPerm.push_back(0);
      for (auto idx : permutation) {
        newPerm.push_back(idx + 1);
      }
      permutation = std::move(newPerm);
    } else {
      permutation.push_back(outType.getRank() - 1);
    }

    auto transOp = rewriter.create<stablehlo::TransposeOp>(
        loc, outType, adaptor.getOperands()[0],
        rewriter.getDenseI64ArrayAttr(permutation));

    rewriter.replaceOp(op, transOp);
    return success();
  }
};

struct ReshapeOpConversion : public OpConversionPattern<stablehlo::ReshapeOp> {
  StringRef concatDimension;

  ReshapeOpConversion(TypeConverter &typeConverter, MLIRContext *context, StringRef concatDimension)
      : OpConversionPattern<stablehlo::ReshapeOp>(typeConverter, context), concatDimension(concatDimension) {}

  LogicalResult matchAndRewrite(stablehlo::ReshapeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    bool isTuple = concatDimension == "tuple";

    if (isTuple) {
      Value high = extractLimb(adaptor.getOperands()[0], 0, rewriter, loc, concatDimension);
      Value low = extractLimb(adaptor.getOperands()[0], 1, rewriter, loc, concatDimension);

      auto outType = cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
      auto partType = cast<TupleType>(outType).getType(0);

      auto reshapeHigh = rewriter.create<stablehlo::ReshapeOp>(loc, partType, high);
      auto reshapeLow = rewriter.create<stablehlo::ReshapeOp>(loc, partType, low);

      Value packed = packLimbs(reshapeHigh, reshapeLow, rewriter, loc, concatDimension);
      rewriter.replaceOp(op, packed);
      return success();
    }

    auto outType = cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));

    auto reshapeOp = rewriter.create<stablehlo::ReshapeOp>(
        loc, outType, adaptor.getOperands()[0]);

    rewriter.replaceOp(op, reshapeOp);
    return success();
  }
};


struct DotGeneralOpConversion : public OpConversionPattern<stablehlo::DotGeneralOp> {
  StringRef concatDimension;

  DotGeneralOpConversion(TypeConverter &typeConverter, MLIRContext *context, StringRef concatDimension)
      : OpConversionPattern<stablehlo::DotGeneralOp>(typeConverter, context), concatDimension(concatDimension) {}

  LogicalResult matchAndRewrite(stablehlo::DotGeneralOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = adaptor.getOperands()[0];
    Value rhs = adaptor.getOperands()[1];

    Value lhs_hi = extractLimb(lhs, 0, rewriter, loc, concatDimension);
    Value lhs_lo = extractLimb(lhs, 1, rewriter, loc, concatDimension);
    Value rhs_hi = extractLimb(rhs, 0, rewriter, loc, concatDimension);
    Value rhs_lo = extractLimb(rhs, 1, rewriter, loc, concatDimension);

    auto origType = cast<RankedTensorType>(op.getType());
    auto origShape = origType.getShape();
    auto f32Type = rewriter.getF32Type();
    auto prodType = RankedTensorType::get(origShape, f32Type);

    Value hi_hi = rewriter.create<stablehlo::DotGeneralOp>(
        loc, prodType, lhs_hi, rhs_hi, op.getDotDimensionNumbers(),
        op.getPrecisionConfigAttr(), op.getAlgorithmAttr());
    Value hi_lo = rewriter.create<stablehlo::DotGeneralOp>(
        loc, prodType, lhs_hi, rhs_lo, op.getDotDimensionNumbers(),
        op.getPrecisionConfigAttr(), op.getAlgorithmAttr());
    Value lo_hi = rewriter.create<stablehlo::DotGeneralOp>(
        loc, prodType, lhs_lo, rhs_hi, op.getDotDimensionNumbers(),
        op.getPrecisionConfigAttr(), op.getAlgorithmAttr());
    Value lo_lo = rewriter.create<stablehlo::DotGeneralOp>(
        loc, prodType, lhs_lo, rhs_lo, op.getDotDimensionNumbers(),
        op.getPrecisionConfigAttr(), op.getAlgorithmAttr());

    Value L = rewriter.create<stablehlo::AddOp>(loc, hi_lo, lo_hi);
    L = rewriter.create<stablehlo::AddOp>(loc, L, lo_lo);

    Value packed = packLimbs(hi_hi, L, rewriter, loc, concatDimension);
    rewriter.replaceOp(op, packed);
    return success();
  }
};

struct ConcatenateOpOptimization : public OpConversionPattern<stablehlo::ConcatenateOp> {
  using OpConversionPattern<stablehlo::ConcatenateOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(stablehlo::ConcatenateOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (op.getOperands().empty()) return failure();

    Operation *firstOp = op.getOperands()[0].getDefiningOp();
    if (!firstOp) return failure();

    for (auto operand : op.getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp) return failure();
      if (defOp->getName() != firstOp->getName()) return failure();
      if (!defOp->hasTrait<mlir::OpTrait::Elementwise>()) return failure();
      if (defOp->getAttrDictionary() != firstOp->getAttrDictionary()) return failure();
      if (defOp->getNumOperands() != firstOp->getNumOperands()) return failure();
    }

    SmallVector<Value> newOperands;
    for (unsigned i = 0; i < firstOp->getNumOperands(); ++i) {
      SmallVector<Value> operandsToConcat;
      for (auto operand : op.getOperands()) {
        operandsToConcat.push_back(operand.getDefiningOp()->getOperand(i));
      }
      
      auto type = cast<RankedTensorType>(operandsToConcat[0].getType());
      SmallVector<int64_t> newShape = llvm::to_vector(type.getShape());
      for (size_t j = 1; j < operandsToConcat.size(); ++j) {
        newShape[op.getDimension()] += cast<RankedTensorType>(operandsToConcat[j].getType()).getShape()[op.getDimension()];
      }
      auto outType = RankedTensorType::get(newShape, type.getElementType());

      auto newConcat = rewriter.create<stablehlo::ConcatenateOp>(
          op.getLoc(), outType, operandsToConcat, op.getDimension());
      newOperands.push_back(newConcat);
    }

    OperationState state(op.getLoc(), firstOp->getName().getStringRef());
    state.addOperands(newOperands);
    state.addAttributes(firstOp->getAttrDictionary().getValue());
    state.addTypes(op.getType());

    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct MultiFloatConversionPass
    : public enzyme::impl::MultiFloatConversionPassBase<MultiFloatConversionPass> {
  using MultiFloatConversionPassBase::MultiFloatConversionPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto op = getOperation();

    Type srcTy = getFloatTypeFromString(sourceType, context);
    Type tgtTy = getFloatTypeFromString(targetType, context);

    if (!srcTy || !tgtTy) {
      op->emitError() << "Invalid source or target type specified.";
      signalPassFailure();
      return;
    }

    ConversionTarget target(*context);
    target.addLegalDialect<stablehlo::StablehloDialect>();
    target.addLegalDialect<arith::ArithDialect>();

    TypeConverter typeConverter;
    
    bool isTuple = concatDimension == "tuple";
    bool isFirst = concatDimension == "first";

    typeConverter.addConversion([&](Type type) -> std::optional<Type> {
      if (type == srcTy) {
        if (isTuple) {
          return TupleType::get(context, {tgtTy, tgtTy});
        }
        return RankedTensorType::get({2}, tgtTy);
      }
      if (auto tensorTy = dyn_cast<RankedTensorType>(type)) {
        if (tensorTy.getElementType() == srcTy) {
          if (isTuple) {
            auto partTy = RankedTensorType::get(tensorTy.getShape(), tgtTy);
            return TupleType::get(context, {partTy, partTy});
          }
          SmallVector<int64_t> newShape;
          if (isFirst) {
            newShape.push_back(2);
            for (auto dim : tensorTy.getShape()) {
              newShape.push_back(dim);
            }
          } else {
            for (auto dim : tensorTy.getShape()) {
              newShape.push_back(dim);
            }
            newShape.push_back(2);
          }
          return RankedTensorType::get(newShape, tgtTy);
        }
      }
      return type;
    });
    
    typeConverter.addSourceMaterialization([](OpBuilder &builder, Type type, ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1) return Value();
      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0]).getResult(0);
    });
    typeConverter.addTargetMaterialization([](OpBuilder &builder, Type type, ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1) return Value();
      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0]).getResult(0);
    });

    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addDynamicallyLegalOp<stablehlo::ConcatenateOp>([&](stablehlo::ConcatenateOp op) {
      if (op.getOperands().size() != 2) return true;
      Operation *lhsOp = op.getOperands()[0].getDefiningOp();
      Operation *rhsOp = op.getOperands()[1].getDefiningOp();
      if (!lhsOp || !rhsOp) return true;
      if (lhsOp->getName() != rhsOp->getName()) return true;
      if (!lhsOp->hasTrait<mlir::OpTrait::Elementwise>()) return true;
      if (lhsOp->getAttrDictionary() != rhsOp->getAttrDictionary()) return true;
      return false; // Not legal if it can be optimized!
    });
    target.addDynamicallyLegalOp<stablehlo::SliceOp>([&](stablehlo::SliceOp op) {
      if (typeConverter.isLegal(op.getType())) return true; // Already legal if its output is legal (meaning it was probably generated by us or is already in the right shape)
      // But wait! If it's are slice of are multi-float tensor, it should be converted!
      // If it's are slice of are standard tensor, it's legal!
      // So it's legal if its INPUT is legal!
      return typeConverter.isLegal(op.getOperand().getType());
    });
    target.addDynamicallyLegalOp<stablehlo::BroadcastInDimOp>([&](stablehlo::BroadcastInDimOp op) {
      if (typeConverter.isLegal(op.getType())) return true;
      return typeConverter.isLegal(op.getOperand().getType());
    });
    target.addDynamicallyLegalOp<stablehlo::TransposeOp>([&](stablehlo::TransposeOp op) {
      if (typeConverter.isLegal(op.getType())) return true;
      return typeConverter.isLegal(op.getOperand().getType());
    });
    target.addDynamicallyLegalOp<stablehlo::ReshapeOp>([&](stablehlo::ReshapeOp op) {
      if (typeConverter.isLegal(op.getType())) return true;
      return typeConverter.isLegal(op.getOperand().getType());
    });
    target.addDynamicallyLegalOp<stablehlo::AddOp>([&](stablehlo::AddOp op) {
      return typeConverter.isLegal(op.getType());
    });
    target.addDynamicallyLegalOp<stablehlo::SubtractOp>([&](stablehlo::SubtractOp op) {
      return typeConverter.isLegal(op.getType());
    });
    target.addDynamicallyLegalOp<stablehlo::MulOp>([&](stablehlo::MulOp op) {
      return typeConverter.isLegal(op.getType());
    });
    target.addDynamicallyLegalOp<stablehlo::SelectOp>([&](stablehlo::SelectOp op) {
      return typeConverter.isLegal(op.getType());
    });

    RewritePatternSet patterns(context);
    patterns.add<AddOpConversion>(typeConverter, context, concatDimension);
    patterns.add<SubOpConversion>(typeConverter, context, concatDimension);
    patterns.add<MulOpConversion>(typeConverter, context, concatDimension);
    patterns.add<SelectOpConversion>(typeConverter, context, concatDimension);
    patterns.add<SliceOpConversion>(typeConverter, context, concatDimension);
    patterns.add<BroadcastInDimOpConversion>(typeConverter, context, concatDimension);
    patterns.add<TransposeOpConversion>(typeConverter, context, concatDimension);
    patterns.add<ReshapeOpConversion>(typeConverter, context, concatDimension);
    patterns.add<DotGeneralOpConversion>(typeConverter, context, concatDimension);
    patterns.add<ConcatenateOpOptimization>(typeConverter, context);


    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
