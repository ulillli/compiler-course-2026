#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

using namespace mlir;

namespace {
class MemRefLoweringPass
    : public PassWrapper<MemRefLoweringPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const final { return "replace-memref-copy"; }
  StringRef getDescription() const final {
    return "lowers memref.copy to scf.for loops";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<scf::SCFDialect, arith::ArithDialect, memref::MemRefDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());
    module.walk([&](memref::CopyOp copyOp) {
      if (failed(lowerCopy(copyOp, rewriter))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }
  LogicalResult lowerCopy(memref::CopyOp copyOp, IRRewriter &rewriter) {
    Location loc = copyOp.getLoc();
    Value src = copyOp.getSource();
    Value dst = copyOp.getTarget();

    auto memrefType = cast<MemRefType>(src.getType());
    auto shape = memrefType.getShape();
    unsigned rank = memrefType.getRank();

    rewriter.setInsertionPoint(copyOp);

    SmallVector<Value, 4> lbs, ubs, steps;
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    for (int64_t dimSize : shape) {
      lbs.push_back(zero);
      steps.push_back(one);
      ubs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, dimSize));
    }

    scf::buildLoopNest(rewriter, loc, lbs, ubs, steps,
                       [&](OpBuilder &b, Location loc, ValueRange ivs) {
                         Value element =
                             b.create<memref::LoadOp>(loc, src, ivs);
                         b.create<memref::StoreOp>(loc, element, dst, ivs);
                       });
    rewriter.eraseOp(copyOp);
    return success();
  }
};
} // namespace

MLIR_DECLARE_EXPLICIT_TYPE_ID(MemRefLoweringPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(MemRefLoweringPass)

mlir::PassPluginLibraryInfo getFunctionCallCounterPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "MemRefLoweringPass", "1.0",
          []() { mlir::PassRegistration<MemRefLoweringPass>(); }};
}

extern "C" LLVM_ATTRIBUTE_WEAK mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return getFunctionCallCounterPassPluginInfo();
}
