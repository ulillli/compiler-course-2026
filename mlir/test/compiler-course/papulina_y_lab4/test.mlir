// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/papulina_y_lab4_MLIR%shlibext --pass-pipeline="builtin.module(replace-memref-copy)" %s | FileCheck %s

// CHECK-LABEL: func.func @test_simple_copy
func.func @test_simple_copy(%arg0: memref<10xf32>, %arg1: memref<10xf32>) {
  
  // CHECK-NOT: memref.copy
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C10:.*]] = arith.constant 10 : index

  // CHECK: scf.for %[[IV:.*]] = %[[C0]] to %[[C10]] step %[[C1]] {
  // CHECK:   %[[VAL:.*]] = memref.load %arg0[%[[IV]]] : memref<10xf32>
  // CHECK:   memref.store %[[VAL]], %arg1[%[[IV]]] : memref<10xf32>
  // CHECK: }
  
  memref.copy %arg0, %arg1 : memref<10xf32> to memref<10xf32>
  return
}

// CHECK-LABEL: func.func @test_2d_copy
func.func @test_2d_copy(%arg0: memref<4x8xf32>, %arg1: memref<4x8xf32>) {
  
  // CHECK-NOT: memref.copy
  
  // CHECK: scf.for %[[I:.*]] = %c0 to %c4
  // CHECK:   scf.for %[[J:.*]] = %c0 to %c8
  // CHECK:     %[[VAL:.*]] = memref.load %arg0[%[[I]], %[[J]]]
  // CHECK:     memref.store %[[VAL]], %arg1[%[[I]], %[[J]]]
  
  memref.copy %arg0, %arg1 : memref<4x8xf32> to memref<4x8xf32>
  return
}
