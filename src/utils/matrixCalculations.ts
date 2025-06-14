// FILE: src/utils/matrixCalculations.ts

// --- RNN Parameters ---
// Simplified vocabulary for the 2-timestep RNN example
export const VOCAB = ['A', 'B'];

// One-Hot Vectors for all input characters
export const ONE_HOT_A = [[1, 0]]; // Vocab size is 2
export const ONE_HOT_B = [[0, 1]];

// --- Weight Matrices & Biases (Initial Values) ---
// W_xh (Input to Hidden): VocabSize x HiddenSize (2x4)
export const W_xh = [
  [ 0.5, -0.2,  0.8,  0.1],
  [ 0.3,  0.6, -0.4,  0.9]
  // Removed third row as vocab size is now 2
];
// W_hh (Hidden to Hidden): HiddenSize x HiddenSize (4x4) - Stays the same
export const W_hh = [
  [ 0.2,  0.5, -0.3,  0.4],
  [-0.1,  0.7,  0.6, -0.2],
  [ 0.8, -0.4,  0.1,  0.9],
  [ 0.3, -0.1,  0.5,  0.7]
];
// W_hy (Hidden to Output): HiddenSize x VocabSize (4x2)
export const W_hy = [
  [ 0.6, -0.3],
  [-0.2,  0.8],
  [ 0.9, -0.1],
  [ 0.3,  0.7]
  // Removed third column as vocab size is now 2
];

// Biases
export const b_h = [[0.1, 0.1, 0.1, 0.1]]; // Hidden bias (1x4) - Stays the same
export const b_y = [[0.1, 0.1]];          // Output bias (1x2) - Adjusted for vocab size 2


// --- Math Helpers ---

export function matrixMultiply(a: number[][], b: number[][]): number[][] {
  const resultRows = a.length;
  const resultCols = b[0].length;
  if (a[0].length !== b.length) {
    console.error("Matrix dimensions incompatible for multiplication.");
    return [[]]; // Return an empty array or throw error
  }
  const result = Array(resultRows).fill(0).map(() => Array(resultCols).fill(0));
  for (let i = 0; i < resultRows; i++) {
    for (let j = 0; j < resultCols; j++) {
      for (let k = 0; k < b.length; k++) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return result;
}

export function matrixAdd(...matrices: number[][][]): number[][] {
  if (matrices.length === 0) return [[]];
  const rows = matrices[0].length;
  const cols = matrices[0][0].length;
  const result = Array(rows).fill(0).map(() => Array(cols).fill(0));
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      for (const m of matrices) {
        if (m.length !== rows || m[i].length !== cols) {
          console.error("Matrix dimensions incompatible for addition.");
          return [[]]; // Return an empty array or throw error
        }
        result[i][j] += m[i][j];
      }
    }
  }
  return result;
}

export function tanh(matrix: number[][]): number[][] {
  return matrix.map(row => row.map(val => Math.tanh(val)));
}

export function softmax(matrix: number[][]): number[][] {
  return matrix.map(row => {
    const maxVal = Math.max(...row);
    const expRow = row.map(x => Math.exp(x - maxVal));
    const sum = expRow.reduce((acc, val) => acc + val, 0);
    if (sum === 0) return row.map(() => 1 / row.length); // Fallback for edge case
    return expRow.map(x => x / sum);
  });
}


// --- Pre-calculated Forward Pass for 2 Timesteps ---

// Initial hidden state h₀ (all zeros)
const h0 = [[0, 0, 0, 0]];

// Timestep 1: Input 'A'
const x1 = ONE_HOT_A;
export const H1 = tanh(matrixAdd(matrixMultiply(x1, W_xh), matrixMultiply(h0, W_hh), b_h));
export const Y1 = matrixAdd(matrixMultiply(H1, W_hy), b_y);
export const PRED1 = softmax(Y1);

// Timestep 2: Input 'B' (previous hidden state is H1)
const x2 = ONE_HOT_B;
export const H2 = tanh(matrixAdd(matrixMultiply(x2, W_xh), matrixMultiply(H1, W_hh), b_h));
export const Y2 = matrixAdd(matrixMultiply(H2, W_hy), b_y);
export const PRED2 = softmax(Y2);

// --- BACKWARD PROPAGATION CALCULATIONS ---

// Target labels for training (simple example: A->B, B->A pattern)
export const TARGET_T1 = [[0, 1]]; // After 'A', we expect 'B'
export const TARGET_T2 = [[1, 0]]; // After 'B', we expect 'A'

// Loss calculation (Cross-entropy loss)
export function crossEntropyLoss(predictions: number[][], targets: number[][]): number {
  let totalLoss = 0;
  for (let i = 0; i < predictions.length; i++) {
    for (let j = 0; j < predictions[i].length; j++) {
      // Prevent log(0) by adding small epsilon
      const pred = Math.max(predictions[i][j], 1e-15);
      totalLoss -= targets[i][j] * Math.log(pred);
    }
  }
  return totalLoss;
}

// Calculate total loss for both timesteps
export const LOSS_T1 = crossEntropyLoss(PRED1, TARGET_T1);
export const LOSS_T2 = crossEntropyLoss(PRED2, TARGET_T2);
export const TOTAL_LOSS = LOSS_T1 + LOSS_T2;

// --- Derivative Functions ---

// Derivative of tanh
export function tanhDerivative(matrix: number[][]): number[][] {
  return matrix.map(row => row.map(val => 1 - Math.tanh(val) * Math.tanh(val)));
}

// Matrix transpose
export function matrixTranspose(matrix: number[][]): number[][] {
  return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
}

// Element-wise matrix subtraction
export function matrixSubtract(a: number[][], b: number[][]): number[][] {
  return a.map((row, i) => row.map((val, j) => val - b[i][j]));
}

// --- BACKWARD PASS GRADIENTS ---

// Step 1: Gradients of loss w.r.t. predictions (softmax output)
export const dL_dPred1 = matrixSubtract(PRED1, TARGET_T1); // ∂L/∂ŷ₁
export const dL_dPred2 = matrixSubtract(PRED2, TARGET_T2); // ∂L/∂ŷ₂

// Step 2: Gradients w.r.t. logits (before softmax)
// For softmax, ∂L/∂y = ∂L/∂ŷ when using cross-entropy loss
export const dL_dY1 = dL_dPred1; // ∂L/∂y₁  
export const dL_dY2 = dL_dPred2; // ∂L/∂y₂

// Step 3: Gradients w.r.t. hidden states
// ∂L/∂h₁ = (∂L/∂y₁)(∂y₁/∂h₁) + (∂L/∂h₂)(∂h₂/∂h₁) 
export const dL_dH1_from_Y1 = matrixMultiply(dL_dY1, matrixTranspose(W_hy)); // From timestep 1 output
// For h₁ flowing to h₂: ∂L/∂h₂ * ∂h₂/∂h₁
const dL_dH2 = matrixMultiply(dL_dY2, matrixTranspose(W_hy)); // ∂L/∂h₂
const h2_preactivation = matrixAdd(matrixMultiply(ONE_HOT_B, W_xh), matrixMultiply(H1, W_hh), b_h);
const dH2_dH1 = matrixMultiply(tanhDerivative(h2_preactivation), matrixTranspose(W_hh));
const dL_dH1_from_H2 = matrixMultiply(dL_dH2, dH2_dH1);
export const dL_dH1 = matrixAdd(dL_dH1_from_Y1, dL_dH1_from_H2); // Total gradient for h₁

export const dL_dH2_final = dL_dH2; // ∂L/∂h₂ (final)

// Step 4: Gradients w.r.t. weights and biases

// Gradients for output weights W_hy and bias b_y
export const dL_dWhy_T1 = matrixMultiply(matrixTranspose(H1), dL_dY1); // From timestep 1
export const dL_dWhy_T2 = matrixMultiply(matrixTranspose(H2), dL_dY2); // From timestep 2
export const dL_dWhy = matrixAdd(dL_dWhy_T1, dL_dWhy_T2); // Combined gradient

export const dL_dby_T1 = dL_dY1; // Bias gradient from timestep 1
export const dL_dby_T2 = dL_dY2; // Bias gradient from timestep 2  
export const dL_dby = matrixAdd(dL_dby_T1, dL_dby_T2); // Combined bias gradient

// Gradients for hidden weights (more complex due to recurrence)
const h1_preactivation = matrixAdd(matrixMultiply(ONE_HOT_A, W_xh), matrixMultiply([[0, 0, 0, 0]], W_hh), b_h);

// For W_xh gradients
const dH1_dWxh_factor = tanhDerivative(h1_preactivation);
const dL_dWxh_T1 = matrixMultiply(matrixTranspose(ONE_HOT_A), matrixMultiply(dL_dH1, dH1_dWxh_factor));

const dH2_dWxh_factor = tanhDerivative(h2_preactivation);
const dL_dWxh_T2 = matrixMultiply(matrixTranspose(ONE_HOT_B), matrixMultiply(dL_dH2_final, dH2_dWxh_factor));

export const dL_dWxh = matrixAdd(dL_dWxh_T1, dL_dWxh_T2); // Combined W_xh gradient

// For W_hh gradients  
const dL_dWhh_T1 = matrixMultiply(matrixTranspose([[0, 0, 0, 0]]), matrixMultiply(dL_dH1, dH1_dWxh_factor));
const dL_dWhh_T2 = matrixMultiply(matrixTranspose(H1), matrixMultiply(dL_dH2_final, dH2_dWxh_factor));

export const dL_dWhh = matrixAdd(dL_dWhh_T1, dL_dWhh_T2); // Combined W_hh gradient

// For hidden bias b_h
export const dL_dbh_T1 = matrixMultiply(dL_dH1, dH1_dWxh_factor);
export const dL_dbh_T2 = matrixMultiply(dL_dH2_final, dH2_dWxh_factor);
export const dL_dbh = matrixAdd(dL_dbh_T1, dL_dbh_T2); // Combined hidden bias gradient