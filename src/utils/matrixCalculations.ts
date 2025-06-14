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