// FILE: src/utils/matrixCalculations.ts

// --- RNN Parameters ---
// Our simple vocabulary for the "RNN" example
export const VOCAB = ['R', 'N', '!'];

// One-Hot Vector for the first input character: 'R'
export const ONE_HOT_R = [[1, 0, 0]];

// --- Weight Matrices & Biases (Initial Values) ---
// W_xh (Input to Hidden): 3x4
export const W_xh = [
  [ 0.5, -0.2,  0.8,  0.1],
  [ 0.3,  0.6, -0.4,  0.9],
  [-0.1,  0.7,  0.3, -0.5]
];
// W_hh (Hidden to Hidden): 4x4
export const W_hh = [
  [ 0.2,  0.5, -0.3,  0.4],
  [-0.1,  0.7,  0.6, -0.2],
  [ 0.8, -0.4,  0.1,  0.9],
  [ 0.3, -0.1,  0.5,  0.7]
];
// W_hy (Hidden to Output): 4x3
export const W_hy = [
  [ 0.6, -0.3,  0.1],
  [-0.2,  0.8,  0.4],
  [ 0.9, -0.1, -0.5],
  [ 0.3,  0.7, -0.2]
];

// Biases
export const b_h = [[0.1, 0.1, 0.1, 0.1]]; // Hidden bias
export const b_y = [[0.1, 0.1, 0.1]];     // Output bias


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


// --- Pre-calculated Forward Pass for Timestep 1 ---

// Initial hidden state h₀ (all zeros)
const h0 = [[0, 0, 0, 0]];
// Current input x₁ ('R')
const x1 = ONE_HOT_R;

// h₁ = tanh((x₁ ⋅ W_xh) + (h₀ ⋅ W_hh) + b_h)
export const H1 = tanh(matrixAdd(matrixMultiply(x1, W_xh), matrixMultiply(h0, W_hh), b_h));

// y₁ = h₁ ⋅ W_hy + b_y
export const Y1 = matrixAdd(matrixMultiply(H1, W_hy), b_y);

// ŷ₁ = Softmax(y₁)
export const PRED1 = softmax(Y1);