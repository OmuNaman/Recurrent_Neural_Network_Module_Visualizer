// FILE: src/utils/matrixCalculations.ts

// --- RNN Parameters ---
// Our simple vocabulary for the "RNN" example
export const VOCAB = ['R', 'N', '!'];

// One-Hot Vector for the first input character: 'R'
// This represents x_t at timestep t=1
export const ONE_HOT_R = [[1, 0, 0]];


// --- Math Helpers (We'll add more as needed) ---

/**
 * A basic matrix multiplication helper.
 * We'll expand this later.
 */
export function matrixMultiply(a: number[][], b: number[][]): number[][] {
  const resultRows = a.length;
  const resultCols = b[0].length;
  if (a[0].length !== b.length) {
    console.error("Matrix dimensions incompatible for multiplication.");
    return [[]];
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