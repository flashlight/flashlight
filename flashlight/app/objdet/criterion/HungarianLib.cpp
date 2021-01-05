#include <limits>
#include <assert.h>
#include <vector>

namespace {

enum Mark: int {
  None = 0,
  Star = 1,
  Prime = 2
};


void findUncoveredZero(float* costs, int* colCover, int* rowCover, int nrows, int ncols, 
    int* row, int* col) {
  bool done = false;
  *row = -1;
  *col = -1;
  for(int c = 0; c < ncols && !done; c++) {
    for(int r = 0; r < nrows && !done; r++) {
      const float cost = costs[c * nrows + r];
      if (cost == 0 && colCover[c] == 0 && rowCover[r] == 0) {
        *row = r;
        *col = c;
        done = true;
      }
    }
  }
}

bool isStarInRow(int* marks, int row, int nrows, int ncols) {
  for(int c = 0; c < ncols; c++) {
    if(marks[c * nrows + row] == Mark::Star) {
      return true;
    };
  }
  return false;
}

int findStarInRow(int* marks, int row, int nrows, int ncols) {
  for(int c = 0; c < ncols; c++) {
    if(marks[c * nrows + row] == Mark::Star) {
      return c;
    };
  }
  return -1;
}

// M x N matrix M = nrows, N = ncols
// For each row, substract it's minimum value
int step_one(float* costs, const int nrows, const int ncols) {
  for (int r = 0; r < nrows; r++) {
    float min_val = std::numeric_limits<float>::max();
    for (int c = 0; c < ncols; c++) {
      float val = costs[c * nrows + r];
      if (val < min_val) {
        min_val = val;
      }
    }
    for (int c = 0; c < ncols; c++) {
      costs[c * nrows + r] -= min_val;
    }
  }
  return 2;
}

// Iterate through rows, and mark 0s with '1' (Star) if they have not already been covered by a previous marking
int step_two(float* costs, int* marks, int* colCover, int* rowCover, const int nrows, const int ncols) {
  for(int r = 0; r < nrows; r++) {
    for(int c = 0; c < ncols; c++) {
      float cost = costs[c * nrows + r];
      if (cost == 0.0 && rowCover[r] == 0 && colCover[c] == 0) {
        marks[c * nrows + r] = Mark::Star;
        rowCover[r] = 1;
        colCover[c] = 1;
      }
    }
  }
  for(int r = 0; r < nrows; r++) {
    rowCover[r] = 0;
  }
  for(int c = 0; c < ncols; c++) {
    colCover[c] = 0;
  }
  return 3;
}

// Count the number of lines needed to cover all "Stars"
int step_three(int* marks, int* colCover, int* rowCover, int nrows, int ncols) {
  for(int r = 0; r < nrows; r++) {
    for(int c = 0; c < ncols; c++) {
      const int mark = marks[c * nrows + r];
      if (mark == 1) {
        colCover[c] = 1;
      }
    }
  }
  int coveredCols = 0;
  for(int c = 0; c < ncols; c++) {
    coveredCols += colCover[c];
  }
  if(coveredCols == ncols || coveredCols >= nrows) {
    return 7;
  } else {
    return 4;
  }
}

// Find a noncovered zero and "prime it". If there are no uncovered zeros in the row containing
// this zero, go to 5. Otherwise, cover this row, and uncovered column containing starred zero.
// Continue until there are no more uncovered zeros left. Then go to 6. 
int step_four(float* costs, int* marks, int* colCover, int* rowCover, int nrows, int ncols, int* firstPathRow,
    int* firstPathCol) {
  bool done = false;
  while(!done) {
    int row, col;
    findUncoveredZero(costs, colCover, rowCover, nrows, ncols, &row, &col);
    if (row < 0 && col < 0) {
      return 6;
    }
    else {
      // "Prime it"
      marks[col * nrows + row] = Mark::Prime;
      if(isStarInRow(marks, row, nrows, ncols)) {
        int col = findStarInRow(marks, row, nrows, ncols);
        rowCover[row] = 1;
        colCover[col] = 0;
      } else {
        *firstPathRow = row;
        *firstPathCol = col;
        done = true;
        return 5;
      }
    }
  }
  return -1;
}

int findStarInCol(int* masks, int col, int nrows, int ncols) {
  for(int r = 0; r < nrows; r++) {
    if(masks[col * nrows + r] == 1) {
      return r;
    }
  }
  return -1;
}

int findPrimeInRow(int* masks, int row, int nrows, int ncols) {
  for(int c = 0; c < ncols; c++) {
    if(masks[c * nrows + row] == 2) {
      return c;
    }
  }
  return -1;

}

void augmentPaths(int* paths, int pathCount, int* marks, int nrows, int ncols) {
  for(int p = 0; p < pathCount; p++) {
    int row = paths[p * 2];
    int col = paths[p * 2 + 1];
    if(marks[col * nrows + row] == Mark::Star) {
      marks[col * nrows + row] = Mark::None;
    } else {
      marks[col * nrows + row] = Mark::Star;
    }
  }
}

void clearCover(int* cover, int n) {
  for(int i = 0; i < n; i++) {
    cover[i] = 0;
  }
}

void erasePrimes(int* marks, int nrows, int ncols) {
  for(int c = 0; c < ncols; c++) {
    for(int r = 0; r < nrows; r++) {
      if(marks[c * nrows + r] == Mark::Prime) {
        marks[c * nrows + r] = Mark::None;
      }
    }
  }
}

int step_five(float* costs, int* marks, int* colCover, int* rowCover, int* path, int firstPathRow, int firstPathCol, int nrows, int ncols) {
  int r = -1;
  int c = -1;
  int pathCount = 1;
  path[(pathCount - 1) * 2] = firstPathRow;
  path[(pathCount - 1) * 2 + 1] = firstPathCol;
  bool done = false;
  while(!done) {
    r = findStarInCol(marks, path[(pathCount - 1) * 2 + 1], nrows, ncols);
    if(r > -1) {
      pathCount += 1;
      path[(pathCount - 1) * 2] = r;
      path[(pathCount - 1) * 2 + 1] = path[(pathCount - 2) * 2 + 1];
    } else {
      done = true;
    }
    if(!done) {
      c = findPrimeInRow(marks, path[(pathCount - 1) * 2], nrows, ncols);
      pathCount += 1;
      path[(pathCount - 1) * 2] = path[(pathCount - 2) * 2];
      path[(pathCount - 1) * 2 + 1] = c;
    }
  }
  augmentPaths(path, pathCount, marks, nrows, ncols);
  clearCover(colCover, ncols);
  clearCover(rowCover, nrows);
  erasePrimes(marks, nrows, ncols);
  return 3;
}

float findSmallestNotCovered(float* costs, int* colCover, int* rowCover, int nrows, int ncols) {
  float minValue = std::numeric_limits<float>::max();
  for(int c = 0; c < ncols; c++) {
    for(int r = 0; r < nrows; r++) {
      if (colCover[c] == 0 && rowCover[r] == 0) {
        const float cost = costs[c * nrows + r];
        if(cost < minValue) {
          minValue = cost;
        }
      }
    }
  }
  return minValue;
}

int step_six(float* costs, int* marks, int* colCover, int* rowCover, int nrows, int ncols) {
  float minVal = findSmallestNotCovered(costs, colCover, rowCover, nrows, ncols);
  for(int c = 0; c < ncols; c++) {
    for(int r = 0; r < nrows; r++) {
      if(rowCover[r] == 1) {
        costs[c * nrows + r] += minVal;
      }
      if (colCover[c] == 0) {
        costs[c * nrows + r] -= minVal;
      }
    }
  }
  return 4;
}

void step_seven(int* marks, int* rowIdxs, int* colIdxs, int M, int N) {
  int i = 0;
  for(int r = 0; r < M; r++) {
    for(int c = 0; c < N; c++) {
      const int mark = marks[c * M + r];
      if(mark == Mark::Star) {
        rowIdxs[i] = r;
        colIdxs[i] = c;
        i += 1;
      }
    }
  }
};

} 

namespace fl {
namespace cv {

void hungarian(float* costs, int* assignments, int M, int N) {
  // Ensure there are more rows than columns
  assert(N >= M);
  std::vector<int> rowCover(M);
  std::vector<int> colCover(N);
  std::vector<int> paths(N * N * 2);
  int firstPathRow, firstPathCol;
  bool done = false;
  int step = 1;
  while(!done) {
    switch(step) {
      case 1:
        step = step_one(costs, M, N);
        break;
      case 2:
        step = step_two(costs, assignments, colCover.data(), rowCover.data(), M, N);
        break;
      case 3:
        step = step_three(assignments, colCover.data(), rowCover.data(), M, N);
        break;
      case 4:
        step = step_four(costs, assignments, colCover.data(), rowCover.data(), M, N, &firstPathRow,
            &firstPathCol);
        break;
      case 5:
        step = step_five(costs, assignments, colCover.data(), rowCover.data(), paths.data(), firstPathRow, firstPathCol, M, N);
        break;
      case 6:
        step = step_six(costs, assignments, colCover.data(), rowCover.data(), M, N);
        break;
      case 7:
        done = true;
        break;
    }
  }
}

void hungarian(float* costs, int* rowIdxs, int* colIdxs, int M, int N) {
  std::vector<int> marks(M * N);
  hungarian(costs, marks.data(), M, N);
  step_seven(marks.data(), rowIdxs, colIdxs, M, N);
}

}
}
